"""演化任务组件 — 周期性分析群聊内容并调整光谱。

周期性地从监控群组获取消息，调用 LLM 分析内容倾向，
根据分析结果调整意识形态光谱四维数值。
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import sqlite3
from datetime import datetime
from typing import Any

from ..models.ideology_model import (
    apply_spectrum_deltas,
    get_or_create_group_evolution,
    get_or_create_spectrum,
)
from ..prompts.thought_prompts import ENHANCED_EVOLUTION_PROMPT
from ..utils.audit_log import log_evolution, log_evolution_cycle, log_evolution_skip
from ..utils.runtime_resolution import (
    generate_soul_text,
    resolve_host_bot_self_ids,
    resolve_monitored_group_stream,
)
from ..utils.spectrum_utils import (
    filter_messages_for_evolution,
    parse_chat_id,
    match_chat,
    sanitize_text,
)

logger = logging.getLogger(__name__)

_bot_filter_warned: set[str] = set()
# 聚合种子通知冷却：记录上次通知时间戳（秒），用于 cooldown 检查
_last_aggregated_notification_ts: float = 0.0
# 连续演化失败计数器 — 超阈值时私信管理员
_consecutive_evolution_failures: int = 0
# 本轮演化产生的种子通知收集列表（每轮开始时清空）
# 多群并行时多个 _analyze_group 协程同时 append，用 asyncio.Lock 保护
_pending_seed_notifications: list[tuple[str, str, str]] = []
_pending_seed_lock = asyncio.Lock()


def reset_aggregation_state() -> None:
    """重置聚合通知的模块级状态，供 on_unload 调用防插件重载间泄漏。"""
    global _last_aggregated_notification_ts, _consecutive_evolution_failures
    _last_aggregated_notification_ts = 0.0
    _consecutive_evolution_failures = 0
    _pending_seed_notifications.clear()


def _warn_no_bot_filter(stream_id: str) -> None:
    """每个群只警告一次：bot 自身账号与 excluded_users 都未配置，自消息会污染演化池。"""
    if stream_id in _bot_filter_warned:
        return
    _bot_filter_warned.add(stream_id)
    logger.warning(
        "群%s：宿主 bot.qq_account 为空且 excluded_users 未配置，bot 自身消息可能混入演化分析池"
        "导致人设自指。建议在配置中填入 bot 账号（格式 平台:ID，如 qq:12345678）。",
        stream_id,
    )


async def run_evolution_loop(plugin) -> None:
    """周期性演化任务主循环。

    由 plugin._evolution_loop() 以 asyncio.create_task 启动。
    循环每 interval_hours 执行一轮分析，遍历所有监控群组。

    Args:
        plugin: MaiSoulEnginePlugin 实例。
    """
    logger.debug("演化循环已启动")
    global _consecutive_evolution_failures
    global _consecutive_evolution_failures

    while True:
        try:
            interval_hours = plugin.config.evolution.evolution_interval_hours
            logger.debug("演化循环等待 %s 小时", interval_hours)
            await asyncio.sleep(interval_hours * 3600)

            if not plugin.config.evolution.evolution_enabled:
                logger.debug("演化已禁用，跳过本轮")
                continue

            spectrum = get_or_create_spectrum("global")
            if not spectrum.initialized:
                logger.debug("光谱未初始化，跳过本轮")
                continue

            # P0-4：过期长期未强化的 active trait
            trait_ttl_days = plugin.config.thought_cabinet.trait_ttl_days
            if trait_ttl_days > 0:
                from ..models.ideology_model import expire_old_traits

                expired = expire_old_traits(trait_ttl_days)
                if expired:
                    logger.info("过期 %s 个超龄 active trait (TTL=%s天)", expired, trait_ttl_days)

            evolution_rate = plugin.config.evolution.evolution_rate
            monitored_groups = list(plugin.config.monitor.monitored_groups or [])
            excluded_groups = list(plugin.config.monitor.excluded_groups or [])
            logger.debug("演化参数: rate=%s, groups=%s", evolution_rate, monitored_groups)

            if not monitored_groups:
                logger.debug("无监控群组，跳过本轮")
                continue

            groups_to_analyze: list[str] = []
            for group_config_id in monitored_groups:
                platform, chat_id, chat_type = parse_chat_id(str(group_config_id))
                if any(match_chat(platform, chat_id, chat_type, str(exc)) for exc in excluded_groups):
                    logger.debug("群组已在排除列表中，跳过: %s", group_config_id)
                    continue
                groups_to_analyze.append(str(group_config_id))

            if not groups_to_analyze:
                logger.debug("监控群组全部被排除，跳过本轮")
                continue

            analyzed = 0
            skipped = 0
            seeds_before = 0
            try:
                from ..models.ideology_model import count_pending_thought_seeds

                seeds_before = int(count_pending_thought_seeds() or 0)
            except (sqlite3.Error, ValueError, TypeError):
                seeds_before = 0

            # 多群并行分析（Semaphore 限流防 LLM 限流）
            max_concurrent = int(getattr(plugin.config.evolution, "max_concurrent_groups", 3) or 3)
            semaphore = asyncio.Semaphore(max(1, max_concurrent))

            async def _analyze_with_sem(gid: str) -> bool:
                async with semaphore:
                    logger.debug("开始分析群组: %s", gid)
                    await _analyze_group(plugin, gid, evolution_rate)
                    return True

            results = await asyncio.gather(
                *[_analyze_with_sem(g) for g in groups_to_analyze],
                return_exceptions=True,
            )
            analyzed = sum(1 for r in results if r is True)

            seeds_after = seeds_before
            try:
                from ..models.ideology_model import count_pending_thought_seeds

                seeds_after = int(count_pending_thought_seeds() or 0)
            except (sqlite3.Error, ValueError, TypeError):
                pass

            await log_evolution_cycle(
                groups_planned=len(groups_to_analyze),
                groups_analyzed=analyzed,
                groups_skipped=max(0, len(groups_to_analyze) - analyzed),
                seeds_created=max(0, seeds_after - seeds_before),
                interval_hours=float(getattr(plugin.config.evolution, "evolution_interval_hours", 0) or 0),
            )

            # 演化成功，重置连续失败计数器
            _consecutive_evolution_failures = 0

            # U-UX-6: 聚合种子通知 — 本轮所有新种子合并为一条通知发送给管理员
            if (
                plugin.config.thought_cabinet.admin_notification_enabled
                and _pending_seed_notifications
            ):
                try:
                    await _send_aggregated_seed_notification(plugin)
                except (RuntimeError, ValueError, OSError):
                    logger.exception("[SeedNotify] 聚合通知发送失败")
                finally:
                    _pending_seed_notifications.clear()

            # P1.5：自评反馈 → 光谱修正（仅 self_reflection.enabled）
            if plugin.config.self_reflection.enabled:
                try:
                    from .reflection_feedback import apply_self_reflection_spectrum_correction

                    apply_self_reflection_spectrum_correction(plugin, evolution_rate)
                except Exception:
                    logger.exception("[SelfReflection] 光谱修正失败（apply_self_reflection_spectrum_correction 内部异常类型不确定，保留兜底）")

        except asyncio.CancelledError:
            logger.info("灵魂光谱演化任务已停止")
            break
        # 顶层兜底：确保演化循环不因意外异常退出，已 log+exc_info
        except Exception as e:
            _consecutive_evolution_failures += 1
            logger.error(
                "灵魂光谱演化任务出错 (连续失败 %s 次): %s",
                _consecutive_evolution_failures, e, exc_info=True,
            )
            if _consecutive_evolution_failures >= 5:
                try:
                    admin_id = plugin.config.admin.admin_user_id
                    if admin_id:
                        from ..utils.spectrum_utils import parse_user_id

                        platform, user_id = parse_user_id(admin_id)
                        stream = await plugin.ctx.chat.get_stream_by_user_id(
                            platform=platform, user_id=user_id
                        )
                        if stream:
                            stream_id = stream.get("stream_id", "") if isinstance(stream, dict) else str(stream)
                            if stream_id:
                                await plugin.ctx.send.text(
                                    f"⚠️ 演化任务已连续失败 {_consecutive_evolution_failures} 次，请检查日志。",
                                    stream_id,
                                )
                except (RuntimeError, ValueError, OSError):
                    logger.exception("发送演化失败通知给管理员时出错")
                _consecutive_evolution_failures = 0
            await asyncio.sleep(60)


async def _analyze_group(plugin, group_config_id: str, evolution_rate: int) -> None:
    """分析单个群组的消息并更新光谱。

    Args:
        plugin: 插件实例。
        group_config_id: 群组配置 ID（如 "qq:12345678:group"）。
        evolution_rate: 单次演化最大变化值。
    """
    try:
        stream_id = await resolve_monitored_group_stream(plugin, group_config_id)
        if not stream_id:
            logger.warning("监控群无法解析到当前宿主会话: %s", group_config_id)
            await log_evolution_skip(group_config_id, "stream_not_found")
            return

        record = get_or_create_group_evolution(group_id=stream_id)
        last_time = record.last_analyzed
        now = datetime.now()

        # 从新 SDK 的 message API 获取消息
        try:
            messages_raw = await plugin.ctx.message.get_by_time_in_chat(
                chat_id=stream_id,
                start_time=str(last_time.timestamp()),
                end_time=str(now.timestamp()),
            )
        except (RuntimeError, ValueError, OSError) as exc:
            logger.exception("获取群%s消息失败", stream_id)
            await log_evolution_skip(stream_id, "fetch_messages_failed", detail=str(exc))
            return

        # 新 SDK 返回的消息列表，每条是 dict
        if not isinstance(messages_raw, list):
            messages_raw = []
        # 过滤掉命令消息（以 / 开头）且长度不足 5 条则跳过
        messages = [m for m in messages_raw if not str(m.get("processed_plain_text", "") or "").strip().startswith("/")]
        logger.debug("群%s获取消息: %s条, 时间范围: %s - %s", stream_id, len(messages), last_time, now)

        if len(messages) < 5:
            logger.debug("群%s消息不足5条，跳过分析", stream_id)
            await log_evolution_skip(stream_id, "messages_lt_5", message_count=len(messages))
            return

        max_messages = plugin.config.evolution.max_messages_per_analysis
        max_chars = plugin.config.evolution.max_chars_per_message

        # Bot identity is owned by the Host (bot.qq_account), never duplicated
        # in plugin configuration.
        host_bot_self_ids = await resolve_host_bot_self_ids(plugin)
        monitor_config = {
            "monitored_users": list(plugin.config.monitor.monitored_users or []),
            "excluded_users": list(plugin.config.monitor.excluded_users or []),
            "bot_self_id": host_bot_self_ids,
        }

        # 过滤发言者：bot 自身消息短路排除（防自指泄漏），再过 monitored/excluded
        messages = filter_messages_for_evolution(messages, monitor_config)

        # 配置卫生提醒：bot 自身账号与排除列表都为空时，自消息会污染演化池
        if not monitor_config["bot_self_id"] and not monitor_config["excluded_users"]:
            _warn_no_bot_filter(stream_id)
        if len(messages) < 5:
            logger.debug("群%s过滤后消息不足5条，跳过分析", stream_id)
            await log_evolution_skip(stream_id, "filtered_messages_lt_5", message_count=len(messages))
            return

        msg_lines = []
        for m in messages[:max_messages]:
            user_info = m.get("user_info", {}) if isinstance(m, dict) else {}
            nickname = (
                str(user_info.get("user_cardname", "") or "")
                or str(user_info.get("user_nickname", "") or "")
                or str(user_info.get("user_id", "") or "")
                or "user"
            )
            content = str(m.get("processed_plain_text", "") or "") or str(m.get("display_message", "") or "") or ""
            sanitized = sanitize_text(str(content), max_chars=max_chars)
            if sanitized:
                msg_lines.append(f"{nickname}: {sanitized}")
        msg_text = "\n".join(msg_lines)

        if not msg_text:
            logger.debug("群%s消息内容为空，跳过分析", stream_id)
            await log_evolution_skip(stream_id, "empty_message_text", message_count=len(messages))
            return

        thought_cabinet_enabled = bool(plugin.config.thought_cabinet.enabled)
        logger.debug("思维阁启用状态: %s", thought_cabinet_enabled)
        prompt = ENHANCED_EVOLUTION_PROMPT.format(rate=evolution_rate, messages=msg_text)

        # 调用新 SDK 的 LLM 接口
        logger.debug("发送LLM请求，prompt长度: %s", len(prompt))
        try:
            llm_result = await generate_soul_text(plugin, prompt)
        except (RuntimeError, ValueError, OSError, asyncio.TimeoutError) as exc:
            logger.exception("LLM 请求失败")
            await log_evolution_skip(
                stream_id, "llm_failed", message_count=len(messages), detail=str(exc)
            )
            return

        response = ""
        if isinstance(llm_result, dict):
            response = str(llm_result.get("response", "") or "")
        elif isinstance(llm_result, str):
            response = llm_result
        logger.debug("LLM响应长度: %s", len(response))

        if not response:
            await log_evolution_skip(stream_id, "llm_empty_response", message_count=len(messages))
            return

        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1].rsplit("```", 1)[0]
            result = _json.loads(response)

            if "spectrum_deltas" in result:
                deltas = result["spectrum_deltas"]
                if thought_cabinet_enabled:
                    thought_seeds = result.get("thought_seeds", [])
                    await _process_thought_seeds(plugin, thought_seeds, stream_id, msg_lines)
            else:
                deltas = result
        except (_json.JSONDecodeError, ValueError):
            logger.warning("无法解析LLM响应: %s", response)
            await log_evolution_skip(
                stream_id,
                "llm_parse_failed",
                message_count=len(messages),
                detail=(response or "")[:240],
            )
            return

        spectrum = get_or_create_spectrum("global")

        before = {
            "sincerity": spectrum.sincerity,
            "engagement": spectrum.engagement,
            "closeness": spectrum.closeness,
            "directness": spectrum.directness,
        }

        ema_alpha = plugin.config.evolution.ema_alpha
        resistance = plugin.config.evolution.direction_resistance

        from ..worldview.service import WorldviewService, config_from_plugin

        wv = WorldviewService(config_from_plugin(plugin))
        raw_deltas = wv.apply_layer_caps_to_deltas(
            {
                "sincerity": int(deltas.get("sincerity", 0) or 0),
                "engagement": int(deltas.get("engagement", 0) or 0),
                "closeness": int(deltas.get("closeness", 0) or 0),
                "directness": int(deltas.get("directness", 0) or 0),
            },
            evolution_rate,
        )

        # 经统一光谱闸门写入（v2.3.0 收口：resistance + EMA + save + history）
        smoothed_deltas = apply_spectrum_deltas(
            "evolution",
            raw_deltas,
            smooth_alpha=ema_alpha,
            resistance=resistance,
            max_per_axis=evolution_rate,
            group_id=stream_id,
            reason=f"分析了{len(messages)}条消息",
        )
        spectrum = get_or_create_spectrum("global")

        after = {
            "sincerity": spectrum.sincerity,
            "engagement": spectrum.engagement,
            "closeness": spectrum.closeness,
            "directness": spectrum.directness,
        }

        await log_evolution(
            group_id=stream_id,
            before=before,
            after=after,
            deltas=smoothed_deltas,
            reason=f"分析了{len(messages)}条消息",
            message_count=len(messages),
        )

        wv.record_local_slice(stream_id, smoothed_deltas, len(messages))
        wv.nudge_mood_from_deltas(smoothed_deltas)

        # P-EVO-1b: 极值告警 — 光谱任一轴在极区间时 log.warning
        spectrum = get_or_create_spectrum("global")
        for dim in ("sincerity", "engagement", "closeness", "directness"):
            val = int(getattr(spectrum, dim))
            if val <= 10 or val >= 90:
                logger.warning(
                    "[SpectrumGuard] %s 极值告警: %s=%d（可能跑偏）",
                    stream_id, dim, val,
                )

        record.last_analyzed = now
        record.save()

        logger.info(
            "群%s演化完成: 真诚=%s, 投入=%s, 亲近=%s, 直率=%s",
            stream_id,
            smoothed_deltas["sincerity"],
            smoothed_deltas["engagement"],
            smoothed_deltas["closeness"],
            smoothed_deltas["directness"],
        )

    # 顶层兜底：单个群分析失败不阻断其他群
    except Exception as e:
        logger.error("分析群%s时出错: %s", group_config_id, e, exc_info=True)


async def _process_thought_seeds(plugin, seeds: list, stream_id: str, msg_lines: list[str]) -> list[str]:
    """处理 LLM 返回的思维种子。

    Args:
        msg_lines: 发送给 LLM 的原始消息行列表，用于提取上下文窗口。

    Returns:
        本轮创建的种子 ID 列表。
    """
    from ..thought.seed_manager import ThoughtSeedManager

    logger.debug("处理思维种子: 收到 %s 个", len(seeds))
    if not seeds:
        return []

    manager = ThoughtSeedManager.from_plugin_config(plugin)
    created_ids: list[str] = []

    for seed_data in seeds[:2]:
        seed_id = await manager.create_seed(seed_data, stream_id=stream_id, context_messages=msg_lines)
        if seed_id:
            logger.info("群%s创建思维种子: %s", stream_id, seed_id)
            created_ids.append(seed_id)
            # 收集到聚合通知列表（不再单独通知）
            if plugin.config.thought_cabinet.admin_notification_enabled:
                async with _pending_seed_lock:
                    _pending_seed_notifications.append(
                        (seed_id, seed_data.get("type", "未知"), seed_data.get("event", "")[:80])
                    )

    return created_ids


async def notify_admin_seed(plugin, manager, seed_id: str) -> bool:
    """向管理员私聊发送思维种子通知（含原始对话上下文）。

    Returns True when a notification text was handed to send.text.
    """
    from ..utils.spectrum_utils import parse_user_id

    admin_config_id = plugin.config.admin.admin_user_id
    if not admin_config_id:
        logger.warning("admin_user_id 未配置，跳过种子通知 seed=%s", seed_id)
        return False

    platform, user_id = parse_user_id(admin_config_id)
    if not platform or not user_id:
        logger.warning("admin_user_id 无法解析，跳过种子通知 seed=%s raw=%s", seed_id, admin_config_id)
        return False

    # 从数据库取存储的种子数据（含上下文窗口）
    seed_data = await manager.get_seed_by_id(seed_id)
    if not seed_data:
        logger.warning("无法找到种子 %s，跳过通知", seed_id)
        return False

    # 通过新 SDK 的 chat API 获取管理员的 stream_id
    try:
        admin_stream_id = await plugin.ctx.chat.get_stream_by_user_id(
            platform=platform, user_id=user_id
        )
    except (RuntimeError, ValueError, OSError):
        logger.exception("获取管理员 stream_id 失败，无法发送种子通知 seed=%s", seed_id)
        return False

    if not admin_stream_id:
        logger.warning(
            "未找到管理员聊天流，无法发送种子通知 seed=%s admin=%s:%s",
            seed_id,
            platform,
            user_id,
        )
        return False

    try:
        await plugin.ctx.send.text(
            text=manager.format_seed_notification(seed_id, seed_data),
            stream_id=admin_stream_id,
        )
        logger.info("已发送思维种子通知 seed=%s admin_stream=%s", seed_id, admin_stream_id)
        return True
    except (RuntimeError, ValueError, OSError):
        logger.exception("发送思维种子通知失败 seed=%s", seed_id)
        return False


# Backward-compatible private alias used by this module.
async def _notify_admin_seed(plugin, manager, seed_id: str) -> bool:
    return await notify_admin_seed(plugin, manager, seed_id)


async def _send_aggregated_seed_notification(plugin) -> bool:
    """向管理员发送本轮聚合种子通知。

    将 _pending_seed_notifications 中收集的种子合并为一条通知，
    受 admin_notification_cooldown_minutes 冷却控制。
    """
    import time as _time

    from ..utils.spectrum_utils import parse_user_id

    global _last_aggregated_notification_ts

    cooldown = plugin.config.thought_cabinet.admin_notification_cooldown_minutes
    now = _time.time()
    if cooldown > 0 and _last_aggregated_notification_ts > 0:
        elapsed = (now - _last_aggregated_notification_ts) / 60.0
        if elapsed < cooldown:
            logger.debug(
                "聚合种子通知冷却中（已过 %.1f / %s 分钟），跳过本轮",
                elapsed, cooldown,
            )
            return False

    admin_config_id = plugin.config.admin.admin_user_id
    if not admin_config_id:
        logger.warning("admin_user_id 未配置，跳过聚合种子通知")
        return False

    platform, user_id = parse_user_id(admin_config_id)
    if not platform or not user_id:
        logger.warning("admin_user_id 无法解析，跳过聚合种子通知 raw=%s", admin_config_id)
        return False

    try:
        admin_stream_id = await plugin.ctx.chat.get_stream_by_user_id(
            platform=platform, user_id=user_id
        )
    except (RuntimeError, ValueError, OSError):
        logger.exception("获取管理员 stream_id 失败，跳过聚合种子通知")
        return False

    if not admin_stream_id:
        logger.warning("未找到管理员聊天流，跳过聚合种子通知 admin=%s:%s", platform, user_id)
        return False

    total = len(_pending_seed_notifications)
    display = _pending_seed_notifications[:10]
    lines: list[str] = []
    for sid, stype, sevent in display:
        lines.append(f"• {stype}：{sevent}（/soul_seed {sid}）")
    if total > 10:
        lines.append(f"…等共 {total} 个")

    text = (
        f"🧠 本轮演化产生 {total} 个新思维种子：\n"
        + "\n".join(lines)
        + "\n\n用 /soul_seed <ID> 查看详情，/soul_approve <ID> 批准内化。"
    )

    try:
        await plugin.ctx.send.text(text=text, stream_id=admin_stream_id)
        _last_aggregated_notification_ts = now
        logger.info("已发送聚合种子通知（%s 个） admin_stream=%s", total, admin_stream_id)
        return True
    except (RuntimeError, ValueError, OSError):
        logger.exception("发送聚合种子通知失败（%s 个）", total)
        return False
