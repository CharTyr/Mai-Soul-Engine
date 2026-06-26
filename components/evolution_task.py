"""演化任务组件 — 周期性分析群聊内容并调整光谱。

周期性地从监控群组获取消息，调用 LLM 分析内容倾向，
根据分析结果调整意识形态光谱四维数值。
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from datetime import datetime
from typing import Any

from ..models.ideology_model import (
    create_evolution_history,
    get_or_create_group_evolution,
    get_or_create_spectrum,
)
from ..prompts.ideology_prompts import EVOLUTION_ANALYSIS_PROMPT
from ..utils.audit_log import log_evolution
from ..utils.spectrum_utils import (
    apply_resistance,
    chat_config_to_stream_id,
    is_user_monitored,
    parse_chat_id,
    match_chat,
    sanitize_text,
    smooth_delta,
    update_spectrum_value,
)

logger = logging.getLogger(__name__)


async def run_evolution_loop(plugin) -> None:
    """周期性演化任务主循环。

    由 plugin._evolution_loop() 以 asyncio.create_task 启动。
    循环每 interval_hours 执行一轮分析，遍历所有监控群组。

    Args:
        plugin: MaiSoulEnginePlugin 实例。
    """
    logger.debug("演化循环已启动")

    while True:
        try:
            interval_hours = float(plugin.config.evolution.evolution_interval_hours or 1.0)
            logger.debug("演化循环等待 %s 小时", interval_hours)
            await asyncio.sleep(interval_hours * 3600)

            if not plugin.config.evolution.evolution_enabled:
                logger.debug("演化已禁用，跳过本轮")
                continue

            spectrum = get_or_create_spectrum("global")
            if not spectrum.initialized:
                logger.debug("光谱未初始化，跳过本轮")
                continue

            evolution_rate = int(plugin.config.evolution.evolution_rate or 5)
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

            for group_config_id in groups_to_analyze:
                logger.debug("开始分析群组: %s", group_config_id)
                await _analyze_group(plugin, group_config_id, evolution_rate)

        except asyncio.CancelledError:
            logger.info("灵魂光谱演化任务已停止")
            break
        except Exception as e:
            logger.error("灵魂光谱演化任务出错: %s", e, exc_info=True)
            await asyncio.sleep(60)


async def _analyze_group(plugin, group_config_id: str, evolution_rate: int) -> None:
    """分析单个群组的消息并更新光谱。

    Args:
        plugin: 插件实例。
        group_config_id: 群组配置 ID（如 "qq:12345678:group"）。
        evolution_rate: 单次演化最大变化值。
    """
    try:
        stream_id = chat_config_to_stream_id(group_config_id)

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
        except Exception:
            logger.exception("获取群%s消息失败", stream_id)
            return

        # 新 SDK 返回的消息列表，每条是 dict
        if not isinstance(messages_raw, list):
            messages_raw = []
        # 过滤掉命令消息（以 / 开头）且长度不足 5 条则跳过
        messages = [m for m in messages_raw if not str(m.get("processed_plain_text", "") or "").strip().startswith("/")]
        logger.debug("群%s获取消息: %s条, 时间范围: %s - %s", stream_id, len(messages), last_time, now)

        if len(messages) < 5:
            logger.debug("群%s消息不足5条，跳过分析", stream_id)
            return

        max_messages = int(plugin.config.evolution.max_messages_per_analysis or 200)
        max_chars = int(plugin.config.evolution.max_chars_per_message or 200)

        monitor_config = {
            "monitored_users": list(plugin.config.monitor.monitored_users or []),
            "excluded_users": list(plugin.config.monitor.excluded_users or []),
        }

        filtered_messages = []
        for m in messages:
            user_info = m.get("user_info", {}) if isinstance(m, dict) else {}
            msg_platform = str(user_info.get("platform", "") or "")
            msg_user_id = str(user_info.get("user_id", "") or "")

            # 跳过不受监控的用户
            if not is_user_monitored(msg_platform, msg_user_id, monitor_config):
                continue

            filtered_messages.append(m)

        messages = filtered_messages
        if len(messages) < 5:
            logger.debug("群%s过滤后消息不足5条，跳过分析", stream_id)
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
            return

        prompt = EVOLUTION_ANALYSIS_PROMPT.format(rate=evolution_rate, messages=msg_text)

        thought_cabinet_enabled = bool(plugin.config.thought_cabinet.enabled)
        logger.debug("思维阁启用状态: %s", thought_cabinet_enabled)
        if thought_cabinet_enabled:
            from ..prompts.thought_prompts import ENHANCED_EVOLUTION_PROMPT

            prompt = ENHANCED_EVOLUTION_PROMPT.format(rate=evolution_rate, messages=msg_text)

        # 调用新 SDK 的 LLM 接口
        logger.debug("发送LLM请求，prompt长度: %s", len(prompt))
        try:
            llm_result = await plugin.ctx.llm.generate(prompt)
        except Exception:
            logger.exception("LLM 请求失败")
            return

        response = ""
        if isinstance(llm_result, dict):
            response = str(llm_result.get("response", "") or "")
        elif isinstance(llm_result, str):
            response = llm_result
        logger.debug("LLM响应长度: %s", len(response))

        if not response:
            return

        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1].rsplit("```", 1)[0]
            result = _json.loads(response)

            if thought_cabinet_enabled and "spectrum_deltas" in result:
                deltas = result["spectrum_deltas"]
                thought_seeds = result.get("thought_seeds", [])
                await _process_thought_seeds(plugin, thought_seeds, stream_id)
            else:
                deltas = result
        except (_json.JSONDecodeError, ValueError):
            logger.warning("无法解析LLM响应: %s", response)
            return

        spectrum = get_or_create_spectrum("global")

        before = {
            "economic": spectrum.economic,
            "social": spectrum.social,
            "diplomatic": spectrum.diplomatic,
            "progressive": spectrum.progressive,
        }

        ema_alpha = float(plugin.config.evolution.ema_alpha or 0.3)
        resistance = float(plugin.config.evolution.direction_resistance or 0.5)

        from ..worldview.service import WorldviewService, config_from_plugin

        wv = WorldviewService(config_from_plugin(plugin))
        raw_deltas = wv.apply_layer_caps_to_deltas(
            {
                "economic": int(deltas.get("economic", 0) or 0),
                "social": int(deltas.get("social", 0) or 0),
                "diplomatic": int(deltas.get("diplomatic", 0) or 0),
                "progressive": int(deltas.get("progressive", 0) or 0),
            },
            evolution_rate,
        )

        resisted_deltas = {}
        new_dirs = {}
        for dim in ["economic", "social", "diplomatic", "progressive"]:
            last_dir = int(getattr(spectrum, f"last_{dim}_dir", 0))
            adj_delta, new_dir = apply_resistance(raw_deltas[dim], last_dir, resistance)
            resisted_deltas[dim] = adj_delta
            new_dirs[dim] = new_dir

        smoothed_deltas = {
            "economic": smooth_delta(spectrum.economic, resisted_deltas["economic"], ema_alpha),
            "social": smooth_delta(spectrum.social, resisted_deltas["social"], ema_alpha),
            "diplomatic": smooth_delta(spectrum.diplomatic, resisted_deltas["diplomatic"], ema_alpha),
            "progressive": smooth_delta(spectrum.progressive, resisted_deltas["progressive"], ema_alpha),
        }

        spectrum.economic = update_spectrum_value(spectrum.economic, smoothed_deltas["economic"])
        spectrum.social = update_spectrum_value(spectrum.social, smoothed_deltas["social"])
        spectrum.diplomatic = update_spectrum_value(spectrum.diplomatic, smoothed_deltas["diplomatic"])
        spectrum.progressive = update_spectrum_value(spectrum.progressive, smoothed_deltas["progressive"])
        spectrum.last_economic_dir = new_dirs["economic"]
        spectrum.last_social_dir = new_dirs["social"]
        spectrum.last_diplomatic_dir = new_dirs["diplomatic"]
        spectrum.last_progressive_dir = new_dirs["progressive"]
        spectrum.last_evolution = now
        spectrum.updated_at = now
        spectrum.save()

        after = {
            "economic": spectrum.economic,
            "social": spectrum.social,
            "diplomatic": spectrum.diplomatic,
            "progressive": spectrum.progressive,
        }

        create_evolution_history(
            timestamp=now,
            group_id=stream_id,
            economic_delta=smoothed_deltas["economic"],
            social_delta=smoothed_deltas["social"],
            diplomatic_delta=smoothed_deltas["diplomatic"],
            progressive_delta=smoothed_deltas["progressive"],
            reason=f"分析了{len(messages)}条消息",
        )

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

        record.last_analyzed = now
        record.save()

        logger.info(
            "群%s演化完成: e=%s, s=%s, d=%s, p=%s",
            stream_id,
            smoothed_deltas["economic"],
            smoothed_deltas["social"],
            smoothed_deltas["diplomatic"],
            smoothed_deltas["progressive"],
        )

    except Exception as e:
        logger.error("分析群%s时出错: %s", group_config_id, e, exc_info=True)


async def _process_thought_seeds(plugin, seeds: list, stream_id: str) -> None:
    """处理 LLM 返回的思维种子。"""
    from ..thought.seed_manager import ThoughtSeedManager

    logger.debug("处理思维种子: 收到 %s 个", len(seeds))
    if not seeds:
        return

    config = {
        "max_seeds": int(plugin.config.thought_cabinet.max_seeds or 20),
        "min_trigger_intensity": float(plugin.config.thought_cabinet.min_trigger_intensity or 0.7),
        "admin_user_id": str(plugin.config.admin.admin_user_id or ""),
    }
    manager = ThoughtSeedManager(config)

    for seed_data in seeds[:2]:
        seed_id = await manager.create_seed(seed_data, stream_id=stream_id)
        if seed_id:
            logger.info("群%s创建思维种子: %s", stream_id, seed_id)
            if plugin.config.thought_cabinet.admin_notification_enabled:
                await _notify_admin_seed(plugin, manager, seed_id, seed_data)


async def _notify_admin_seed(plugin, manager, seed_id: str, seed_data: dict) -> None:
    """向管理员私聊发送思维种子通知。"""
    from ..utils.spectrum_utils import parse_user_id

    admin_config_id = str(plugin.config.admin.admin_user_id or "")
    if not admin_config_id:
        return

    platform, user_id = parse_user_id(admin_config_id)
    if not platform or not user_id:
        return

    # 通过新 SDK 的 chat API 获取管理员的 stream_id
    try:
        admin_stream_id = await plugin.ctx.chat.get_stream_by_user_id(
            platform=platform, user_id=user_id
        )
    except Exception:
        logger.exception("获取管理员 stream_id 失败，无法发送种子通知")
        return

    if not admin_stream_id:
        logger.warning("未找到管理员的聊天流，无法发送种子通知")
        return

    try:
        await plugin.ctx.send.text(
            text=manager.format_seed_notification(seed_id, seed_data),
            stream_id=admin_stream_id,
        )
    except Exception:
        logger.exception("发送思维种子通知失败")
