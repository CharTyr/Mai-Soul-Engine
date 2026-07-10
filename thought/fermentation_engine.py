"""发酵引擎 — 种子批准后持续收集群聊输入，到期后触发最终内化。

v2.4.0 新增。受 ``thought_cabinet.fermentation_enabled`` 控制。

工作流：
1. ``run_fermentation_loop`` 异步协程，每 ``fermentation_check_interval_minutes`` 分钟执行一轮
2. 每轮扫描所有 ``status='fermenting'`` 的种子
3. 对每个种子取所在群的新消息，用 L1 关键词过滤 + L3 LLM 批量判断关联度
4. 关联度 ≥ 阈值的消息存入 ``soul_fermentation_inputs``
5. 检查发酵窗口是否到期 + 最小输入数
6. 到期且有足够输入 → 触发最终内化
7. 到期但输入不足 → 延长窗口（最多 ``fermentation_max_extensions`` 次）
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any

from ..models._conn import _dt_to_str, _str_to_dt
from ..utils.runtime_resolution import generate_soul_text
from ..prompts.fermentation_prompts import (
    FERMENTATION_RELEVANCE_PROMPT,
    FERMENTED_INTERNALIZATION_PROMPT,
)

logger = logging.getLogger(__name__)

# 关键词过滤：L1 预筛的最低相似度（低于此值直接跳过，不送 LLM）
_KEYWORD_MATCH_THRESHOLD = 0.15
# 单次取消息的时间窗口上限（防重启后积压大量消息）
_MAX_MESSAGE_FETCH_HOURS = 6
# 单轮发酵检查的 LLM 消息上限
_LLM_BATCH_MAX_MESSAGES = 30


def _extract_keywords(text: str) -> set[str]:
    """从文本中提取关键词（用于 L1 预筛）。

    简单分词：按标点/空格分割，取长度 ≥2 的片段。
    中文连续文本无法有效分词，这里返回原文片段供子串匹配使用。
    """
    if not text:
        return set()
    tokens = re.split(r'[^\w\u4e00-\u9fff]+', text)
    return {t.casefold() for t in tokens if len(t) >= 2}


def _keyword_overlap_score(seed_keywords: set[str], message: str) -> float:
    """计算消息与种子关键词的重叠度。

    对于中文（无空格分词），使用子串匹配：种子关键词作为子串在消息中出现即算命中。
    """
    if not seed_keywords or not message:
        return 0.0
    msg_lower = message.casefold()
    overlap = 0
    for kw in seed_keywords:
        if kw in msg_lower:
            overlap += 1
    if overlap == 0:
        return 0.0
    return overlap / len(seed_keywords)


def _build_seed_keywords(seed: Any) -> set[str]:
    """从种子的 type + event + reasoning 提取关键词集合。"""
    parts = [
        getattr(seed, "seed_type", "") or "",
        getattr(seed, "event", "") or "",
        getattr(seed, "reasoning", "") or "",
    ]
    return _extract_keywords(" ".join(parts))


async def run_fermentation_loop(plugin: Any) -> None:
    """发酵循环主入口。

    在 ``plugin.on_load`` 中以 ``asyncio.create_task`` 启动，
    在 ``plugin.on_unload`` 中 cancel。
    """
    cfg = plugin.config.thought_cabinet
    interval = float(cfg.fermentation_check_interval_minutes) * 60.0
    logger.info("[Fermentation] 发酵循环已启动，间隔 %.0f 分钟", interval / 60.0)

    while True:
        try:
            await asyncio.sleep(interval)
            await _fermentation_cycle(plugin)
        except asyncio.CancelledError:
            logger.info("[Fermentation] 发酵循环被取消")
            raise
        except Exception as e:
            logger.error("[Fermentation] 发酵循环异常: %s", e, exc_info=True)
            await asyncio.sleep(interval)  # 出错后等一轮再试


async def _fermentation_cycle(plugin: Any) -> None:
    """单轮发酵检查。"""
    from ..models.seeds import get_fermenting_seeds

    seeds = get_fermenting_seeds()
    if not seeds:
        return

    logger.debug("[Fermentation] 本轮检查 %d 个发酵中种子", len(seeds))

    for seed in seeds:
        try:
            await _process_fermenting_seed(plugin, seed)
        except Exception as e:
            logger.error("[Fermentation] 处理种子 %s 失败: %s", seed.seed_id, e, exc_info=True)


async def _process_fermenting_seed(plugin: Any, seed: Any) -> None:
    """处理单个发酵中种子：取消息 → L1 过滤 → L3 LLM 判断 → 累积 → 检查到期。"""
    from ..models.seeds import (
        add_fermentation_input,
        count_fermentation_inputs,
        update_fermentation_checked,
        extend_fermentation_window,
    )

    cfg = plugin.config.thought_cabinet
    stream_id = seed.stream_id or ""
    if not stream_id:
        logger.warning("[Fermentation] 种子 %s 无 stream_id，跳过", seed.seed_id)
        return

    # 已达最大输入数，不再收集
    max_inputs = int(cfg.fermentation_max_inputs)
    current_inputs = count_fermentation_inputs(seed.seed_id)
    if current_inputs >= max_inputs:
        logger.debug("[Fermentation] 种子 %s 已达最大输入数 %d", seed.seed_id, max_inputs)
        await _check_completion(plugin, seed)
        return

    # 取上次检查以来的新消息（限制窗口上限防积压）
    checked_at = seed.fermentation_checked_at or seed.fermentation_started_at or datetime.now()
    if isinstance(checked_at, str):
        checked_at = _str_to_dt(checked_at) or datetime.now()

    now = datetime.now()
    # 限制单次取消息的时间窗口
    fetch_start = max(checked_at, now - timedelta(hours=_MAX_MESSAGE_FETCH_HOURS))

    try:
        messages_raw = await plugin.ctx.message.get_by_time_in_chat(
            chat_id=stream_id,
            start_time=str(fetch_start.timestamp()),
            end_time=str(now.timestamp()),
        )
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("[Fermentation] 取种子 %s 群消息失败: %s", seed.seed_id, exc)
        return  # 不更新 checked_at，下轮重试

    if not isinstance(messages_raw, list):
        messages_raw = []

    # 过滤命令消息和 bot 自身消息
    from ..utils.spectrum_utils import filter_messages_for_evolution, sanitize_text
    from ..utils.runtime_resolution import resolve_host_bot_self_ids

    messages = [m for m in messages_raw if not str(m.get("processed_plain_text", "") or "").strip().startswith("/")]
    if not messages:
        update_fermentation_checked(seed.seed_id, _dt_to_str(now))
        await _check_completion(plugin, seed)
        return

    host_bot_self_ids = await resolve_host_bot_self_ids(plugin)
    monitor_config = {
        "monitored_users": list(plugin.config.monitor.monitored_users or []),
        "excluded_users": list(plugin.config.monitor.excluded_users or []),
        "bot_self_id": host_bot_self_ids,
    }
    messages = filter_messages_for_evolution(messages, monitor_config)
    if not messages:
        update_fermentation_checked(seed.seed_id, _dt_to_str(now))
        await _check_completion(plugin, seed)
        return

    # 构建消息行
    max_chars = int(plugin.config.evolution.max_chars_per_message)
    msg_lines = []
    for m in messages[:_LLM_BATCH_MAX_MESSAGES]:
        user_info = m.get("user_info", {}) if isinstance(m, dict) else {}
        nickname = (
            str(user_info.get("user_cardname", "") or "")
            or str(user_info.get("user_nickname", "") or "")
            or str(user_info.get("user_id", "") or "")
            or "user"
        )
        content = str(m.get("processed_plain_text", "") or "") or str(m.get("display_message", "") or "") or ""
        sanitized = sanitize_text(content, max_chars=max_chars)
        if sanitized:
            msg_lines.append(f"{nickname}: {sanitized}")

    if not msg_lines:
        update_fermentation_checked(seed.seed_id, _dt_to_str(now))
        await _check_completion(plugin, seed)
        return

    # L1 关键词预筛
    seed_keywords = _build_seed_keywords(seed)
    threshold = float(cfg.fermentation_relevance_threshold)

    candidates = []
    for line in msg_lines:
        if threshold <= 0:
            candidates.append(line)
            continue
        score = _keyword_overlap_score(seed_keywords, line)
        if score >= _KEYWORD_MATCH_THRESHOLD:
            candidates.append(line)

    if not candidates:
        logger.debug("[Fermentation] 种子 %s L1 过滤后无候选消息", seed.seed_id)
        update_fermentation_checked(seed.seed_id, _dt_to_str(now))
        await _check_completion(plugin, seed)
        return

    # L3 LLM 批量判断关联度
    relevance_scores = await _llm_judge_relevance(plugin, seed, candidates)

    # LLM 失败（空列表）：不推进 checked_at，下轮重试
    if not relevance_scores:
        logger.warning(
            "[Fermentation] 种子 %s 关联度判断 LLM 失败/返回空，不推进 checked_at，下轮重试",
            seed.seed_id,
        )
        await _check_completion(plugin, seed)
        return

    # 收录关联度达标的消息
    added = 0
    for i, line in enumerate(candidates):
        score = relevance_scores[i] if i < len(relevance_scores) else 0.0
        if score >= threshold:
            remaining = max_inputs - count_fermentation_inputs(seed.seed_id)
            if remaining <= 0:
                break
            add_fermentation_input(seed.seed_id, stream_id, line, score)
            added += 1

    if added:
        logger.info("[Fermentation] 种子 %s 新增 %d 条发酵输入", seed.seed_id, added)

    # 更新检查时间
    update_fermentation_checked(seed.seed_id, _dt_to_str(now))

    # 检查是否到期
    await _check_completion(plugin, seed)


async def _llm_judge_relevance(plugin: Any, seed: Any, messages: list[str]) -> list[float]:
    """用 LLM 批量判断消息与种子的关联度，返回每条消息的分数列表。"""
    if not messages:
        return []

    messages_text = "\n".join(f"[{i}] {msg}" for i, msg in enumerate(messages))
    prompt = FERMENTATION_RELEVANCE_PROMPT.format(
        seed_type=getattr(seed, "seed_type", "") or "",
        seed_event=getattr(seed, "event", "") or "",
        seed_reasoning=getattr(seed, "reasoning", "") or "",
        messages=messages_text,
    )

    try:
        result = await generate_soul_text(plugin, prompt)
        response = result.get("response", "") if isinstance(result, dict) else str(result)
    except (RuntimeError, ValueError, OSError, asyncio.TimeoutError) as exc:
        logger.warning("[Fermentation] 关联度判断 LLM 失败: %s", exc)
        return []

    # 解析 JSON 数组
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(response)
        if not isinstance(parsed, list):
            return []
    except (json.JSONDecodeError, ValueError):
        logger.warning("[Fermentation] 关联度判断响应解析失败: %s", response[:200])
        return []

    # 构建 score 列表，按 index 对齐
    scores = [0.0] * len(messages)
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("index", -1))
        if 0 <= idx < len(messages):
            try:
                scores[idx] = float(item.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass

    return scores


async def _try_notify_admin_insufficient(plugin: Any, seed: Any, input_count: int, min_inputs: int) -> None:
    """尝试通知管理员种子因证据不足无法内化。失败仅记日志，不抛出。"""
    from ..utils.spectrum_utils import parse_user_id

    admin_config_id = plugin.config.admin.admin_user_id
    if not admin_config_id:
        logger.debug("[Fermentation] admin_user_id 未配置，跳过通知 seed=%s", seed.seed_id)
        return

    platform, user_id = parse_user_id(admin_config_id)
    if not platform or not user_id:
        logger.debug("[Fermentation] admin_user_id 无法解析，跳过通知 seed=%s", seed.seed_id)
        return

    try:
        admin_stream_id = await plugin.ctx.chat.get_stream_by_user_id(
            platform=platform, user_id=user_id
        )
    except (RuntimeError, ValueError, OSError):
        logger.debug("[Fermentation] 获取管理员 stream_id 失败 seed=%s", seed.seed_id)
        return

    if not admin_stream_id:
        logger.debug("[Fermentation] 未找到管理员聊天流 seed=%s", seed.seed_id)
        return

    text = (
        f"🧬 发酵种子 {seed.seed_id} 证据不足\n"
        f"收集到 {input_count}/{min_inputs} 条相关输入，已达最大延长次数\n"
        f"种子保持发酵状态，可手动处理 (approve/reject)"
    )
    try:
        await plugin.ctx.send.text(text=text, stream_id=admin_stream_id)
    except (RuntimeError, ValueError, OSError):
        logger.debug("[Fermentation] 发送管理员通知失败 seed=%s", seed.seed_id)


async def _try_notify_admin_fermented(plugin: Any, seed: Any, trait_id: str) -> None:
    """发酵内化成功后通知管理员，含槽位建议。失败仅记日志，不抛出。"""
    from ..utils.spectrum_utils import parse_user_id
    from ..models.traits import query_crystallized_traits

    admin_config_id = plugin.config.admin.admin_user_id
    if not admin_config_id:
        logger.info("[Fermentation] 种子 %s 发酵内化: trait=%s（admin_user_id 未配置）", seed.seed_id, trait_id)
        return

    platform, user_id = parse_user_id(admin_config_id)
    if not platform or not user_id:
        logger.info("[Fermentation] 种子 %s 发酵内化: trait=%s", seed.seed_id, trait_id)
        return

    # 计算可用空槽
    _all_t = query_crystallized_traits(deleted=False, enabled=True, limit=200)
    used = {t.cabinet_slot_no for t in _all_t if t.cabinet_slot_no is not None}
    free = [i for i in range(1, 13) if i not in used]
    if free:
        slot_hint = f"建议：/soul_slot {trait_id} {free[0]}  将观点放入思维阁槽位（推荐空槽 {free[0]}）"
    else:
        slot_hint = f"12 槽已满，可用 /soul_slot {trait_id} <1-12> 替换已有"

    text = (
        f"🧬 发酵种子 {seed.seed_id} 已内化完成\n"
        f"trait_id: {trait_id}\n"
        f"{slot_hint}"
    )

    try:
        admin_stream_id = await plugin.ctx.chat.get_stream_by_user_id(
            platform=platform, user_id=user_id
        )
    except (RuntimeError, ValueError, OSError):
        logger.info("[Fermentation] 种子 %s 发酵内化: trait=%s（获取管理员流失败，仅日志）", seed.seed_id, trait_id)
        return

    if not admin_stream_id:
        logger.info("[Fermentation] 种子 %s 发酵内化: trait=%s（未找到管理员流，仅日志）", seed.seed_id, trait_id)
        return

    try:
        await plugin.ctx.send.text(text=text, stream_id=admin_stream_id)
    except (RuntimeError, ValueError, OSError):
        logger.debug("[Fermentation] 发送管理发酵成功通知失败 seed=%s", seed.seed_id)


async def _check_completion(plugin: Any, seed: Any) -> None:
    """检查种子是否发酵到期，触发最终内化或延长窗口。"""
    from ..models.seeds import (
        count_fermentation_inputs,
        extend_fermentation_window,
    )

    cfg = plugin.config.thought_cabinet
    window_hours = float(cfg.fermentation_window_hours)
    started_at = seed.fermentation_started_at or datetime.now()
    if isinstance(started_at, str):
        started_at = _str_to_dt(started_at) or datetime.now()

    now = datetime.now()
    elapsed = (now - started_at).total_seconds() / 3600.0

    if elapsed < window_hours:
        return  # 窗口未到期

    # 窗口到期
    input_count = count_fermentation_inputs(seed.seed_id)
    min_inputs = int(cfg.fermentation_min_inputs)
    max_extensions = int(cfg.fermentation_max_extensions)

    if min_inputs > 0 and input_count < min_inputs:
        # 输入不足，尝试延长
        if seed.fermentation_extension_count < max_extensions:
            extend_fermentation_window(seed.seed_id)
            logger.info(
                "[Fermentation] 种子 %s 输入不足(%d<%d)，延长窗口（第%d次）",
                seed.seed_id, input_count, min_inputs, seed.fermentation_extension_count + 1,
            )
            return
        else:
            logger.warning(
                "[Fermentation] 种子 %s 输入不足(%d<%d)且已达最大延长次数(%d)，证据不足，保持发酵",
                seed.seed_id, input_count, min_inputs, max_extensions,
            )
            # 不触发内化：种子保持 fermenting，等待管理员处理或未来新消息。
            # 正常路径下调用前已推进 checked_at；LLM 失败路径未推进 checked_at，下轮会重试同批消息。
            await _try_notify_admin_insufficient(plugin, seed, input_count, min_inputs)
            return

    # 触发最终内化
    logger.info("[Fermentation] 种子 %s 发酵到期，触发最终内化（%d 条输入）", seed.seed_id, input_count)
    await _finalize_fermentation(plugin, seed)


async def _finalize_fermentation(plugin: Any, seed: Any) -> None:
    """发酵到期后执行最终内化。"""
    from ..models.seeds import get_fermentation_inputs, mark_seed_internalized
    from ..thought.internalization_engine import InternalizationEngine

    # 获取发酵输入
    inputs = get_fermentation_inputs(seed.seed_id)
    fermentation_texts = [fi.message_text for fi in inputs]

    # 用内化引擎执行发酵后内化
    engine = InternalizationEngine(plugin)

    # 构建种子 dict（internalize_seed 期望 dict 格式）
    from ..utils.evidence_utils import parse_evidence_json
    seed_dict = {
        "seed_id": seed.seed_id,
        "stream_id": seed.stream_id or "",
        "type": seed.seed_type,
        "event": seed.event,
        "intensity": float(seed.intensity) / 100.0,
        "confidence": float(seed.confidence or 0) / 100.0,
        "evidence": parse_evidence_json(seed.evidence_json or "[]"),
        "context": json.loads(seed.context_json or "[]") if seed.context_json else [],
        "reasoning": seed.reasoning,
        "potential_impact": json.loads(seed.potential_impact_json or "{}"),
        "created_at": seed.created_at.isoformat() if seed.created_at else None,
    }

    dedup_cfg = {
        "enabled": bool(plugin.config.thought_cabinet.auto_dedup_enabled),
        "threshold": float(plugin.config.thought_cabinet.auto_dedup_threshold),
    }

    try:
        result = await engine.internalize_seed(
            seed_dict,
            dedup=dedup_cfg,
            fermentation_inputs=fermentation_texts,
        )
    except Exception as e:
        logger.error("[Fermentation] 种子 %s 最终内化异常: %s", seed.seed_id, e, exc_info=True)
        return  # 保持 fermenting 状态，下轮重试

    if result.get("success"):
        mark_seed_internalized(seed.seed_id)
        trait_id = result.get("trait_id", "")
        logger.info(
            "[Fermentation] 种子 %s 发酵内化完成: trait=%s, thought=%s...",
            seed.seed_id, trait_id, str(result.get("thought", ""))[:50],
        )
        # P1.1: 发酵成功后尽量通知管理员含槽位建议
        if trait_id:
            await _try_notify_admin_fermented(plugin, seed, trait_id)
    else:
        logger.warning(
            "[Fermentation] 种子 %s 发酵内化失败: %s（保持 fermenting，下轮重试）",
            seed.seed_id, result.get("error", "unknown"),
        )
