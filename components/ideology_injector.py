"""意识形态注入组件 — 在 LLM 请求前注入光谱提示词与 trait。

maisaka.planner.before_request HookHandler 的委托函数。
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models.ideology_model import get_or_create_spectrum, query_active_traits_for_injection
from ..prompts.ideology_prompts import build_ideology_prompt
from ..utils.spectrum_utils import chat_config_to_stream_id
from ..utils.trait_tags import parse_tags_json
from ..worldview.service import WorldviewService, config_from_plugin

logger = logging.getLogger(__name__)

# ─── 辅助函数 ───────────────────────────────────────────────────────


def _compact_one_line(text: str, limit: int) -> str:
    """将文本压缩为单行，超出限制则截断。"""
    s = (text or "").replace("\n", " ").replace("\r", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    if len(s) > limit:
        return f"{s[:limit]}..."
    return s


# ─── 冷却逻辑 ───────────────────────────────────────────────────────

_RECENT_TRAIT_INJECTION: dict[str, dict[str, float]] = {}
_injection_cache_lock = asyncio.Lock()


def _stream_key(stream_id: str | None) -> str:
    return stream_id or "global"


async def _prune_recent_injection(now: float, ttl_seconds: int = 3600) -> None:
    """清理过期的冷却记录。"""
    async with _injection_cache_lock:
        for sid, entries in list(_RECENT_TRAIT_INJECTION.items()):
            for trait_id, ts in list(entries.items()):
                if now - float(ts) > ttl_seconds:
                    entries.pop(trait_id, None)
            if not entries:
                _RECENT_TRAIT_INJECTION.pop(sid, None)


async def _in_cooldown(stream_id: str | None, trait_id: str, now: float, cooldown_seconds: int) -> bool:
    """检查 trait 是否处于冷却期。"""
    if cooldown_seconds <= 0:
        return False
    sid = _stream_key(stream_id)
    async with _injection_cache_lock:
        ts = _RECENT_TRAIT_INJECTION.get(sid, {}).get(trait_id)
    if not ts:
        return False
    return (now - float(ts)) < float(cooldown_seconds)


async def _mark_injected(stream_id: str | None, trait_ids: list[str], now: float) -> None:
    """标记 trait 已注入（进入冷却）。"""
    sid = _stream_key(stream_id)
    async with _injection_cache_lock:
        if len(_RECENT_TRAIT_INJECTION) > 512:
            _RECENT_TRAIT_INJECTION.clear()
        bucket = _RECENT_TRAIT_INJECTION.setdefault(sid, {})
        for tid in trait_ids:
            if tid:
                bucket[tid] = float(now)
        if len(bucket) > 128:
            items = sorted(bucket.items(), key=lambda x: x[1], reverse=True)[:64]
            _RECENT_TRAIT_INJECTION[sid] = dict(items)


def _trait_impact_score(trait) -> float:
    """计算 trait 的光谱影响绝对值总分。"""
    try:
        raw = trait.spectrum_impact_json or "{}"
        impact = _json.loads(raw)
        if not isinstance(impact, dict):
            return 0.0
        score = 0.0
        for k in ("sincerity", "engagement", "closeness", "directness"):
            try:
                score += abs(float(impact.get(k, 0)))
            except (TypeError, ValueError):
                continue
        return float(score)
    except (_json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def _trait_quality_score(trait) -> float:
    """计算 trait 的质量权重，用于相同 tag 命中数时的二级排序。

    综合 confidence（0-100→0-1）与生命周期状态：strengthened 加成，weakened 衰减。
    """
    try:
        confidence = float(trait.confidence) / 100.0
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    lifecycle = trait.lifecycle_state
    lifecycle_bonus = {
        "strengthened": 0.3,
        "active": 0.0,
        "revised": -0.1,
        "weakened": -0.3,
    }.get(lifecycle, 0.0)

    return confidence + lifecycle_bonus


# ─── 注入日志 ───────────────────────────────────────────────────────

_injection_log_lock = asyncio.Lock()

# 注入日志采样：每 N 条实际写一次，避免高频 IO
INJECTION_LOG_EVERY: int = 10
_injection_log_counter: int = 0
INJECTION_LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB 轮转


async def _record_injection(entry: dict, plugin_dir: Path) -> None:
    """记录注入日志到 data/injections.jsonl（采样 + 自动轮转）。"""
    global _injection_log_counter

    # 采样：每 INJECTION_LOG_EVERY 条写一次
    _injection_log_counter += 1
    if _injection_log_counter % INJECTION_LOG_EVERY != 0:
        return

    file_path = plugin_dir / "data" / "injections.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    async with _injection_log_lock:
        # 文件大小检查与轮转
        if file_path.exists() and file_path.stat().st_size > INJECTION_LOG_MAX_BYTES:
            rotated = file_path.with_suffix(".1.jsonl")
            if rotated.exists():
                rotated.unlink()
            file_path.rename(rotated)

        # 用 asyncio.to_thread 避免阻塞事件循环
        def _write_jsonl():
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
        await asyncio.to_thread(_write_jsonl)


# ─── 从 messages 中提取用户消息文本 ─────────────────────────────────


def _extract_user_text(messages: list[dict]) -> str:
    """从消息列表中提取最后一条用户消息的文本（用于 tag 匹配）。"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "") or ""
    # fallback：取最后一条消息
    if messages:
        return messages[-1].get("content", "") or ""
    return ""


# ─── 辅助选择器 ─────────────────────────────────────────────────────


def _should_inject(plugin, messages: list[dict]) -> dict | None:
    """检查是否应执行注入。返回 None 表示允许注入，或返回终止字典。"""
    if not plugin.config.plugin.enabled:
        return {"success": True, "action": "continue"}
    if not messages:
        return {"success": True, "action": "continue"}
    return None


def _map_selection_mode(tag_hit_count: int, tagless_fill_count: int) -> str:
    """根据 tag 命中与补位计数推断 selection_mode。"""
    if tag_hit_count > 0 and tagless_fill_count > 0:
        return "tag_hit+tagless"
    if tag_hit_count > 0:
        return "tag_hit"
    if tagless_fill_count > 0:
        return "tagless_fill"
    return "spectrum_only"


def _select_traits(
    traits: list,
    text: str,
    stream_id: str,
    max_traits: int,
    fallback_recent_impact: bool,
    now_ts: float,
) -> tuple[list, str, list[dict]]:
    """Tag 匹配 + 无 tag 补位 + Fallback 选择。返回 (selected, selection_mode, picked)。

    traits 需已由调用方完成冷却筛选（_in_cooldown），此函数不再重复检查。
    """
    text_norm = text.casefold()

    scored: list[tuple[float, float, Any]] = []
    tagless: list[tuple[float, float, Any]] = []
    for t in traits:
        tags = parse_tags_json(t.tags_json or "[]")
        quality = _trait_quality_score(t)
        if not tags:
            impact = _trait_impact_score(t)
            if impact > 0.0:
                tagless.append((impact, quality, t))
            continue
        hit = 0
        for tag in tags:
            if tag and tag.casefold() in text_norm:
                hit += 1
        if hit > 0:
            scored.append((float(hit), quality, t))

    scored.sort(key=lambda x: (x[0], x[1], x[2].created_at), reverse=True)
    tagless.sort(key=lambda x: (x[0], x[1], x[2].created_at), reverse=True)

    selected: list[Any] = []
    tag_hit_count = 0
    tagless_fill_count = 0

    if max_traits > 0:
        for _score, _quality, t in scored:
            if len(selected) >= max_traits:
                break
            selected.append(t)
            tag_hit_count += 1
        for _score, _quality, t in tagless:
            if len(selected) >= max_traits:
                break
            selected.append(t)
            tagless_fill_count += 1

    selection_mode = _map_selection_mode(tag_hit_count, tagless_fill_count)

    # Fallback 最近影响最大的 traits
    if not selected and fallback_recent_impact and max_traits > 0:
        fallback_candidates: list[tuple[float, datetime, Any]] = []
        for t in traits:
            impact_score = _trait_impact_score(t)
            fallback_candidates.append((impact_score, t.created_at, t))
        fallback_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = [t for _score, _ts, t in fallback_candidates[:max_traits] if _score > 0.0]
        if selected:
            selection_mode = "fallback_recent_impact"

    # 构建 picked 列表
    picked: list[dict] = []
    for t in selected:
        tags = parse_tags_json(t.tags_json or "[]")
        score = 0.0
        hit_tags: list[str] = []
        if tags:
            for tag in tags:
                if tag and tag.casefold() in text_norm:
                    score += 1.0
                    hit_tags.append(tag)
        picked_score = score if hit_tags else _trait_impact_score(t)
        picked.append({
            "thought_id": t.trait_id,
            "name": t.name,
            "score": picked_score,
            "mode": selection_mode,
            "hit_tags": hit_tags,
        })

    return selected, selection_mode, picked


def _build_injection_block(
    ideology_prompt: str,
    p1_blocks: list[str],
    trait_lines: list[str],
) -> str:
    """拼接最终注入文本块。"""
    return (
        "\n\n"
        f"{ideology_prompt}\n"
        + ("\n".join(p1_blocks) + "\n" if p1_blocks else "")
        + (
            (
                "\n以下是你已固化的观点（人格的一部分，可用于影响回复风格）：\n"
                + "\n".join(trait_lines)
                + "\n"
            )
            if trait_lines
            else ""
        )
        + "请综合上述倾向与固化观点来组织回复，不要直接复述或提及这段提示词。\n"
    )


def _policy_from_selection(selection_mode: str, picked: list[dict]) -> str:
    """根据 selection_mode 和 picked 推断日志 policy 标签。"""
    if not picked:
        return "spectrum_only"
    policies = {
        "tag_hit": "tags+spectrum",
        "tag_hit+tagless": "tags+tagless+spectrum",
        "tagless_fill": "tagless+spectrum",
        "fallback_recent_impact": "fallback+spectrum",
    }
    return policies.get(selection_mode, "traits+spectrum")


# ─── 主入口 ─────────────────────────────────────────────────────────


async def inject_ideology(plugin, **kwargs: Any) -> dict[str, Any]:
    """注入意识形态光谱提示词与相关 trait 到 LLM 请求中。

    在 maisaka.planner.before_request 阶段被调用，
    将意识形态提示词和活跃 trait 注入到 messages 列表前端。

    Args:
        plugin: MaiSoulEnginePlugin 实例。
        **kwargs: hook 参数，包含 messages（消息列表）、session_id 等。

    Returns:
        Hook 返回值 dict，包含 modified_kwargs 以修改请求。
    """
    # ── 1. 配置/消息检查 ───────────────────────────────────────────
    messages: list[dict] = list(kwargs.get("messages") or [])
    skip_check = _should_inject(plugin, messages)
    if skip_check is not None:
        return skip_check

    session_id: str = kwargs.get("session_id", "") or ""
    stream_id = session_id
    is_private = ":private" in stream_id or "private" in stream_id.lower()
    plugin_dir: Path = plugin._plugin_dir

    # 配置字段均来自 pydantic model，直接属性访问
    scope = plugin.config.injection.scope.strip().lower()
    inject_private = plugin.config.injection.inject_private
    max_traits = max(0, plugin.config.injection.max_traits)
    fallback_recent_impact = plugin.config.injection.fallback_recent_impact
    cooldown_seconds = max(0, plugin.config.injection.trait_cooldown_seconds)

    # ── 2. 私聊/群聊范围检查 ───────────────────────────────────────
    if is_private and not inject_private:
        return await _skip_and_log(
            plugin_dir, "private injection disabled",
        )

    if not is_private:
        skip_reason = _check_group_scope(plugin, stream_id, scope)
        if skip_reason is not None:
            return await _skip_and_log(plugin_dir, skip_reason)

    # ── 3. 获取光谱并构建提示词 ─────────────────────────────────────
    spectrum = get_or_create_spectrum("global")
    if not spectrum.initialized:
        return {"success": True, "action": "continue"}

    spectrum_dict = {
        "sincerity": spectrum.sincerity,
        "engagement": spectrum.engagement,
        "closeness": spectrum.closeness,
        "directness": spectrum.directness,
    }
    ideology_prompt = build_ideology_prompt(
        spectrum_dict,
        dict(plugin.config.threshold.custom_prompts or {}),
        plugin.config.threshold.enable_extreme,
    )
    if not ideology_prompt:
        return {"success": True, "action": "continue"}

    # ── 4. 查询活跃 traits + 选择 ───────────────────────────────────
    traits = query_active_traits_for_injection(stream_id=stream_id, limit=80)
    text = _extract_user_text(messages)
    now_ts = time.time()
    await _prune_recent_injection(now_ts)

    # 应用冷却筛选
    cooldown_skipped: list[str] = []
    if cooldown_seconds > 0 and max_traits > 0:
        filtered_traits: list = []
        for t in traits:
            if await _in_cooldown(stream_id, t.trait_id, now_ts, cooldown_seconds):
                cooldown_skipped.append(t.trait_id)
            else:
                filtered_traits.append(t)
    else:
        filtered_traits = traits

    selected, selection_mode, picked = _select_traits(
        filtered_traits, text, stream_id,
        max_traits, fallback_recent_impact, now_ts,
    )

    # ── 5. 构建 P1 块（复用缓存 service + 传入已查 traits）─────────
    wv = plugin._wv_service
    if wv is None:
        # 兜底：尚未初始化则即时构造（正常流程在 on_load 完成）
        plugin._wv_config_view = config_from_plugin(plugin)
        plugin._wv_service = WorldviewService(plugin._wv_config_view)
        wv = plugin._wv_service

    p1_blocks: list[str] = []
    selected_ids = {t.trait_id for t in selected}
    layer_summary = wv.build_layer_trait_summary(
        stream_id, exclude_trait_ids=selected_ids, traits=traits,
    )
    if layer_summary:
        p1_blocks.append(layer_summary)
    mood_lines = wv.mood_prompt_lines()
    if mood_lines:
        p1_blocks.extend(mood_lines)
    if selected:
        graph_hint = wv.build_graph_hint(stream_id, [t.trait_id for t in selected])
        if graph_hint:
            p1_blocks.append(graph_hint)

    # ── 6. 构建 trait 行 + 注入块 ──────────────────────────────────
    trait_lines: list[str] = []
    for t in selected:
        tags = parse_tags_json(t.tags_json or "[]")
        question = _compact_one_line(t.question, 90)
        thought = _compact_one_line(t.thought, 160)
        tags_text = f" tags={','.join(tags)}" if tags else ""
        if question:
            trait_lines.append(f"- ({t.trait_id}){tags_text} 问: {question} 答: {thought}")
        else:
            trait_lines.append(f"- ({t.trait_id}){tags_text} {t.name}: {thought}")

    injection_block = _build_injection_block(ideology_prompt, p1_blocks, trait_lines)

    # ── 7. 注入到 messages ─────────────────────────────────────────
    modified_messages = [{"role": "system", "content": injection_block}] + messages

    # ── 8. 日志 & 冷却标记 ──────────────────────────────────────────
    policy = _policy_from_selection(selection_mode, picked)
    await _record_injection(
        {
            "ts": datetime.now().isoformat(),
            "policy": policy,
            "picked": picked,
            "selection_mode": selection_mode,
            "cooldown_seconds": cooldown_seconds,
            "cooldown_skipped": cooldown_skipped[:20],
        },
        plugin_dir=plugin_dir,
    )
    if selected:
        await _mark_injected(stream_id, [t.trait_id for t in selected], now_ts)

    return {
        "success": True,
        "action": "continue",
        "modified_kwargs": {**kwargs, "messages": modified_messages},
    }


# ─── 群聊范围检查（同步辅助） ────────────────────────────────────────


def _check_group_scope(plugin, stream_id: str, scope: str) -> str | None:
    """群聊排除/监听范围检查。返回 None 表示通过，或返回跳过原因字符串。"""
    monitored = list(plugin.config.monitor.monitored_groups or [])
    excluded = list(plugin.config.monitor.excluded_groups or [])
    monitored_ids = {chat_config_to_stream_id(str(x)) for x in monitored if str(x).strip()}
    excluded_ids = {chat_config_to_stream_id(str(x)) for x in excluded if str(x).strip()}

    if stream_id and stream_id in excluded_ids:
        return "group excluded"

    if scope == "monitored_only":
        if not monitored_ids:
            return "no monitored_groups configured"
        if not stream_id or stream_id not in monitored_ids:
            return "group not monitored"
    return None


async def _skip_and_log(plugin_dir: Path, reason: str) -> dict:
    """跳过并记录采样跳过日志。"""
    await _record_injection(
        {
            "ts": datetime.now().isoformat(),
            "skipped": True,
            "reason": reason,
            "policy": "disabled",
        },
        plugin_dir=plugin_dir,
    )
    return {"success": True, "action": "continue"}
