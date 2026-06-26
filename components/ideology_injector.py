"""意识形态注入组件 — 在 LLM 请求前注入光谱提示词与 trait。

maisaka.planner.before_request HookHandler 的委托函数。
"""

from __future__ import annotations

import json as _json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models.ideology_model import get_or_create_spectrum, query_active_traits_for_injection
from ..prompts.ideology_prompts import build_ideology_prompt
from ..utils.spectrum_utils import chat_config_to_stream_id
from ..utils.trait_tags import parse_tags_json

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
_injection_cache_lock = threading.Lock()


def _stream_key(stream_id: str | None) -> str:
    return stream_id or "global"


def _prune_recent_injection(now: float, ttl_seconds: int = 3600) -> None:
    """清理过期的冷却记录。"""
    with _injection_cache_lock:
        for sid, entries in list(_RECENT_TRAIT_INJECTION.items()):
            for trait_id, ts in list(entries.items()):
                if now - float(ts) > ttl_seconds:
                    entries.pop(trait_id, None)
            if not entries:
                _RECENT_TRAIT_INJECTION.pop(sid, None)


def _in_cooldown(stream_id: str | None, trait_id: str, now: float, cooldown_seconds: int) -> bool:
    """检查 trait 是否处于冷却期。"""
    if cooldown_seconds <= 0:
        return False
    sid = _stream_key(stream_id)
    with _injection_cache_lock:
        ts = _RECENT_TRAIT_INJECTION.get(sid, {}).get(trait_id)
    if not ts:
        return False
    return (now - float(ts)) < float(cooldown_seconds)


def _mark_injected(stream_id: str | None, trait_ids: list[str], now: float) -> None:
    """标记 trait 已注入（进入冷却）。"""
    sid = _stream_key(stream_id)
    with _injection_cache_lock:
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
        raw = getattr(trait, "spectrum_impact_json", "") or "{}"
        impact = _json.loads(raw) if isinstance(raw, str) else {}
        if not isinstance(impact, dict):
            return 0.0
        score = 0.0
        for k in ("economic", "social", "diplomatic", "progressive"):
            try:
                score += abs(float(impact.get(k, 0) or 0))
            except Exception:
                continue
        return float(score)
    except Exception:
        return 0.0


# ─── 注入日志 ───────────────────────────────────────────────────────

_injection_log_lock = threading.Lock()


def _record_injection(entry: dict, plugin_dir: Path) -> None:
    """记录注入日志到 data/injections.jsonl。"""
    file_path = plugin_dir / "data" / "injections.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with _injection_log_lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")


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
    # ── 1. 配置检查 ────────────────────────────────────────────────
    if not plugin.config.plugin.enabled:
        return {"success": True, "action": "continue"}

    messages: list[dict] = list(kwargs.get("messages") or [])
    if not messages:
        return {"success": True, "action": "continue"}

    session_id: str = kwargs.get("session_id", "") or ""
    stream_id = session_id  # session_id 即为 stream_id

    # 判断是否为私聊（通过 session_id 格式推断，私聊通常包含 ":private"）
    is_private = ":private" in stream_id or "private" in stream_id.lower()
    is_group = not is_private

    scope = str(plugin.config.injection.scope or "global").strip().lower()
    inject_private = bool(plugin.config.injection.inject_private)
    max_traits = max(0, int(plugin.config.injection.max_traits or 3))
    fallback_recent_impact = bool(plugin.config.injection.fallback_recent_impact)
    cooldown_seconds = max(0, int(plugin.config.injection.trait_cooldown_seconds or 180))

    plugin_dir: Path = plugin._plugin_dir

    # ── 2. 私聊排除 ────────────────────────────────────────────────
    if is_private and not inject_private:
        _record_injection(
            {
                "ts": datetime.now().isoformat(),
                "skipped": True,
                "reason": "private injection disabled",
                "policy": "disabled",
            },
            plugin_dir=plugin_dir,
        )
        return {"success": True, "action": "continue"}

    # ── 3. 群聊范围/排除检查 ──────────────────────────────────────
    if is_group:
        monitored = list(plugin.config.monitor.monitored_groups or [])
        excluded = list(plugin.config.monitor.excluded_groups or [])
        monitored_ids = {chat_config_to_stream_id(str(x)) for x in monitored if str(x).strip()}
        excluded_ids = {chat_config_to_stream_id(str(x)) for x in excluded if str(x).strip()}

        if stream_id and stream_id in excluded_ids:
            _record_injection(
                {
                    "ts": datetime.now().isoformat(),
                    "skipped": True,
                    "reason": "group excluded",
                    "policy": "disabled",
                },
                plugin_dir=plugin_dir,
            )
            return {"success": True, "action": "continue"}

        if scope == "monitored_only":
            if not monitored_ids:
                _record_injection(
                    {
                        "ts": datetime.now().isoformat(),
                        "skipped": True,
                        "reason": "no monitored_groups configured",
                        "policy": "disabled",
                    },
                    plugin_dir=plugin_dir,
                )
                return {"success": True, "action": "continue"}
            if not stream_id or stream_id not in monitored_ids:
                _record_injection(
                    {
                        "ts": datetime.now().isoformat(),
                        "skipped": True,
                        "reason": "group not monitored",
                        "policy": "disabled",
                    },
                    plugin_dir=plugin_dir,
                )
                return {"success": True, "action": "continue"}

    # ── 4. 获取光谱 ────────────────────────────────────────────────
    spectrum = get_or_create_spectrum("global")

    if not spectrum.initialized:
        return {"success": True, "action": "continue"}

    spectrum_dict = {
        "economic": spectrum.economic,
        "social": spectrum.social,
        "diplomatic": spectrum.diplomatic,
        "progressive": spectrum.progressive,
    }

    custom_prompts = dict(plugin.config.threshold.custom_prompts or {})
    enable_extreme = bool(plugin.config.threshold.enable_extreme)
    ideology_prompt = build_ideology_prompt(spectrum_dict, custom_prompts, enable_extreme)

    if not ideology_prompt:
        return {"success": True, "action": "continue"}

    # ── 5. 查询活跃 traits ────────────────────────────────────────
    traits = query_active_traits_for_injection(stream_id=stream_id, limit=80)

    # ── 6. Tag 匹配选择 ────────────────────────────────────────────
    text = _extract_user_text(messages)
    text_norm = str(text).casefold()

    now_ts = time.time()
    _prune_recent_injection(now_ts)

    scored: list[tuple[float, Any]] = []
    for t in traits:
        tags = parse_tags_json(getattr(t, "tags_json", "[]") or "[]")
        if not tags:
            continue
        hit = 0
        for tag in tags:
            if tag and tag.casefold() in text_norm:
                hit += 1
        if hit > 0:
            scored.append((float(hit), t))

    scored.sort(key=lambda x: (x[0], x[1].created_at), reverse=True)
    selected: list[Any] = []
    cooldown_skipped: list[str] = []
    if max_traits > 0:
        for _score, t in scored:
            if len(selected) >= max_traits:
                break
            if _in_cooldown(stream_id, t.trait_id, now_ts, cooldown_seconds):
                cooldown_skipped.append(t.trait_id)
                continue
            selected.append(t)

    selection_mode = "tag_hit" if selected else "spectrum_only"

    # ── 7. Fallback 最近影响最大的 traits ──────────────────────────
    if not selected and fallback_recent_impact and max_traits > 0:
        fallback_candidates: list[tuple[float, datetime, Any]] = []
        for t in traits:
            if _in_cooldown(stream_id, t.trait_id, now_ts, cooldown_seconds):
                continue
            impact_score = _trait_impact_score(t)
            fallback_candidates.append((impact_score, t.created_at, t))
        fallback_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = [t for _score, _ts, t in fallback_candidates[:max_traits] if _score > 0.0]
        if selected:
            selection_mode = "fallback_recent_impact"

    # ── 8. 构建注入块 ──────────────────────────────────────────────
    picked: list[dict] = []
    trait_lines: list[str] = []
    for t in selected:
        tags = parse_tags_json(getattr(t, "tags_json", "[]") or "[]")
        question = _compact_one_line(getattr(t, "question", "") or "", 90)
        thought = _compact_one_line(t.thought or "", 160)
        tags_text = f" tags={','.join(tags)}" if tags else ""
        if question:
            trait_lines.append(f"- ({t.trait_id}){tags_text} 问: {question} 答: {thought}")
        else:
            trait_lines.append(f"- ({t.trait_id}){tags_text} {t.name}: {thought}")

        score = 0.0
        hit_tags: list[str] = []
        if tags:
            for tag in tags:
                if tag and tag.casefold() in text_norm:
                    score += 1.0
                    hit_tags.append(tag)
        picked.append(
            {
                "thought_id": t.trait_id,
                "name": t.name,
                "score": score if selection_mode == "tag_hit" else _trait_impact_score(t),
                "mode": selection_mode,
                "hit_tags": hit_tags,
            }
        )

    from ..worldview.service import WorldviewService, config_from_plugin

    wv = WorldviewService(config_from_plugin(plugin))
    p1_blocks: list[str] = []
    layer_summary = wv.build_layer_trait_summary(stream_id)
    if layer_summary:
        p1_blocks.append(layer_summary)
    mood_lines = wv.mood_prompt_lines()
    if mood_lines:
        p1_blocks.extend(mood_lines)
    if selected:
        graph_hint = wv.build_graph_hint(stream_id, [t.trait_id for t in selected])
        if graph_hint:
            p1_blocks.append(graph_hint)

    injection_block = (
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

    # ── 9. 注入到 messages ─────────────────────────────────────────
    modified_messages = [{"role": "system", "content": injection_block}] + list(messages)

    # ── 10. 记录注入日志 ────────────────────────────────────────────
    if not picked:
        policy = "spectrum_only"
    elif selection_mode == "tag_hit":
        policy = "tags+spectrum"
    elif selection_mode == "fallback_recent_impact":
        policy = "fallback+spectrum"
    else:
        policy = "traits+spectrum"

    _record_injection(
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
        _mark_injected(stream_id, [t.trait_id for t in selected], now_ts)

    return {
        "success": True,
        "action": "continue",
        "modified_kwargs": {**kwargs, "messages": modified_messages},
    }
