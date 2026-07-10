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
from .reflection_capture import cache_session_context, maybe_write_injection_snapshot

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


# ── 注入指标计数器（可观测性）──────────────────────────────────────
_injection_metrics = {
    "total": 0,           # 总注入次数
    "traits_hit": 0,      # 命中 trait 的注入次数
    "skipped_cooldown": 0,  # 因冷却跳过的次数
    "skipped_no_traits": 0,  # 无可用 trait 跳过的次数
}

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


async def _batch_cooldown_filter(
    stream_id: str | None,
    traits: list,
    now: float,
    cooldown_seconds: int,
) -> tuple[list, list[str]]:
    """批量冷却筛选：一次性获取锁 copy snapshot，锁外批查。

    返回 (filtered_traits, cooldown_skipped_ids)。
    替代循环调用 _in_cooldown 的 80 次锁获取。

    软竞态：snapshot 是锁内拷贝的，锁外批查期间其他协程可能已通过 _mark_injected
    标记了新 trait。本协程的 snapshot 看不到该标记，可能导致 trait 被重复注入一次。
    冷却机制是"尽力而为"的软约束，重复注入一次无功能影响，可接受。
    """
    if cooldown_seconds <= 0 or not traits:
        return traits, []
    sid = _stream_key(stream_id)
    async with _injection_cache_lock:
        snapshot = dict(_RECENT_TRAIT_INJECTION.get(sid, {}))
    # 锁外批查
    filtered: list = []
    skipped: list[str] = []
    for t in traits:
        ts = snapshot.get(t.trait_id)
        if ts and (now - float(ts)) < float(cooldown_seconds):
            skipped.append(t.trait_id)
        else:
            filtered.append(t)
    return filtered, skipped


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
        # contradicted trait 理论上已 enabled=0 不会进入注入池
        # （query_active_traits_for_injection 过滤 enabled=1），
        # 但显式声明 -1.0 是防御性兜底，防止未来变更遗漏导致矛盾观点被注入。
        "contradicted": -1.0,
    }.get(lifecycle, 0.0)

    return confidence + lifecycle_bonus


# ─── 注入日志 ───────────────────────────────────────────────────────

_injection_log_lock = asyncio.Lock()

# 注入日志采样：每 N 条实际写一次，避免高频 IO
# 1=全量记录（开发观察），生产建议改为 10
INJECTION_LOG_EVERY: int = 1
_injection_log_counter: int = 0
# 注入日志轮转阈值（MB）
_INJECTION_LOG_MAX_SIZE_MB: int = 5
INJECTION_LOG_MAX_BYTES: int = _INJECTION_LOG_MAX_SIZE_MB * 1024 * 1024


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


def _extract_terms(trait, max_terms: int = 12) -> list[str]:
    """从 trait 中提取关键词 terms（name / tags / question / thought 前 80 字）。

    用于轻量文本相关召回，避免 FTS/LLM。
    返回去重、casefold 后的 term 列表（至少 2 字），最多 max_terms 个。
    """
    seen: set[str] = set()
    terms: list[str] = []

    def _add(raw: str) -> None:
        token = raw.casefold()
        if len(token) >= 2 and token not in seen:
            seen.add(token)
            terms.append(token)

    def _split_add(text: str) -> None:
        """从文本中提取 ≥2 字的子串。
        对含空白文本按 token 切分；
        对连续 CJK 文本提取 2-gram（兼顾中文词匹配）。
        """
        cleaned = text.strip(".,;:!?\"'()[]{}<>/\\|`~@#$%^&*+-=《》，。！？、；：""''（）【】")
        if not cleaned:
            return
        parts = cleaned.split()
        if len(parts) > 1:
            for part in parts:
                _add(part)
        else:
            # Continuous text (likely CJK) — extract 2-grams
            for i in range(len(cleaned) - 1):
                bigram = cleaned[i:i + 2]
                _add(bigram)

    # 1. tags（全部）
    try:
        tags = parse_tags_json(trait.tags_json or "[]")
        for tag in tags:
            _add(tag)
    except Exception:
        pass

    # 2. name 整串（≥2 字）
    name = (getattr(trait, "name", None) or "").strip()
    if len(name) >= 2:
        _add(name)

    # 3. question 中 ≥2 字 token
    question = (getattr(trait, "question", None) or "").strip()
    if question:
        _split_add(question)

    # 4. thought 前 80 字中 ≥2 字 token
    thought = (getattr(trait, "thought", None) or "").strip()[:80]
    if thought:
        _split_add(thought)

    return terms[:max_terms]


def _text_relevance_score(trait, text_norm: str) -> tuple[float, list[str]]:
    """返回 (score, matched_terms)。

    在 name / tags / question / thought 中提取关键词 terms，
    检查每个 term 是否以子串形式出现在 text_norm 中。
    Tag 已命中的 trait 不应调用此函数（已在 tag 阶段选中）。
    每个 term 匹配计 1 分，score ≥ 1 才算相关。
    """
    terms = _extract_terms(trait)
    matched: list[str] = []
    for term in terms:
        if term in text_norm:
            matched.append(term)
    if matched:
        return float(len(matched)), matched
    return 0.0, []


def _is_inject_enabled(plugin, messages: list[dict]) -> dict | None:
    """检查是否应执行注入。返回 None 表示允许注入，或返回终止字典。"""
    if not plugin.config.plugin.enabled:
        return {"success": True, "action": "continue"}
    if not messages:
        return {"success": True, "action": "continue"}
    return None


def _map_selection_mode(
    tag_hit_count: int, keyword_fill_count: int = 0, tagless_fill_count: int = 0,
) -> str:
    """根据 tag 命中、关键词补位与无 tag 补位计数推断 selection_mode。"""
    if tag_hit_count > 0 and keyword_fill_count > 0 and tagless_fill_count > 0:
        return "tag_hit+keyword+tagless"
    if tag_hit_count > 0 and keyword_fill_count > 0:
        return "tag_hit+keyword"
    if keyword_fill_count > 0 and tagless_fill_count > 0:
        return "keyword+tagless"
    if tag_hit_count > 0 and tagless_fill_count > 0:
        return "tag_hit+tagless"
    if keyword_fill_count > 0:
        return "keyword_fill"
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
    """Tag 匹配 + 关键词补位 + 无 tag 补位 + Fallback 选择。返回 (selected, selection_mode, picked)。

    traits 需已由调用方完成冷却筛选（_in_cooldown），此函数不再重复检查。

    picked 每项包含：
    - thought_id, name, score, mode, hit_tags
    - activation_reason: "tag_hit:tag1,tag2" / "keyword:term1,term2" / "tagless_impact" / "fallback_recent_impact"
    """
    text_norm = text.casefold()

    # ── Phase 1: Tag 命中 ────────────────────────────────────────
    scored: list[tuple[float, float, float, Any]] = []
    tag_hit_trait_ids: set[str] = set()
    tag_hit_map: dict[str, list[str]] = {}  # trait_id -> hit tags

    for t in traits:
        tags = parse_tags_json(t.tags_json or "[]")
        if not tags:
            continue
        quality = _trait_quality_score(t)
        has_slot = 1.0 if t.cabinet_slot_no else 0.0
        hit = 0
        hit_tag_list: list[str] = []
        for tag in tags:
            if tag and tag.casefold() in text_norm:
                hit += 1
                hit_tag_list.append(tag)
        if hit > 0:
            scored.append((has_slot, float(hit), quality, t))
            tag_hit_trait_ids.add(t.trait_id)
            tag_hit_map[t.trait_id] = hit_tag_list

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3].created_at), reverse=True)

    selected: list[Any] = []
    picked: list[dict] = []
    tag_hit_count = 0
    keyword_fill_count = 0
    tagless_fill_count = 0

    if max_traits > 0:
        # Phase 1: Tag 命中
        for _has_slot, _score, _quality, t in scored:
            if len(selected) >= max_traits:
                break
            selected.append(t)
            tag_hit_count += 1
            hit_tags = tag_hit_map.get(t.trait_id, [])
            picked.append({
                "thought_id": t.trait_id,
                "name": t.name,
                "score": float(_score),
                "mode": "tag_hit",
                "hit_tags": hit_tags,
                "activation_reason": f"tag_hit:{','.join(hit_tags)}" if hit_tags else "tag_hit",
                "cabinet_slot_no": t.cabinet_slot_no,
            })

        # Phase 2: 关键词补位（有 tag 但未命中，用文本相关召回）
        if len(selected) < max_traits:
            keyword_candidates: list[tuple[float, float, float, Any, list[str]]] = []
            for t in traits:
                if t.trait_id in tag_hit_trait_ids:
                    continue
                tags = parse_tags_json(t.tags_json or "[]")
                if not tags:
                    continue  # 有 tag 的 trait 才进入关键词补位
                rel_score, matched_terms = _text_relevance_score(t, text_norm)
                if rel_score > 0:
                    quality = _trait_quality_score(t)
                    has_slot = 1.0 if t.cabinet_slot_no else 0.0
                    keyword_candidates.append((has_slot, rel_score, quality, t, matched_terms))

            keyword_candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3].created_at), reverse=True)
            for _has_slot, rel_score, _quality, t, matched_terms in keyword_candidates:
                if len(selected) >= max_traits:
                    break
                selected.append(t)
                keyword_fill_count += 1
                picked.append({
                    "thought_id": t.trait_id,
                    "name": t.name,
                    "score": rel_score,
                    "mode": "keyword_fill",
                    "hit_tags": [],
                    "activation_reason": f"keyword:{','.join(matched_terms)}",
                    "cabinet_slot_no": t.cabinet_slot_no,
                })

        # Phase 3: 无 tag 补位（按 impact）
        tagless: list[tuple[float, float, float, Any]] = []
        seen_selected_ids = {t.trait_id for t in selected}
        for t in traits:
            if t.trait_id in seen_selected_ids:
                continue
            tags = parse_tags_json(t.tags_json or "[]")
            if tags:
                continue
            impact = _trait_impact_score(t)
            if impact > 0.0:
                quality = _trait_quality_score(t)
                has_slot = 1.0 if t.cabinet_slot_no else 0.0
                tagless.append((has_slot, impact, quality, t))

        tagless.sort(key=lambda x: (x[0], x[1], x[2], x[3].created_at), reverse=True)
        for _has_slot, _score, _quality, t in tagless:
            if len(selected) >= max_traits:
                break
            selected.append(t)
            tagless_fill_count += 1
            picked.append({
                "thought_id": t.trait_id,
                "name": t.name,
                "score": _score,
                "mode": "tagless_fill",
                "hit_tags": [],
                "activation_reason": "tagless_impact",
                "cabinet_slot_no": t.cabinet_slot_no,
            })

    selection_mode = _map_selection_mode(tag_hit_count, keyword_fill_count, tagless_fill_count)

    # Fallback 最近影响最大的 traits
    if not selected and fallback_recent_impact and max_traits > 0:
        fallback_candidates: list[tuple[float, float, datetime, Any]] = []
        for t in traits:
            impact_score = _trait_impact_score(t)
            has_slot = 1.0 if t.cabinet_slot_no else 0.0
            fallback_candidates.append((has_slot, impact_score, t.created_at, t))
        fallback_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        selected = [t for _has_slot, _score, _ts, t in fallback_candidates[:max_traits] if _score > 0.0]
        if selected:
            selection_mode = "fallback_recent_impact"
            picked = []
            for t in selected:
                picked.append({
                    "thought_id": t.trait_id,
                    "name": t.name,
                    "score": _trait_impact_score(t),
                    "mode": "fallback_recent_impact",
                    "hit_tags": [],
                    "activation_reason": "fallback_recent_impact",
                    "cabinet_slot_no": t.cabinet_slot_no,
                })

    # 补全 picked 中未设 mode 的项（兜底）
    for item in picked:
        item.setdefault("mode", selection_mode)

    return selected, selection_mode, picked


def _build_injection_block(
    ideology_prompt: str,
    p1_blocks: list[str],
    trait_lines: list[str],
    reflection_summary: str = "",
) -> str:
    """拼接最终注入文本块。

    自评摘要按是否有 trait 分场景插入（oracle 修订点 5）：
    - 有 trait：放 trait 块下方，语态"低优先级自查，以固化观点为准"
    - 无 trait：放光谱提示后、收束指令前，语态"无特定观点时的补充参考"

    注入块体积控制：trait 行总字符超过 1500 时从尾部裁剪。
    """
    # 体积控制：trait 行总字符上限 1500
    MAX_TRAIT_CHARS = 1500
    total_trait_chars = sum(len(line) for line in trait_lines)
    if total_trait_chars > MAX_TRAIT_CHARS:
        trimmed: list[str] = []
        acc = 0
        for line in trait_lines:
            acc += len(line)
            if acc > MAX_TRAIT_CHARS:
                break
            trimmed.append(line)
        trait_lines = trimmed

    has_traits = bool(trait_lines)
    reflection_block = ""
    if reflection_summary:
        if has_traits:
            reflection_block = f"\n近期自我反思提示（低优先级，以固化观点为准）：{reflection_summary}\n"
        else:
            reflection_block = f"\n最近自我评价洞察（无特定观点时的补充参考）：{reflection_summary}\n"
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
        + reflection_block
        + "请综合上述倾向与固化观点来组织回复，不要直接复述或提及这段提示词。\n"
    )


def _policy_from_selection(selection_mode: str, picked: list[dict]) -> str:
    """根据 selection_mode 和 picked 推断日志 policy 标签。"""
    if not picked:
        return "spectrum_only"
    policies = {
        "tag_hit": "tags+spectrum",
        "tag_hit+keyword": "tags+keyword+spectrum",
        "tag_hit+keyword+tagless": "tags+keyword+tagless+spectrum",
        "tag_hit+tagless": "tags+tagless+spectrum",
        "keyword_fill": "keyword+spectrum",
        "keyword+tagless": "keyword+tagless+spectrum",
        "tagless_fill": "tagless+spectrum",
        "fallback_recent_impact": "fallback+spectrum",
    }
    return policies.get(selection_mode, "traits+spectrum")


# ─── 消息合并 ────────────────────────────────────────────────────────


def _apply_soul_injection_to_messages(
    messages: list[dict],
    injection_block: str,
) -> tuple[list[dict] | None, str]:
    """将 injection_block 安全合并到宿主 messages 中（追加到首条 system）。

    Returns:
        (new_messages, strategy):
        - 有首条 system → (复制并追加后的列表, "append_host_system")
        - 无 system     → (None, "skip_no_host_system")
    """
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role and role.lower() == "system":
            # 深复制：复制整个列表及首个 system dict
            new_messages = [dict(m) for m in messages]
            sys_content = new_messages[i].get("content", "")
            sys_suffix = (
                "\n\n---\n[Mai-Soul 动态层 | 受上方固定人设与表达风格约束，不得覆盖身份事实与 reply_style]\n"
                f"{injection_block.lstrip()}"
            )
            new_messages[i] = {
                **new_messages[i],
                "content": sys_content + sys_suffix,
            }
            return (new_messages, "append_host_system")

    return (None, "skip_no_host_system")


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
    skip_check = _is_inject_enabled(plugin, messages)
    if skip_check is not None:
        return skip_check

    # 进入注入流程 → 计数
    _injection_metrics["total"] += 1

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
    traits = query_active_traits_for_injection(stream_id=stream_id, limit=40)
    text = _extract_user_text(messages)
    now_ts = time.time()
    await _prune_recent_injection(now_ts)

    # 应用冷却筛选（批量查，单次锁获取）
    if cooldown_seconds > 0 and max_traits > 0:
        filtered_traits, cooldown_skipped = await _batch_cooldown_filter(
            stream_id, traits, now_ts, cooldown_seconds
        )
    else:
        filtered_traits = traits
        cooldown_skipped = []

    # 冷却跳过计数
    if cooldown_skipped:
        _injection_metrics["skipped_cooldown"] += len(cooldown_skipped)

    selected, selection_mode, picked = _select_traits(
        filtered_traits, text, stream_id,
        max_traits, fallback_recent_impact, now_ts,
    )

    # 命中/无 trait 计数
    if selected:
        _injection_metrics["traits_hit"] += 1
    else:
        _injection_metrics["skipped_no_traits"] += 1

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

    # ── 自评反馈摘要（仅 self_reflection.enabled，按 selection_mode 分场景注入）──
    reflection_summary = ""
    if plugin.config.self_reflection.enabled:
        from .reflection_feedback import build_recent_reflection_summary

        reflection_summary = build_recent_reflection_summary(stream_id)

    # ── v2.4.0: 发酵中种子"思考中"提示（仅 fermentation_enabled）──
    fermenting_hint = ""
    if getattr(plugin.config.thought_cabinet, "fermentation_enabled", False):
        from ..models.seeds import get_fermenting_seeds

        fermenting_seeds = get_fermenting_seeds()
        # 只取与当前群相关的发酵种子
        relevant = [s for s in fermenting_seeds if s.stream_id == stream_id or s.stream_id == "global"]
        if relevant:
            hints = [f"「{s.seed_type}: {s.event[:40]}」" for s in relevant[:2]]
            fermenting_hint = "\n近期正在思考的问题（尚未形成结论，仅作背景参考）：" + " ".join(hints) + "\n"

    injection_block = _build_injection_block(ideology_prompt, p1_blocks, trait_lines, reflection_summary)
    if fermenting_hint:
        # 插入到 trait 块之后、收束指令之前
        injection_block = injection_block.replace(
            "请综合上述倾向与固化观点来组织回复",
            fermenting_hint + "请综合上述倾向与固化观点来组织回复",
        )

    # ── 7. 注入到 messages ─────────────────────────────────────────
    modified_messages, inject_strategy = _apply_soul_injection_to_messages(messages, injection_block)
    if modified_messages is None:
        # 无法安全合并：记录 skip 日志后 continue（不改 messages）
        await _record_injection(
            {
                "ts": datetime.now().isoformat(),
                "skipped": True,
                "reason": inject_strategy,
                "policy": "disabled",
                "prompt_version": "v2.4.0",
            },
            plugin_dir=plugin_dir,
        )
        return {"success": True, "action": "continue"}

    # ── 8. 日志 & 冷却标记 ──────────────────────────────────────────
    policy = _policy_from_selection(selection_mode, picked)
    await _record_injection(
        {
            "ts": datetime.now().isoformat(),
            "policy": policy,
            "picked": picked,
            "selection_mode": selection_mode,
            "inject_strategy": inject_strategy,
            "cooldown_seconds": cooldown_seconds,
            "cooldown_skipped": cooldown_skipped[:20],
            "prompt_version": "v2.4.0",
        },
        plugin_dir=plugin_dir,
    )
    if selected:
        await _mark_injected(stream_id, [t.trait_id for t in selected], now_ts)

    # ── 9. 自评捕获：缓存上下文 + 落注入快照（仅 self_reflection.enabled）──
    # 缓存始终使用原始 messages（未注入），保持现有语义
    cache_session_context(session_id, messages)
    maybe_write_injection_snapshot(
        plugin, session_id, stream_id, selected, spectrum_dict, mood_lines, selection_mode,
    )

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
            "prompt_version": "v2.4.0",
        },
        plugin_dir=plugin_dir,
    )
    return {"success": True, "action": "continue"}
