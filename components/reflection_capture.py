"""自我评价捕获层：after_response hook 委托 + before_request 上下文缓存/快照。

两个 OBSERVE 模式 HookHandler（不改写输出，零干扰宿主主流程）：
- ``maisaka.planner.after_response`` → 拿 planner LLM 决策输出
- ``maisaka.replyer.after_response`` → 拿最终回复文本

before_request 侧（由 ``ideology_injector.inject_ideology`` 调用）：
- ``cache_session_context``：缓存触发上文（session_id 作 key，TTL bound）
- ``maybe_write_injection_snapshot``：仅 ``[self_reflection].enabled`` 时落注入快照

after_response 侧（``capture_after_response``）：
- 取最近 snapshot 配对（1:N）+ 取缓存 context → 入队 ``soul_pending_reflections``

设计要点（见 .slim/deepwork/self-reflection.md oracle 审查）：
- after_response payload 不含触发消息 → context 从 before_request 内存缓存取；
  缓存缺失/超龄 → context_json 空，是合法降级路径（评估只基于 response 文本）。
- snapshot 仅 enabled 时写，防表膨胀（oracle 修订点 4）。
- OBSERVE 模式 + error_policy=SKIP → 捕获失败不影响 bot 正常回复。
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import time
from typing import Any

from ..utils.spectrum_utils import sanitize_text

logger = logging.getLogger(__name__)

# 注：models.self_reflection 的导入放在函数内懒加载，避免在模块加载期触发
# ``models._conn → worldview → worldview.service → models.ideology_model → models.history → _conn``
# 的预存循环导入（仅在测试隔离首个导入 models 时暴露）。热路径 cache_session_context
# 不依赖 models，无懒导入开销；maybe_write_injection_snapshot / capture_after_response
# 仅在 [self_reflection].enabled 时调用，此时插件已完全加载。

# ─── 上下文缓存（before → after 配对用）────────────────────────────

_CONTEXT_TTL_SECONDS: int = 600  # 10 分钟
_CONTEXT_MAX_ENTRIES: int = 256
_CONTEXT_MAX_LINES: int = 6
_CONTEXT_LINE_MAX_CHARS: int = 200

# session_id -> (context_lines, timestamp)
_context_cache: dict[str, tuple[list[str], float]] = {}


def cache_session_context(session_id: str, messages: list[dict]) -> None:
    """从 before_request 的 messages 提取最近用户消息，缓存供 after_response 用。

    after_response payload 不含触发消息，故在此缓存。空 session_id 不缓存。
    """
    if not session_id:
        return
    lines: list[str] = []
    for msg in reversed(messages):
        if len(lines) >= _CONTEXT_MAX_LINES:
            break
        if msg.get("role") == "user":
            content = str(msg.get("content", "") or "")
            sanitized = sanitize_text(content, max_chars=_CONTEXT_LINE_MAX_CHARS)
            if sanitized:
                lines.append(sanitized)
    lines.reverse()
    # 容量控制：超上限删最旧一半
    if len(_context_cache) > _CONTEXT_MAX_ENTRIES:
        sorted_items = sorted(_context_cache.items(), key=lambda x: x[1][1])
        for k, _ in sorted_items[: len(sorted_items) // 2]:
            _context_cache.pop(k, None)
    _context_cache[session_id] = (lines, time.time())


def take_cached_context(session_id: str) -> list[str]:
    """取并清除缓存（一次性）。超龄或缺失返回空列表（合法降级）。"""
    if not session_id:
        return []
    entry = _context_cache.pop(session_id, None)
    if not entry:
        return []
    lines, ts = entry
    if time.time() - float(ts) > _CONTEXT_TTL_SECONDS:
        return []
    return lines


# ─── 注入快照（before 侧）─────────────────────────────────────────


def maybe_write_injection_snapshot(
    plugin,
    session_id: str,
    stream_id: str,
    selected_traits: list,
    spectrum_dict: dict,
    mood_lines: list[str],
    selection_mode: str,
) -> str:
    """仅 ``[self_reflection].enabled`` 时落注入快照，返回 snapshot_id（否则空串）。

    防表膨胀：调用方无需判断 enabled，本函数内部守卫（oracle 修订点 4）。
    """
    if not plugin.config.self_reflection.enabled:
        return ""
    trait_ids = [t.trait_id for t in selected_traits if t.trait_id]
    # context_fingerprint：session_id + trait_ids + selection_mode 简单哈希，用于去重
    fp_src = f"{session_id}|{','.join(trait_ids)}|{selection_mode}"
    fingerprint = hashlib.md5(fp_src.encode()).hexdigest()[:16]
    mood_json = _json.dumps({"lines": mood_lines}, ensure_ascii=False) if mood_lines else "{}"
    try:
        from ..models.self_reflection import create_injection_snapshot

        return create_injection_snapshot(
            stream_id=stream_id or "global",
            session_id=session_id,
            trait_ids_json=_json.dumps(trait_ids, ensure_ascii=False),
            spectrum_json=_json.dumps(spectrum_dict, ensure_ascii=False),
            mood_json=mood_json,
            selection_mode=selection_mode,
            context_fingerprint=fingerprint,
        )
    except Exception:
        logger.exception("[SelfReflection] 写注入快照失败")
        return ""


# ─── after_response 捕获 ──────────────────────────────────────────


async def capture_after_response(plugin, source: str, **kwargs: Any) -> dict[str, Any]:
    """after_response hook 委托：捕获 bot 回复入待评队列。

    OBSERVE 模式——不改写输出，原样返回。失败由 error_policy=SKIP 兜底。

    Args:
        plugin: 插件实例。
        source: ``"planner"`` 或 ``"replyer"``，区分决策输出与最终回复。
        **kwargs: hook payload（response / session_id / reply_message_id 等）。
    """
    if not plugin.config.self_reflection.enabled:
        return {"success": True, "action": "continue"}
    response = str(kwargs.get("response", "") or "").strip()
    if not response:
        return {"success": True, "action": "continue"}
    session_id = str(kwargs.get("session_id", "") or "")
    reply_message_id = str(kwargs.get("reply_message_id", "") or "")
    stream_id = session_id or "global"
    try:
        from ..models.self_reflection import (
            create_pending_reflection,
            get_latest_snapshot_for_session,
        )

        snapshot = get_latest_snapshot_for_session(session_id)
        snapshot_id = snapshot.snapshot_id if snapshot else ""
        context_lines = take_cached_context(session_id)
        context_json = _json.dumps(context_lines, ensure_ascii=False) if context_lines else "[]"
        create_pending_reflection(
            stream_id=stream_id,
            session_id=session_id,
            reply_message_id=reply_message_id,
            snapshot_id=snapshot_id,
            source=source,
            response_text=response,
            context_json=context_json,
        )
    except Exception:
        logger.exception("[SelfReflection] 捕获 %s after_response 失败", source)
    # OBSERVE：不改写，原样返回（无 modified_kwargs → 宿主保留原输出）
    return {"success": True, "action": "continue"}
