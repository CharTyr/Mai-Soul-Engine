"""Read host persona configuration for injection into thought prompts.

Provides a snapshot of the host bot's configured personality / reply style
so that internalization and discovery prompts can reference the bot's fixed
persona baseline — preventing the soul engine from generating thoughts that
contradict the operator's explicit bot design.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── module-level cache ──────────────────────────────────────────────
_cache: tuple[str, Any, float] | None = None  # (hash, snapshot, ts)
_CACHE_TTL: float = 300.0  # 5 minutes


@dataclass
class HostPersonaSnapshot:
    """Immutable view of the host bot's persona fields at a point in time."""

    personality_text: str = ""
    reply_style_text: str = ""
    profile_hash: str = ""
    source: str = "empty"
    raw: dict[str, str] = field(default_factory=dict)


async def fetch_host_persona(plugin: Any) -> HostPersonaSnapshot:
    """Read host bot persona via ``config.get`` capabilities.

    Reads ``personality.personality`` and ``personality.reply_style``
    from host configuration.  Missing keys produce empty strings (not None).

    Results are cached at module level for ``_CACHE_TTL`` seconds because
    persona config rarely changes at runtime.
    """
    global _cache
    now = time.time()
    if _cache is not None and (now - _cache[2]) < _CACHE_TTL:
        return _cache[1]

    snap = await _do_fetch(plugin)
    _cache = (snap.profile_hash, snap, now)
    return snap


def invalidate_host_persona_cache() -> None:
    """Force next ``fetch_host_persona`` to re-read config."""
    global _cache
    _cache = None


async def _do_fetch(plugin: Any) -> HostPersonaSnapshot:
    personality = ""
    reply_style = ""

    # personality.personality
    try:
        result = await plugin.ctx.call_capability(
            "config.get", key="personality.personality", default="",
        )
        if isinstance(result, dict) and result.get("success"):
            personality = str(result.get("value", "") or "").strip()
    except (RuntimeError, ValueError, OSError, AttributeError):
        pass

    # personality.reply_style
    try:
        result = await plugin.ctx.call_capability(
            "config.get", key="personality.reply_style", default="",
        )
        if isinstance(result, dict) and result.get("success"):
            reply_style = str(result.get("value", "") or "").strip()
    except (RuntimeError, ValueError, OSError, AttributeError):
        pass

    raw = {"personality": personality, "reply_style": reply_style}

    if not personality and not reply_style:
        return HostPersonaSnapshot(source="empty", raw=raw)

    profile_hash = _normalize_hash(personality, reply_style)
    return HostPersonaSnapshot(
        personality_text=personality,
        reply_style_text=reply_style,
        profile_hash=profile_hash,
        source="config.get",
        raw=raw,
    )


def _normalize_hash(personality: str, reply_style: str) -> str:
    """Stable SHA-256 fingerprint (first 16 hex chars) of the persona."""
    text = (personality.strip() + "|" + reply_style.strip()).replace(
        "\n", " "
    ).replace("\r", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def format_persona_for_prompt(
    snap: HostPersonaSnapshot, max_chars: int = 800,
) -> str:
    """Short summary for insertion into LLM prompts.

    Returns an empty string when the snapshot is empty (no persona configured).
    """
    if not snap.personality_text and not snap.reply_style_text:
        return ""

    parts: list[str] = []
    if snap.personality_text:
        parts.append(f"人格设定：{snap.personality_text[:400]}")
    if snap.reply_style_text:
        parts.append(f"回复风格：{snap.reply_style_text[:400]}")

    block = "；".join(parts)
    if len(block) > max_chars:
        block = block[:max_chars]
    return (
        "\n\n【宿主固定人设基底（不可与身份事实冲突；可延伸细化）】\n"
        f"{block}"
    )
