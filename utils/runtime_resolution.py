"""Host-integrated runtime resolution for Mai-Soul-Engine."""

from __future__ import annotations

import logging
from typing import Any

from .spectrum_utils import chat_config_to_stream_id, parse_chat_id

logger = logging.getLogger(__name__)


async def resolve_monitored_group_stream(plugin: Any, config_id: str) -> str:
    """Resolve a configured group target to the host's current stream ID.

    The host owns session-ID construction because it may include account and
    routing scope.  The historical MD5 form is retained only as a compatibility
    fallback for hosts where no stream discovery capability is available.

    Resolution order:
      1. get_stream_by_group_id (fast-path from host's in-memory sessions)
      2. open_session (persistent session creation / recovery)
      3. chat_config_to_stream_id (MD5 fallback, only when no platform given)
      4. "" (empty = skip this target)
    """
    raw = str(config_id or "").strip()
    platform, group_id, chat_type = parse_chat_id(raw)
    if not platform:
        return chat_config_to_stream_id(raw)

    if group_id and chat_type == "group":
        stream_id = await _resolve_via_get_stream(plugin, platform, group_id)
        if stream_id:
            return stream_id

        stream_id = await _resolve_via_open_session(plugin, platform, group_id)
        if stream_id:
            return stream_id

    return ""


async def _resolve_via_get_stream(
    plugin: Any, platform: str, group_id: str,
) -> str:
    """Try get_stream_by_group_id (in-memory fast path)."""
    try:
        result = await plugin.ctx.chat.get_stream_by_group_id(
            group_id=group_id,
            platform=platform,
        )
        if isinstance(result, dict) and result.get("success"):
            stream = result.get("stream")
            if isinstance(stream, dict):
                stream_id = str(stream.get("session_id", "") or "").strip()
                if stream_id:
                    return stream_id
    except (RuntimeError, ValueError, OSError, AttributeError):
        pass
    return ""


async def _resolve_via_open_session(
    plugin: Any, platform: str, group_id: str,
) -> str:
    """Fallback: open_session to create / recover a persistent session.

    Tries two paths in order:
      1. plugin.ctx.chat.open_session (direct attribute)
      2. plugin.ctx.call_capability("chat.open_session", …)
    """
    # Path A — direct attribute
    try:
        result = await plugin.ctx.chat.open_session(
            platform=platform,
            group_id=group_id,
            chat_type="group",
        )
        sid = _extract_session_id(result)
        if sid:
            logger.info(
                "open_session resolved session %s for %s:%s",
                sid, platform, group_id,
            )
            return sid
    except (RuntimeError, ValueError, OSError, AttributeError):
        pass

    # Path B — call_capability
    try:
        result = await plugin.ctx.call_capability(
            "chat.open_session",
            platform=platform,
            group_id=group_id,
            chat_type="group",
        )
        sid = _extract_session_id(result)
        if sid:
            logger.info(
                "call_capability open_session resolved %s for %s:%s",
                sid, platform, group_id,
            )
            return sid
    except (RuntimeError, ValueError, OSError, AttributeError):
        pass

    return ""


def _extract_session_id(result: Any) -> str:
    """Extract session_id from various open_session return shapes."""
    if result is None:
        return ""
    if isinstance(result, dict):
        for key in ("session_id", "stream_id"):
            sid = str(result.get(key, "") or "").strip()
            if sid:
                return sid
        stream = result.get("stream") or result.get("session")
        if isinstance(stream, dict):
            sid = str(stream.get("session_id", "") or "").strip()
            if sid:
                return sid
    elif hasattr(result, "session_id"):
        sid = str(result.session_id or "").strip()
        if sid:
            return sid
    return ""


async def resolve_host_bot_self_ids(plugin: Any) -> list[str]:
    """Read the bot's QQ identity from host configuration, not plugin config."""
    try:
        result = await plugin.ctx.call_capability(
            "config.get",
            key="bot.qq_account",
            default="",
        )
    except (RuntimeError, ValueError, OSError, AttributeError):
        return []
    if not isinstance(result, dict) or not result.get("success"):
        return []
    account_id = str(result.get("value", "") or "").strip()
    return [f"qq:{account_id}"] if account_id else []


async def generate_soul_text(plugin: Any, prompt: str) -> Any:
    """Generate Soul analysis with the host Planner task and RPC timeout budget."""
    return await plugin.ctx.call_capability(
        "llm.generate",
        timeout_ms=120_000,
        prompt=prompt,
        model="planner",
    )
