"""Host-integrated runtime resolution for Mai-Soul-Engine."""

from __future__ import annotations

from typing import Any

from .spectrum_utils import chat_config_to_stream_id, parse_chat_id


async def resolve_monitored_group_stream(plugin: Any, config_id: str) -> str:
    """Resolve a configured group target to the host's current stream ID.

    The host owns session-ID construction because it may include account and
    routing scope.  The historical MD5 form is retained only as a compatibility
    fallback for hosts where no stream discovery capability is available.
    """
    platform, group_id, chat_type = parse_chat_id(config_id)
    if platform and group_id and chat_type == "group":
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
    return chat_config_to_stream_id(config_id)


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
