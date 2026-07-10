"""Runtime-regression tests for Soul host integration."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from .conftest import _import_soul_submodule


def test_group_resolution_uses_host_account_scoped_stream() -> None:
    runtime = _import_soul_submodule("utils.runtime_resolution")

    class Chat:
        async def get_stream_by_group_id(self, group_id: str, platform: str):
            assert (platform, group_id) == ("qq", "902106123")
            return {"success": True, "stream": {"session_id": "current-account-scoped-stream"}}

    plugin = SimpleNamespace(ctx=SimpleNamespace(chat=Chat()))
    stream_id = asyncio.run(runtime.resolve_monitored_group_stream(plugin, "qq:902106123:group"))

    assert stream_id == "current-account-scoped-stream"


def test_group_resolution_returns_empty_when_host_has_no_stream() -> None:
    runtime = _import_soul_submodule("utils.runtime_resolution")

    class Chat:
        async def get_stream_by_group_id(self, group_id: str, platform: str):
            return {"success": True, "stream": None}

    plugin = SimpleNamespace(ctx=SimpleNamespace(chat=Chat()))
    stream_id = asyncio.run(runtime.resolve_monitored_group_stream(plugin, "qq:902106123:group"))

    assert stream_id == ""


def test_group_resolution_falls_back_to_open_session() -> None:
    """get_stream_by_group_id returns empty → open_session resolves → stream_id."""
    runtime = _import_soul_submodule("utils.runtime_resolution")

    class Chat:
        async def get_stream_by_group_id(self, group_id: str, platform: str):
            return {"success": True, "stream": None}

        async def open_session(self, platform: str, group_id: str, chat_type: str):
            return {"session_id": "session-from-open-session"}

    plugin = SimpleNamespace(ctx=SimpleNamespace(chat=Chat()))
    stream_id = asyncio.run(runtime.resolve_monitored_group_stream(plugin, "qq:12345:group"))

    assert stream_id == "session-from-open-session"


def test_group_resolution_open_session_extracts_stream_key() -> None:
    """open_session returns nested stream dict → extracts session_id."""
    runtime = _import_soul_submodule("utils.runtime_resolution")

    class Chat:
        async def get_stream_by_group_id(self, group_id: str, platform: str):
            return {"success": True, "stream": None}

        async def open_session(self, platform: str, group_id: str, chat_type: str):
            return {"stream": {"session_id": "nested-session"}}

    plugin = SimpleNamespace(ctx=SimpleNamespace(chat=Chat()))
    stream_id = asyncio.run(runtime.resolve_monitored_group_stream(plugin, "qq:12345:group"))

    assert stream_id == "nested-session"


def test_group_resolution_get_stream_raises_open_session_succeeds() -> None:
    """get_stream_by_group_id raises → open_session fallback succeeds."""
    runtime = _import_soul_submodule("utils.runtime_resolution")

    class Chat:
        async def get_stream_by_group_id(self, group_id: str, platform: str):
            raise RuntimeError("host not ready")

        async def open_session(self, platform: str, group_id: str, chat_type: str):
            return {"session_id": "fallback-session"}

    plugin = SimpleNamespace(ctx=SimpleNamespace(chat=Chat()))
    stream_id = asyncio.run(runtime.resolve_monitored_group_stream(plugin, "qq:12345:group"))

    assert stream_id == "fallback-session"


def test_group_resolution_both_fail_returns_empty() -> None:
    """Both get_stream_by_group_id and open_session fail → returns ''."""
    runtime = _import_soul_submodule("utils.runtime_resolution")

    class Chat:
        async def get_stream_by_group_id(self, group_id: str, platform: str):
            return {"success": True, "stream": None}

    plugin = SimpleNamespace(ctx=SimpleNamespace(chat=Chat()))
    stream_id = asyncio.run(runtime.resolve_monitored_group_stream(plugin, "qq:12345:group"))

    assert stream_id == ""


def test_group_resolution_no_platform_uses_md5_fallback() -> None:
    """No platform prefix → chat_config_to_stream_id fallback."""
    runtime = _import_soul_submodule("utils.runtime_resolution")

    plugin = SimpleNamespace(ctx=SimpleNamespace(chat=SimpleNamespace()))
    stream_id = asyncio.run(
        runtime.resolve_monitored_group_stream(plugin, "plain_group_id")
    )

    # chat_config_to_stream_id with no platform → returns raw group_id
    assert stream_id == "plain_group_id"


def test_soul_text_generation_uses_planner_task_with_long_rpc_timeout() -> None:
    runtime = _import_soul_submodule("utils.runtime_resolution")
    captured: dict[str, object] = {}

    class Context:
        async def call_capability(self, capability: str, timeout_ms: int, **kwargs: object):
            captured["capability"] = capability
            captured["timeout_ms"] = timeout_ms
            captured.update(kwargs)
            return {"success": True, "response": "ok"}

    plugin = SimpleNamespace(ctx=Context())
    result = asyncio.run(runtime.generate_soul_text(plugin, "evaluate this"))

    assert captured == {
        "capability": "llm.generate",
        "timeout_ms": 120000,
        "prompt": "evaluate this",
        "model": "planner",
    }
    assert result["response"] == "ok"


def test_host_bot_identity_is_read_from_main_config() -> None:
    runtime = _import_soul_submodule("utils.runtime_resolution")

    class Context:
        async def call_capability(self, capability: str, **kwargs):
            assert capability == "config.get"
            assert kwargs == {"key": "bot.qq_account", "default": ""}
            return {"success": True, "value": "3430049585"}

    plugin = SimpleNamespace(ctx=Context())
    identities = asyncio.run(runtime.resolve_host_bot_self_ids(plugin))

    assert identities == ["qq:3430049585"]


def test_monitor_schema_has_no_plugin_side_bot_identity() -> None:
    schema = _import_soul_submodule("plugin_ui_schema")
    assert "bot_self_id" not in schema.MonitorConfig.model_fields
