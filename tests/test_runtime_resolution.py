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


def test_group_resolution_falls_back_to_legacy_hash_when_host_has_no_stream() -> None:
    runtime = _import_soul_submodule("utils.runtime_resolution")

    class Chat:
        async def get_stream_by_group_id(self, group_id: str, platform: str):
            return {"success": True, "stream": None}

    plugin = SimpleNamespace(ctx=SimpleNamespace(chat=Chat()))
    stream_id = asyncio.run(runtime.resolve_monitored_group_stream(plugin, "qq:902106123:group"))

    assert stream_id == "931772ff504085850356d5e82c5d4438"


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
