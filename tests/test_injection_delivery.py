"""注入链路端到端测试：真实宿主 hook 载荷 → 注入 → 回写 items。

这是 issue #2 引出的核心回归：旧实现读 `messages`、回写 `modified_kwargs["messages"]`，
而宿主传/读的是 Context Item 的 `items`，导致**注入全程空转且无报错**。
本测试锁定「注入真的落到宿主会读取的键上」这一行为。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from .conftest import _import_soul_submodule

_TS = "2026-09-17T10:00:00+00:00"


def _sys_item(item_id: str, text: str) -> dict:
    return {
        "item_type": "SystemMessageItem",
        "meta": {"item_id": item_id, "logical_turn_id": None, "timestamp": _TS},
        "parts": [{"type": "text", "text": text}],
    }


def _user_item(item_id: str, text: str) -> dict:
    return {
        "item_type": "UserMessageItem",
        "meta": {"item_id": item_id, "logical_turn_id": None, "timestamp": _TS},
        "parts": [{"type": "text", "text": text}],
    }


def _host_kwargs() -> dict:
    return {
        "hook_name": "maisaka.planner.before_request",
        "items": [
            _sys_item("sys1", "你是 Mai，一个友善的群聊助手。\n"),
            _user_item("u1", "你怎么看这件事"),
        ],
        "item_schema_version": 3,
        "tool_definitions": [],
        "selected_history_count": 4,
        "built_message_count": 5,
        "selection_reason": "recent",
        "session_id": "qq-123-group",
    }


class _WVStub:
    """WorldviewService 桩：本测试只关心注入投递，不关心 P1 分层内容。"""

    def build_layer_trait_summary(self, *a: Any, **k: Any) -> str:
        return ""

    def mood_prompt_lines(self) -> list[str]:
        return []

    def build_graph_hint(self, *a: Any, **k: Any) -> str:
        return ""


def _plugin(tmp_path: Path, *, mode: str = "apply", enabled: bool = True) -> SimpleNamespace:
    """构造测试插件。

    必须提供 ``ctx.chat``（宿主的显式流列表接口）——注入器用它判定会话类型，
    不再猜 session_id 字符串。
    """

    class _Chat:
        async def get_group_streams(self, platform: str = "qq") -> list[str]:
            return ["qq-123-group"]

        async def get_private_streams(self, platform: str = "qq") -> list[str]:
            return ["qq-999-private"]

    return SimpleNamespace(
        ctx=SimpleNamespace(chat=_Chat()),
        config=SimpleNamespace(
            plugin=SimpleNamespace(enabled=enabled, mode=mode),
            injection=SimpleNamespace(
                scope="all",
                inject_private=False,
                max_traits=3,
                fallback_recent_impact=False,
                trait_cooldown_seconds=0,
            ),
            threshold=SimpleNamespace(custom_prompts={}, enable_extreme=False),
            monitor=SimpleNamespace(monitored_groups=[], excluded_groups=[]),
            self_reflection=SimpleNamespace(enabled=False),
            thought_cabinet=SimpleNamespace(fermentation_enabled=False),
        ),
        _plugin_dir=tmp_path,
        _data_dir=tmp_path / "data",
        _wv_service=_WVStub(),
        _wv_config_view=None,
    )


def _initialized_spectrum() -> SimpleNamespace:
    return SimpleNamespace(
        initialized=True,
        sincerity=70,
        engagement=60,
        closeness=50,
        directness=40,
    )


def test_injection_writes_back_to_items_key(tmp_path: Path) -> None:
    """真实 item 载荷 → 返回 modified_kwargs，且回写键是 items。"""
    injector = _import_soul_submodule("components.ideology_injector")
    plugin = _plugin(tmp_path)
    kwargs = _host_kwargs()

    with patch.object(injector, "get_or_create_spectrum", lambda *a, **k: _initialized_spectrum()), \
         patch.object(injector, "query_active_traits_for_injection", lambda *a, **k: []):
        result = asyncio.run(injector.inject_ideology(plugin, **kwargs))

    assert result.get("modified_kwargs") is not None, "注入未回写 → 宿主拿不到任何改动"
    modified = result["modified_kwargs"]
    assert "items" in modified, "回写键必须是 items（宿主只读 items）"
    assert "messages" not in modified

    sys_text = "".join(
        p.get("text", "") for p in modified["items"][0]["parts"] if p.get("type") == "text"
    )
    assert "[Mai-Soul 动态层" in sys_text
    assert "倾向：真诚+70" in sys_text or "真诚" in sys_text
    # 宿主会在 items 变化时重新反序列化，schema 版本必须原样带回
    assert modified["item_schema_version"] == kwargs["item_schema_version"]
    # 用户项不得被改动
    assert modified["items"][1] == kwargs["items"][1]


def test_injection_skips_when_no_system_item(tmp_path: Path) -> None:
    """无 system item → 放弃注入（fail-open），不得凭空 prepend。"""
    injector = _import_soul_submodule("components.ideology_injector")
    plugin = _plugin(tmp_path)
    kwargs = _host_kwargs()
    kwargs["items"] = [_user_item("u1", "只有用户消息")]

    with patch.object(injector, "get_or_create_spectrum", lambda *a, **k: _initialized_spectrum()), \
         patch.object(injector, "query_active_traits_for_injection", lambda *a, **k: []):
        result = asyncio.run(injector.inject_ideology(plugin, **kwargs))

    assert "modified_kwargs" not in result
    assert result.get("action") == "continue"


def test_injection_skipped_when_plugin_disabled(tmp_path: Path) -> None:
    """插件 disabled → 不注入。"""
    injector = _import_soul_submodule("components.ideology_injector")
    plugin = _plugin(tmp_path, mode="off", enabled=False)

    result = asyncio.run(injector.inject_ideology(plugin, **_host_kwargs()))

    assert "modified_kwargs" not in result


def test_injection_skipped_in_observe_mode(tmp_path: Path) -> None:
    """观察模式：可以学习，但**不得影响真实回复**。"""
    injector = _import_soul_submodule("components.ideology_injector")
    plugin = _plugin(tmp_path, mode="observe")

    result = asyncio.run(injector.inject_ideology(plugin, **_host_kwargs()))

    assert "modified_kwargs" not in result, "observe 模式不得注入回复请求"


def test_injection_skipped_for_legacy_enabled_without_mode(tmp_path: Path) -> None:
    """旧配置 enabled=true 且未写 mode → 不注入（不得隐式生效）。"""
    injector = _import_soul_submodule("components.ideology_injector")
    plugin = _plugin(tmp_path, mode="", enabled=True)

    result = asyncio.run(injector.inject_ideology(plugin, **_host_kwargs()))

    assert "modified_kwargs" not in result


def test_injection_skipped_when_spectrum_uninitialized(tmp_path: Path) -> None:
    """光谱未初始化 → 不注入（要先 /soul_setup）。"""
    injector = _import_soul_submodule("components.ideology_injector")
    plugin = _plugin(tmp_path)
    uninit = _initialized_spectrum()
    uninit.initialized = False

    with patch.object(injector, "get_or_create_spectrum", lambda *a, **k: uninit), \
         patch.object(injector, "query_active_traits_for_injection", lambda *a, **k: []):
        result = asyncio.run(injector.inject_ideology(plugin, **_host_kwargs()))

    assert "modified_kwargs" not in result


def test_injection_records_snapshot_with_items_payload(tmp_path: Path) -> None:
    """注入日志与自评上下文缓存能用 item 载荷工作（不再取空）。"""
    injector = _import_soul_submodule("components.ideology_injector")
    capture = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin(tmp_path)
    kwargs = _host_kwargs()

    with patch.object(injector, "get_or_create_spectrum", lambda *a, **k: _initialized_spectrum()), \
         patch.object(injector, "query_active_traits_for_injection", lambda *a, **k: []):
        asyncio.run(injector.inject_ideology(plugin, **kwargs))

    # 自评上下文缓存应从 item 载荷里取到触发消息
    lines = capture.take_cached_context("qq-123-group")
    assert lines == ["你怎么看这件事"]
