"""P1.2+P1.3: 捕获+配对层测试。

验证：
- 上下文缓存 TTL + 一次性取 + 缺失降级
- 注入快照仅 enabled 时写（防膨胀守卫）
- after_response 捕获入队 + enabled 守卫 + 空回复跳过 + 缺失 context 降级

从宿主仓根运行：``uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/test_reflection_capture.py -q``
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from .conftest import _import_soul_submodule


def _plugin_with_self_reflection(enabled: bool) -> SimpleNamespace:
    """构造一个带 config.self_reflection.enabled 的最小 plugin mock。"""
    return SimpleNamespace(
        config=SimpleNamespace(self_reflection=SimpleNamespace(enabled=enabled))
    )


# ─── 上下文缓存 ───────────────────────────────────────────────────


def test_context_cache_roundtrip() -> None:
    rc = _import_soul_submodule("components.reflection_capture")
    rc.cache_session_context(
        "s1", [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "hi"}]
    )
    ctx = rc.take_cached_context("s1")
    assert ctx == ["你好"]


def test_context_cache_one_time() -> None:
    rc = _import_soul_submodule("components.reflection_capture")
    rc.cache_session_context("s2", [{"role": "user", "content": "在吗"}])
    assert rc.take_cached_context("s2") == ["在吗"]
    assert rc.take_cached_context("s2") == []  # 一次性取走


def test_context_cache_missing_returns_empty() -> None:
    rc = _import_soul_submodule("components.reflection_capture")
    assert rc.take_cached_context("nope") == []


def test_context_cache_empty_session_not_cached() -> None:
    rc = _import_soul_submodule("components.reflection_capture")
    rc.cache_session_context("", [{"role": "user", "content": "x"}])
    assert rc.take_cached_context("") == []


# ─── 注入快照守卫 ─────────────────────────────────────────────────


def test_snapshot_not_written_when_disabled(soul_db: Any) -> None:
    """disabled 时不写快照（防膨胀，oracle 修订点 4）。"""
    rc = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin_with_self_reflection(enabled=False)
    sid = rc.maybe_write_injection_snapshot(
        plugin, "sess", "g", [], {"sincerity": 50}, [], "spectrum_only"
    )
    assert sid == ""
    assert soul_db.get_latest_snapshot_for_session("sess") is None


def test_snapshot_written_when_enabled(soul_db: Any) -> None:
    rc = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin_with_self_reflection(enabled=True)
    trait = SimpleNamespace(trait_id="t1")
    sid = rc.maybe_write_injection_snapshot(
        plugin, "sess", "g", [trait], {"sincerity": 60}, ["情绪行"], "tag_hit"
    )
    assert sid
    snap = soul_db.get_latest_snapshot_for_session("sess")
    assert snap is not None
    assert snap.snapshot_id == sid
    assert snap.selection_mode == "tag_hit"


# ─── after_response 捕获 ──────────────────────────────────────────


def test_capture_after_response_creates_pending(soul_db: Any) -> None:
    rc = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin_with_self_reflection(enabled=True)
    rc.cache_session_context("sess", [{"role": "user", "content": "你觉得呢"}])
    rc.maybe_write_injection_snapshot(
        plugin, "sess", "g", [], {"sincerity": 50}, [], "spectrum_only"
    )
    result = asyncio.run(
        rc.capture_after_response(plugin, "planner", response="我建议...", session_id="sess")
    )
    assert result["action"] == "continue"
    pendings = soul_db.list_pending_reflections(limit=10)
    assert len(pendings) == 1
    assert pendings[0].source == "planner"
    assert pendings[0].response_text == "我建议..."
    assert pendings[0].snapshot_id  # 配对到了快照
    assert "你觉得呢" in pendings[0].context_json


def test_capture_after_response_disabled_skips(soul_db: Any) -> None:
    rc = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin_with_self_reflection(enabled=False)
    asyncio.run(rc.capture_after_response(plugin, "replyer", response="hi", session_id="s"))
    assert len(soul_db.list_pending_reflections(limit=10)) == 0


def test_capture_after_response_empty_response_skips(soul_db: Any) -> None:
    rc = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin_with_self_reflection(enabled=True)
    asyncio.run(rc.capture_after_response(plugin, "planner", response="", session_id="s"))
    asyncio.run(rc.capture_after_response(plugin, "planner", response="   ", session_id="s"))
    assert len(soul_db.list_pending_reflections(limit=10)) == 0


def test_capture_after_response_missing_context_degrades(soul_db: Any) -> None:
    """缓存缺失时 context_json 为空，仍入队（合法降级，oracle 修订点 2）。"""
    rc = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin_with_self_reflection(enabled=True)
    # 不缓存上下文，直接捕获
    asyncio.run(rc.capture_after_response(plugin, "replyer", response="好的", session_id="sess"))
    pendings = soul_db.list_pending_reflections(limit=10)
    assert len(pendings) == 1
    assert pendings[0].context_json == "[]"


def test_capture_pairs_with_latest_snapshot(soul_db: Any) -> None:
    """同 session 多次注入，after_response 配对最近一条 snapshot（oracle 补充测试）。"""
    rc = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin_with_self_reflection(enabled=True)
    # 写两条快照，第二条 selection_mode 不同
    rc.maybe_write_injection_snapshot(plugin, "sess", "g", [], {"sincerity": 50}, [], "spectrum_only")
    rc.maybe_write_injection_snapshot(plugin, "sess", "g", [], {"sincerity": 50}, [], "tag_hit")
    asyncio.run(rc.capture_after_response(plugin, "planner", response="回复", session_id="sess"))
    pendings = soul_db.list_pending_reflections(limit=10)
    assert len(pendings) == 1
    # 配对的应是最近那条（tag_hit）
    snap = soul_db.get_injection_snapshot(pendings[0].snapshot_id)
    assert snap is not None
    assert snap.selection_mode == "tag_hit"
