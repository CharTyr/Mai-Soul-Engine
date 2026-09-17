"""P1.2+P1.3: 捕获+配对层测试。

验证：
- 上下文缓存 TTL + 一次性取 + 缺失降级
- 注入快照仅 enabled 时写（防膨胀守卫）
- after_response 捕获入队 + enabled 守卫 + 空回复跳过 + 缺失 context 降级

从宿主仓根运行：``uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/test_reflection_capture.py -q``
"""

from __future__ import annotations

import json

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


def test_planner_after_response_is_not_enqueued(soul_db: Any) -> None:
    rc = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin_with_self_reflection(enabled=True)
    rc.cache_session_context("planner-sess", [{"role": "user", "content": "你觉得呢"}])
    rc.maybe_write_injection_snapshot(
        plugin, "planner-sess", "g", [], {"sincerity": 50}, [], "spectrum_only"
    )
    result = asyncio.run(
        rc.capture_after_response(plugin, "planner", response="我建议...", session_id="planner-sess")
    )
    assert result["action"] == "continue"
    assert soul_db.list_pending_reflections(limit=10) == []


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


def test_capture_pairs_with_oldest_unclaimed_snapshot(soul_db: Any) -> None:
    """同 session 多次注入，after_response 认领**最旧的未认领**快照（FIFO）。

    原测试断言「配对最近一条快照」（写两条 → 配第二条）。该行为在并发下是错的：
    轮次 A 的回复会配上轮次 B 刚写入的快照，导致评估用错 trait 集合。
    新契约是 FIFO 认领 + 1:1 归属，故此处按新行为重写。
    """
    rc = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin_with_self_reflection(enabled=True)
    # 写两条快照，第二条 selection_mode 不同
    rc.maybe_write_injection_snapshot(plugin, "sess", "g", [], {"sincerity": 50}, [], "spectrum_only")
    rc.maybe_write_injection_snapshot(plugin, "sess", "g", [], {"sincerity": 50}, [], "tag_hit")
    asyncio.run(rc.capture_after_response(plugin, "replyer", response="回复", session_id="sess", reply_message_id="m1"))
    pendings = soul_db.list_pending_reflections(limit=10)
    assert len(pendings) == 1
    # 认领的是最旧那条（spectrum_only），而不是最近写入的 tag_hit
    snap = soul_db.get_injection_snapshot(pendings[0].snapshot_id)
    assert snap is not None
    assert snap.selection_mode == "spectrum_only"

    # 第二条回复认领剩下那条 —— 两条互不重叠
    asyncio.run(rc.capture_after_response(plugin, "replyer", response="回复2", session_id="sess", reply_message_id="m2"))
    pendings = soul_db.list_pending_reflections(limit=10)
    assert len(pendings) == 2
    by_reply = {p.reply_message_id: p for p in pendings}
    second = soul_db.get_injection_snapshot(by_reply["m2"].snapshot_id)
    assert second is not None
    assert second.selection_mode == "tag_hit"
    first = soul_db.get_injection_snapshot(by_reply["m1"].snapshot_id)
    assert first.selection_mode == "spectrum_only"


def test_capture_uses_context_bound_to_its_snapshot(soul_db: Any) -> None:
    """触发上下文来自快照本身，不是被后一轮覆盖的 session 缓存。"""
    rc = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin_with_self_reflection(enabled=True)
    rc.maybe_write_injection_snapshot(
        plugin, "sess-ctx", "g", [], {"sincerity": 50}, [], "spectrum_only",
        context_lines=["第一轮触发"],
    )
    rc.maybe_write_injection_snapshot(
        plugin, "sess-ctx", "g", [], {"sincerity": 50}, [], "spectrum_only",
        context_lines=["第二轮触发"],
    )
    asyncio.run(rc.capture_after_response(plugin, "replyer", response="回复", session_id="sess-ctx", reply_message_id="m1"))
    pendings = soul_db.list_pending_reflections(limit=10)
    assert json.loads(pendings[0].context_json) == ["第一轮触发"]


def test_replyer_capture_deduplicates_reply_message_id(soul_db: Any) -> None:
    rc = _import_soul_submodule("components.reflection_capture")
    plugin = _plugin_with_self_reflection(enabled=True)
    kwargs = {"response": "最终可见回复", "session_id": "sess", "reply_message_id": "same-message"}
    asyncio.run(rc.capture_after_response(plugin, "replyer", **kwargs))
    asyncio.run(rc.capture_after_response(plugin, "replyer", **kwargs))
    pendings = soul_db.list_pending_reflections(limit=10)
    assert len(pendings) == 1
    assert pendings[0].reply_message_id == "same-message"
