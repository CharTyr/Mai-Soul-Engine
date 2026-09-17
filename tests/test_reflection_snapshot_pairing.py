"""注入快照 ↔ 回复 的配对正确性（T03）与投递阶段可观测（T04）。

问题：旧配对是「取该 session **最新** 一条快照」，且触发上下文放在
**session 键**的内存缓存里一次性取走。同会话两轮并发时：

- 轮次 A 的回复可能配上轮次 B 的快照（trait 归属错）
- 轮次 A 的回复可能取走轮次 B 缓存的触发上文（上下文错）

修复方向：上下文随快照一起落库；配对改为「未消费且最旧」的 FIFO 认领，
并对同一 reply_message_id 的重试复用同一快照。
"""

from __future__ import annotations

import json
from typing import Any

from .conftest import _import_soul_submodule


def _sr() -> Any:
    return _import_soul_submodule("models.self_reflection")


def _mk_snapshot(sr: Any, session_id: str, trait_ids: list[str], context_lines: list[str]) -> str:
    return sr.create_injection_snapshot(
        stream_id=session_id,
        session_id=session_id,
        trait_ids_json=json.dumps(trait_ids, ensure_ascii=False),
        spectrum_json=json.dumps({"sincerity": 60}),
        mood_json="{}",
        selection_mode="tag_hit",
        context_fingerprint="",
        context_json=json.dumps(context_lines, ensure_ascii=False),
    )


# ─── T03：并发两轮不混用 ────────────────────────────────────────────


def test_overlapping_turns_pair_to_own_snapshot(soul_db: Any) -> None:
    """两轮并发：先到的回复认领第一轮快照，后到的认领第二轮。"""
    sr = _sr()
    s1 = _mk_snapshot(sr, "sess-1", ["trait-A"], ["第一轮的触发消息"])
    s2 = _mk_snapshot(sr, "sess-1", ["trait-B"], ["第二轮的触发消息"])

    first = sr.claim_snapshot_for_response("sess-1", "msg-1")
    second = sr.claim_snapshot_for_response("sess-1", "msg-2")

    assert first is not None and second is not None
    assert first.snapshot_id == s1
    assert second.snapshot_id == s2
    assert json.loads(first.trait_ids_json) == ["trait-A"]
    assert json.loads(second.trait_ids_json) == ["trait-B"]


def test_context_is_bound_to_snapshot_not_session(soul_db: Any) -> None:
    """触发上下文必须随快照走——否则后一轮的上下文会顶掉前一轮。"""
    sr = _sr()
    _mk_snapshot(sr, "sess-2", ["trait-A"], ["第一轮触发"])
    _mk_snapshot(sr, "sess-2", ["trait-B"], ["第二轮触发"])

    first = sr.claim_snapshot_for_response("sess-2", "msg-1")
    assert json.loads(first.context_json) == ["第一轮触发"]


def test_retry_of_same_reply_reuses_snapshot(soul_db: Any) -> None:
    """同一 reply_message_id 的重试必须复用同一快照（1:N 的合法场景）。"""
    sr = _sr()
    s1 = _mk_snapshot(sr, "sess-3", ["trait-A"], ["触发"])

    a = sr.claim_snapshot_for_response("sess-3", "msg-9")
    b = sr.claim_snapshot_for_response("sess-3", "msg-9")

    assert a.snapshot_id == s1
    assert b.snapshot_id == s1


def test_no_snapshot_returns_none_without_crash(soul_db: Any) -> None:
    """无可用快照 → None（合法降级），不得抛异常。"""
    sr = _sr()
    assert sr.claim_snapshot_for_response("sess-none", "msg-1") is None


def test_consumed_snapshot_not_reclaimed(soul_db: Any) -> None:
    """已被认领的快照不会被另一条回复再次认领。"""
    sr = _sr()
    _mk_snapshot(sr, "sess-4", ["trait-A"], ["触发"])

    first = sr.claim_snapshot_for_response("sess-4", "msg-1")
    second = sr.claim_snapshot_for_response("sess-4", "msg-2")

    assert first is not None
    assert second is None


def test_claim_only_touches_own_session(soul_db: Any) -> None:
    """不同 session 的快照互不干扰。"""
    sr = _sr()
    sa = _mk_snapshot(sr, "sess-A", ["trait-A"], ["A 触发"])
    _mk_snapshot(sr, "sess-B", ["trait-B"], ["B 触发"])

    claimed = sr.claim_snapshot_for_response("sess-A", "msg-1")

    assert claimed.snapshot_id == sa


# ─── T04：投递阶段可观测（selected / hook_applied / final_request_verified）──


def test_snapshot_defaults_to_selected_state(soul_db: Any) -> None:
    """新快照初始投递态是 selected（尚未确认进入请求）。"""
    sr = _sr()
    sid = _mk_snapshot(sr, "sess-5", ["trait-A"], ["触发"])
    snap = sr.get_injection_snapshot(sid)
    assert snap.delivery_state == "selected"


def test_delivery_state_transitions_are_recorded(soul_db: Any) -> None:
    """投递态可推进到 hook_applied / final_request_verified。"""
    sr = _sr()
    sid = _mk_snapshot(sr, "sess-6", ["trait-A"], ["触发"])

    assert sr.mark_snapshot_delivery_state(sid, "hook_applied") is True
    assert sr.get_injection_snapshot(sid).delivery_state == "hook_applied"

    assert sr.mark_snapshot_delivery_state(sid, "final_request_verified") is True
    assert sr.get_injection_snapshot(sid).delivery_state == "final_request_verified"


def test_unverified_state_is_distinguishable(soul_db: Any) -> None:
    """无法确认最终请求时必须落 unverified，不能记成成功。"""
    sr = _sr()
    sid = _mk_snapshot(sr, "sess-7", ["trait-A"], ["触发"])
    sr.mark_snapshot_delivery_state(sid, "unverified")
    assert sr.get_injection_snapshot(sid).delivery_state == "unverified"


def test_invalid_delivery_state_rejected(soul_db: Any) -> None:
    """未知投递态不得写入。"""
    sr = _sr()
    sid = _mk_snapshot(sr, "sess-8", ["trait-A"], ["触发"])
    assert sr.mark_snapshot_delivery_state(sid, "made_up_state") is False
    assert sr.get_injection_snapshot(sid).delivery_state == "selected"
