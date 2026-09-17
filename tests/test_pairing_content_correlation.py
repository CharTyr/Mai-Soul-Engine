"""快照配对：用**内容证据**替代顺序推断（T03 的插件侧补强）。

背景：宿主 planner/replyer 两侧没有共同请求标识（已核对源码），且宿主不可改。
旧规则只看"谁更旧"（FIFO）+ "窗口内多于一条就算歧义"。它有个漏洞：

    第 1 轮注入后回复没发出来 → 快照滞留；第 2 轮回复**自己没注入**
    → 窗口内恰好只剩 1 条未认领 → 被误配给第 2 轮 → 拿别人的人格反馈改光谱。

补强：replyer 腿本来就能看到**真实的生成请求 items**（宿主传的），从中取"本轮
回复在回答哪条消息"（尾行），在认领时与候选快照的触发上文尾行做内容比对。
证据唯一命中 → 精确认领（并发两轮也能分开）；全不命中 → 弃权（标歧义）；
无证据 → 退回旧规则。**不猜时间窗，不猜顺序。**
"""

from __future__ import annotations

import json
from typing import Any

from .conftest import _import_soul_submodule


def _sr() -> Any:
    return _import_soul_submodule("models.self_reflection")


def _snapshot(session: str, tail: str, *, mode: str = "tag_hit") -> str:
    """造一条带触发上文的快照。"""
    return _sr().create_injection_snapshot(
        "group-A", session, '["trait-x"]', "{}", "{}", mode,
        context_json=json.dumps([f"用户A: {tail}"], ensure_ascii=False),
    )


def _item(text: str) -> dict[str, Any]:
    return {"item_type": "user_message", "parts": [{"type": "text", "text": text}]}


# ─── 1. 堵漏：滞留的单条快照不得被误配 ──────────────────────────


def test_stale_single_snapshot_is_not_misattributed(soul_db: Any) -> None:
    """窗口内只有一条滞留快照，但其触发消息**不是**本轮在答的 → 必须弃权。

    这是旧规则的真实漏洞：只有一条候选时它无条件认领并判为"无歧义"，
    于是拿上一轮的人格反馈去评这一轮的回复。
    """
    _snapshot("sess-1", "旧话题：今天中午吃什么")

    claimed = _sr().claim_snapshot_for_response(
        "sess-1", "reply-2", reply_tail="用户B: 新话题：几点开会",
    )
    assert claimed is not None
    assert claimed.pairing_ambiguous is True, (
        "证据表明这条快照不是本轮注入的，却仍被判为无歧义——会拿错人格改光谱"
    )


def test_matching_single_snapshot_is_claimed_precisely(soul_db: Any) -> None:
    """唯一候选且触发消息对得上 → 精确认领（不算歧义）。"""
    _snapshot("sess-2", "这事你怎么看")

    claimed = _sr().claim_snapshot_for_response(
        "sess-2", "reply-1", reply_tail="用户A: 这事你怎么看",
    )
    assert claimed is not None
    assert claimed.pairing_ambiguous is False


# ─── 2. 精度提升：并发两轮用内容分开 ────────────────────────────


def test_two_pending_are_disambiguated_by_content(soul_db: Any) -> None:
    """两条未认领候选，内容证据唯一命中第二条 → 认领第二条且**不算歧义**。

    旧规则在这里只能按"最旧"消费并标歧义（放弃反馈）；内容证据能分清。
    """
    _snapshot("sess-3", "第一轮：周末去哪玩")
    second = _snapshot("sess-3", "第二轮：这个 bug 怎么修")

    claimed = _sr().claim_snapshot_for_response(
        "sess-3", "reply-2", reply_tail="用户B: 第二轮：这个 bug 怎么修",
    )
    assert claimed is not None
    assert claimed.snapshot_id == second, "认领的不是内容证据命中的那条"
    assert claimed.pairing_ambiguous is False


def test_two_pending_matching_both_is_ambiguous(soul_db: Any) -> None:
    """两条候选内容都命中（同一条消息触发过两次）→ 分不清 → 歧义。"""
    _snapshot("sess-4", "同样的问题")
    _snapshot("sess-4", "同样的问题")

    claimed = _sr().claim_snapshot_for_response(
        "sess-4", "reply-1", reply_tail="用户A: 同样的问题",
    )
    assert claimed is not None and claimed.pairing_ambiguous is True


# ─── 3. 无证据时退回旧规则（行为不回退）────────────────────────


def test_without_evidence_legacy_fifo_still_applies(soul_db: Any) -> None:
    """没有内容证据（replyer 腿未观测到）→ 旧规则：单条认领、多条标歧义。"""
    only = _snapshot("sess-5", "只有一条")
    claimed = _sr().claim_snapshot_for_response("sess-5", "reply-1")
    assert claimed is not None and claimed.snapshot_id == only
    assert claimed.pairing_ambiguous is False

    first = _snapshot("sess-6", "先来的")
    _snapshot("sess-6", "后来的")
    claimed2 = _sr().claim_snapshot_for_response("sess-6", "reply-9")
    assert claimed2 is not None and claimed2.snapshot_id == first, "应消费最旧的一条"
    assert claimed2.pairing_ambiguous is True


def test_retry_reuses_same_snapshot(soul_db: Any) -> None:
    """同一 reply_message_id 重试 → 复用同一快照（不重复消费）。"""
    snap = _snapshot("sess-7", "重试场景")
    first = _sr().claim_snapshot_for_response("sess-7", "reply-x", reply_tail="用户A: 重试场景")
    second = _sr().claim_snapshot_for_response("sess-7", "reply-x", reply_tail="用户A: 重试场景")
    assert first is not None and second is not None
    assert first.snapshot_id == second.snapshot_id == snap


# ─── 4. 提取与匹配工具 ───────────────────────────────────────


def test_tails_match_rules() -> None:
    """匹配规则：相等 → 命中；短串只认同等；长串包含也算。"""
    m = _sr()._tails_match
    assert m("用户A: 你好", "用户A: 你好") is True
    assert m("好", "好吧") is False, "短消息不得靠包含关系乱配"
    long_a = "这是一条超过八个字的用户消息内容"
    assert m(long_a, "前缀 " + long_a + " 后缀") is True
    assert m("", "任何") is False


def test_tail_of_context_reads_last_line() -> None:
    sr = _sr()
    assert sr._tail_of_context(json.dumps(["第一行", "第二行"])) == "第二行"
    assert sr._tail_of_context("[]") == ""
    assert sr._tail_of_context("not-json") == ""


def test_reply_tail_cache_roundtrip(soul_db: Any) -> None:
    """replyer 腿记下的尾行能取回；没记过则空串。"""
    cap = _import_soul_submodule("components.reflection_capture")
    cap._reply_tail_cache.clear()

    tail = cap.cache_reply_tail("sess-8", "reply-8", [_item("用户C: 交给我来"), _item("用户D: 最后一条")])
    assert tail == "用户D: 最后一条"
    assert cap.take_reply_tail("sess-8", "reply-8") == "用户D: 最后一条"
    assert cap.take_reply_tail("sess-8", "不存在") == ""
    assert cap.take_reply_tail("别的会话", "reply-8") == ""


def test_reply_tail_cache_empty_items_is_noop(soul_db: Any) -> None:
    """items 里没有用户文本 → 不写缓存（宁缺毋滥，别拿空证据去比对）。"""
    cap = _import_soul_submodule("components.reflection_capture")
    cap._reply_tail_cache.clear()

    assert cap.cache_reply_tail("sess-9", "reply-9", []) == ""
    assert cap.take_reply_tail("sess-9", "reply-9") == ""


def test_prefix_rendering_differences_do_not_break_matching(soul_db: Any) -> None:
    """两侧昵称前缀渲染不同（快照 vs replyer items）时仍应命中同一条消息。"""
    _snapshot("sess-10", "把日志贴出来")

    claimed = _sr().claim_snapshot_for_response(
        "sess-10", "reply-1", reply_tail="某位群友: 把日志贴出来",
    )
    assert claimed is not None and claimed.pairing_ambiguous is False, (
        "前缀差异导致漏配——应剥前缀后比内容"
    )
