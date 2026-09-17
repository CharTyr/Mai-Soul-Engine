"""通知 outbox：发送失败可重试、可查询，不能静默丢。

管理员通知（新种子、发酵需要人工定夺）以前走即时 `send.text`，失败只留一行
日志。于是「管理员没收到通知」与「没有需要通知的事」从外部看完全一样——
需要人工介入的事就这么沉了。

现在：先试发，失败入 outbox，后台循环重放；`dedupe_key` 唯一防刷屏，
`attempts` 达上限转 failed 保留待查，不无限重试。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from .conftest import _import_soul_submodule


def _ntf() -> Any:
    return _import_soul_submodule("models.notifications")


def _notify() -> Any:
    return _import_soul_submodule("utils.notify")


class _Ctx:
    """假 ctx：send.text 行为可按脚本变化。"""

    def __init__(self, outcomes: list[Any] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[dict] = []

        class _Send:
            async def text(inner_self, text: str, stream_id: str = "") -> Any:
                self.calls.append({"text": text, "stream_id": stream_id})
                if not self.outcomes:
                    return {"success": True}
                outcome = self.outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        self.send = _Send()


def _plugin(outcomes: list[Any] | None = None) -> Any:
    return SimpleNamespace(ctx=_Ctx(outcomes))


# ─── 模型层 ─────────────────────────────────────────────────────────


def test_enqueue_is_idempotent_by_dedupe_key(soul_db: Any) -> None:
    """同一 dedupe_key 只入队一条（重排队不刷屏）。"""
    ntf = _ntf()
    first = ntf.enqueue_notification("seed:abc", "qq:p:1", "有种子")
    second = ntf.enqueue_notification("seed:abc", "qq:p:1", "有种子")

    assert first == second
    assert ntf.count_notifications() == 1


def test_pending_list_is_fifo(soul_db: Any) -> None:
    """按入队顺序取（最旧的先发），保证通知顺序不颠倒。"""
    ntf = _ntf()
    ntf.enqueue_notification("k1", "s", "第一条")
    ntf.enqueue_notification("k2", "s", "第二条")

    pending = ntf.list_pending_notifications()
    assert [p.text for p in pending] == ["第一条", "第二条"]


def test_mark_sent_moves_out_of_pending(soul_db: Any) -> None:
    """标记已发后不再出现在待发队列。"""
    ntf = _ntf()
    nid = ntf.enqueue_notification("k1", "s", "内容")
    assert ntf.mark_notification_sent(nid) is True

    assert ntf.list_pending_notifications() == []
    assert ntf.get_notification(nid).status == ntf.NOTIFICATION_SENT


def test_failed_attempt_increments_and_stays_retryable(soul_db: Any) -> None:
    """未达上限的失败留在待发队列，并记下错误。"""
    ntf = _ntf()
    nid = ntf.enqueue_notification("k1", "s", "内容", max_attempts=3)

    status = ntf.mark_notification_failed(nid, "网络断了")

    assert status == ntf.NOTIFICATION_PENDING
    record = ntf.get_notification(nid)
    assert record.attempts == 1
    assert "网络断了" in record.last_error
    assert len(ntf.list_pending_notifications()) == 1


def test_failed_attempt_becomes_terminal_at_limit(soul_db: Any) -> None:
    """达上限转 failed，不再重试（但记录保留待查）。"""
    ntf = _ntf()
    nid = ntf.enqueue_notification("k1", "s", "内容", max_attempts=2)

    assert ntf.mark_notification_failed(nid, "错1") == ntf.NOTIFICATION_PENDING
    assert ntf.mark_notification_failed(nid, "错2") == ntf.NOTIFICATION_FAILED

    assert ntf.list_pending_notifications() == []
    assert ntf.count_notifications(ntf.NOTIFICATION_FAILED) == 1


def test_sent_notification_is_not_marked_failed(soul_db: Any) -> None:
    """已发的通知不会被后续失败标记回退（幂等）。"""
    ntf = _ntf()
    nid = ntf.enqueue_notification("k1", "s", "内容")
    ntf.mark_notification_sent(nid)

    ntf.mark_notification_failed(nid, "迟到的错误")

    assert ntf.get_notification(nid).status == ntf.NOTIFICATION_SENT


def test_purge_keeps_only_recent_sent(soul_db: Any) -> None:
    """已发历史按条数上限清理；待发的绝不误删。"""
    ntf = _ntf()
    for i in range(5):
        nid = ntf.enqueue_notification(f"k{i}", "s", f"内容{i}")
        ntf.mark_notification_sent(nid)
    ntf.enqueue_notification("kp", "s", "还没发")

    removed = ntf.purge_old_notifications(keep_sent=2)

    assert removed == 3
    assert ntf.count_notifications(ntf.NOTIFICATION_SENT) == 2
    assert ntf.count_notifications(ntf.NOTIFICATION_PENDING) == 1


# ─── 发送层 ─────────────────────────────────────────────────────────


def test_send_success_does_not_queue(soul_db: Any) -> None:
    """发送成功不落 outbox。"""
    notify = _notify()
    ntf = _ntf()
    plugin = _plugin([{"success": True}])

    ok = asyncio.run(notify.send_or_queue(plugin, "文本", "qq:p:1", dedupe_key="k"))

    assert ok is True
    assert ntf.count_notifications() == 0
    assert plugin.ctx.calls[0]["stream_id"] == "qq:p:1"


def test_send_exception_queues_for_retry(soul_db: Any) -> None:
    """发送抛异常 → 入队并记一次失败。"""
    notify = _notify()
    ntf = _ntf()
    plugin = _plugin([RuntimeError("宿主不在线")])

    ok = asyncio.run(notify.send_or_queue(plugin, "重要通知", "qq:p:1", dedupe_key="k1"))

    assert ok is False
    pending = ntf.list_pending_notifications()
    assert len(pending) == 1
    assert pending[0].text == "重要通知"
    assert pending[0].attempts == 1
    assert "宿主不在线" in pending[0].last_error


def test_send_envelope_failure_queues(soul_db: Any) -> None:
    """send.text 返回 success=False 也算失败 → 入队。"""
    notify = _notify()
    ntf = _ntf()
    plugin = _plugin([{"success": False, "error": "无权限"}])

    ok = asyncio.run(notify.send_or_queue(plugin, "文本", "qq:p:1", dedupe_key="k2"))

    assert ok is False
    assert ntf.count_notifications() == 1


def test_send_without_stream_still_queues(soul_db: Any) -> None:
    """目标 stream 未知也不能丢：入队等待（宁可留着也不静默丢弃）。"""
    notify = _notify()
    ntf = _ntf()
    plugin = _plugin()

    ok = asyncio.run(notify.send_or_queue(plugin, "文本", "", dedupe_key="k3"))

    assert ok is False
    assert ntf.count_notifications() == 1


def test_drain_sends_pending_and_marks_sent(soul_db: Any) -> None:
    """重放：待发通知发出后标记 sent。"""
    notify = _notify()
    ntf = _ntf()
    ntf.enqueue_notification("k1", "qq:p:1", "第一条")
    ntf.enqueue_notification("k2", "qq:p:1", "第二条")

    stats = asyncio.run(notify.drain_notifications(_plugin([{"success": True}, {"success": True}])))

    assert stats == {"sent": 2, "retry": 0, "failed": 0}
    assert ntf.list_pending_notifications() == []


def test_drain_keeps_failed_for_next_round(soul_db: Any) -> None:
    """重放失败 → 留在队列下轮再试。"""
    notify = _notify()
    ntf = _ntf()
    ntf.enqueue_notification("k1", "qq:p:1", "内容")

    stats = asyncio.run(notify.drain_notifications(_plugin([RuntimeError("又断了")])))

    assert stats == {"sent": 0, "retry": 1, "failed": 0}
    assert ntf.get_notification(ntf.list_pending_notifications()[0].notification_id).attempts == 1


def test_drain_gives_up_after_max_attempts(soul_db: Any) -> None:
    """连续失败到上限 → failed，不再无限重试。"""
    notify = _notify()
    ntf = _ntf()
    ntf.enqueue_notification("k1", "qq:p:1", "内容", max_attempts=2)

    asyncio.run(notify.drain_notifications(_plugin([RuntimeError("断")])))
    stats = asyncio.run(notify.drain_notifications(_plugin([RuntimeError("断")])))

    assert stats["failed"] == 1
    assert ntf.list_pending_notifications() == []


def test_drain_one_failure_does_not_block_others(soul_db: Any) -> None:
    """单条发送失败不影响其余通知（逐条隔离）。"""
    notify = _notify()
    ntf = _ntf()
    ntf.enqueue_notification("k1", "qq:p:1", "会失败")
    ntf.enqueue_notification("k2", "qq:p:1", "会成功")

    stats = asyncio.run(
        notify.drain_notifications(_plugin([RuntimeError("断"), {"success": True}]))
    )

    assert stats["sent"] == 1
    assert stats["retry"] == 1
