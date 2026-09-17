"""通知 outbox：发送失败必须可重试、可查询，不能静默丢。

**问题**：管理员通知（新种子、发酵异常、需要人工定夺）走的是即时 `send.text`，
失败只记一行日志就丢了。于是「管理员没收到通知」和「没有需要通知的事」
在外部看起来完全一样——需要人工介入的事就这么沉了。

**做法**：先试发；失败（或抛异常）就落 outbox，之后由后台循环重放。
``dedupe_key`` 唯一，重复排队不会变成刷屏；``attempts`` 达上限转 ``failed``，
保留在表里等人工查看，不无限重试。
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime

from ._conn import _dt_to_str, _get_conn, _str_to_dt

__all__ = [
    "NOTIFICATION_DONE",
    "NOTIFICATION_FAILED",
    "NOTIFICATION_PENDING",
    "Notification",
    "count_notifications",
    "enqueue_notification",
    "get_notification",
    "list_pending_notifications",
    "mark_notification_failed",
    "mark_notification_sent",
    "purge_old_notifications",
]

NOTIFICATION_PENDING = "pending"
NOTIFICATION_SENT = "sent"
NOTIFICATION_FAILED = "failed"


@dataclass
class Notification:
    """一条待发/已发通知。"""

    notification_id: str = ""
    dedupe_key: str = ""
    stream_id: str = ""
    text: str = ""
    status: str = NOTIFICATION_PENDING
    attempts: int = 0
    max_attempts: int = 3
    last_error: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    sent_at: datetime | None = None


def _row_to_notification(row) -> Notification:
    return Notification(
        notification_id=row["notification_id"],
        dedupe_key=row["dedupe_key"],
        stream_id=row["stream_id"],
        text=row["text"],
        status=row["status"],
        attempts=int(row["attempts"] or 0),
        max_attempts=int(row["max_attempts"] or 3),
        last_error=row["last_error"] or "",
        created_at=_str_to_dt(row["created_at"]),
        updated_at=_str_to_dt(row["updated_at"]),
        sent_at=_str_to_dt(row["sent_at"]),
    )


def enqueue_notification(
    dedupe_key: str,
    stream_id: str,
    text: str,
    *,
    max_attempts: int = 3,
) -> str:
    """入队一条通知；``dedupe_key`` 相同则复用已有记录（幂等，不刷屏）。

    Returns:
        通知记录 id（新建或已存在的）。
    """
    conn = _get_conn()
    now = _dt_to_str(datetime.now())
    key = str(dedupe_key or "").strip()
    if not key:
        # 无去重键时退化为按内容去重。必须用**稳定哈希**：
        # Python 的 hash() 受 PYTHONHASHSEED 影响、每个进程都不同，
        # 用它做键会让插件重启后同一条通知被重复入队（去重形同虚设）。
        digest = hashlib.sha256(f"{stream_id}\x00{text}".encode("utf-8")).hexdigest()[:24]
        key = f"auto:{digest}"

    existing = conn.execute(
        "SELECT notification_id FROM soul_notifications WHERE dedupe_key = ?",
        (key,),
    ).fetchone()
    if existing is not None:
        return existing["notification_id"]

    notification_id = f"ntf_{uuid.uuid4().hex[:12]}"
    try:
        conn.execute(
            """
            INSERT INTO soul_notifications
                (notification_id, dedupe_key, stream_id, text, status,
                 attempts, max_attempts, last_error, created_at, updated_at, sent_at)
            VALUES (?, ?, ?, ?, ?, 0, ?, '', ?, ?, '')
            """,
            (
                notification_id, key, stream_id, text, NOTIFICATION_PENDING,
                int(max_attempts), now, now,
            ),
        )
        conn.commit()
    except Exception:
        # 并发插入撞唯一索引：视为已存在
        conn.rollback()
        row = conn.execute(
            "SELECT notification_id FROM soul_notifications WHERE dedupe_key = ?",
            (key,),
        ).fetchone()
        if row is not None:
            return row["notification_id"]
        raise
    return notification_id


def get_notification(notification_id: str) -> Notification | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_notifications WHERE notification_id = ?",
        (notification_id,),
    ).fetchone()
    return _row_to_notification(row) if row is not None else None


def list_pending_notifications(limit: int = 20) -> list[Notification]:
    """按入队顺序取待发通知（FIFO，最旧的先发）。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM soul_notifications WHERE status = ? "
        "ORDER BY created_at ASC, rowid ASC LIMIT ?",
        (NOTIFICATION_PENDING, int(limit)),
    ).fetchall()
    return [_row_to_notification(r) for r in rows]


def mark_notification_sent(notification_id: str) -> bool:
    """标记已发送。"""
    conn = _get_conn()
    now = _dt_to_str(datetime.now())
    cursor = conn.execute(
        "UPDATE soul_notifications SET status = ?, sent_at = ?, updated_at = ? "
        "WHERE notification_id = ? AND status != ?",
        (NOTIFICATION_SENT, now, now, notification_id, NOTIFICATION_SENT),
    )
    conn.commit()
    return cursor.rowcount > 0


def mark_notification_failed(notification_id: str, error: str) -> str:
    """记录一次发送失败。返回更新后的状态（``pending`` 可重试 / ``failed`` 放弃）。"""
    conn = _get_conn()
    now = _dt_to_str(datetime.now())
    row = conn.execute(
        "SELECT attempts, max_attempts, status FROM soul_notifications "
        "WHERE notification_id = ?",
        (notification_id,),
    ).fetchone()
    if row is None or row["status"] == NOTIFICATION_SENT:
        return row["status"] if row is not None else ""

    attempts = int(row["attempts"] or 0) + 1
    max_attempts = int(row["max_attempts"] or 3)
    status = NOTIFICATION_FAILED if attempts >= max_attempts else NOTIFICATION_PENDING
    conn.execute(
        "UPDATE soul_notifications SET attempts = ?, status = ?, last_error = ?, updated_at = ? "
        "WHERE notification_id = ?",
        (attempts, status, str(error)[:300], now, notification_id),
    )
    conn.commit()
    return status


def count_notifications(status: str | None = None) -> int:
    conn = _get_conn()
    if status is None:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM soul_notifications").fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM soul_notifications WHERE status = ?", (status,)
        ).fetchone()
    return int(row["cnt"] or 0)


def purge_old_notifications(*, keep_sent: int = 200) -> int:
    """清理历史已发通知，只保留最近 ``keep_sent`` 条（防表无限增长）。"""
    conn = _get_conn()
    cursor = conn.execute(
        """
        DELETE FROM soul_notifications
        WHERE status = ?
          AND notification_id NOT IN (
              SELECT notification_id FROM soul_notifications
              WHERE status = ? ORDER BY sent_at DESC LIMIT ?
          )
        """,
        (NOTIFICATION_SENT, NOTIFICATION_SENT, int(keep_sent)),
    )
    conn.commit()
    return cursor.rowcount
