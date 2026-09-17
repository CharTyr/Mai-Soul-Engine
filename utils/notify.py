"""管理员通知发送：先试发，失败落 outbox 由后台重放。

调用点统一走 ``send_or_queue``，不再直接 ``send.text`` ——这样「发送失败」
是一个可追踪、可重试的状态，而不是一行日志。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["drain_notifications", "send_or_queue"]


def _send_succeeded(result: Any) -> bool:
    """判断 send.text 的返回值是否表示成功。

    SDK 不同能力的返回形状不一致：``send.*`` 是**完整 envelope**（带 ``success``），
    但也可能返回 None 或裸值。只有明确 ``success is False`` 才算失败。
    """
    if isinstance(result, dict):
        if result.get("success") is False:
            return False
    return True


async def send_or_queue(
    plugin: Any,
    text: str,
    stream_id: str,
    *,
    dedupe_key: str,
    max_attempts: int = 3,
) -> bool:
    """发送一条管理员通知；失败则入 outbox 等待重放。

    Returns:
        True 表示已直接发出；False 表示已入队（之后会重试）。
    """
    from ..models.notifications import enqueue_notification

    if not stream_id:
        logger.warning("通知缺少目标 stream_id，转入 outbox 等待（key=%s）", dedupe_key)
        enqueue_notification(dedupe_key, stream_id, text, max_attempts=max_attempts)
        return False

    try:
        result = await plugin.ctx.send.text(text=text, stream_id=stream_id)
        if _send_succeeded(result):
            return True
        error = f"send.text 返回失败: {result}"
    except Exception as e:  # noqa: BLE001 — 发送失败必须降级为入队，不能丢
        error = f"{type(e).__name__}: {e}"

    logger.warning("通知发送失败，转入 outbox（key=%s）: %s", dedupe_key, error)
    notification_id = enqueue_notification(
        dedupe_key, stream_id, text, max_attempts=max_attempts,
    )
    from ..models.notifications import mark_notification_failed

    mark_notification_failed(notification_id, error)
    return False


async def drain_notifications(plugin: Any, limit: int = 20) -> dict[str, int]:
    """重放 outbox 中的待发通知。

    Returns:
        ``{"sent": n, "retry": n, "failed": n}``
    """
    from ..models.notifications import (
        list_pending_notifications,
        mark_notification_failed,
        mark_notification_sent,
        purge_old_notifications,
    )

    stats = {"sent": 0, "retry": 0, "failed": 0}
    for notification in list_pending_notifications(limit=limit):
        try:
            result = await plugin.ctx.send.text(
                text=notification.text, stream_id=notification.stream_id,
            )
            ok = _send_succeeded(result)
            error = "" if ok else f"send.text 返回失败: {result}"
        except Exception as e:  # noqa: BLE001 — 单条失败不影响其余
            ok = False
            error = f"{type(e).__name__}: {e}"

        if ok:
            mark_notification_sent(notification.notification_id)
            stats["sent"] += 1
            continue

        status = mark_notification_failed(notification.notification_id, error)
        if status == "failed":
            stats["failed"] += 1
            logger.error(
                "通知重试已达上限，停止重试（id=%s）: %s",
                notification.notification_id, error,
            )
        else:
            stats["retry"] += 1

    purge_old_notifications()
    return stats
