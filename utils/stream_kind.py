"""会话类型判定：用宿主**显式接口**，不猜 session_id 字符串。

**为什么不能猜**：宿主 planner hook 的载荷里没有会话类型字段，旧实现只能看
`session_id` 里有没有 "private" 字样。这种启发式一旦宿主改了 id 编码就会静默
失效——最坏情况是把私聊当群聊（或反过来），而 `inject_private=False` 这类
设置就是靠它兜底的。

宿主提供了显式的流列表接口：``chat.get_group_streams`` / ``chat.get_private_streams``。
本模块据此判定，并带 TTL 缓存（流列表变化不频繁，注入在热路径上）。

判定不出时返回 ``"unknown"``，调用方**保守处理**（不假装知道）。
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["STREAM_KIND_GROUP", "STREAM_KIND_PRIVATE", "STREAM_KIND_UNKNOWN", "resolve_stream_kind"]

STREAM_KIND_GROUP = "group"
STREAM_KIND_PRIVATE = "private"
STREAM_KIND_UNKNOWN = "unknown"

# stream_id -> (kind, 记录时间)
_CACHE: dict[str, tuple[str, float]] = {}
DEFAULT_TTL_SECONDS = 300.0

# 未判定出且频繁调用时，避免刷屏日志
_UNKNOWN_WARNED: set[str] = set()


def _extract_stream_ids(payload: Any) -> set[str]:
    """把宿主返回值里的流标识抽成集合。

    兼容三种形状：裸 list、``{success, value: [...]}`` 包装、元素为 dict
    （含 session_id / stream_id / id）。
    """
    data = payload
    if isinstance(data, dict):
        for key in ("value", "streams", "data", "result"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, (list, tuple, set)):
        return set()

    ids: set[str] = set()
    for entry in data:
        if isinstance(entry, str):
            if entry:
                ids.add(entry)
        elif isinstance(entry, dict):
            for key in ("session_id", "stream_id", "id"):
                value = entry.get(key)
                if isinstance(value, str) and value:
                    ids.add(value)
    return ids


async def resolve_stream_kind(
    plugin: Any,
    stream_id: str,
    *,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
) -> str:
    """判定 stream 属于群聊还是私聊。

    Returns:
        ``"group"`` / ``"private"`` / ``"unknown"``（判定不出时由调用方保守处理）。
    """
    sid = str(stream_id or "").strip()
    if not sid:
        return STREAM_KIND_UNKNOWN

    now = time.time()
    cached = _CACHE.get(sid)
    if cached is not None and (now - cached[1]) < ttl_seconds:
        return cached[0]

    chat = getattr(getattr(plugin, "ctx", None), "chat", None)
    if chat is None:
        return STREAM_KIND_UNKNOWN

    kind = STREAM_KIND_UNKNOWN
    try:
        group_ids = _extract_stream_ids(await chat.get_group_streams())
        if sid in group_ids:
            kind = STREAM_KIND_GROUP
        else:
            private_ids = _extract_stream_ids(await chat.get_private_streams())
            if sid in private_ids:
                kind = STREAM_KIND_PRIVATE
    except Exception as e:  # noqa: BLE001 — 判定失败不能影响注入主流程
        logger.debug("[Soul] 会话类型判定失败（按 unknown 处理）: %s: %s", type(e).__name__, e)
        return STREAM_KIND_UNKNOWN

    if kind != STREAM_KIND_UNKNOWN:
        _CACHE[sid] = (kind, now)
    elif sid not in _UNKNOWN_WARNED:
        _UNKNOWN_WARNED.add(sid)
        logger.info(
            "[Soul] 无法从宿主流列表判定会话类型（session=%s）："
            "注入将按保守策略处理，可能是新会话或未注册流", sid,
        )
    return kind


def clear_stream_kind_cache() -> None:
    """清缓存（插件卸载/配置热更时调用，避免跨实例泄漏）。"""
    _CACHE.clear()
    _UNKNOWN_WARNED.clear()
