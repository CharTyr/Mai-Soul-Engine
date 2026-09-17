"""会话类型判定：用宿主**显式接口**，不猜 session_id 字符串。

**为什么不能猜**：宿主 planner hook 的载荷里没有会话类型字段，旧实现只能看
`session_id` 里有没有 "private" 字样。这种启发式一旦宿主改了 id 编码就会静默
失效——最坏情况是把私聊当群聊（或反过来），而 `inject_private=False` 这类
设置就是靠它兜底的。

宿主提供了显式的流列表接口：``chat.get_group_streams`` / ``chat.get_private_streams``。
本模块据此判定，并带 TTL 缓存（流列表变化不频繁，注入在热路径上）。

判定不出时返回 ``"unknown"``，调用方**保守处理**（不假装知道）。

**平台（作用域字段）**：宿主既不返回平台、也没有「枚举平台」的能力
（已核对 hook payload 与 SDK 签名），但 ``chat.get_*_streams`` **接受 platform 参数**。
因此插件侧唯一诚实解是：由**配置声明**平台列表，再按平台逐个探测归属——
数据是宿主给的，**禁止**按 session_id 字符串猜平台。探不到就留空（未知）。
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "STREAM_KIND_GROUP",
    "STREAM_KIND_PRIVATE",
    "STREAM_KIND_UNKNOWN",
    "configured_platforms",
    "resolve_stream_kind",
    "resolve_stream_scope",
]

STREAM_KIND_GROUP = "group"
STREAM_KIND_PRIVATE = "private"
STREAM_KIND_UNKNOWN = "unknown"

DEFAULT_PLATFORM = "qq"
DEFAULT_TTL_SECONDS = 300.0

# stream_id -> (kind, platform, 记录时间)
_CACHE: dict[str, tuple[str, str, float]] = {}

# 未判定出且频繁调用时，避免刷屏日志
_UNKNOWN_WARNED: set[str] = set()


def configured_platforms(plugin: Any) -> list[str]:
    """读配置声明的平台列表；缺省 ``["qq"]``（宿主自身的默认平台）。"""
    raw = getattr(getattr(getattr(plugin, "config", None), "plugin", None), "platforms", None)
    if isinstance(raw, (list, tuple)):
        cleaned = [str(x).strip() for x in raw if str(x).strip()]
        if cleaned:
            return cleaned
    if isinstance(raw, str) and raw.strip():
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [DEFAULT_PLATFORM]


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
    kind, _platform = await resolve_stream_scope(plugin, stream_id, ttl_seconds=ttl_seconds)
    return kind


async def resolve_stream_scope(
    plugin: Any,
    stream_id: str,
    *,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
) -> tuple[str, str]:
    """判定 stream 的 ``(会话类型, 平台)``。

    平台由「按配置声明的平台逐个探测宿主流列表」得出；探测不到 → 空串（未知，不编造）。
    """
    sid = str(stream_id or "").strip()
    if not sid:
        return STREAM_KIND_UNKNOWN, ""

    now = time.time()
    cached = _CACHE.get(sid)
    if cached is not None and (now - cached[2]) < ttl_seconds:
        return cached[0], cached[1]

    chat = getattr(getattr(plugin, "ctx", None), "chat", None)
    if chat is None:
        return STREAM_KIND_UNKNOWN, ""

    kind = STREAM_KIND_UNKNOWN
    platform = ""
    for candidate in configured_platforms(plugin):
        try:
            group_ids = _extract_stream_ids(await chat.get_group_streams(platform=candidate))
            if sid in group_ids:
                kind, platform = STREAM_KIND_GROUP, candidate
                break
            private_ids = _extract_stream_ids(await chat.get_private_streams(platform=candidate))
            if sid in private_ids:
                kind, platform = STREAM_KIND_PRIVATE, candidate
                break
        except Exception as e:  # noqa: BLE001 — 单个平台判定失败不能影响主流程
            logger.debug(
                "[Soul] 平台 %s 会话类型判定失败（继续下一个）: %s: %s",
                candidate, type(e).__name__, e,
            )
            continue

    if kind != STREAM_KIND_UNKNOWN:
        _CACHE[sid] = (kind, platform, now)
    elif sid not in _UNKNOWN_WARNED:
        _UNKNOWN_WARNED.add(sid)
        logger.info(
            "[Soul] 无法从宿主流列表判定会话类型（session=%s）："
            "注入将按保守策略处理，可能是新会话或未注册流", sid,
        )
    return kind, platform


def clear_stream_kind_cache() -> None:
    """清缓存（插件卸载/配置热更时调用，避免跨实例泄漏）。"""
    _CACHE.clear()
    _UNKNOWN_WARNED.clear()
