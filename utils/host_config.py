"""宿主 config.get 返回值读取（SDK 2.x 解包契约）。

SDK 的 ``PluginContext._normalize_capability_result`` 会按能力映射表把
``config.get`` 的返回**先解包**再交给插件：

    宿主原始返回: {"success": True, "value": "3659592968"}
    SDK 交给插件: "3659592968"        ← 裸值，没有 success 键

因此插件侧若仍判断 ``result.get("success")`` 会永远为假，静默拿到空配置。
本模块是唯一的取值入口，同时兼容裸值与旧式 success/value 包装。
"""

from __future__ import annotations

from typing import Any

# 允许包装形式的取值键（按优先级）。
_VALUE_KEYS = ("value",)


def read_config_value(result: Any, default: str = "") -> str:
    """把 config.get 的返回值归一成字符串。

    Args:
        result: ``ctx.call_capability("config.get", ...)`` 的返回值，
            可能是裸值（SDK 2.x）或 ``{"success": bool, "value": ...}`` 包装（旧 SDK）。
        default: 无法取到值时返回的字符串。

    Returns:
        去除两端空白的字符串；无值、失败包装或未预期类型时返回 ``default``。
    """
    if result is None:
        return default

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, bool):
        return str(result)

    if isinstance(result, (int, float)):
        return str(result)

    if isinstance(result, dict):
        if "success" in result and not result.get("success"):
            return default
        for key in _VALUE_KEYS:
            if key in result:
                return read_config_value(result.get(key), default)
        return default

    return default


async def fetch_config_value(
    ctx: Any,
    key: str,
    default: str = "",
) -> str:
    """经宿主 ``config.get`` 读取单个配置项并归一化。

    能力调用本身的异常与失败包装都降级为 ``default``：调用方据此走 fallback，
    而不是被异常打断整个 hook。
    """
    try:
        result = await ctx.call_capability("config.get", key=key, default=default)
    except (RuntimeError, ValueError, OSError, AttributeError, TypeError):
        return default
    return read_config_value(result, default)
