"""重置命令模块 — maibot_sdk 2.x 版本。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

_RESET_CONFIRM_TIMEOUT = 300  # 5 分钟


async def handle_reset(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """重置灵魂光谱为中立状态（管理员）。

    两步确认机制：
    1. 第一次 /soul_reset → 提示确认
    2. /soul_reset confirm → 5 分钟内有效，执行重置
    """
    from ..utils.spectrum_utils import check_admin_permission
    from ..utils.audit_log import log_reset
    from ..models.ideology_model import get_or_create_spectrum

    ok, err = check_admin_permission(plugin, kwargs, "重置灵魂光谱")
    if not ok:
        await plugin.ctx.send.text(err, stream_id)
        return True, err, True

    from ..utils.spectrum_utils import extract_command_actor
    platform, user_id = extract_command_actor(kwargs)

    # 检查是否为确认指令
    message_text = (kwargs.get("message") or {}).get("text", "") or ""
    is_confirm = "confirm" in message_text.casefold()

    if is_confirm:
        # 检查是否在确认窗口内
        last_ts = plugin._reset_confirm_ts.get(stream_id, 0.0)
        if time.time() - last_ts > _RESET_CONFIRM_TIMEOUT:
            msg = "确认超时，请重新发送 /soul_reset 开始重置流程"
            await plugin.ctx.send.text(msg, stream_id)
            return True, msg, True

        # 执行重置
        spectrum = get_or_create_spectrum("global")
        spectrum.sincerity = 50
        spectrum.engagement = 50
        spectrum.closeness = 50
        spectrum.directness = 50
        spectrum.initialized = False
        spectrum.updated_at = datetime.now()
        spectrum.save()

        await log_reset(f"{platform}:{user_id}")

        # 清除确认状态
        plugin._reset_confirm_ts.pop(stream_id, None)

        msg = "灵魂光谱已重置为中立状态，请使用 /soul_setup 重新初始化"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # 第一次请求：记录确认状态，提示二次确认
    plugin._reset_confirm_ts[stream_id] = time.time()
    msg = (
        "⚠️ 确认重置光谱？这将清空所有光谱数据。\n"
        "回复 /soul_reset confirm 确认，5 分钟内有效。"
    )
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
