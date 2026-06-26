"""重置命令模块 — maibot_sdk 2.x 版本。"""

from __future__ import annotations

from datetime import datetime
from typing import Any


async def handle_reset(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """重置灵魂光谱为中立状态（管理员）。"""
    from ..utils.spectrum_utils import match_user
    from ..utils.audit_log import log_reset
    from ..models.ideology_model import get_or_create_spectrum

    admin_user_id = plugin.config.admin.admin_user_id
    # 从 kwargs 中获取平台和用户信息
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以重置灵魂光谱"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    spectrum = get_or_create_spectrum("global")
    spectrum.sincerity = 50
    spectrum.engagement = 50
    spectrum.closeness = 50
    spectrum.directness = 50
    spectrum.initialized = False
    spectrum.updated_at = datetime.now()
    spectrum.save()

    await log_reset(f"{platform}:{user_id}")

    msg = "灵魂光谱已重置为中立状态，请使用 /soul_setup 重新初始化"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
