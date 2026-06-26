"""状态查看命令模块 — maibot_sdk 2.x 版本。"""

from __future__ import annotations

from typing import Any


async def handle_status(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """查看当前意识形态光谱状态。"""
    from ..utils.spectrum_utils import format_spectrum_display
    from ..models.ideology_model import get_or_create_spectrum

    spectrum = get_or_create_spectrum("global")

    if not spectrum.initialized:
        msg = "灵魂光谱尚未初始化，请管理员使用 /soul_setup 进行初始化"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    spectrum_dict = {
        "sincerity": spectrum.sincerity,
        "engagement": spectrum.engagement,
        "closeness": spectrum.closeness,
        "directness": spectrum.directness,
    }

    display = format_spectrum_display(spectrum_dict)
    last_update = spectrum.updated_at.strftime("%Y-%m-%d %H:%M:%S") if spectrum.updated_at else "未知"

    from ..worldview.service import WorldviewService, config_from_plugin

    extras = WorldviewService(config_from_plugin(plugin)).format_status_extras(stream_id)
    msg = f"当前灵魂光谱：\n\n{display}\n\n上次更新: {last_update}"
    if extras:
        msg = f"{msg}\n\n{extras}"

    # 思维阁启用时显示待审种子数
    if plugin.config.thought_cabinet.enabled:
        from ..models.ideology_model import count_pending_thought_seeds

        pending = count_pending_thought_seeds()
        if pending > 0:
            msg = f"{msg}\n\n待审思维种子: {pending} 个（用 /soul_seeds 查看）"

    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
