"""Soul 引擎全状态可视化仪表盘命令处理。"""

from __future__ import annotations

from typing import Any

import logging

logger = logging.getLogger(__name__)


async def handle_dashboard(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """聚合 Soul 引擎状态并渲染为可视化卡片图片发送；渲染关闭或失败时回退纯文本。

    Args:
        plugin: 插件实例。
        stream_id: 消息流 ID。
        **kwargs: 额外参数。

    Returns:
        (success, message, should_send_response) 元组。
    """
    from .dashboard_data import collect_dashboard_data

    data = collect_dashboard_data(plugin, stream_id=stream_id)

    # ── 判断是否启用卡片渲染 ─────────────────────────────────────────
    card_enabled = plugin.config.render.card_enabled

    if card_enabled:
        # ── 出图路径 ──────────────────────────────────────────────
        from .dashboard_renderer import DashboardRenderer

        renderer = DashboardRenderer(
            plugin.ctx,
            plugin.config.render.viewport_width,
            plugin.config.render.device_scale_factor,
            plugin.config.render.render_timeout_ms,
        )
        image_base64 = await renderer.render(data)

        if image_base64:
            # 渲染成功，发送图片
            try:
                await plugin.ctx.send.image(image_base64, stream_id)
            except (OSError, RuntimeError) as exc:
                logger.exception("发送 Soul 引擎状态卡片图片失败: %s", exc)
                # 降级文本
                from .dashboard_renderer import build_dashboard_text

                fallback_text = f"卡片渲染失败，以下为文本状态：\n{build_dashboard_text(data)}"
                await plugin.ctx.send.text(fallback_text, stream_id)
                return (True, "Soul 引擎状态(文本)", True)
            return (True, "已生成 Soul 引擎状态卡片", True)

        # 渲染返回空串——渲染失败，降级文本
        from .dashboard_renderer import build_dashboard_text

        fallback_text = f"卡片渲染失败，以下为文本状态：\n{build_dashboard_text(data)}"
        await plugin.ctx.send.text(fallback_text, stream_id)
        return (True, "Soul 引擎状态(文本)", True)

    # ── 文本降级路径（card_enabled=False） ───────────────────────────
    from .dashboard_renderer import build_dashboard_text

    text = build_dashboard_text(data)
    await plugin.ctx.send.text(text, stream_id)
    return (True, "Soul 引擎状态(文本)", True)
