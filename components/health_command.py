"""健康状态命令 — 将 api_health 格式化为用户可读文本。"""

from __future__ import annotations

from typing import Any


async def handle_health(plugin, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """插件健康状态。"""
    from ..utils.spectrum_utils import check_admin_permission

    ok, err = check_admin_permission(plugin, kwargs, "查看插件健康状态")
    if not ok:
        await plugin.ctx.send.text(err, stream_id)
        return True, err, True

    lines = ["插件健康状态：\n"]
    lines.append(f"演化任务: {'运行中' if plugin._evolution_task else '已停止'}")
    lines.append(f"Notion同步: {'运行中' if plugin._notion_sync_task else '已停止'}")
    lines.append(f"自评任务: {'运行中' if plugin._self_reflection_task else '已停止'}")

    # DB 大小
    try:
        db_path = plugin._data_dir / "soul.db"
        if db_path.exists():
            size_kb = db_path.stat().st_size // 1024
            lines.append(f"数据库大小: {size_kb} KB")
    except Exception:
        pass

    # 待审种子
    try:
        from ..models.ideology_model import count_pending_thought_seeds

        pending = count_pending_thought_seeds()
        lines.append(f"待审种子: {pending}")
    except Exception:
        pass

    # 注入指标
    try:
        from .ideology_injector import _injection_metrics

        m = _injection_metrics
        lines.append(
            f"注入统计: 总{m['total']} 命中{m['traits_hit']} "
            f"冷却跳过{m['skipped_cooldown']} 无trait跳过{m['skipped_no_traits']}"
        )
    except Exception:
        pass

    msg = "\n".join(lines)
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
