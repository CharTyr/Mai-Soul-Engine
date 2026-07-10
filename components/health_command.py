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
    lines.append(f"引擎开关: {'开启' if plugin.config.plugin.enabled else '关闭'}")
    lines.append(f"演化任务: {'运行中' if plugin._evolution_task else '已停止'}")
    lines.append(f"Notion同步: {'运行中' if plugin._notion_sync_task else '已停止'}")
    lines.append(f"自评任务: {'运行中' if plugin._self_reflection_task else '已停止'}")
    lines.append(f"发酵任务: {'运行中' if plugin._fermentation_task else '已停止'}")
    lines.append(f"数据目录: {plugin._data_dir}")
    lines.append(f"数据来源: {'宿主' if plugin._data_dir_source == 'host' else '插件目录'}")

    # P1.4: data_dir 迁移错误提示
    data_dir_info = getattr(plugin, '_data_dir_info', None) or {}
    mig_detail = data_dir_info.get("migration_detail") or {}
    if mig_detail.get("error"):
        lines.append(f"⚠️ 数据迁移失败: {mig_detail['error']}，当前使用 plugin_dir/data")
        lines.append("状态: degraded")
    else:
        lines.append("状态: ok")

    # P1.4: 当使用 plugin_dir 时提示宿主 data_dir 可用性
    if plugin._data_dir_source == 'plugin_dir':
        try:
            from ..utils.data_dir import _try_get_host_data_dir
            host_path = _try_get_host_data_dir(plugin)
            if host_path is not None:
                lines.append("提示: 宿主 data_dir 可用但未使用（见日志），可通过删除 plugin_dir/data 重启触发迁移")
        except Exception:
            pass

    # schema user_version（可选）
    try:
        from ..models._conn import _get_conn
        conn = _get_conn()
        uv = conn.execute("PRAGMA user_version").fetchone()
        if uv:
            lines.append(f"Schema版本: {uv[0]}")
    except Exception:
        pass

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

    # 发酵中种子（可选）
    try:
        from ..models.seeds import count_fermenting_seeds

        fermenting = count_fermenting_seeds()
        if fermenting > 0:
            lines.append(f"发酵中种子: {fermenting}")
    except Exception:
        pass

    # 槽位占用（可选简短）
    try:
        from ..models.traits import query_crystallized_traits

        all_traits = query_crystallized_traits(deleted=False, limit=200)
        slotted = sum(1 for t in all_traits if t.cabinet_slot_no is not None)
        if slotted > 0:
            lines.append(f"槽位占用: {slotted}/12")
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
