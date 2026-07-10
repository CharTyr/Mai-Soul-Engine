"""帮助命令 — 列出所有可用 Soul 命令及用法。"""

from __future__ import annotations

from typing import Any


async def handle_help(plugin, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """列出所有可用命令。"""
    from ..utils.spectrum_utils import check_admin_permission

    ok, _ = check_admin_permission(plugin, kwargs, "")

    lines = ["可用命令列表：\n"]

    # 所有用户可用
    lines.append("📊 /soul_status — 查看光谱状态")
    lines.append("📈 /soul_dashboard — 可视化状态卡片")

    # 管理员命令
    if ok:
        lines.append("\n🔒 管理员命令：\n")
        lines.append("/soul_setup [--restart [--yes]] — 初始化/重做问卷")
        lines.append("/soul_answer <1-5> — 问卷答题")
        lines.append("/soul_reset [confirm] — 重置光谱（需二次确认）")
        lines.append("/soul_seeds — 查看待审种子列表")
        lines.append("/soul_seed <ID> — 查看种子详情")
        lines.append("/soul_approve <ID> — 批准种子内化")
        lines.append("/soul_reject <ID> — 拒绝种子")
        lines.append("/soul_reject_all — 批量拒绝所有种子")
        lines.append("/soul_traits [stream_id] — 查看 trait 列表")
        lines.append("/soul_trait <ID> — 查看 trait 详情卡片")
        lines.append("/soul_trait_set_tags <ID> <tags> — 设置 trait 标签")
        lines.append("/soul_trait_merge <from> <to> — 合并 trait")
        lines.append("/soul_trait_disable <ID> — 禁用 trait")
        lines.append("/soul_trait_enable <ID> — 启用 trait")
        lines.append("/soul_trait_delete <ID> — 删除 trait")
        lines.append("/soul_slot <ID> <1-12|clear> — 设置/清空 trait 思维阁槽位")
        lines.append("/soul_promote_global <ID> — 将群锁 trait 提升为全局作用域")
        lines.append("/soul_inspect <文本> — 注入命中预览")
        lines.append("/soul_reflect [N] — 查看自评记录")
        lines.append("/soul_observe — 运维状态摘要")
        lines.append("/soul_health — 插件健康状态")
    else:
        lines.append("\n管理员命令需配置 admin_user_id 后使用。")

    msg = "\n".join(lines)
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
