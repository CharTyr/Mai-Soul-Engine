"""自我评价查看命令 — /soul_reflect [N]（管理员）。

查看近期自我评价记录：待评队列、评价统计、最近 N 条自评详情。
纯文本输出（不渲染卡片，保持轻量）。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_AXIS_LABELS = {
    "sincerity": "真诚度",
    "engagement": "投入度",
    "closeness": "亲密度",
    "directness": "直率度",
}
_REPLY_TYPE_LABELS = {
    "social_glue": "闲聊(跳过)",
    "reactive": "反应(语气)",
    "substantive": "表态(完整)",
}


async def handle_reflect(
    plugin: Any, stream_id: str = "", **kwargs: Any
) -> tuple[bool, str, bool]:
    """/soul_reflect [N] — 查看近期自我评价（管理员）。

    Args:
        plugin: 插件实例。
        stream_id: 回复目标流。
        **kwargs: 含 message（鉴权）与可选 count（命名捕获组）。
    """
    from ..utils.spectrum_utils import match_user

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以查看自我评价"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.self_reflection.enabled:
        msg = "自我评价反馈回路未启用（[self_reflection].enabled=false）"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # 解析可选 N
    count = 10
    raw_count = kwargs.get("count")
    if raw_count:
        try:
            count = max(1, min(50, int(raw_count)))
        except (TypeError, ValueError):
            count = 10

    from ..models.self_reflection import (
        count_pending_reflections,
        count_self_reflections,
        list_recent_reflections,
    )

    pending = count_pending_reflections()
    stats = count_self_reflections()
    reflections = list_recent_reflections(stream_id or "global", limit=count)

    lines: list[str] = ["【自我评价反馈回路】"]
    # 待评队列
    pending_parts = [f"{k}={v}" for k, v in sorted(pending.items())]
    lines.append(f"待评队列：{', '.join(pending_parts) if pending_parts else '空'}")
    # 评价统计
    lines.append(
        f"评价统计：共{stats['total']}条（已评{stats['evaluated']}，跳过{stats['skipped']}）"
    )
    by_axis = stats.get("by_axis", {})
    if by_axis:
        axis_parts = [f"{_AXIS_LABELS.get(k, k)}={v}" for k, v in sorted(by_axis.items())]
        lines.append(f"偏离轴分布：{', '.join(axis_parts)}")

    # 最近 N 条详情
    if reflections:
        lines.append(f"\n最近 {len(reflections)} 条自评：")
        for r in reflections:
            rt = _REPLY_TYPE_LABELS.get(r.reply_type, r.reply_type or "?")
            if not r.evaluated:
                lines.append(f"  [{rt}] 跳过（{r.reply_type}）")
                continue
            axis = _AXIS_LABELS.get(r.deviating_axis, r.deviating_axis or "无")
            direction = r.deviating_direction or ""
            reason = (r.reason or "")[:40]
            lines.append(f"  [{rt}] 一致性={r.consistency_score} 偏离={axis}{direction} {reason}")
    else:
        lines.append("\n暂无自评记录（开启后需等评价周期跑完）")

    msg = "\n".join(lines)
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
