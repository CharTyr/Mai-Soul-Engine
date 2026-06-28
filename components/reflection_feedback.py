"""自我评价双路反馈：planner 注入摘要 + 光谱演化修正。

两路反馈（见 .slim/deepwork/self-reflection.md）：
- **planner 反馈路**：``build_recent_reflection_summary`` 聚合近期自评为一行摘要，
  由 ``ideology_injector`` 按 selection_mode 分场景注入（oracle 修订点 5）。
- **演化路**：``apply_self_reflection_spectrum_correction`` 把自评偏离信号 ×weight
  折算成光谱 delta 修正，dead zone（净偏离≥3 才触发）+ magnitude 受 weight 缩放
  （0.5→±1，1.0→±2）防自指闭环。

方向语义：deviating_direction=low（bot 比人设弱）→ 人设该维向下校准（向实际表现靠拢）；
high → 向上校准。这是"人设向可兑现的现实靠拢"，非"拔高要求"。gentle，每周期每轴最多 ±2。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_AXIS_LABELS = {
    "sincerity": "真诚度",
    "engagement": "投入度",
    "closeness": "亲密度",
    "directness": "直率度",
}
_DIR_LABELS = {"high": "偏高", "low": "偏低"}

# 光谱修正 dead zone：某轴净偏离次数需 ≥ 此值才修正，防噪声
_CORRECTION_DEAD_ZONE: int = 3
# planner 摘要：某轴净偏离次数需 ≥ 此值才提及
_SUMMARY_MIN_DEVIATIONS: int = 2


def _aggregate_deviations(reflections: list) -> dict[str, dict[str, int]]:
    """聚合自评记录的偏离信号，返回 {axis: {"high": n, "low": n}}。"""
    agg: dict[str, dict[str, int]] = {}
    for r in reflections:
        if not getattr(r, "evaluated", 0):
            continue
        axis = getattr(r, "deviating_axis", "") or ""
        direction = getattr(r, "deviating_direction", "") or ""
        if axis not in _AXIS_LABELS or direction not in _DIR_LABELS:
            continue
        bucket = agg.setdefault(axis, {"high": 0, "low": 0})
        bucket[direction] = bucket.get(direction, 0) + 1
    return agg


def build_recent_reflection_summary(stream_id: str, limit: int = 10) -> str:
    """聚合近期自评为一行摘要（供 planner 注入）。

    返回空串表示无显著偏离，调用方应跳过注入。
    """
    from ..models.self_reflection import list_recent_reflections

    reflections = list_recent_reflections(stream_id, limit=limit)
    agg = _aggregate_deviations(reflections)
    if not agg:
        return ""
    parts: list[str] = []
    for axis, counts in agg.items():
        net = counts.get("high", 0) - counts.get("low", 0)
        if abs(net) < _SUMMARY_MIN_DEVIATIONS:
            continue
        direction = "偏高" if net > 0 else "偏低"
        total = counts.get("high", 0) + counts.get("low", 0)
        parts.append(f"{_AXIS_LABELS[axis]}{direction}（{abs(net)}/{total}次）")
    if not parts:
        return ""
    return "近期自我评价：你在表态时" + "；".join(parts) + "。可酌情校准。"


def apply_self_reflection_spectrum_correction(plugin, evolution_rate: int) -> None:
    """把自评偏离信号折算成光谱 delta 修正，应用到全局光谱。

    仅在 ``[self_reflection].enabled`` 且 weight>0 时生效。dead zone（净偏离≥3）防噪声，
    magnitude 受 weight 缩放（0.5→±1，1.0→±2，受 evolution_rate 上限）。
    """
    cfg = plugin.config.self_reflection
    weight = float(cfg.self_reflection_weight)
    if weight <= 0:
        return
    from ..models.self_reflection import list_recent_reflections
    from ..models.ideology_model import create_evolution_history, get_or_create_spectrum
    from ..utils.spectrum_utils import update_spectrum_value
    from ..worldview.constants import GLOBAL_STREAM

    reflections = list_recent_reflections(GLOBAL_STREAM, limit=30)
    agg = _aggregate_deviations(reflections)
    if not agg:
        return

    spectrum = get_or_create_spectrum("global")
    now = datetime.now()
    corrections: dict[str, int] = {}
    reason_parts: list[str] = []

    for axis, counts in agg.items():
        net = counts.get("high", 0) - counts.get("low", 0)
        if abs(net) < _CORRECTION_DEAD_ZONE:
            continue
        # weight 缩放修正幅度：0.5→±1，1.0→±2（ora-2 建议）
        magnitude = max(1, int(round(weight * 2)))
        if magnitude > evolution_rate:
            magnitude = evolution_rate
        delta = magnitude if net > 0 else -magnitude
        # 直接应用原始 delta（不再 smooth_delta：±1 经 EMA 平滑会被归零，
        # 自评修正本身已足够 gentle，dead zone + weight 已是护栏）
        corrections[axis] = delta
        direction = "偏高" if net > 0 else "偏低"
        reason_parts.append(f"{_AXIS_LABELS[axis]}{direction}{abs(net)}次")

    if not corrections:
        return

    # 应用修正
    for axis, delta in corrections.items():
        current = int(getattr(spectrum, axis))
        setattr(spectrum, axis, update_spectrum_value(current, delta))
    spectrum.last_evolution = now
    spectrum.updated_at = now
    spectrum.save()

    # 记录演化历史（标注含自评成分）
    create_evolution_history(
        timestamp=now,
        group_id="global",
        sincerity_delta=corrections.get("sincerity", 0),
        engagement_delta=corrections.get("engagement", 0),
        closeness_delta=corrections.get("closeness", 0),
        directness_delta=corrections.get("directness", 0),
        reason=f"自评修正：{', '.join(reason_parts)}",
    )
    logger.info("[SelfReflection] 光谱自评修正：%s", corrections)
