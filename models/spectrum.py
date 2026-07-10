"""IdeologySpectrum / GroupEvolutionRecord 数据类与 CRUD。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import logging
import sqlite3

from ._conn import _dt_to_str, _get_conn, _str_to_dt

logger = logging.getLogger(__name__)

__all__ = [
    "GroupEvolutionRecord",
    "IdeologySpectrum",
    "apply_spectrum_deltas",
    "get_or_create_group_evolution",
    "get_or_create_spectrum",
    "save_group_evolution_record",
    "save_spectrum",
]


@dataclass
class IdeologySpectrum:
    """意识形态光谱 — 群聊社交四维 (0-100)。

    v2.1.0 起四维从政治光谱换为群聊社交轴：
    - sincerity（真诚度）：真诚直率 ↔ 重视场面与分寸
    - engagement（投入度）：克制怕消耗 ↔ 热情投入
    - closeness（亲密度）：保持距离 ↔ 容易亲近
    - directness（直率度）：含蓄绕弯 ↔ 有话直说
    """

    scope_id: str = "global"
    sincerity: int = 50
    engagement: int = 50
    closeness: int = 50
    directness: int = 50
    last_sincerity_dir: int = 0
    last_engagement_dir: int = 0
    last_closeness_dir: int = 0
    last_directness_dir: int = 0
    initialized: bool = False
    last_evolution: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def save(self) -> None:
        """持久化当前光谱状态。"""
        save_spectrum(self)


@dataclass
class GroupEvolutionRecord:
    """群组演化记录 — 记录每个群上次分析的时间。"""

    group_id: str = ""
    last_analyzed: datetime = field(default_factory=datetime.now)

    def save(self) -> None:
        """持久化当前记录。"""
        save_group_evolution_record(self)


# ─── IdeologySpectrum CRUD ──────────────────────────────────────────


def get_or_create_spectrum(scope_id: str = "global") -> IdeologySpectrum:
    """获取或创建光谱记录。DB 异常时返回默认中性光谱（降级不崩）。"""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM soul_ideology_spectrum WHERE scope_id = ?", (scope_id,)
        ).fetchone()
        if row:
            return _row_to_spectrum(row)
        # 不存在则创建
        now = datetime.now()
        conn.execute(
            """INSERT INTO soul_ideology_spectrum
               (scope_id, sincerity, engagement, closeness, directness,
                last_sincerity_dir, last_engagement_dir, last_closeness_dir, last_directness_dir,
                initialized, last_evolution, updated_at)
               VALUES (?, 50, 50, 50, 50, 0, 0, 0, 0, 0, ?, ?)""",
            (scope_id, _dt_to_str(now), _dt_to_str(now)),
        )
        conn.commit()
        return IdeologySpectrum(scope_id=scope_id, last_evolution=now, updated_at=now)
    except Exception as e:
        logger.warning("[Spectrum] get_or_create_spectrum 降级: %s", e)
        return IdeologySpectrum(scope_id=scope_id)


def save_spectrum(s: IdeologySpectrum) -> None:
    """更新光谱记录。"""
    conn = _get_conn()
    conn.execute(
        """UPDATE soul_ideology_spectrum SET
           sincerity = ?, engagement = ?, closeness = ?, directness = ?,
           last_sincerity_dir = ?, last_engagement_dir = ?, last_closeness_dir = ?, last_directness_dir = ?,
           initialized = ?, last_evolution = ?, updated_at = ?
           WHERE scope_id = ?""",
        (
            s.sincerity, s.engagement, s.closeness, s.directness,
            s.last_sincerity_dir, s.last_engagement_dir, s.last_closeness_dir, s.last_directness_dir,
            int(s.initialized), _dt_to_str(s.last_evolution), _dt_to_str(s.updated_at),
            s.scope_id,
        ),
    )
    conn.commit()


def _row_to_spectrum(row: sqlite3.Row) -> IdeologySpectrum:
    """sqlite3.Row → IdeologySpectrum。"""
    return IdeologySpectrum(
        scope_id=row["scope_id"],
        sincerity=row["sincerity"],
        engagement=row["engagement"],
        closeness=row["closeness"],
        directness=row["directness"],
        last_sincerity_dir=row["last_sincerity_dir"],
        last_engagement_dir=row["last_engagement_dir"],
        last_closeness_dir=row["last_closeness_dir"],
        last_directness_dir=row["last_directness_dir"],
        initialized=bool(row["initialized"]),
        last_evolution=_str_to_dt(row["last_evolution"]) or datetime.now(),
        updated_at=_str_to_dt(row["updated_at"]) or datetime.now(),
    )


# ─── 统一光谱写入闸门（v2.3.0 三回路收口）────────────────────────


def apply_spectrum_deltas(
    source: str,
    deltas: dict[str, int],
    *,
    smooth_alpha: float = 0.0,
    resistance: float = 0.0,
    max_per_axis: int | None = None,
    group_id: str = "",
    reason: str = "",
    write_history: bool = True,
    commit: bool = True,
) -> dict[str, int]:
    """统一光谱写入闸门 — 所有改人设数值的路径都经此（ora-2 三回路瘦身高优先级建议）。

    收口三条写入路径，使单表三路叠写可观测、可统一限幅：
    - 回路1 群聊演化：``smooth_alpha=ema_alpha, resistance=direction_resistance, max=evolution_rate``
    - 回路2 内化：``max_per_axis=10``（保留现行 ±10；如需收紧改此一处）
    - 回路3 自评修正：``max=evolution_rate``（weight 缩放的 magnitude 由调用方先算好传入）

    顺序：clamp(max_per_axis) → resistance → smooth(EMA) → update_spectrum_value → save → history。

    Args:
        source: 写入来源标签（"evolution"/"internalize"/"self_reflection"），写入 history reason 与日志。
        deltas: 四轴原始 delta（调用方已完成 layer caps 等预处理）。
        smooth_alpha: EMA 平滑系数，0=不平滑（回路2/3）。
        resistance: 反向变动阻力系数，0=不阻力（回路2/3）。
        max_per_axis: 单轴 delta 绝对值上限（clamp）。
        group_id: 演化历史 group_id（回路1 传 stream_id；回路3 传 "global"；回路2 留空）。
        reason: 附加到 history reason 的说明。
        write_history: 是否写 soul_evolution_history（默认 True；回路2 经此获得可观测性）。
        commit: 是否立即提交。``internalize_seed`` 等的原子化路径传 ``commit=False``（由外层事务统一 commit/rollback）。

    Returns:
        实际应用的 deltas（经 resistance/smooth 后），供调用方做 slice/mood/audit 副作用。
    """
    from ..utils.spectrum_utils import apply_resistance, smooth_delta, update_spectrum_value
    from .history import create_evolution_history

    spectrum = get_or_create_spectrum("global")
    now = datetime.now()
    applied: dict[str, int] = {}
    new_dirs: dict[str, int] = {}

    for dim in ("sincerity", "engagement", "closeness", "directness"):
        delta = int(deltas.get(dim, 0) or 0)
        if max_per_axis is not None and max_per_axis >= 0:
            delta = max(-max_per_axis, min(max_per_axis, delta))
        if resistance > 0:
            last_dir = int(getattr(spectrum, f"last_{dim}_dir", 0))
            delta, new_dir = apply_resistance(delta, last_dir, resistance)
        elif delta == 0:
            # 与 apply_resistance 对齐：delta=0 时保留 last_dir（不清零）
            new_dir = int(getattr(spectrum, f"last_{dim}_dir", 0))
        else:
            new_dir = 1 if delta > 0 else -1
        if smooth_alpha > 0:
            delta = smooth_delta(int(getattr(spectrum, dim)), delta, smooth_alpha)
            # 回归均值力：超 80 或低 20 时微弱向 50 回归（每轮 ±1），防极值卡死。
            # 仅演化回路触发（smooth_alpha > 0）；内化/自评回路 smooth_alpha=0 不触发。
            # 阈值 20/80 比极值告警阈值 10/90 更宽松，是有意设计：回归更早介入，告警更极端才响。
            current_val = int(getattr(spectrum, dim))
            if current_val > 80:
                delta -= 1
            elif current_val < 20:
                delta += 1
        current = int(getattr(spectrum, dim))
        setattr(spectrum, dim, update_spectrum_value(current, delta))
        setattr(spectrum, f"last_{dim}_dir", new_dir)
        applied[dim] = delta
        new_dirs[dim] = new_dir

    spectrum.last_evolution = now
    spectrum.updated_at = now

    # 事务原子化：直接内联 SQL，绕过 save_spectrum/create_evolution_history 各自的 commit，
    # 确保 UPDATE 光谱 + INSERT 历史在同一事务内（with conn: 退出时统一 commit/rollback）。
    # commit=False 模式：由外层事务统一管理，此处不 with conn: 不 commit。
    conn = _get_conn()
    if commit:
        with conn:
            conn.execute(
                """UPDATE soul_ideology_spectrum SET
                   sincerity = ?, engagement = ?, closeness = ?, directness = ?,
                   last_sincerity_dir = ?, last_engagement_dir = ?, last_closeness_dir = ?, last_directness_dir = ?,
                   initialized = ?, last_evolution = ?, updated_at = ?
                   WHERE scope_id = ?""",
                (
                    spectrum.sincerity, spectrum.engagement, spectrum.closeness, spectrum.directness,
                    spectrum.last_sincerity_dir, spectrum.last_engagement_dir,
                    spectrum.last_closeness_dir, spectrum.last_directness_dir,
                    int(spectrum.initialized), _dt_to_str(now), _dt_to_str(now),
                    spectrum.scope_id,
                ),
            )
            if write_history:
                tagged_reason = f"[{source}] {reason}" if reason else f"[{source}]"
                conn.execute(
                    """INSERT INTO soul_evolution_history
                       (timestamp, group_id, sincerity_delta, engagement_delta, closeness_delta, directness_delta, reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        _dt_to_str(now), group_id,
                        applied.get("sincerity", 0), applied.get("engagement", 0),
                        applied.get("closeness", 0), applied.get("directness", 0),
                        tagged_reason,
                    ),
                )
    else:
        conn.execute(
            """UPDATE soul_ideology_spectrum SET
               sincerity = ?, engagement = ?, closeness = ?, directness = ?,
               last_sincerity_dir = ?, last_engagement_dir = ?, last_closeness_dir = ?, last_directness_dir = ?,
               initialized = ?, last_evolution = ?, updated_at = ?
               WHERE scope_id = ?""",
            (
                spectrum.sincerity, spectrum.engagement, spectrum.closeness, spectrum.directness,
                spectrum.last_sincerity_dir, spectrum.last_engagement_dir,
                spectrum.last_closeness_dir, spectrum.last_directness_dir,
                int(spectrum.initialized), _dt_to_str(now), _dt_to_str(now),
                spectrum.scope_id,
            ),
        )
        if write_history:
            tagged_reason = f"[{source}] {reason}" if reason else f"[{source}]"
            conn.execute(
                """INSERT INTO soul_evolution_history
                   (timestamp, group_id, sincerity_delta, engagement_delta, closeness_delta, directness_delta, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    _dt_to_str(now), group_id,
                    applied.get("sincerity", 0), applied.get("engagement", 0),
                    applied.get("closeness", 0), applied.get("directness", 0),
                    tagged_reason,
                ),
            )
    logger.info("[SpectrumGate] %s applied: %s", source, applied)
    return applied


# ─── GroupEvolutionRecord CRUD ──────────────────────────────────────


def get_or_create_group_evolution(group_id: str) -> GroupEvolutionRecord:
    """获取或创建群组演化记录。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_group_evolution WHERE group_id = ?", (group_id,)
    ).fetchone()
    if row:
        return GroupEvolutionRecord(group_id=row["group_id"], last_analyzed=_str_to_dt(row["last_analyzed"]) or datetime.now())
    now = datetime.now()
    conn.execute(
        "INSERT INTO soul_group_evolution (group_id, last_analyzed) VALUES (?, ?)",
        (group_id, _dt_to_str(now)),
    )
    conn.commit()
    return GroupEvolutionRecord(group_id=group_id, last_analyzed=now)


def save_group_evolution_record(r: GroupEvolutionRecord) -> None:
    """更新群组演化记录。"""
    conn = _get_conn()
    conn.execute(
        "UPDATE soul_group_evolution SET last_analyzed = ? WHERE group_id = ?",
        (_dt_to_str(r.last_analyzed), r.group_id),
    )
    conn.commit()
