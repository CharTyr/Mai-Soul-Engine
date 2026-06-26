"""IdeologySpectrum / GroupEvolutionRecord 数据类与 CRUD。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import sqlite3

from ._conn import _dt_to_str, _get_conn, _str_to_dt

__all__ = [
    "GroupEvolutionRecord",
    "IdeologySpectrum",
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
    """获取或创建光谱记录。"""
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
        last_evolution=_str_to_dt(row["last_evolution"]),
        updated_at=_str_to_dt(row["updated_at"]),
    )


# ─── GroupEvolutionRecord CRUD ──────────────────────────────────────


def get_or_create_group_evolution(group_id: str) -> GroupEvolutionRecord:
    """获取或创建群组演化记录。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_group_evolution WHERE group_id = ?", (group_id,)
    ).fetchone()
    if row:
        return GroupEvolutionRecord(group_id=row["group_id"], last_analyzed=_str_to_dt(row["last_analyzed"]))
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
