"""EvolutionHistory 数据类与 CRUD。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ._conn import _dt_to_str, _get_conn, _str_to_dt

__all__ = [
    "EvolutionHistory",
    "create_evolution_history",
    "get_evolution_history",
]


@dataclass
class EvolutionHistory:
    """演化历史 — 每次光谱变化的 delta 记录。"""

    id: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    group_id: str = ""
    sincerity_delta: int = 0
    engagement_delta: int = 0
    closeness_delta: int = 0
    directness_delta: int = 0
    reason: str = ""


# ─── EvolutionHistory CRUD ──────────────────────────────────────────


def create_evolution_history(
    timestamp: datetime,
    group_id: str,
    sincerity_delta: int,
    engagement_delta: int,
    closeness_delta: int,
    directness_delta: int,
    reason: str,
) -> None:
    """创建演化历史记录。"""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO soul_evolution_history
           (timestamp, group_id, sincerity_delta, engagement_delta, closeness_delta, directness_delta, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            _dt_to_str(timestamp), group_id,
            sincerity_delta, engagement_delta, closeness_delta, directness_delta,
            reason,
        ),
    )
    conn.commit()


def get_evolution_history(limit: int = 50) -> list[EvolutionHistory]:
    """获取最近的演化历史。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM soul_evolution_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [
        EvolutionHistory(
            id=row["id"],
            timestamp=_str_to_dt(row["timestamp"]),
            group_id=row["group_id"],
            sincerity_delta=row["sincerity_delta"],
            engagement_delta=row["engagement_delta"],
            closeness_delta=row["closeness_delta"],
            directness_delta=row["directness_delta"],
            reason=row["reason"],
        )
        for row in rows
    ]
