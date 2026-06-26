"""ThoughtSeed 数据类与 CRUD。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import sqlite3

from ._conn import _dt_to_str, _get_conn, _str_to_dt

__all__ = [
    "ThoughtSeed",
    "count_pending_thought_seeds",
    "count_reviewed_seeds",
    "create_thought_seed",
    "delete_oldest_reviewed_seeds",
    "delete_thought_seed",
    "expire_old_pending_seeds",
    "get_pending_thought_seeds",
    "get_thought_seed_by_id",
    "update_seed_status",
]


@dataclass
class ThoughtSeed:
    """思维种子 — 待审核的潜在观点。"""

    seed_id: str = ""
    stream_id: str = ""
    seed_type: str = ""
    event: str = ""
    intensity: int = 0
    confidence: int = 0
    evidence_json: str = "[]"
    reasoning: str = ""
    potential_impact_json: str = "{}"
    context_json: str = "[]"
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"

    def delete_instance(self) -> None:
        """删除当前种子记录。"""
        delete_thought_seed(self.seed_id)


# ─── ThoughtSeed CRUD ───────────────────────────────────────────────


def create_thought_seed(
    seed_id: str,
    stream_id: str,
    seed_type: str,
    event: str,
    intensity: int,
    confidence: int,
    evidence_json: str,
    reasoning: str,
    potential_impact_json: str,
    context_json: str = "[]",
    status: str = "pending",
) -> None:
    """创建思维种子。"""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO soul_thought_seeds
           (seed_id, stream_id, seed_type, event, intensity, confidence,
            evidence_json, reasoning, potential_impact_json, context_json, created_at, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            seed_id, stream_id, seed_type, event, intensity, confidence,
            evidence_json, reasoning, potential_impact_json, context_json,
            _dt_to_str(datetime.now()), status,
        ),
    )
    conn.commit()


def get_pending_thought_seeds(stream_id: str | None = None) -> list[ThoughtSeed]:
    """获取待审核种子列表。"""
    conn = _get_conn()
    if stream_id and stream_id != "global":
        rows = conn.execute(
            "SELECT * FROM soul_thought_seeds WHERE status = 'pending' AND stream_id = ? ORDER BY created_at DESC",
            (stream_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM soul_thought_seeds WHERE status = 'pending' ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_seed(row) for row in rows]


def get_thought_seed_by_id(seed_id: str) -> ThoughtSeed | None:
    """根据 ID 获取种子。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_thought_seeds WHERE seed_id = ?", (seed_id,)
    ).fetchone()
    return _row_to_seed(row) if row else None


def delete_thought_seed(seed_id: str) -> bool:
    """删除种子。"""
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM soul_thought_seeds WHERE seed_id = ?", (seed_id,))
    conn.commit()
    return cursor.rowcount > 0


def count_pending_thought_seeds() -> int:
    """统计待审核种子数量。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM soul_thought_seeds WHERE status = 'pending'"
    ).fetchone()
    return int(row["cnt"]) if row else 0


def update_seed_status(seed_id: str, status: str, expected_status: Optional[str] = "pending") -> bool:
    """更新种子状态（approved/rejected/expired），不删除记录以保留审计链。

    expected_status 非 None 时做原子校验，仅当当前状态匹配才更新，避免竞态下复活已过期/已审核种子。
    """
    conn = _get_conn()
    if expected_status is not None:
        cursor = conn.execute(
            "UPDATE soul_thought_seeds SET status = ? WHERE seed_id = ? AND status = ?",
            (status, seed_id, expected_status),
        )
    else:
        cursor = conn.execute(
            "UPDATE soul_thought_seeds SET status = ? WHERE seed_id = ?",
            (status, seed_id),
        )
    conn.commit()
    return cursor.rowcount > 0


def expire_old_pending_seeds(ttl_hours: float) -> int:
    """将超过 TTL 的 pending 种子标记为 expired，返回过期数量。"""
    if ttl_hours <= 0:
        return 0
    cutoff = _dt_to_str(datetime.now() - timedelta(hours=ttl_hours))
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE soul_thought_seeds SET status = 'expired' WHERE status = 'pending' AND created_at < ?",
        (cutoff,),
    )
    conn.commit()
    return cursor.rowcount


def count_reviewed_seeds() -> int:
    """统计已审核种子数量（approved/rejected/expired）。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM soul_thought_seeds WHERE status != 'pending'"
    ).fetchone()
    return int(row["cnt"]) if row else 0


def delete_oldest_reviewed_seeds(keep_count: int) -> int:
    """删除最旧的已审核种子（approved/rejected/expired），保留最近 keep_count 条。"""
    if keep_count <= 0:
        return 0
    conn = _get_conn()
    rows = conn.execute(
        "SELECT seed_id FROM soul_thought_seeds WHERE status != 'pending' ORDER BY created_at DESC"
    ).fetchall()
    to_delete = [row["seed_id"] for row in rows[keep_count:]]
    for seed_id in to_delete:
        conn.execute("DELETE FROM soul_thought_seeds WHERE seed_id = ?", (seed_id,))
    conn.commit()
    return len(to_delete)


def _row_to_seed(row: sqlite3.Row) -> ThoughtSeed:
    """sqlite3.Row → ThoughtSeed。"""
    return ThoughtSeed(
        seed_id=row["seed_id"],
        stream_id=row["stream_id"],
        seed_type=row["seed_type"],
        event=row["event"],
        intensity=row["intensity"],
        confidence=row["confidence"],
        evidence_json=row["evidence_json"],
        reasoning=row["reasoning"],
        potential_impact_json=row["potential_impact_json"],
        context_json=row["context_json"] if "context_json" in row.keys() else "[]",
        created_at=_str_to_dt(row["created_at"]),
        status=row["status"],
    )
