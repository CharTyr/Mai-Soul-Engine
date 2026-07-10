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
    "FermentationInput",
    "count_fermenting_seeds",
    "count_pending_thought_seeds",
    "count_reviewed_seeds",
    "count_seeds_created_today",
    "count_self_observation_seeds_created_today",
    "create_thought_seed",
    "delete_oldest_reviewed_seeds",
    "delete_thought_seed",
    "delete_fermentation_inputs",
    "expire_old_pending_seeds",
    "extend_fermentation_window",
    "add_fermentation_input",
    "count_fermentation_inputs",
    "get_fermentation_inputs",
    "get_fermenting_seeds",
    "get_pending_thought_seeds",
    "get_thought_seed_by_id",
    "mark_seed_fermenting",
    "mark_seed_internalized",
    "update_fermentation_checked",
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
    fermentation_started_at: datetime | None = None
    fermentation_checked_at: datetime | None = None
    fermentation_extension_count: int = 0

    def delete_instance(self) -> None:
        """删除当前种子记录。"""
        delete_thought_seed(self.seed_id)


@dataclass
class FermentationInput:
    """发酵输入 — 发酵期间收集的与种子相关的群聊消息。"""
    input_id: str = ""
    seed_id: str = ""
    stream_id: str = ""
    message_text: str = ""
    relevance_score: float = 0.0
    added_at: datetime = field(default_factory=datetime.now)


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


def count_fermenting_seeds() -> int:
    """统计发酵中种子数量（status='fermenting'）。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM soul_thought_seeds WHERE status = 'fermenting'"
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
        created_at=_str_to_dt(row["created_at"]) or datetime.now(),
        status=row["status"],
        fermentation_started_at=_str_to_dt(row["fermentation_started_at"]) if "fermentation_started_at" in row.keys() and row["fermentation_started_at"] else None,
        fermentation_checked_at=_str_to_dt(row["fermentation_checked_at"]) if "fermentation_checked_at" in row.keys() and row["fermentation_checked_at"] else None,
        fermentation_extension_count=int(row["fermentation_extension_count"]) if "fermentation_extension_count" in row.keys() else 0,
    )


# ─── Fermentation CRUD ─────────────────────────────────────────────


def add_fermentation_input(seed_id: str, stream_id: str, message_text: str, relevance_score: float = 0.0) -> str:
    """添加一条发酵输入记录，返回 input_id。"""
    import uuid
    conn = _get_conn()
    input_id = f"fi_{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO soul_fermentation_inputs (input_id, seed_id, stream_id, message_text, relevance_score, added_at) VALUES (?, ?, ?, ?, ?, ?)",
        (input_id, seed_id, stream_id, message_text, relevance_score, _dt_to_str(datetime.now())),
    )
    conn.commit()
    return input_id


def get_fermentation_inputs(seed_id: str) -> list[FermentationInput]:
    """获取指定种子的所有发酵输入，按时间升序。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM soul_fermentation_inputs WHERE seed_id = ? ORDER BY added_at ASC",
        (seed_id,),
    ).fetchall()
    return [_row_to_fermentation_input(row) for row in rows]


def count_fermentation_inputs(seed_id: str) -> int:
    """统计指定种子的发酵输入数量。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM soul_fermentation_inputs WHERE seed_id = ?",
        (seed_id,),
    ).fetchone()
    return int(row["cnt"]) if row else 0


def delete_fermentation_inputs(seed_id: str) -> int:
    """删除指定种子的所有发酵输入，返回删除数量。"""
    conn = _get_conn()
    cursor = conn.execute(
        "DELETE FROM soul_fermentation_inputs WHERE seed_id = ?",
        (seed_id,),
    )
    conn.commit()
    return cursor.rowcount


def get_fermenting_seeds() -> list[ThoughtSeed]:
    """获取所有发酵中的种子。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM soul_thought_seeds WHERE status = 'fermenting' ORDER BY fermentation_started_at ASC"
    ).fetchall()
    return [_row_to_seed(row) for row in rows]


def mark_seed_fermenting(seed_id: str) -> bool:
    """将种子标记为发酵中，记录发酵开始时间。原子守卫：仅 pending 状态可转发酵。"""
    conn = _get_conn()
    now = _dt_to_str(datetime.now())
    cursor = conn.execute(
        "UPDATE soul_thought_seeds SET status = 'fermenting', fermentation_started_at = ?, fermentation_checked_at = ? WHERE seed_id = ? AND status = 'pending'",
        (now, now, seed_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def mark_seed_internalized(seed_id: str) -> bool:
    """将种子标记为已内化（发酵完成后的终态）。"""
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE soul_thought_seeds SET status = 'internalized' WHERE seed_id = ? AND status = 'fermenting'",
        (seed_id,),
    )
    conn.commit()
    return cursor.rowcount > 0


def update_fermentation_checked(seed_id: str, checked_at: str) -> bool:
    """更新种子的发酵检查时间戳。"""
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE soul_thought_seeds SET fermentation_checked_at = ? WHERE seed_id = ? AND status = 'fermenting'",
        (checked_at, seed_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def extend_fermentation_window(seed_id: str) -> bool:
    """延长发酵窗口：递增 extension_count，重置 fermentation_started_at 为当前时间。"""
    conn = _get_conn()
    now = _dt_to_str(datetime.now())
    cursor = conn.execute(
        "UPDATE soul_thought_seeds SET fermentation_extension_count = fermentation_extension_count + 1, fermentation_started_at = ?, fermentation_checked_at = ? WHERE seed_id = ? AND status = 'fermenting'",
        (now, now, seed_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def count_seeds_created_today(stream_id: str) -> int:
    """统计今天在指定群创建的种子数量（所有状态）。"""
    conn = _get_conn()
    today_start = _dt_to_str(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM soul_thought_seeds WHERE stream_id = ? AND created_at >= ?",
        (stream_id, today_start),
    ).fetchone()
    return int(row["cnt"]) if row else 0


def count_self_observation_seeds_created_today() -> int:
    """统计今天创建的 self_observation 种子数（所有 stream、所有状态）。"""
    conn = _get_conn()
    today_start = _dt_to_str(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM soul_thought_seeds WHERE seed_type = 'self_observation' AND created_at >= ?",
        (today_start,),
    ).fetchone()
    return int(row["cnt"]) if row else 0


def _row_to_fermentation_input(row: sqlite3.Row) -> FermentationInput:
    """sqlite3.Row → FermentationInput。"""
    return FermentationInput(
        input_id=row["input_id"],
        seed_id=row["seed_id"],
        stream_id=row["stream_id"],
        message_text=row["message_text"],
        relevance_score=float(row["relevance_score"] or 0.0),
        added_at=_str_to_dt(row["added_at"]) or datetime.now(),
    )
