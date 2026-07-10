"""CrystallizedTrait 数据类与 CRUD。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import sqlite3

from ..worldview.constants import GLOBAL_STREAM
from ._conn import _dt_to_str, _get_conn, _str_to_dt

__all__ = [
    "CrystallizedTrait",
    "count_traits_by_layer",
    "create_crystallized_trait",
    "expire_old_traits",
    "get_crystallized_trait_by_id",
    "get_crystallized_traits_by_ids",
    "query_active_traits_for_injection",
    "query_crystallized_traits",
    "save_crystallized_trait",
    "set_trait_lifecycle_state",
]


@dataclass
class CrystallizedTrait:
    """固化 trait — 已内化的观点，可禁用/软删除。"""

    trait_id: str = ""
    stream_id: str = ""
    seed_id: str = ""
    name: str = ""
    question: str = ""
    thought: str = ""
    tags_json: str = "[]"
    confidence: int = 0
    evidence_json: str = "[]"
    spectrum_impact_json: str = "{}"
    created_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True
    deleted: bool = False
    ideology_layer: str = "conduct"
    lifecycle_state: str = "active"
    origin_stream_id: str = ""
    cabinet_slot_no: int | None = None

    def save(self, commit: bool = True) -> None:
        """持久化当前 trait。"""
        save_crystallized_trait(self, commit=commit)


# ─── CrystallizedTrait CRUD ─────────────────────────────────────────


def create_crystallized_trait(
    trait_id: str,
    stream_id: str,
    seed_id: str,
    name: str,
    question: str,
    thought: str,
    tags_json: str,
    confidence: int,
    evidence_json: str,
    spectrum_impact_json: str,
    created_at: datetime | None = None,
    enabled: bool = True,
    deleted: bool = False,
    ideology_layer: str = "conduct",
    lifecycle_state: str = "active",
    origin_stream_id: str = "",
    cabinet_slot_no: int | None = None,
    commit: bool = True,
) -> None:
    """创建固化 trait。

    Args:
        origin_stream_id: 来源流 ID（内化种子所属的群 stream_id），
            用于溯源该 trait 最初来源于哪个群的讨论。
            空串表示无特定来源。
        commit: 是否立即提交。atomic 路径传 ``commit=False``（由外层事务统一管理）。
    """
    # 空串 = 未设置 = 全局作用域，归一为显式 GLOBAL_STREAM，避免与"异常空值"混淆
    if not stream_id:
        stream_id = GLOBAL_STREAM
    conn = _get_conn()
    conn.execute(
        """INSERT INTO soul_crystallized_traits
           (trait_id, stream_id, seed_id, name, question, thought, tags_json,
            confidence, evidence_json, spectrum_impact_json, created_at, enabled, deleted,
            ideology_layer, lifecycle_state, origin_stream_id, cabinet_slot_no)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trait_id, stream_id, seed_id, name, question, thought, tags_json,
            confidence, evidence_json, spectrum_impact_json,
            _dt_to_str(created_at or datetime.now()),
            int(enabled), int(deleted),
            ideology_layer or "conduct",
            lifecycle_state or "active",
            origin_stream_id or "",
            cabinet_slot_no,
        ),
    )
    if commit:
        conn.commit()


def get_crystallized_trait_by_id(trait_id: str) -> CrystallizedTrait | None:
    """根据 ID 获取 trait（不含已软删除的）。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_crystallized_traits WHERE trait_id = ? AND deleted = 0", (trait_id,)
    ).fetchone()
    return _row_to_trait(row) if row else None


def save_crystallized_trait(t: CrystallizedTrait, commit: bool = True) -> None:
    """更新 trait。"""
    conn = _get_conn()
    conn.execute(
        """UPDATE soul_crystallized_traits SET
           stream_id = ?, seed_id = ?, name = ?, question = ?, thought = ?,
           tags_json = ?, confidence = ?, evidence_json = ?, spectrum_impact_json = ?,
           created_at = ?, enabled = ?, deleted = ?,
           ideology_layer = ?, lifecycle_state = ?, origin_stream_id = ?,
           cabinet_slot_no = ?
           WHERE trait_id = ?""",
        (
            t.stream_id, t.seed_id, t.name, t.question, t.thought,
            t.tags_json, t.confidence, t.evidence_json, t.spectrum_impact_json,
            _dt_to_str(t.created_at), int(t.enabled), int(t.deleted),
            t.ideology_layer or "conduct",
            t.lifecycle_state or "active",
            t.origin_stream_id or "",
            t.cabinet_slot_no,
            t.trait_id,
        ),
    )
    if commit:
        conn.commit()


def query_crystallized_traits(
    *,
    deleted: bool = False,
    enabled: bool | None = None,
    stream_id: str | None = None,
    order_by_created_desc: bool = True,
    limit: int = 50,
) -> list[CrystallizedTrait]:
    """查询固化 trait 列表。"""
    conn = _get_conn()
    conditions = ["deleted = ?"]
    params: list[Any] = [int(deleted)]
    if enabled is not None:
        conditions.append("enabled = ?")
        params.append(int(enabled))
    if stream_id is not None:
        # 空串 = 未设置 = 全局作用域，归一为显式 GLOBAL_STREAM
        if not stream_id:
            stream_id = GLOBAL_STREAM
        conditions.append("stream_id = ?")
        params.append(stream_id)
    where = " AND ".join(conditions)
    order = "created_at DESC" if order_by_created_desc else "created_at ASC"
    sql = f"SELECT * FROM soul_crystallized_traits WHERE {where} ORDER BY {order} LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_trait(row) for row in rows]


def query_active_traits_for_injection(
    stream_id: str = "",
    limit: int = 80,
) -> list[CrystallizedTrait]:
    """查询用于注入的活跃 trait（按 stream_id 匹配或全局）。

    使用 UNION ALL 拆分为两条查询，确保 idx_traits_stream_enabled 索引对两条子查询都有效
    （SQLite 对 OR 条件通常退化为全表扫描，UNION ALL 各自走索引）。
    """
    conn = _get_conn()
    if stream_id and stream_id != GLOBAL_STREAM:
        rows = conn.execute(
            """SELECT * FROM soul_crystallized_traits
               WHERE deleted = 0 AND enabled = 1 AND stream_id = ?
               UNION ALL
               SELECT * FROM soul_crystallized_traits
               WHERE deleted = 0 AND enabled = 1 AND stream_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (stream_id, GLOBAL_STREAM, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM soul_crystallized_traits
               WHERE deleted = 0 AND enabled = 1 AND stream_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (GLOBAL_STREAM, limit),
        ).fetchall()
    return [_row_to_trait(row) for row in rows]


def get_crystallized_traits_by_ids(trait_ids: list[str]) -> dict[str, CrystallizedTrait]:
    """批量按 ID 查 trait，返回 {trait_id: trait}。

    用于替代 N+1 场景中循环调用 get_crystallized_trait_by_id。
    单次 SQL IN 查询，空列表返回空 dict。
    """
    if not trait_ids:
        return {}
    conn = _get_conn()
    placeholders = ",".join("?" * len(trait_ids))
    rows = conn.execute(
        f"""SELECT * FROM soul_crystallized_traits
            WHERE trait_id IN ({placeholders}) AND deleted = 0""",
        trait_ids,
    ).fetchall()
    return {row["trait_id"]: _row_to_trait(row) for row in rows}


def _row_to_trait(row: sqlite3.Row) -> CrystallizedTrait:
    """sqlite3.Row → CrystallizedTrait。"""
    return CrystallizedTrait(
        trait_id=row["trait_id"],
        stream_id=row["stream_id"],
        seed_id=row["seed_id"],
        name=row["name"],
        question=row["question"],
        thought=row["thought"],
        tags_json=row["tags_json"],
        confidence=row["confidence"],
        evidence_json=row["evidence_json"],
        spectrum_impact_json=row["spectrum_impact_json"],
        created_at=_str_to_dt(row["created_at"]) or datetime.now(),
        enabled=bool(row["enabled"]),
        deleted=bool(row["deleted"]),
        ideology_layer=row["ideology_layer"] if "ideology_layer" in row.keys() else "conduct",
        lifecycle_state=row["lifecycle_state"] if "lifecycle_state" in row.keys() else "active",
        origin_stream_id=row["origin_stream_id"] if "origin_stream_id" in row.keys() else "",
        cabinet_slot_no=row["cabinet_slot_no"] if "cabinet_slot_no" in row.keys() else None,
    )


# ─── 统计工具 ────────────────────────────────────────────────────────


def count_traits_by_layer() -> dict[str, int]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT ideology_layer, COUNT(*) AS cnt FROM soul_crystallized_traits
           WHERE deleted = 0 AND enabled = 1 GROUP BY ideology_layer"""
    ).fetchall()
    out: dict[str, int] = {"values": 0, "worldview": 0, "conduct": 0}
    for row in rows:
        layer = row["ideology_layer"] or "conduct"
        out[layer] = int(row["cnt"])
    return out


def expire_old_traits(ttl_days: int) -> int:
    """将超过 TTL 且未被强化的 active trait 标记为 expired 并禁用注入。

    只处理 lifecycle_state='active' 的 trait（strengthened 的不受影响——
    被强化说明有重复证据支撑，不应轻易过期）。

    Args:
        ttl_days: 过期阈值天数。

    Returns:
        过期的 trait 数量。
    """
    if ttl_days <= 0:
        return 0
    cutoff = _dt_to_str(datetime.now() - timedelta(days=ttl_days))
    conn = _get_conn()
    cursor = conn.execute(
        """UPDATE soul_crystallized_traits
           SET lifecycle_state = 'expired', enabled = 0
           WHERE deleted = 0 AND enabled = 1
             AND lifecycle_state = 'active'
             AND created_at < ?""",
        (cutoff,),
    )
    conn.commit()
    return cursor.rowcount


def set_trait_lifecycle_state(
    trait_id: str,
    lifecycle_state: str,
    *,
    enabled: bool | None = None,
    commit: bool = True,
) -> bool:
    """设置 trait 的生命周期状态，可选同时改 enabled。

    用于内省矛盾检测：contradicted 时传 enabled=False 禁用，
    weakened/revised 时 enabled 不变（保留可见但降权）。

    返回是否更新成功（rowcount > 0）。
    """
    conn = _get_conn()
    if enabled is not None:
        cursor = conn.execute(
            "UPDATE soul_crystallized_traits SET lifecycle_state = ?, enabled = ? WHERE trait_id = ?",
            (lifecycle_state, 1 if enabled else 0, trait_id),
        )
    else:
        cursor = conn.execute(
            "UPDATE soul_crystallized_traits SET lifecycle_state = ? WHERE trait_id = ?",
            (lifecycle_state, trait_id),
        )
    if commit:
        conn.commit()
    return cursor.rowcount > 0
