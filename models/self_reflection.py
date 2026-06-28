"""自我评价反馈回路数据类与 CRUD。

三张表：
- ``soul_injection_snapshots``：每次 before_request 注入时落一条快照，用于事后配对
  "这次注入命中了哪些 trait ↔ 这次 LLM 输出了什么"。
- ``soul_pending_reflections``：after_response hook 捕获的待评价回复队列（异步消费）。
- ``soul_self_reflections``：评价完成后的一致性记录（带偏离轴/原因/关联种子）。

设计要点（见 .slim/deepwork/self-reflection.md oracle 审查）：
- snapshot 仅在 ``[self_reflection].enabled=True`` 时写入（防膨胀）。
- pending 有 TTL/上限 + ``expired`` 状态，防队列堆积静默故障。
- pending.context_json 可空——after_response payload 不含触发消息，context 从
  before_request 内存缓存取；空 context 是合法降级路径（评估只基于 response 文本）。
- 一个 snapshot 可配对多条 pending（planner + replyer，或 replyer 多次重试）→ 1:N。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ._conn import _dt_to_str, _get_conn, _str_to_dt

__all__ = [
    "InjectionSnapshot",
    "PendingReflection",
    "SelfReflection",
    "cleanup_expired_pending",
    "cleanup_orphan_snapshots",
    "count_pending_reflections",
    "count_self_reflections",
    "create_injection_snapshot",
    "create_pending_reflection",
    "create_self_reflection",
    "get_injection_snapshot",
    "get_latest_snapshot_for_session",
    "list_pending_reflections",
    "list_recent_reflections",
    "update_pending_status",
]


# ─── 数据类 ───────────────────────────────────────────────────────


@dataclass
class InjectionSnapshot:
    """一次 before_request 注入的快照（命中的 trait / 光谱 / 情绪 / 选择模式）。"""

    snapshot_id: str = ""
    stream_id: str = ""
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    trait_ids_json: str = "[]"
    spectrum_json: str = "{}"
    mood_json: str = "{}"
    selection_mode: str = ""
    context_fingerprint: str = ""


@dataclass
class PendingReflection:
    """after_response 捕获的待评价回复（消费队列项）。"""

    pending_id: int = 0
    stream_id: str = ""
    session_id: str = ""
    reply_message_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    snapshot_id: str = ""
    source: str = ""  # planner | replyer
    response_text: str = ""
    context_json: str = "[]"
    status: str = "pending"  # pending | done | skipped | expired


@dataclass
class SelfReflection:
    """评价完成后的一致性记录。"""

    reflection_id: int = 0
    stream_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    pending_id: int = 0
    snapshot_id: str = ""
    reply_type: str = ""  # social_glue | reactive | substantive
    evaluated: int = 0  # 0=未评(跳过) | 1=已评
    consistency_score: int = 0  # 0-100
    deviating_axis: str = ""  # sincerity | engagement | closeness | directness | ""
    deviating_direction: str = ""  # high | low | ""
    reason: str = ""
    user_reaction_signal: str = ""
    seed_id: str = ""


# ─── injection_snapshots CRUD ─────────────────────────────────────


def create_injection_snapshot(
    stream_id: str,
    session_id: str,
    trait_ids_json: str,
    spectrum_json: str,
    mood_json: str,
    selection_mode: str,
    context_fingerprint: str = "",
) -> str:
    """落一条注入快照，返回 snapshot_id。仅在 [self_reflection].enabled 时调用。"""
    conn = _get_conn()
    snapshot_id = uuid.uuid4().hex
    conn.execute(
        """INSERT INTO soul_injection_snapshots
           (snapshot_id, stream_id, session_id, created_at, trait_ids_json,
            spectrum_json, mood_json, selection_mode, context_fingerprint)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            snapshot_id,
            stream_id,
            session_id,
            _dt_to_str(datetime.now()),
            trait_ids_json,
            spectrum_json,
            mood_json,
            selection_mode,
            context_fingerprint,
        ),
    )
    conn.commit()
    return snapshot_id


def get_latest_snapshot_for_session(session_id: str) -> InjectionSnapshot | None:
    """取该 session 最近一条注入快照（after_response 配对用，1:N 的"1"端）。

    时序安全：inject_ideology 是 BLOCKING hook，宿主在 before_request 完成后才调
    LLM 再触发 after_response，故 snapshot 落库先于本查询。
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_injection_snapshots WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if not row:
        return None
    return _row_to_snapshot(row)


def get_injection_snapshot(snapshot_id: str) -> InjectionSnapshot | None:
    """按 snapshot_id 取注入快照（评价器按 pending.snapshot_id 精确配对用）。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_injection_snapshots WHERE snapshot_id = ?",
        (snapshot_id,),
    ).fetchone()
    if not row:
        return None
    return _row_to_snapshot(row)


# ─── pending_reflections CRUD ─────────────────────────────────────


def create_pending_reflection(
    stream_id: str,
    session_id: str,
    reply_message_id: str,
    snapshot_id: str,
    source: str,
    response_text: str,
    context_json: str = "[]",
) -> int:
    """入队一条待评价回复，返回 pending_id。"""
    conn = _get_conn()
    cursor = conn.execute(
        """INSERT INTO soul_pending_reflections
           (stream_id, session_id, reply_message_id, created_at, snapshot_id,
            source, response_text, context_json, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
        (
            stream_id,
            session_id,
            reply_message_id,
            _dt_to_str(datetime.now()),
            snapshot_id,
            source,
            response_text,
            context_json,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid is not None else 0


def list_pending_reflections(limit: int = 20, max_age_hours: int = 48) -> list[PendingReflection]:
    """取待评价队列（仅 pending 且未超龄），按时间正序（先入先评）。"""
    conn = _get_conn()
    cutoff = _dt_to_str(datetime.now() - timedelta(hours=max_age_hours))
    rows = conn.execute(
        """SELECT * FROM soul_pending_reflections
           WHERE status = 'pending' AND created_at >= ?
           ORDER BY created_at ASC LIMIT ?""",
        (cutoff, limit),
    ).fetchall()
    return [_row_to_pending(row) for row in rows]


def update_pending_status(pending_id: int, status: str) -> bool:
    """更新待评价记录状态（done/skipped/expired）。返回是否更新成功。"""
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE soul_pending_reflections SET status = ? WHERE pending_id = ?",
        (status, pending_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def cleanup_expired_pending(max_age_hours: int = 48, max_rows: int = 5000) -> int:
    """清理过期/超量 pending：超龄 pending 标 expired，超上限删最旧。

    返回清理条数。防队列堆积静默故障（oracle 修订点 3）。
    """
    conn = _get_conn()
    cutoff = _dt_to_str(datetime.now() - timedelta(hours=max_age_hours))
    # 1) 超龄 pending 标 expired（保留记录，不物理删）
    cur = conn.execute(
        "UPDATE soul_pending_reflections SET status = 'expired' WHERE status = 'pending' AND created_at < ?",
        (cutoff,),
    )
    cleaned = cur.rowcount
    # 2) 总行数超上限时物理删最旧（含已 done/skipped/expired）
    total_row = conn.execute("SELECT COUNT(*) FROM soul_pending_reflections").fetchone()
    total = int(total_row[0]) if total_row else 0
    if total > max_rows:
        del_cur = conn.execute(
            """DELETE FROM soul_pending_reflections WHERE pending_id IN (
                   SELECT pending_id FROM soul_pending_reflections
                   ORDER BY created_at ASC LIMIT ?
               )""",
            (total - max_rows,),
        )
        cleaned += del_cur.rowcount
    conn.commit()
    return int(cleaned)


def cleanup_orphan_snapshots(max_age_hours: int = 48) -> int:
    """删除超龄且未被 pending / self_reflections 引用的注入快照，防表膨胀（ora-2 建议）。"""
    conn = _get_conn()
    cutoff = _dt_to_str(datetime.now() - timedelta(hours=max_age_hours))
    cur = conn.execute(
        """DELETE FROM soul_injection_snapshots
           WHERE created_at < ?
             AND snapshot_id NOT IN (
                 SELECT snapshot_id FROM soul_pending_reflections
                 WHERE snapshot_id IS NOT NULL AND snapshot_id != ''
             )
             AND snapshot_id NOT IN (
                 SELECT snapshot_id FROM soul_self_reflections
                 WHERE snapshot_id IS NOT NULL AND snapshot_id != ''
             )""",
        (cutoff,),
    )
    conn.commit()
    return int(cur.rowcount)


def count_pending_reflections() -> dict[str, int]:
    """各状态 pending 计数（dashboard 用）。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT status, COUNT(*) AS c FROM soul_pending_reflections GROUP BY status"
    ).fetchall()
    return {row["status"]: int(row["c"]) for row in rows}


# ─── self_reflections CRUD ────────────────────────────────────────


def create_self_reflection(
    stream_id: str,
    pending_id: int,
    snapshot_id: str,
    reply_type: str,
    evaluated: int,
    consistency_score: int,
    deviating_axis: str = "",
    deviating_direction: str = "",
    reason: str = "",
    user_reaction_signal: str = "",
    seed_id: str = "",
) -> int:
    """落一条评价结果，返回 reflection_id。"""
    conn = _get_conn()
    cursor = conn.execute(
        """INSERT INTO soul_self_reflections
           (stream_id, created_at, pending_id, snapshot_id, reply_type, evaluated,
            consistency_score, deviating_axis, deviating_direction, reason,
            user_reaction_signal, seed_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            stream_id,
            _dt_to_str(datetime.now()),
            pending_id,
            snapshot_id,
            reply_type,
            evaluated,
            consistency_score,
            deviating_axis,
            deviating_direction,
            reason,
            user_reaction_signal,
            seed_id,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid is not None else 0


def list_recent_reflections(stream_id: str, limit: int = 20) -> list[SelfReflection]:
    """取近 N 条已评价记录（planner 反馈路聚合 / /soul_reflect 命令用）。

    stream_id 传 GLOBAL_STREAM 表示全局；否则按群。含全局作用域记录。
    """
    from ..worldview.constants import GLOBAL_STREAM

    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM soul_self_reflections
           WHERE stream_id = ? OR stream_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (stream_id, GLOBAL_STREAM, limit),
    ).fetchall()
    return [_row_to_reflection(row) for row in rows]


def count_self_reflections() -> dict[str, Any]:
    """评价统计（dashboard 用）：总数 / 已评 / 跳过 / 各偏离轴计数。"""
    conn = _get_conn()

    def _count(where: str = "") -> int:
        row = conn.execute(f"SELECT COUNT(*) FROM soul_self_reflections {where}").fetchone()
        return int(row[0]) if row else 0

    total = _count()
    evaluated = _count("WHERE evaluated = 1")
    skipped = _count("WHERE evaluated = 0")
    axis_rows = conn.execute(
        """SELECT deviating_axis, COUNT(*) AS c FROM soul_self_reflections
           WHERE evaluated = 1 AND deviating_axis != '' GROUP BY deviating_axis"""
    ).fetchall()
    by_axis = {row["deviating_axis"]: int(row["c"]) for row in axis_rows}
    return {"total": total, "evaluated": evaluated, "skipped": skipped, "by_axis": by_axis}


# ─── 行映射 ───────────────────────────────────────────────────────


def _row_to_snapshot(row) -> InjectionSnapshot:
    return InjectionSnapshot(
        snapshot_id=row["snapshot_id"],
        stream_id=row["stream_id"],
        session_id=row["session_id"],
        created_at=_str_to_dt(row["created_at"]),
        trait_ids_json=row["trait_ids_json"],
        spectrum_json=row["spectrum_json"],
        mood_json=row["mood_json"],
        selection_mode=row["selection_mode"],
        context_fingerprint=row["context_fingerprint"],
    )


def _row_to_pending(row) -> PendingReflection:
    return PendingReflection(
        pending_id=int(row["pending_id"]),
        stream_id=row["stream_id"],
        session_id=row["session_id"],
        reply_message_id=row["reply_message_id"],
        created_at=_str_to_dt(row["created_at"]),
        snapshot_id=row["snapshot_id"],
        source=row["source"],
        response_text=row["response_text"],
        context_json=row["context_json"],
        status=row["status"],
    )


def _row_to_reflection(row) -> SelfReflection:
    return SelfReflection(
        reflection_id=int(row["reflection_id"]),
        stream_id=row["stream_id"],
        created_at=_str_to_dt(row["created_at"]),
        pending_id=int(row["pending_id"]),
        snapshot_id=row["snapshot_id"],
        reply_type=row["reply_type"],
        evaluated=int(row["evaluated"]),
        consistency_score=int(row["consistency_score"]),
        deviating_axis=row["deviating_axis"],
        deviating_direction=row["deviating_direction"],
        reason=row["reason"],
        user_reaction_signal=row["user_reaction_signal"],
        seed_id=row["seed_id"],
    )
