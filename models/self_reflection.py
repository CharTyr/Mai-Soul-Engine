"""自我评价反馈回路数据类与 CRUD。

三张表：
- ``soul_injection_snapshots``：每次 before_request 注入时落一条快照，用于事后配对
  "这次注入命中了哪些 trait ↔ 这次 LLM 输出了什么"。
- ``soul_pending_reflections``：after_response hook 捕获的待评价回复队列（异步消费）。
- ``soul_self_reflections``：评价完成后的一致性记录（带偏离轴/原因/关联种子）。

设计要点：
- snapshot 仅在 ``[self_reflection].enabled=True`` 时写入（防膨胀）。
- pending 有 TTL/上限 + ``expired`` 状态，防队列堆积静默故障。
- **触发上文随快照落库**（``snapshot.context_json``）：旧实现放在 session 键的内存
  缓存里，同会话两轮并发会互相顶掉，导致回复配上别人的触发消息。
- **配对是 FIFO 认领**：``claim_snapshot_for_response`` 取该 session 最旧的未认领
  快照并标记认领；同一 ``reply_message_id`` 的重试复用同一快照（1:N 仅限同轮重试）。
- **投递阶段三段可观测**：``selected`` → ``hook_applied`` → ``final_request_verified``；
  无法确认最终请求时落 ``unverified``，不得当成成功。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ._conn import _dt_to_str, _get_conn, _str_to_dt

# 投递阶段（T04）：区分「已选中」「已交回宿主」「已确认进入最终请求」
DELIVERY_SELECTED = "selected"
SNAPSHOT_MAX_CLAIM_AGE_SECONDS = 1800
DELIVERY_HOOK_APPLIED = "hook_applied"
DELIVERY_FINAL_VERIFIED = "final_request_verified"
DELIVERY_UNVERIFIED = "unverified"
SNAPSHOT_DELIVERY_STATES = (
    DELIVERY_SELECTED,
    DELIVERY_HOOK_APPLIED,
    DELIVERY_FINAL_VERIFIED,
    DELIVERY_UNVERIFIED,
)

__all__ = [
    "SNAPSHOT_MAX_CLAIM_AGE_SECONDS",
    "DELIVERY_FINAL_VERIFIED",
    "DELIVERY_HOOK_APPLIED",
    "DELIVERY_SELECTED",
    "DELIVERY_UNVERIFIED",
    "SNAPSHOT_DELIVERY_STATES",
    "InjectionSnapshot",
    "PendingReflection",
    "SelfReflection",
    "claim_snapshot_for_response",
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
    "list_unconsumed_reflections_for_correction",
    "mark_reflections_correction_consumed",
    "mark_snapshot_delivery_state",
    "update_pending_status",
]


# ─── 数据类 ───────────────────────────────────────────────────────


@dataclass
class InjectionSnapshot:
    """一次 before_request 注入的快照（命中的 trait / 光谱 / 情绪 / 选择模式）。

    ``context_json``：该轮注入时的触发上文，随快照走（不同轮次不互相覆盖）。
    ``consumed_at`` / ``consumed_by_reply``：被哪条回复认领，保证 1:1 归属。
    ``delivery_state``：投递阶段（详见 ``SNAPSHOT_DELIVERY_STATES``）。
    """

    snapshot_id: str = ""
    stream_id: str = ""
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    trait_ids_json: str = "[]"
    spectrum_json: str = "{}"
    mood_json: str = "{}"
    selection_mode: str = ""
    context_fingerprint: str = ""
    context_json: str = "[]"
    consumed_at: str = ""
    consumed_by_reply: str = ""
    delivery_state: str = DELIVERY_SELECTED
    # 配对歧义：同会话存在多条未认领快照时，无法确定这条回复对应哪次注入。
    # 下游**不得**据此改写人格（标记而不是猜）。
    pairing_ambiguous: bool = False


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
    consistency_score: int = 0  # 0-100（主分 = raw，兼容旧读路径）
    raw_consistency_score: int | None = None  # 归一化前原始分
    normalized_consistency_score: int | None = None  # 归一化后分（若无归一化则 = raw）
    correction_consumed_at: str = ""  # 空串=未消费，非空时间戳=已消费
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
    context_json: str = "[]",
) -> str:
    """落一条注入快照，返回 snapshot_id。仅在 [self_reflection].enabled 时调用。

    ``context_json`` 是该轮的触发上文，随快照一起落库，保证并发轮次互不覆盖。
    """
    conn = _get_conn()
    snapshot_id = uuid.uuid4().hex
    conn.execute(
        """INSERT INTO soul_injection_snapshots
           (snapshot_id, stream_id, session_id, created_at, trait_ids_json,
            spectrum_json, mood_json, selection_mode, context_fingerprint,
            context_json, delivery_state)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            context_json,
            DELIVERY_SELECTED,
        ),
    )
    conn.commit()
    return snapshot_id


def claim_snapshot_for_response(
    session_id: str,
    reply_message_id: str = "",
    max_age_seconds: int = SNAPSHOT_MAX_CLAIM_AGE_SECONDS,
) -> InjectionSnapshot | None:
    """为一条回复认领注入快照（FIFO，1:1 归属）。

    规则：
      1. 同一 ``reply_message_id`` 若已认领过快照 → 复用（replyer 重试属同一轮）。
      2. 否则取该 session **最旧的未认领且不过期**快照并标记认领。
      3. 无可认领快照 → 返回 None（合法降级：评估只基于 response 文本）。

    为什么 FIFO 而不是"最新一条"：旧实现取最新，同会话两轮并发时会把轮次 A 的
    回复配到轮次 B 的快照上——trait 归属和触发上文一起串味。

    为什么有 ``max_age_seconds``：宿主 planner/after_response 两个 payload 之间
    **没有关联 id**（已核对宿主源码），所以配对只能是启发式。若某一轮生成失败、
    after_response 没触发，它的快照会滞留；没有窗口限制的话，它会去配很久以后
    另一轮的回复。超过窗口的快照不再被认领（返回 None = 合法降级），
    但保留在表里可审计。
    """
    conn = _get_conn()
    if reply_message_id:
        row = conn.execute(
            "SELECT * FROM soul_injection_snapshots "
            "WHERE session_id = ? AND consumed_by_reply = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (session_id, reply_message_id),
        ).fetchone()
        if row:
            return _row_to_snapshot(row)

    cutoff = _dt_to_str(datetime.now() - timedelta(seconds=int(max_age_seconds)))
    row = conn.execute(
        "SELECT * FROM soul_injection_snapshots "
        "WHERE session_id = ? AND (consumed_at IS NULL OR consumed_at = '') "
        "AND created_at >= ? "
        "ORDER BY created_at ASC, rowid ASC LIMIT 1",
        (session_id, cutoff),
    ).fetchone()
    if not row:
        return None

    # 配对歧义：窗口内还有别的未认领快照 → 无法确定这条回复对应哪一次注入。
    # 仍然按 FIFO 消费（否则队列会滞留），但**打上标记**，
    # 下游据此跳过会改写人格的自评反馈——不允许拿猜出来的关联去改人格。
    pending_count = conn.execute(
        "SELECT COUNT(*) AS n FROM soul_injection_snapshots "
        "WHERE session_id = ? AND (consumed_at IS NULL OR consumed_at = '') "
        "AND created_at >= ?",
        (session_id, cutoff),
    ).fetchone()
    ambiguous = int(pending_count["n"] or 0) > 1

    now = _dt_to_str(datetime.now())
    cursor = conn.execute(
        "UPDATE soul_injection_snapshots "
        "SET consumed_at = ?, consumed_by_reply = ?, pairing_ambiguous = ? "
        "WHERE snapshot_id = ? AND (consumed_at IS NULL OR consumed_at = '')",
        (now, reply_message_id, 1 if ambiguous else 0, row["snapshot_id"]),
    )
    conn.commit()
    if cursor.rowcount != 1:
        # 并发下被抢走：本次不强行复用，退回无快照降级
        return None
    fresh = conn.execute(
        "SELECT * FROM soul_injection_snapshots WHERE snapshot_id = ?",
        (row["snapshot_id"],),
    ).fetchone()
    return _row_to_snapshot(fresh) if fresh else None


def mark_snapshot_delivery_state(snapshot_id: str, state: str) -> bool:
    """推进注入快照的投递阶段；未知状态拒绝写入。"""
    if state not in SNAPSHOT_DELIVERY_STATES:
        return False
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE soul_injection_snapshots SET delivery_state = ? WHERE snapshot_id = ?",
        (state, snapshot_id),
    )
    conn.commit()
    return cursor.rowcount == 1


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
    normalized_reply_message_id = str(reply_message_id or "").strip()
    with conn:
        if normalized_reply_message_id:
            existing = conn.execute(
                """SELECT pending_id FROM soul_pending_reflections
                   WHERE source = ? AND reply_message_id = ?
                   ORDER BY pending_id DESC LIMIT 1""",
                (source, normalized_reply_message_id),
            ).fetchone()
            if existing:
                return int(existing[0])
        cursor = conn.execute(
            """INSERT INTO soul_pending_reflections
               (stream_id, session_id, reply_message_id, created_at, snapshot_id,
                source, response_text, context_json, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
            (
                stream_id,
                session_id,
                normalized_reply_message_id,
                _dt_to_str(datetime.now()),
                snapshot_id,
                source,
                response_text,
                context_json,
            ),
        )
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
                   WHERE status != 'pending'
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
    raw_consistency_score: int | None = None,
    normalized_consistency_score: int | None = None,
) -> int:
    """落一条评价结果，返回 reflection_id。

    consistency_score 为主分（写入 raw 列兼容旧读路径），同时写 raw 与 normalized 列。
    """
    conn = _get_conn()
    # 归一化数据库也带自省值
    raw = raw_consistency_score if raw_consistency_score is not None else consistency_score
    norm = normalized_consistency_score if normalized_consistency_score is not None else raw
    cursor = conn.execute(
        """INSERT INTO soul_self_reflections
           (stream_id, created_at, pending_id, snapshot_id, reply_type, evaluated,
            consistency_score, raw_consistency_score, normalized_consistency_score,
            deviating_axis, deviating_direction, reason,
            user_reaction_signal, seed_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            stream_id,
            _dt_to_str(datetime.now()),
            pending_id,
            snapshot_id,
            reply_type,
            evaluated,
            consistency_score,
            raw,
            norm,
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


def list_unconsumed_reflections_for_correction(limit: int = 30) -> list[SelfReflection]:
    """取未消费的评价记录（供光谱修正用），不限制 stream_id（修复跨 session 空转）。

    条件：已评(evaluated=1) + substantive + correction_consumed_at 为空串/NULL。
    按 created_at 降序，取最新 limit 条。
    """
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM soul_self_reflections
           WHERE evaluated = 1 AND reply_type = 'substantive'
             AND (correction_consumed_at IS NULL OR correction_consumed_at = '')
           ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [_row_to_reflection(row) for row in rows]


def mark_reflections_correction_consumed(reflection_ids: list[int], consumed_at: str) -> int:
    """批量标记自评记录为已消费（correction_consumed_at 写入时间戳）。

    Args:
        reflection_ids: 参与聚合的 reflection_id 列表。
        consumed_at: 消费时间戳（ISO 格式）。

    Returns: 受影响行数。
    """
    if not reflection_ids:
        return 0
    conn = _get_conn()
    placeholders = ",".join("?" for _ in reflection_ids)
    cursor = conn.execute(
        f"UPDATE soul_self_reflections SET correction_consumed_at = ? WHERE reflection_id IN ({placeholders})",
        (consumed_at, *reflection_ids),
    )
    conn.commit()
    return int(cursor.rowcount)


# ─── 行映射 ───────────────────────────────────────────────────────


def _row_to_snapshot(row) -> InjectionSnapshot:
    return InjectionSnapshot(
        snapshot_id=row["snapshot_id"],
        stream_id=row["stream_id"],
        session_id=row["session_id"],
        created_at=_str_to_dt(row["created_at"]) or datetime.now(),
        trait_ids_json=row["trait_ids_json"],
        spectrum_json=row["spectrum_json"],
        mood_json=row["mood_json"],
        selection_mode=row["selection_mode"],
        context_fingerprint=row["context_fingerprint"],
        context_json=row["context_json"] if "context_json" in row.keys() else "[]",
        consumed_at=row["consumed_at"] if "consumed_at" in row.keys() else "",
        consumed_by_reply=row["consumed_by_reply"] if "consumed_by_reply" in row.keys() else "",
        delivery_state=(
            row["delivery_state"] if "delivery_state" in row.keys() else DELIVERY_SELECTED
        ),
        pairing_ambiguous=bool(
            row["pairing_ambiguous"] if "pairing_ambiguous" in row.keys() else 0
        ),
    )


def _row_to_pending(row) -> PendingReflection:
    return PendingReflection(
        pending_id=int(row["pending_id"]),
        stream_id=row["stream_id"],
        session_id=row["session_id"],
        reply_message_id=row["reply_message_id"],
        created_at=_str_to_dt(row["created_at"]) or datetime.now(),
        snapshot_id=row["snapshot_id"],
        source=row["source"],
        response_text=row["response_text"],
        context_json=row["context_json"],
        status=row["status"],
    )


def _row_to_reflection(row) -> SelfReflection:
    raw = row["raw_consistency_score"]
    norm = row["normalized_consistency_score"]
    raw_val: int | None = int(raw) if raw is not None else None
    norm_val: int | None = int(norm) if norm is not None else None
    return SelfReflection(
        reflection_id=int(row["reflection_id"]),
        stream_id=row["stream_id"],
        created_at=_str_to_dt(row["created_at"]) or datetime.now(),
        pending_id=int(row["pending_id"]),
        snapshot_id=row["snapshot_id"],
        reply_type=row["reply_type"],
        evaluated=int(row["evaluated"]),
        consistency_score=int(row["consistency_score"]),
        raw_consistency_score=raw_val,
        normalized_consistency_score=norm_val,
        correction_consumed_at=row["correction_consumed_at"] or "",
        deviating_axis=row["deviating_axis"],
        deviating_direction=row["deviating_direction"],
        reason=row["reason"],
        user_reaction_signal=row["user_reaction_signal"],
        seed_id=row["seed_id"],
    )
