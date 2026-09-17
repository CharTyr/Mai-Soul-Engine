"""P1.1: 自评数据模型 CRUD 测试。

验证三张表（injection_snapshots / pending_reflections / self_reflections）的
建表、写入、配对查询、TTL 清理、计数。覆盖 oracle 补的"pending 溢出/过期清理"项。

从宿主仓根运行：``uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/test_self_reflection_model.py -q``
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .conftest import _import_soul_submodule


# ─── injection_snapshots ──────────────────────────────────────────


def test_create_and_get_snapshot(soul_db: Any) -> None:
    sid = soul_db.create_injection_snapshot(
        stream_id="global",
        session_id="sess-1",
        trait_ids_json='["t1","t2"]',
        spectrum_json='{"sincerity":60}',
        mood_json='{"valence":5}',
        selection_mode="tag_hit",
        context_fingerprint="abc123",
    )
    assert sid
    snap = soul_db.get_latest_snapshot_for_session("sess-1")
    assert snap is not None
    assert snap.snapshot_id == sid
    assert snap.selection_mode == "tag_hit"
    assert snap.trait_ids_json == '["t1","t2"]'


def test_get_latest_snapshot_orders_by_created(soul_db: Any) -> None:
    """同 session 多次注入，取最近一条（1:N 配对的 1 端）。"""
    soul_db.create_injection_snapshot("g", "sess-2", "[]", "{}", "{}", "spectrum_only")
    # 手动改 created_at 让第二条更晚
    soul_db.create_injection_snapshot("g", "sess-2", '["t9"]', "{}", "{}", "tag_hit")
    snap = soul_db.get_latest_snapshot_for_session("sess-2")
    assert snap is not None
    assert snap.selection_mode == "tag_hit"


def test_get_snapshot_missing_session(soul_db: Any) -> None:
    assert soul_db.get_latest_snapshot_for_session("nope") is None


# ─── pending_reflections ──────────────────────────────────────────


def test_create_and_list_pending(soul_db: Any) -> None:
    pid = soul_db.create_pending_reflection(
        stream_id="g",
        session_id="sess-1",
        reply_message_id="r1",
        snapshot_id="snap-1",
        source="planner",
        response_text="你好",
        context_json='["用户: 在吗"]',
    )
    assert pid > 0
    pendings = soul_db.list_pending_reflections(limit=10)
    assert len(pendings) == 1
    assert pendings[0].pending_id == pid
    assert pendings[0].status == "pending"
    assert pendings[0].response_text == "你好"


def test_update_pending_status(soul_db: Any) -> None:
    pid = soul_db.create_pending_reflection("g", "s", "r", "snap", "replyer", "hi")
    assert soul_db.update_pending_status(pid, "done") is True
    pendings = soul_db.list_pending_reflections(limit=10)
    assert len(pendings) == 0  # done 不在 pending 列表


def test_list_pending_excludes_expired_by_age(soul_db: Any) -> None:
    """超龄 pending 不被 list_pending_reflections 取出（虽未标 expired）。"""
    pid = soul_db.create_pending_reflection("g", "s", "r", "snap", "planner", "old")
    # 手动把 created_at 改成 100 小时前
    conn_mod = _import_soul_submodule("models._conn")

    old = conn_mod._dt_to_str(datetime.now() - timedelta(hours=100))
    conn_mod._get_conn().execute(
        "UPDATE soul_pending_reflections SET created_at = ? WHERE pending_id = ?",
        (old, pid),
    )
    conn_mod._get_conn().commit()
    assert len(soul_db.list_pending_reflections(limit=10, max_age_hours=48)) == 0


def test_cleanup_expired_marks_old_pending(soul_db: Any) -> None:
    """cleanup_expired_pending 把超龄 pending 标 expired。"""
    pid = soul_db.create_pending_reflection("g", "s", "r", "snap", "planner", "old")
    conn_mod = _import_soul_submodule("models._conn")

    old = conn_mod._dt_to_str(datetime.now() - timedelta(hours=100))
    conn_mod._get_conn().execute(
        "UPDATE soul_pending_reflections SET created_at = ? WHERE pending_id = ?",
        (old, pid),
    )
    conn_mod._get_conn().commit()
    cleaned = soul_db.cleanup_expired_pending(max_age_hours=48, max_rows=5000)
    assert cleaned >= 1
    counts = soul_db.count_pending_reflections()
    assert counts.get("expired", 0) >= 1


def test_cleanup_orphan_snapshots(soul_db: Any) -> None:
    """超龄且未被引用的 snapshot 被清理（ora-2）。"""
    conn_mod = _import_soul_submodule("models._conn")

    soul_db.create_injection_snapshot("g", "s1", "[]", "{}", "{}", "spectrum_only")
    soul_db.create_injection_snapshot("g", "s2", "[]", "{}", "{}", "tag_hit")
    old = conn_mod._dt_to_str(datetime.now() - timedelta(hours=100))
    conn_mod._get_conn().execute(
        "UPDATE soul_injection_snapshots SET created_at = ? WHERE session_id = 's1'",
        (old,),
    )
    conn_mod._get_conn().commit()
    n = soul_db.cleanup_orphan_snapshots(max_age_hours=48)
    assert n >= 1
    assert soul_db.get_latest_snapshot_for_session("s1") is None
    assert soul_db.get_latest_snapshot_for_session("s2") is not None


def test_cleanup_pending_cap_deletes_oldest(soul_db: Any) -> None:
    """pending 总数超上限时物理删最旧非 pending 记录（D-DATA-3：保留 pending 业务记录）。"""
    conn_mod = _import_soul_submodule("models._conn")

    # 写 6 条，前 3 条标记为 done，后 3 条保持 pending
    for i in range(6):
        pid = soul_db.create_pending_reflection("g", "s", f"r{i}", "snap", "planner", f"m{i}")
        ts = conn_mod._dt_to_str(datetime.now() + timedelta(seconds=i))
        conn_mod._get_conn().execute(
            "UPDATE soul_pending_reflections SET created_at = ? WHERE pending_id = ?",
            (ts, pid),
        )
        if i < 3:
            soul_db.update_pending_status(pid, "done")
    conn_mod._get_conn().commit()
    # 上限设 3，当前 6 条（3 done + 3 pending），应删最旧 3 条 done
    soul_db.cleanup_expired_pending(max_age_hours=9999, max_rows=3)
    total = conn_mod._get_conn().execute("SELECT COUNT(*) FROM soul_pending_reflections").fetchone()[0]
    assert int(total) == 3
    # 验证剩下的是 3 条 pending
    remaining = conn_mod._get_conn().execute(
        "SELECT status FROM soul_pending_reflections ORDER BY created_at ASC"
    ).fetchall()
    assert all(r["status"] == "pending" for r in remaining)


# ─── self_reflections ─────────────────────────────────────────────


def test_create_and_list_reflections(soul_db: Any) -> None:
    rid = soul_db.create_self_reflection(
        stream_id="g",
        pending_id=1,
        snapshot_id="snap",
        reply_type="substantive",
        evaluated=1,
        consistency_score=72,
        deviating_axis="directness",
        deviating_direction="high",
        reason="过于直率",
    )
    assert rid > 0
    refs = soul_db.list_recent_reflections("g", limit=10)
    assert len(refs) == 1
    assert refs[0].consistency_score == 72
    assert refs[0].deviating_axis == "directness"


def test_list_reflections_includes_global_scope(soul_db: Any) -> None:
    """按群查询时也包含全局作用域记录。"""
    soul_db.create_self_reflection("g", 1, "snap", "substantive", 1, 80, "sincerity", "low", "x")
    soul_db.create_self_reflection("global", 2, "snap", "reactive", 1, 70, "", "", "y")
    refs = soul_db.list_recent_reflections("g", limit=10)
    assert len(refs) == 2  # 群 g + 全局 global


def test_count_self_reflections(soul_db: Any) -> None:
    soul_db.create_self_reflection("g", 1, "snap", "substantive", 1, 80, "directness", "high", "x")
    soul_db.create_self_reflection("g", 2, "snap", "social_glue", 0, 0, "", "", "skip")
    soul_db.create_self_reflection("g", 3, "snap", "substantive", 1, 60, "sincerity", "low", "y")
    counts = soul_db.count_self_reflections()
    assert counts["total"] == 3
    assert counts["evaluated"] == 2
    assert counts["skipped"] == 1
    assert counts["by_axis"].get("directness") == 1
    assert counts["by_axis"].get("sincerity") == 1


# ─── 新列 CRUD（Phase 0A R2） ──────────────────────────────────────


def test_create_reflection_with_raw_normalized_scores(soul_db: Any) -> None:
    """创建时传入 raw/normalized 分数，验证持久化。"""
    rid = soul_db.create_self_reflection(
        stream_id="g",
        pending_id=1,
        snapshot_id="snap",
        reply_type="substantive",
        evaluated=1,
        consistency_score=80,  # 主分=raw
        deviating_axis="sincerity",
        deviating_direction="low",
        reason="test",
        seed_id="",
        raw_consistency_score=80,
        normalized_consistency_score=65,
    )
    assert rid > 0
    refs = soul_db.list_recent_reflections("g", limit=10)
    ref = next(r for r in refs if r.reflection_id == rid)
    assert ref.consistency_score == 80  # 主分兼容
    assert ref.raw_consistency_score == 80
    assert ref.normalized_consistency_score == 65
    assert ref.correction_consumed_at == ""  # 默认空串


def test_create_reflection_without_raw_normalized(soul_db: Any) -> None:
    """不传 raw/normalized 时，raw 从 consistency_score 继承，normalized 从 raw 继承。"""
    rid = soul_db.create_self_reflection(
        stream_id="g",
        pending_id=1,
        snapshot_id="snap",
        reply_type="substantive",
        evaluated=1,
        consistency_score=75,
    )
    refs = soul_db.list_recent_reflections("g", limit=10)
    ref = next(r for r in refs if r.reflection_id == rid)
    # consistency_score=75 → raw=75, normalized=75
    assert ref.consistency_score == 75
    assert ref.raw_consistency_score == 75
    assert ref.normalized_consistency_score == 75


def test_row_to_reflection_fallback_to_consistency_score(soul_db: Any) -> None:
    """新列为 NULL 时，_row_to_reflection 的 raw/normalized 为 None（不是 0），
    correction_consumed_at 为空串。"""
    conn = _import_soul_submodule("models._conn")._get_conn()
    conn.execute(
        """INSERT INTO soul_self_reflections
           (stream_id, created_at, pending_id, snapshot_id, reply_type, evaluated,
            consistency_score, deviating_axis, deviating_direction, reason, seed_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("g", "2024-01-01T00:00:00", 1, "snap", "substantive", 1, 72, "directness", "high", "x", ""),
    )
    conn.commit()
    refs = soul_db.list_recent_reflections("g", limit=10)
    ref = next(r for r in refs if r.pending_id == 1 and r.created_at is not None)
    assert ref.consistency_score == 72
    assert ref.raw_consistency_score is None  # NULL 列 → None
    assert ref.normalized_consistency_score is None
    assert ref.correction_consumed_at == ""


# ─── list_unconsumed_reflections_for_correction ──────────────────


def test_list_unconsumed_returns_only_substantive_evaluated(soul_db: Any) -> None:
    """仅返回 evaluated=1 + reply_type='substantive' 的记录。"""
    sr = _import_soul_submodule("models.self_reflection")

    soul_db.create_self_reflection("g", 1, "", "substantive", 1, 50, "directness", "low", "x")
    soul_db.create_self_reflection("g", 2, "", "reactive", 1, 80, "", "", "")
    soul_db.create_self_reflection("g", 3, "", "social_glue", 0, 0, "", "", "skip")
    soul_db.create_self_reflection("g", 4, "", "substantive", 1, 30, "sincerity", "high", "y")
    results = sr.list_unconsumed_reflections_for_correction(limit=10)
    assert len(results) == 2
    assert all(r.reply_type == "substantive" and r.evaluated == 1 for r in results)


def test_list_unconsumed_excludes_consumed(soul_db: Any) -> None:
    """已消费（correction_consumed_at 非空）的记录不被返回。"""
    sr = _import_soul_submodule("models.self_reflection")

    r1 = soul_db.create_self_reflection("g", 1, "", "substantive", 1, 50, "directness", "low", "x")
    r2 = soul_db.create_self_reflection("g", 2, "", "substantive", 1, 40, "sincerity", "high", "y")
    # 标记 r1 已消费
    sr.mark_reflections_correction_consumed([r1], "2024-01-01T00:00:00")
    results = sr.list_unconsumed_reflections_for_correction(limit=10)
    assert len(results) == 1
    assert results[0].reflection_id == r2


def test_list_unconsumed_cross_stream(soul_db: Any) -> None:
    """不限制 stream_id，不同群的记录都能被消费。"""
    sr = _import_soul_submodule("models.self_reflection")

    soul_db.create_self_reflection("group_a", 1, "", "substantive", 1, 50, "directness", "low", "x")
    soul_db.create_self_reflection("group_b", 2, "", "substantive", 1, 40, "sincerity", "high", "y")
    results = sr.list_unconsumed_reflections_for_correction(limit=10)
    assert len(results) == 2


# ─── mark_reflections_correction_consumed ──────────────────────────


def test_mark_reflections_correction_consumed(soul_db: Any) -> None:
    """批量标记已消费。"""
    sr = _import_soul_submodule("models.self_reflection")

    r1 = soul_db.create_self_reflection("g", 1, "", "substantive", 1, 50, "directness", "low", "x")
    r2 = soul_db.create_self_reflection("g", 2, "", "substantive", 1, 40, "sincerity", "high", "y")
    affected = sr.mark_reflections_correction_consumed([r1, r2], "2024-06-01T12:00:00")
    assert affected == 2
    assert len(sr.list_unconsumed_reflections_for_correction(limit=10)) == 0


def test_mark_reflections_correction_consumed_empty_list(soul_db: Any) -> None:
    """空列表不报错，返回 0。"""
    sr = _import_soul_submodule("models.self_reflection")
    assert sr.mark_reflections_correction_consumed([], "2024-01-01T00:00:00") == 0
