"""Schema 迁移框架测试。

验证：
1. 新库 init 后 CURRENT_SCHEMA_VERSION 正确
2. soul_schema_migrations 有对应 success 记录
3. 二次 init_db 幂等（version 不变、不炸）
4. cabinet_slot_no 列存在
5. 唯一部分索引：同 slot + enabled=1 + deleted=0 → UNIQUE 冲突
6. 不同 slot 或 disabled 可共存
7. 迁移失败不推进 user_version
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import _import_soul_submodule


def _get_conn_mod() -> Any:
    return _import_soul_submodule("models._conn")


def _get_im() -> Any:
    return _import_soul_submodule("models.ideology_model")


# ─── 新库基础验证 ────────────────────────────────────────────────


def test_fresh_init_schema_version(soul_db: Any) -> None:
    """新库 init_db 后 PRAGMA user_version == CURRENT_SCHEMA_VERSION。"""
    conn_mod = _get_conn_mod()
    assert conn_mod.CURRENT_SCHEMA_VERSION == 2
    assert conn_mod._get_schema_version() == 2


def test_fresh_init_migration_records(soul_db: Any) -> None:
    """新库有 soul_schema_migrations 表且两条均为 success。"""
    conn_mod = _get_conn_mod()
    conn = conn_mod._get_conn()
    rows = conn.execute(
        "SELECT version, name, status FROM soul_schema_migrations ORDER BY version"
    ).fetchall()
    assert len(rows) >= 2, "应有至少两条迁移记录"
    assert rows[0]["version"] == 1
    assert rows[0]["name"] == "v1_legacy_bootstrap"
    assert rows[0]["status"] == "success"
    assert rows[1]["version"] == 2
    assert rows[1]["name"] == "v2_cabinet_slot_no"
    assert rows[1]["status"] == "success"


# ─── 二次 init_db 幂等 ───────────────────────────────────────────


def test_reinit_is_idempotent(soul_db: Any) -> None:
    """二次 init_db 不炸、不推进版本。"""
    conn_mod = _get_conn_mod()
    # 首次 init 已在 fixture 完成
    version_before = conn_mod._get_schema_version()
    # 模拟重新初始化（同一连接）
    conn_mod._create_tables()
    conn_mod._run_migrations()
    conn_mod._create_indexes()
    version_after = conn_mod._get_schema_version()
    assert version_after == version_before, "二次 init 不应推进版本"


def test_reinit_migration_count_stable(soul_db: Any) -> None:
    """二次 init 后 soul_schema_migrations 记录数不变。"""
    conn_mod = _get_conn_mod()
    conn = conn_mod._get_conn()
    before = conn.execute(
        "SELECT COUNT(*) AS cnt FROM soul_schema_migrations"
    ).fetchone()["cnt"]
    conn_mod._create_tables()
    conn_mod._run_migrations()
    conn_mod._create_indexes()
    after = conn.execute(
        "SELECT COUNT(*) AS cnt FROM soul_schema_migrations"
    ).fetchone()["cnt"]
    assert after == before


# ─── cabinet_slot_no 列 ──────────────────────────────────────────


def test_cabinet_slot_no_column_exists(soul_db: Any) -> None:
    """soul_crystallized_traits 表有 cabinet_slot_no 列。"""
    conn_mod = _get_conn_mod()
    assert conn_mod._has_column("soul_crystallized_traits", "cabinet_slot_no")


def test_cabinet_slot_no_default_null(soul_db: Any) -> None:
    """新 trait 不传 cabinet_slot_no 时默认 NULL。"""
    im = _get_im()
    im.create_crystallized_trait(
        trait_id="t_null_slot",
        stream_id="global",
        seed_id="",
        name="无槽位",
        question="",
        thought="测试",
        tags_json="[]",
        confidence=50,
        evidence_json="[]",
        spectrum_impact_json="{}",
    )
    trait = im.get_crystallized_trait_by_id("t_null_slot")
    assert trait is not None
    assert trait.cabinet_slot_no is None


def test_cabinet_slot_no_set_and_read(soul_db: Any) -> None:
    """传 cabinet_slot_no=5 可以写回并读取。"""
    im = _get_im()
    im.create_crystallized_trait(
        trait_id="t_slot_5",
        stream_id="global",
        seed_id="",
        name="槽位5",
        question="",
        thought="测试",
        tags_json="[]",
        confidence=50,
        evidence_json="[]",
        spectrum_impact_json="{}",
        cabinet_slot_no=5,
    )
    trait = im.get_crystallized_trait_by_id("t_slot_5")
    assert trait is not None
    assert trait.cabinet_slot_no == 5


# ─── 唯一部分索引（UNIQUE 冲突 vs 共存） ─────────────────────────


def test_same_slot_active_conflict(soul_db: Any) -> None:
    """两条 enabled=1 + deleted=0 同 cabinet_slot_no → IntegrityError。"""
    im = _get_im()
    im.create_crystallized_trait(
        trait_id="t_slot_1_a",
        stream_id="global",
        seed_id="",
        name="槽位1-A",
        question="",
        thought="测试A",
        tags_json="[]",
        confidence=50,
        evidence_json="[]",
        spectrum_impact_json="{}",
        cabinet_slot_no=10,
    )
    with pytest.raises(Exception) as excinfo:
        im.create_crystallized_trait(
            trait_id="t_slot_1_b",
            stream_id="global",
            seed_id="",
            name="槽位1-B",
            question="",
            thought="测试B",
            tags_json="[]",
            confidence=50,
            evidence_json="[]",
            spectrum_impact_json="{}",
            cabinet_slot_no=10,
        )
    # SQLite 唯一约束冲突错误码 2067
    assert "UNIQUE" in str(excinfo.value) or "2067" in str(excinfo.value)


def test_different_slot_no_conflict(soul_db: Any) -> None:
    """不同 cabinet_slot_no 可共存。"""
    im = _get_im()
    im.create_crystallized_trait(
        trait_id="t_slot_2_a",
        stream_id="global",
        seed_id="",
        name="槽位2",
        question="",
        thought="测试2",
        tags_json="[]",
        confidence=50,
        evidence_json="[]",
        spectrum_impact_json="{}",
        cabinet_slot_no=20,
    )
    im.create_crystallized_trait(
        trait_id="t_slot_2_b",
        stream_id="global",
        seed_id="",
        name="槽位3",
        question="",
        thought="测试3",
        tags_json="[]",
        confidence=50,
        evidence_json="[]",
        spectrum_impact_json="{}",
        cabinet_slot_no=30,
    )
    # 不应抛异常
    t1 = im.get_crystallized_trait_by_id("t_slot_2_a")
    t2 = im.get_crystallized_trait_by_id("t_slot_2_b")
    assert t1 is not None and t1.cabinet_slot_no == 20
    assert t2 is not None and t2.cabinet_slot_no == 30


def test_disabled_same_slot_allowed(soul_db: Any) -> None:
    """同 slot + enabled=0 可共存（索引只约束 enabled=1）。"""
    im = _get_im()
    im.create_crystallized_trait(
        trait_id="t_ds_1",
        stream_id="global",
        seed_id="",
        name="禁用同槽1",
        question="",
        thought="测试",
        tags_json="[]",
        confidence=50,
        evidence_json="[]",
        spectrum_impact_json="{}",
        cabinet_slot_no=40,
        enabled=False,
    )
    im.create_crystallized_trait(
        trait_id="t_ds_2",
        stream_id="global",
        seed_id="",
        name="禁用同槽2",
        question="",
        thought="测试",
        tags_json="[]",
        confidence=50,
        evidence_json="[]",
        spectrum_impact_json="{}",
        cabinet_slot_no=40,
        enabled=False,
    )
    # 不应抛异常


def test_deleted_same_slot_allowed(soul_db: Any) -> None:
    """同 slot + deleted=1 可共存（索引只约束 deleted=0）。"""
    im = _get_im()
    im.create_crystallized_trait(
        trait_id="t_dl_1",
        stream_id="global",
        seed_id="",
        name="删除同槽1",
        question="",
        thought="测试",
        tags_json="[]",
        confidence=50,
        evidence_json="[]",
        spectrum_impact_json="{}",
        cabinet_slot_no=50,
        deleted=True,
    )
    im.create_crystallized_trait(
        trait_id="t_dl_2",
        stream_id="global",
        seed_id="",
        name="删除同槽2",
        question="",
        thought="测试",
        tags_json="[]",
        confidence=50,
        evidence_json="[]",
        spectrum_impact_json="{}",
        cabinet_slot_no=50,
        deleted=True,
    )
    # 不应抛异常


# ─── NULL slot 不参与索引 ─────────────────────────────────────


def test_null_slot_no_unique_conflict(soul_db: Any) -> None:
    """cabinet_slot_no=NULL 的多条 trait 可共存（索引不包含 NULL）。"""
    im = _get_im()
    im.create_crystallized_trait(
        trait_id="t_n1",
        stream_id="global",
        seed_id="",
        name="空槽1",
        question="",
        thought="测试",
        tags_json="[]",
        confidence=50,
        evidence_json="[]",
        spectrum_impact_json="{}",
    )
    im.create_crystallized_trait(
        trait_id="t_n2",
        stream_id="global",
        seed_id="",
        name="空槽2",
        question="",
        thought="测试",
        tags_json="[]",
        confidence=50,
        evidence_json="[]",
        spectrum_impact_json="{}",
    )
    # 不应抛异常


# ─── 迁移失败不推进版本（模拟 v2 失败场景） ──────────────────────


def test_migration_failure_does_not_advance_version(soul_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """模拟 v2 迁移途中抛异常，PRAGMA user_version 停留在 1。"""
    conn_mod = _get_conn_mod()

    # 模拟 v2 迁移第一步成功（加列）但第二步（建索引）失败
    original_run_v2 = conn_mod._run_v2_migration
    call_count = 0

    def _faulty_v2():
        nonlocal call_count
        call_count += 1
        # 先正常执行到加列
        conn = conn_mod._get_conn()
        if not conn_mod._has_column("soul_crystallized_traits", "cabinet_slot_no"):
            conn_mod._add_column("soul_crystallized_traits", "cabinet_slot_no", "INTEGER DEFAULT NULL")
        conn.commit()
        # 然后抛异常模拟失败
        raise RuntimeError("模拟迁移失败")

    monkeypatch.setattr(conn_mod, "_run_v2_migration", _faulty_v2)

    # 回退到 v1 再跑迁移
    conn_mod._set_schema_version(1)

    with pytest.raises(RuntimeError, match="模拟迁移失败"):
        conn_mod._run_migrations()

    # user_version 应停留在 1（不前进到 2）
    assert conn_mod._get_schema_version() == 1

    # soul_schema_migrations 应有 v2 failed 记录
    conn = conn_mod._get_conn()
    row = conn.execute(
        "SELECT status, error FROM soul_schema_migrations WHERE version = 2"
    ).fetchone()
    assert row is not None
    assert row["status"] == "failed"
    assert "模拟迁移失败" in row["error"]

    # 恢复后再跑应正常
    monkeypatch.undo()
    conn_mod._run_migrations()
    assert conn_mod._get_schema_version() == 2
    row = conn.execute(
        "SELECT status FROM soul_schema_migrations WHERE version = 2 AND name = 'v2_cabinet_slot_no'"
    ).fetchone()
    assert row is not None
    assert row["status"] == "success"
