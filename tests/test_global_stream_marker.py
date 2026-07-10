"""全局作用域 stream_id 标记测试。

验证：
- `create_crystallized_trait(stream_id="")` 后 stream_id 归一化为 "global"
- `query_active_traits_for_injection(stream_id="群A")` 同时返回群A trait 和 global trait
- `query_active_traits_for_injection(stream_id="")` 只返回 global trait（不含群A）
- 迁移：旧 `stream_id=""` 数据通过 _run_migrations 自动变为 "global"
- `origin_stream_id` 列：create 时可设置，_row_to_trait 可读取
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest


def _create_trait(
    im: Any,
    trait_id: str,
    stream_id: str = "",
    *,
    layer: str = "conduct",
    lifecycle: str = "active",
    origin_stream_id: str = "",
) -> None:
    im.create_crystallized_trait(
        trait_id=trait_id,
        stream_id=stream_id,
        seed_id="",
        name=f"测试_{trait_id}",
        question="问题",
        thought="观点",
        tags_json="[]",
        confidence=80,
        evidence_json="[]",
        spectrum_impact_json="{}",
        ideology_layer=layer,
        lifecycle_state=lifecycle,
        origin_stream_id=origin_stream_id,
    )


# ─── 创建时空串归一化为 global ─────────────────────────────────────


def test_empty_stream_id_normalized_to_global(soul_db: Any) -> None:
    """create_crystallized_trait(stream_id='') 后 DB 中 stream_id 为 'global'。"""
    _create_trait(soul_db, "gt1", stream_id="")
    # 直接查 DB 确认（用 soul_db 的 row_factory，或用索引访问）
    conn = sqlite3.connect(str(soul_db._db_path))  # type: ignore[attr-defined]
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT stream_id FROM soul_crystallized_traits WHERE trait_id = ?", ("gt1",)
    ).fetchone()
    conn.close()
    assert row is not None
    assert row["stream_id"] == "global"


# ─── query_active_traits_for_injection 双匹配 ───────────────────────


def test_active_injection_returns_both_group_and_global(soul_db: Any) -> None:
    """query_active_traits_for_injection(stream_id='群A') 同时返回群A trait 和 global trait。"""
    _create_trait(soul_db, "group_a", stream_id="群A")
    _create_trait(soul_db, "global_t1", stream_id="global")

    results = soul_db.query_active_traits_for_injection(stream_id="群A")
    ids = {t.trait_id for t in results}
    assert "group_a" in ids
    assert "global_t1" in ids


def test_active_injection_empty_stream_only_global(soul_db: Any) -> None:
    """query_active_traits_for_injection(stream_id='') 只返回 global trait。"""
    _create_trait(soul_db, "group_b", stream_id="群B")
    _create_trait(soul_db, "global_t2", stream_id="global")

    results = soul_db.query_active_traits_for_injection(stream_id="")
    ids = {t.trait_id for t in results}
    assert "global_t2" in ids
    assert "group_b" not in ids


# ─── 迁移：旧空串 → global ──────────────────────────────────────────


def test_migration_converts_empty_stream_to_global(soul_db: Any) -> None:
    """旧版 stream_id='' 的 trait 在 init_db 迁移后变为 'global'。

    绕过 create 的归一化，直接用 SQL 插入 `stream_id=''`，然后重新 init_db
    触发 _run_migrations（使用新数据库再 init，迁移包含 UPDATE .. WHERE stream_id=''）。
    """
    # 直接 SQL 插入空串 stream_id
    conn = sqlite3.connect(str(soul_db._db_path))  # type: ignore[attr-defined]
    conn.execute(
        """INSERT INTO soul_crystallized_traits
           (trait_id, stream_id, name, thought, confidence, evidence_json,
            spectrum_impact_json, created_at, enabled, deleted,
            ideology_layer, lifecycle_state)
           VALUES (?, '', ?, ?, ?, ?, ?, ?, 1, 0, 'conduct', 'active')""",
        ("legacy_empty", "旧版空串trait", "旧版观点", 50, "[]", "{}", "2024-01-01T00:00:00"),
    )
    conn.commit()
    conn.close()

    # 回退 PRAGMA user_version 到 0，模拟 legacy v0 库升级场景
    conn_rw = sqlite3.connect(str(soul_db._db_path))  # type: ignore[attr-defined]
    conn_rw.execute("PRAGMA user_version = 0")
    conn_rw.close()

    # 重新 init_db——迁移逻辑会把 stream_id='' 改为 'global'
    soul_db.close_db()
    soul_db.init_db(soul_db._db_path)  # type: ignore[attr-defined]

    # 验证
    conn2 = sqlite3.connect(str(soul_db._db_path))  # type: ignore[attr-defined]
    conn2.row_factory = sqlite3.Row
    row = conn2.execute(
        "SELECT stream_id FROM soul_crystallized_traits WHERE trait_id = ?", ("legacy_empty",)
    ).fetchone()
    conn2.close()
    assert row is not None
    assert row["stream_id"] == "global", "迁移应把 '' 改为 'global'"


# ─── origin_stream_id 溯源列 ──────────────────────────────────────────


def test_origin_stream_id_stored_in_db(soul_db: Any) -> None:
    """create_crystallized_trait 传入 origin_stream_id 后 DB 中持久化。"""
    _create_trait(soul_db, "origin_t1", stream_id="global", origin_stream_id="群A")

    conn = sqlite3.connect(str(soul_db._db_path))  # type: ignore[attr-defined]
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT origin_stream_id FROM soul_crystallized_traits WHERE trait_id = ?", ("origin_t1",)
    ).fetchone()
    conn.close()
    assert row is not None
    assert row["origin_stream_id"] == "群A"


def test_origin_stream_id_default_empty(soul_db: Any) -> None:
    """不传 origin_stream_id 时 DB 中默认空串。"""
    _create_trait(soul_db, "origin_t2", stream_id="global")

    conn = sqlite3.connect(str(soul_db._db_path))  # type: ignore[attr-defined]
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT origin_stream_id FROM soul_crystallized_traits WHERE trait_id = ?", ("origin_t2",)
    ).fetchone()
    conn.close()
    assert row is not None
    assert row["origin_stream_id"] == ""


def test_origin_stream_id_row_to_trait(soul_db: Any) -> None:
    """_row_to_trait 能正确读取 origin_stream_id。"""
    _create_trait(soul_db, "origin_t3", stream_id="global", origin_stream_id="群B")

    trait = soul_db.get_crystallized_trait_by_id("origin_t3")
    assert trait is not None
    assert trait.origin_stream_id == "群B"


def test_origin_stream_id_migration_adds_column(soul_db: Any) -> None:
    """迁移幂等添加 origin_stream_id 列（_run_migrations 中已有）。"""
    # 直接从 DB 读列名验证 origin_stream_id 存在
    conn = sqlite3.connect(str(soul_db._db_path))  # type: ignore[attr-defined]
    cols = {row[1] for row in conn.execute("PRAGMA table_info('soul_crystallized_traits')").fetchall()}
    conn.close()
    assert "origin_stream_id" in cols
