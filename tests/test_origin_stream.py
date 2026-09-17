"""origin_stream_id 溯源列 + 内化思想默认 global 归属（0B.1）测试。

验证：
- create_crystallized_trait(stream_id=GLOBAL, origin_stream_id="群A"):
  → DB stream_id="global", origin_stream_id="群A"
- query_active_traits_for_injection("群B") 能查到该 global trait
- save_crystallized_trait 持久化 origin_stream_id
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest


def test_create_with_origin_and_query_global(soul_db: Any) -> None:
    """create_crystallized_trait(stream_id='global', origin_stream_id='群A')
    后，query_active_traits_for_injection('群B') 能查到。
    """
    soul_db.create_crystallized_trait(
        trait_id="origin_g1",
        stream_id="global",
        seed_id="",
        name="测试",
        question="问题",
        thought="观点",
        tags_json="[]",
        confidence=80,
        evidence_json="[]",
        spectrum_impact_json="{}",
        origin_stream_id="群A",
    )

    # 其他群也能查到
    results = soul_db.query_active_traits_for_injection(stream_id="群B")
    ids = {t.trait_id for t in results}
    assert "origin_g1" in ids

    # 验证 origin
    trait = soul_db.get_crystallized_trait_by_id("origin_g1")
    assert trait is not None
    assert trait.stream_id == "global"
    assert trait.origin_stream_id == "群A"


def test_save_persists_origin_stream_id(soul_db: Any) -> None:
    """save_crystallized_trait 更新 origin_stream_id。"""
    # 先创建（不传 origin_stream_id，默认空串）
    soul_db.create_crystallized_trait(
        trait_id="origin_s1",
        stream_id="global",
        seed_id="",
        name="测试",
        question="问题",
        thought="观点",
        tags_json="[]",
        confidence=80,
        evidence_json="[]",
        spectrum_impact_json="{}",
    )
    trait = soul_db.get_crystallized_trait_by_id("origin_s1")
    assert trait is not None
    assert trait.origin_stream_id == ""

    # 更新 origin_stream_id
    trait.origin_stream_id = "群X"
    trait.save()

    # 重新读取验证
    trait2 = soul_db.get_crystallized_trait_by_id("origin_s1")
    assert trait2 is not None
    assert trait2.origin_stream_id == "群X"


def test_origin_stream_id_preserved_through_row_to_trait(soul_db: Any) -> None:
    """直接 SQL 更新 origin_stream_id 后，_row_to_trait 能正确读取。"""
    soul_db.create_crystallized_trait(
        trait_id="origin_r1",
        stream_id="global",
        seed_id="",
        name="测试",
        question="问题",
        thought="观点",
        tags_json="[]",
        confidence=80,
        evidence_json="[]",
        spectrum_impact_json="{}",
    )

    # 直接 SQL 更新
    conn = sqlite3.connect(str(soul_db._db_path))  # type: ignore[attr-defined]
    conn.execute(
        "UPDATE soul_crystallized_traits SET origin_stream_id = ? WHERE trait_id = ?",
        ("群Y", "origin_r1"),
    )
    conn.commit()
    conn.close()

    # 通过代码层读取
    trait = soul_db.get_crystallized_trait_by_id("origin_r1")
    assert trait is not None
    assert trait.origin_stream_id == "群Y"
