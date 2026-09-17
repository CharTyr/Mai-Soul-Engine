"""promote_trait_to_global 提升群锁 trait 为全局测试。

验证：
- 群锁 trait → stream_id 变为 global，origin_stream_id 保留旧值
- 已全局 trait → 幂等，不变
- 不存在的 trait → 返回 False
"""

from __future__ import annotations

from typing import Any

from .conftest import _import_soul_submodule


def _create_group_locked_trait(im: Any, trait_id: str = "locked_t1", stream_id: str = "群A") -> None:
    im.create_crystallized_trait(
        trait_id=trait_id,
        stream_id=stream_id,
        seed_id="",
        name="群锁测试",
        question="测试问题",
        thought="测试观点",
        tags_json="[]",
        confidence=80,
        evidence_json="[]",
        spectrum_impact_json="{}",
        ideology_layer="conduct",
        lifecycle_state="active",
    )


def test_promote_group_locked_to_global(soul_db: Any) -> None:
    """群锁 trait → stream_id=global, origin_stream_id=旧值。"""
    _create_group_locked_trait(soul_db, "p_g1", stream_id="群A")
    ok = soul_db.promote_trait_to_global("p_g1")
    assert ok is True
    t = soul_db.get_crystallized_trait_by_id("p_g1")
    assert t is not None
    assert t.stream_id == "global"
    assert t.origin_stream_id == "群A"


def test_promote_with_existing_origin(soul_db: Any) -> None:
    """已有 origin_stream_id 时 promote 不覆盖它。"""
    soul_db.create_crystallized_trait(
        trait_id="p_g2",
        stream_id="群B",
        seed_id="",
        name="测试",
        question="问题",
        thought="观点",
        tags_json="[]",
        confidence=80,
        evidence_json="[]",
        spectrum_impact_json="{}",
        origin_stream_id="群C",
    )
    ok = soul_db.promote_trait_to_global("p_g2")
    assert ok is True
    t = soul_db.get_crystallized_trait_by_id("p_g2")
    assert t is not None
    assert t.stream_id == "global"
    assert t.origin_stream_id == "群C"


def test_promote_already_global(soul_db: Any) -> None:
    """已全局的 trait → promote 幂等（返回 True，不变）。"""
    _create_group_locked_trait(soul_db, "p_g3", stream_id="global")
    ok = soul_db.promote_trait_to_global("p_g3")
    assert ok is True
    t = soul_db.get_crystallized_trait_by_id("p_g3")
    assert t is not None
    assert t.stream_id == "global"


def test_promote_nonexistent_returns_false(soul_db: Any) -> None:
    """不存在的 trait → False。"""
    ok = soul_db.promote_trait_to_global("p_nonexistent")
    assert ok is False
