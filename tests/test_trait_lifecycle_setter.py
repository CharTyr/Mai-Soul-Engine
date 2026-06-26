"""trait 生命周期状态设置器测试。

验证 `set_trait_lifecycle_state`：
- 同时改 lifecycle 和 enabled（contradicted 场景）
- 只改 lifecycle，enabled 不变（weakened 场景）
- 不存在的 trait_id 返回 False
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import _import_soul_submodule


def _create_trait(im: Any, trait_id: str, stream_id: str = "") -> None:
    im.create_crystallized_trait(
        trait_id=trait_id,
        stream_id=stream_id,
        seed_id="",
        name="测试特质",
        question="测试问题",
        thought="测试观点",
        tags_json="[]",
        confidence=80,
        evidence_json="[]",
        spectrum_impact_json="{}",
        ideology_layer="conduct",
        lifecycle_state="active",
    )


# ─── 同时改 lifecycle + enabled ─────────────────────────────────────


def test_set_contradicted_disables_trait(soul_db: Any) -> None:
    """set_trait_lifecycle_state(id, 'contradicted', enabled=False) 同时改 lifecycle 和 enabled。"""
    _create_trait(soul_db, "t1")
    result = soul_db.set_trait_lifecycle_state("t1", "contradicted", enabled=False)
    assert result is True
    trait = soul_db.get_crystallized_trait_by_id("t1")
    assert trait is not None
    assert trait.lifecycle_state == "contradicted"
    assert trait.enabled is False


# ─── 只改 lifecycle，enabled 不变 ───────────────────────────────────


def test_set_weakened_keeps_enabled(soul_db: Any) -> None:
    """set_trait_lifecycle_state(id, 'weakened') 只改 lifecycle，enabled 保持不变(1)。"""
    _create_trait(soul_db, "t2")
    result = soul_db.set_trait_lifecycle_state("t2", "weakened")
    assert result is True
    trait = soul_db.get_crystallized_trait_by_id("t2")
    assert trait is not None
    assert trait.lifecycle_state == "weakened"
    assert trait.enabled is True


def test_set_revised_keeps_enabled(soul_db: Any) -> None:
    """set_trait_lifecycle_state(id, 'revised') 只改 lifecycle，enabled 保持不变。"""
    _create_trait(soul_db, "t_revised")
    assert soul_db.set_trait_lifecycle_state("t_revised", "revised") is True
    trait = soul_db.get_crystallized_trait_by_id("t_revised")
    assert trait is not None
    assert trait.lifecycle_state == "revised"
    assert trait.enabled is True


# ─── 不存在的 trait_id ──────────────────────────────────────────────


def test_set_lifecycle_nonexistent_returns_false(soul_db: Any) -> None:
    """不存在的 trait_id 调用 set_trait_lifecycle_state 返回 False。"""
    assert soul_db.set_trait_lifecycle_state("nonexistent", "expired") is False
    assert soul_db.set_trait_lifecycle_state("nonexistent", "contradicted", enabled=False) is False
