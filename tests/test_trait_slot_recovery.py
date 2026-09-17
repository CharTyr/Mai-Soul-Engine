"""槽位恢复冲突（T11）。

唯一索引只覆盖启用中的 trait：

    CREATE UNIQUE INDEX idx_unique_slot_active ON soul_crystallized_traits(cabinet_slot_no)
    WHERE cabinet_slot_no IS NOT NULL AND enabled = 1 AND deleted = 0

于是「禁用 → 他人占用该槽 → 重新启用」会撞索引：
禁用的 trait 保留旧槽号，占槽时不会被清（只清 enabled=1 的占用者），
恢复启用时两条 enabled trait 争同一槽 → IntegrityError。

约定（方案 §5）：恢复默认**不挤走当前占用者**；槽被占则本次以无槽启用。
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import _import_soul_submodule


def _traits() -> Any:
    return _import_soul_submodule("models.traits")


def _mk(traits: Any, trait_id: str) -> str:
    traits.create_crystallized_trait(
        trait_id=trait_id,
        name=f"观点{trait_id}",
        thought="某个结论",
        stream_id="global",
        seed_id="",
        question="",
        tags_json="[]",
        spectrum_impact_json="{}",
        confidence=80,
        evidence_json="[]",
    )
    return trait_id


def _get(traits: Any, trait_id: str) -> Any:
    return traits.get_crystallized_trait_by_id(trait_id)


# ─── 冲突场景 ───────────────────────────────────────────────────────


def test_reenable_after_slot_taken_does_not_raise(soul_db: Any) -> None:
    """禁用 → 他人占槽 → 重新启用：不抛异常，且不挤走现占用者。"""
    traits = _traits()
    a = _mk(traits, "trait-a")
    b = _mk(traits, "trait-b")

    assert traits.set_trait_slot(a, 3)
    traits.set_trait_lifecycle_state(a, "contradicted", enabled=False)

    # B 占用 3 号槽（A 已禁用，不会被清槽）
    assert traits.set_trait_slot(b, 3)

    # 恢复 A：不得抛 IntegrityError
    assert traits.set_trait_lifecycle_state(a, "active", enabled=True) is True

    assert _get(traits, a).enabled is True
    # B 仍然是 3 号槽的主人
    assert _get(traits, b).cabinet_slot_no == 3
    # A 改为无槽启用
    assert _get(traits, a).cabinet_slot_no is None


def test_reenable_keeps_slot_when_still_free(soul_db: Any) -> None:
    """槽仍然空着 → 恢复时保留原槽。"""
    traits = _traits()
    a = _mk(traits, "trait-a")
    assert traits.set_trait_slot(a, 5)
    traits.set_trait_lifecycle_state(a, "expired", enabled=False)

    assert traits.set_trait_lifecycle_state(a, "active", enabled=True) is True

    assert _get(traits, a).cabinet_slot_no == 5


def test_reenable_without_slot_unaffected(soul_db: Any) -> None:
    """本来就没槽的 trait 恢复不受影响。"""
    traits = _traits()
    a = _mk(traits, "trait-a")
    traits.set_trait_lifecycle_state(a, "contradicted", enabled=False)

    assert traits.set_trait_lifecycle_state(a, "active", enabled=True) is True
    assert _get(traits, a).cabinet_slot_no is None


def test_disable_does_not_clear_slot(soul_db: Any) -> None:
    """禁用保留槽号（历史信息），只是恢复时可能需要让位。"""
    traits = _traits()
    a = _mk(traits, "trait-a")
    assert traits.set_trait_slot(a, 7)
    traits.set_trait_lifecycle_state(a, "contradicted", enabled=False)

    assert _get(traits, a).cabinet_slot_no == 7


def test_repeated_enable_disable_cycles_stay_consistent(soul_db: Any) -> None:
    """反复禁用/启用 + 他人占槽，最终仍只有一条 enabled trait 持有该槽。"""
    traits = _traits()
    a = _mk(traits, "trait-a")
    b = _mk(traits, "trait-b")
    traits.set_trait_slot(a, 2)

    for _ in range(3):
        traits.set_trait_lifecycle_state(a, "contradicted", enabled=False)
        traits.set_trait_slot(b, 2)
        assert traits.set_trait_lifecycle_state(a, "active", enabled=True) is True

    holders = [
        t.trait_id
        for t in traits.query_crystallized_traits(deleted=False, limit=50)
        if t.enabled and t.cabinet_slot_no == 2
    ]
    assert holders == ["trait-b"]


def test_integrity_error_never_propagates(soul_db: Any) -> None:
    """即使底层约束真的被触发，也必须被消化为返回值，而不是抛给命令层。"""
    traits = _traits()
    a = _mk(traits, "trait-a")
    b = _mk(traits, "trait-b")
    traits.set_trait_slot(a, 1)
    traits.set_trait_slot(b, 1)
    traits.set_trait_lifecycle_state(a, "expired", enabled=False)
    traits.set_trait_slot(b, 1)

    # 不应抛异常
    try:
        traits.set_trait_lifecycle_state(a, "active", enabled=True)
    except Exception as exc:  # pragma: no cover - 失败时给出清晰信息
        pytest.fail(f"启用 trait 不应抛异常: {exc!r}")
