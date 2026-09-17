"""trait 质量分 `_trait_quality_score` 单元测试。

验证不同生命周期状态下的加权分数：
- strengthened +0.3
- active +0.0
- weakened -0.3
- revised -0.1
- contradicted -1.0（防御性兜底）
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from .conftest import _import_soul_submodule


# ─── mock trait 工厂 ────────────────────────────────────────────────


def _mock_trait(confidence: int = 80, lifecycle_state: str = "active") -> SimpleNamespace:
    """构造替代 CrystallizedTrait 的轻量 mock，只暴露 `_trait_quality_score` 需要的字段。"""
    return SimpleNamespace(
        confidence=confidence,
        lifecycle_state=lifecycle_state,
        tags_json="[]",
        spectrum_impact_json="{}",
    )


# ─── 各生命周期分数验证 ─────────────────────────────────────────────


def test_quality_strengthened_bonus() -> None:
    """strengthened 生命周期加成 +0.3。"""
    injector = _import_soul_submodule("components.ideology_injector")
    score = injector._trait_quality_score(_mock_trait(confidence=80, lifecycle_state="strengthened"))
    assert score == pytest.approx(0.8 + 0.3)


def test_quality_active_no_bonus() -> None:
    """active 生命周期无加成，score == confidence/100。"""
    injector = _import_soul_submodule("components.ideology_injector")
    score = injector._trait_quality_score(_mock_trait(confidence=80, lifecycle_state="active"))
    assert score == pytest.approx(0.8)


def test_quality_weakened_penalty() -> None:
    """weakened 生命周期衰减 -0.3。"""
    injector = _import_soul_submodule("components.ideology_injector")
    score = injector._trait_quality_score(_mock_trait(confidence=80, lifecycle_state="weakened"))
    assert score == pytest.approx(0.8 - 0.3)


def test_quality_revised_penalty() -> None:
    """revised 生命周期衰减 -0.1。"""
    injector = _import_soul_submodule("components.ideology_injector")
    score = injector._trait_quality_score(_mock_trait(confidence=80, lifecycle_state="revised"))
    assert score == pytest.approx(0.8 - 0.1)


def test_quality_contradicted_heavy_penalty() -> None:
    """contradicted 生命周期衰减 -1.0（防御性兜底）。"""
    injector = _import_soul_submodule("components.ideology_injector")
    score = injector._trait_quality_score(_mock_trait(confidence=80, lifecycle_state="contradicted"))
    assert score == pytest.approx(0.8 - 1.0)


# ─── 边界值 ─────────────────────────────────────────────────────────


def test_quality_clamped_confidence() -> None:
    """confidence 被钳制到 0-1 范围。"""
    injector = _import_soul_submodule("components.ideology_injector")
    # 负值钳制到 0
    low = injector._trait_quality_score(_mock_trait(confidence=-10, lifecycle_state="active"))
    assert low >= 0.0
    # 超过 100 钳制到 1.0
    high = injector._trait_quality_score(_mock_trait(confidence=200, lifecycle_state="active"))
    assert high == pytest.approx(1.0)


def test_quality_unknown_lifecycle_defaults_to_zero() -> None:
    """未知的 lifecycle_state 按 0.0 处理。"""
    injector = _import_soul_submodule("components.ideology_injector")
    score = injector._trait_quality_score(_mock_trait(confidence=50, lifecycle_state="unknown_state"))
    assert score == pytest.approx(0.5)
