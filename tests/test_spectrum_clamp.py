"""测试 spectrum_utils.update_spectrum_value 硬 clamp 行为。

验证从越界反弹改为硬 clamp 后：上界 100、下界 0、正常区间均正确。
"""

from __future__ import annotations

from typing import Any

from .conftest import _import_soul_submodule


def _get_update_spectrum_value() -> Any:
    su = _import_soul_submodule("utils.spectrum_utils")
    return su.update_spectrum_value


# ─── 上界 clamp ──────────────────────────────────────────────────


def test_clamp_upper_bound_no_bounce() -> None:
    """current=100, delta=5 → 100（硬 clamp，不再反弹到 95）。"""
    fn = _get_update_spectrum_value()
    assert fn(100, 5) == 100


def test_clamp_upper_bound_at_limit() -> None:
    """current=100, delta=0 → 100。"""
    fn = _get_update_spectrum_value()
    assert fn(100, 0) == 100


def test_clamp_upper_bound_small_delta() -> None:
    """current=98, delta=5 → 100。"""
    fn = _get_update_spectrum_value()
    assert fn(98, 5) == 100


# ─── 下界 clamp ──────────────────────────────────────────────────


def test_clamp_lower_bound_no_bounce() -> None:
    """current=0, delta=-5 → 0（硬 clamp，不再反弹到 5）。"""
    fn = _get_update_spectrum_value()
    assert fn(0, -5) == 0


def test_clamp_lower_bound_at_limit() -> None:
    """current=0, delta=0 → 0。"""
    fn = _get_update_spectrum_value()
    assert fn(0, 0) == 0


def test_clamp_lower_bound_small_negative() -> None:
    """current=3, delta=-5 → 0。"""
    fn = _get_update_spectrum_value()
    assert fn(3, -5) == 0


# ─── 正常区间 ────────────────────────────────────────────────────


def test_clamp_normal_increase() -> None:
    """current=50, delta=5 → 55。"""
    fn = _get_update_spectrum_value()
    assert fn(50, 5) == 55


def test_clamp_normal_decrease() -> None:
    """current=50, delta=-5 → 45。"""
    fn = _get_update_spectrum_value()
    assert fn(50, -5) == 45


def test_clamp_zero_delta() -> None:
    """current=42, delta=0 → 42。"""
    fn = _get_update_spectrum_value()
    assert fn(42, 0) == 42


def test_clamp_large_delta_within_bounds() -> None:
    """current=50, delta=50 → 100（达上限但没超）。"""
    fn = _get_update_spectrum_value()
    assert fn(50, 50) == 100


def test_clamp_large_negative_delta_within_bounds() -> None:
    """current=50, delta=-50 → 0（达下限但没超）。"""
    fn = _get_update_spectrum_value()
    assert fn(50, -50) == 0


# ─── 边界附近的精确值 ────────────────────────────────────────────


def test_clamp_exactly_one_hundred() -> None:
    """current=95, delta=5 → 100（边界精确）。"""
    fn = _get_update_spectrum_value()
    assert fn(95, 5) == 100


def test_clamp_exactly_zero() -> None:
    """current=5, delta=-5 → 0（边界精确）。"""
    fn = _get_update_spectrum_value()
    assert fn(5, -5) == 0
