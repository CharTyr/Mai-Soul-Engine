"""种子状态原子守卫测试。

验证 `update_seed_status` 的 expected_status 原子守卫行为：
- pending 种子可正常 approve
- 显式 expected_status="pending" 守卫成功
- 已 expired 种子无法被复活（默认守卫拦截）
- 已 approved 种子无法再次 approve（默认守卫拦截）
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import _import_soul_submodule


# ─── 辅助 ───────────────────────────────────────────────────────────


def _create_pending_seed(im: Any, seed_id: str, stream_id: str = "") -> None:
    im.create_thought_seed(
        seed_id=seed_id,
        stream_id=stream_id,
        seed_type="真诚与虚伪的冲突",
        event="测试事件",
        intensity=80,
        confidence=75,
        evidence_json="[]",
        reasoning="测试原子守卫",
        potential_impact_json="{}",
        context_json="[]",
        status="pending",
    )


# ─── P0: pending → approved 成功 ───────────────────────────────────


def test_approve_pending_seed_default_guard(soul_db: Any) -> None:
    """默认 expected_status="pending" 时，对 pending 种子 approve 成功。"""
    _create_pending_seed(soul_db, "s1")
    assert soul_db.update_seed_status("s1", "approved") is True
    seed = soul_db.get_thought_seed_by_id("s1")
    assert seed is not None
    assert seed.status == "approved"


def test_approve_pending_seed_explicit_guard(soul_db: Any) -> None:
    """显式 expected_status="pending" 守卫通过，approve 成功。"""
    _create_pending_seed(soul_db, "s2")
    assert soul_db.update_seed_status("s2", "approved", expected_status="pending") is True
    seed = soul_db.get_thought_seed_by_id("s2")
    assert seed is not None
    assert seed.status == "approved"


# ─── P0: 已 expired 种子无法再 approve ─────────────────────────────


def test_cannot_approve_expired_seed(soul_db: Any) -> None:
    """对已 expired 种子调用 approve，默认 expected_status="pending" 守卫拦截，返回 False。"""
    _create_pending_seed(soul_db, "s3")
    # 先标 expired
    assert soul_db.update_seed_status("s3", "expired") is True
    # 再尝试 approve——应失败（当前 status=expired ≠ pending）
    assert soul_db.update_seed_status("s3", "approved") is False
    # 记录仍存在，status 仍为 expired
    seed = soul_db.get_thought_seed_by_id("s3")
    assert seed is not None
    assert seed.status == "expired"


# ─── P0: 已 approved 种子无法再次 approve ──────────────────────────


def test_cannot_reapprove_approved_seed(soul_db: Any) -> None:
    """对已 approved 种子再次 approve，默认守卫拦截，返回 False。"""
    _create_pending_seed(soul_db, "s4")
    assert soul_db.update_seed_status("s4", "approved") is True
    # 再次 approve——应失败
    assert soul_db.update_seed_status("s4", "approved") is False
    seed = soul_db.get_thought_seed_by_id("s4")
    assert seed is not None
    assert seed.status == "approved"


# ─── 边界：不存在的 seed_id 返回 False ─────────────────────────────


def test_update_nonexistent_seed_returns_false(soul_db: Any) -> None:
    """不存在的 seed_id 调用 update_seed_status 返回 False。"""
    assert soul_db.update_seed_status("nonexistent", "approved") is False
