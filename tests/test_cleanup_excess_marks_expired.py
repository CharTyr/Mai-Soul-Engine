"""_cleanup_excess_seeds 标记过期而非删除测试。

验证：pending 种子数超出 max_seeds 时，最旧的种子被标记为 expired
而非物理删除，保留审计记录。
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from .conftest import _import_soul_submodule


def _create_pending_seed(im: Any, seed_id: str, event: str, stream_id: str = "群A") -> None:
    im.create_thought_seed(
        seed_id=seed_id,
        stream_id=stream_id,
        seed_type="真诚与虚伪的冲突",
        event=event,
        intensity=80,
        confidence=75,
        evidence_json="[]",
        reasoning="测试清理",
        potential_impact_json="{}",
        context_json="[]",
        status="pending",
    )


def test_cleanup_excess_marks_expired_not_deleted(soul_db: Any) -> None:
    """超出 max_seeds 时，_cleanup_excess_seeds 将最旧种子标 expired 而非物理删除。

    清理逻辑：保留 max_seeds-1 个最新 pending 种子，其余标记 expired。
    所以 4 个种子 (max=3) → 保持 2 个最新 pending，标记 2 个最旧为 expired。
    """
    sm = _import_soul_submodule("thought.seed_manager")

    # 构造 manager，max_seeds=3
    manager = sm.ThoughtSeedManager({"max_seeds": 3, "seed_dedup_threshold": 0.0})

    # 建 4 个 pending 种子（超 1 个）
    _create_pending_seed(soul_db, "s_oldest", "最旧种子")
    _create_pending_seed(soul_db, "s_mid1", "中间种子1")
    _create_pending_seed(soul_db, "s_mid2", "中间种子2")
    _create_pending_seed(soul_db, "s_newest", "最新种子")

    # 把种子 created_at 调成不同时间，确保排序顺序可控
    conn = sqlite3.connect(str(soul_db._db_path))  # type: ignore[attr-defined]
    times = [
        ("s_oldest", "2024-01-01T00:00:00"),
        ("s_mid1", "2024-02-01T00:00:00"),
        ("s_mid2", "2024-03-01T00:00:00"),
        ("s_newest", "2024-04-01T00:00:00"),
    ]
    for sid, t in times:
        conn.execute("UPDATE soul_thought_seeds SET created_at = ? WHERE seed_id = ?", (t, sid))
    conn.commit()
    conn.close()

    # 执行清理
    import asyncio
    asyncio.run(manager._cleanup_excess_seeds())

    # 验证：所有 4 个种子仍可查到（未物理删除）
    for sid in ("s_oldest", "s_mid1", "s_mid2", "s_newest"):
        s = soul_db.get_thought_seed_by_id(sid)
        assert s is not None, f"{sid} 不应被物理删除"

    # 最旧的 2 个种子（s_oldest, s_mid1）被标 expired
    seed_oldest = soul_db.get_thought_seed_by_id("s_oldest")
    assert seed_oldest is not None
    assert seed_oldest.status == "expired"

    seed_mid1 = soul_db.get_thought_seed_by_id("s_mid1")
    assert seed_mid1 is not None
    assert seed_mid1.status == "expired"

    # 最新的 2 个种子仍为 pending
    seed_mid2 = soul_db.get_thought_seed_by_id("s_mid2")
    assert seed_mid2 is not None
    assert seed_mid2.status == "pending"

    seed_newest = soul_db.get_thought_seed_by_id("s_newest")
    assert seed_newest is not None
    assert seed_newest.status == "pending"


def test_cleanup_excess_not_triggered_when_under_limit(soul_db: Any) -> None:
    """种子数未超 max_seeds 时，不应执行清理。"""
    sm = _import_soul_submodule("thought.seed_manager")

    manager = sm.ThoughtSeedManager({"max_seeds": 5, "seed_dedup_threshold": 0.0})

    # 建 3 个种子（< 5）
    for i in range(3):
        _create_pending_seed(soul_db, f"s_under_{i}", f"种子{i}")

    import asyncio
    asyncio.run(manager._cleanup_excess_seeds())

    # 全部应为 pending
    for i in range(3):
        s = soul_db.get_thought_seed_by_id(f"s_under_{i}")
        assert s is not None
        assert s.status == "pending"
