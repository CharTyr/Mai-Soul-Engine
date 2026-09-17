"""种子保留策略：清理只能碰**终态**记录（T10）。

旧实现用 ``status != 'pending'`` 选可删除记录，把 ``fermenting``（在途发酵，
已经积累发酵输入）也算了进去。一旦超过 ``reviewed_keep_count``，
正在发酵的种子会被删掉：状态、发酵输入、审计链一起丢失。

终态定义：approved / rejected / expired / internalized。
非终态：pending（待审）、fermenting（发酵中）。
"""

from __future__ import annotations

from typing import Any

from .conftest import _import_soul_submodule


def _seeds() -> Any:
    return _import_soul_submodule("models.seeds")


def _mk(seeds: Any, *, status: str, created_at: str | None = None, suffix: str = "") -> str:
    seed_id = f"seed{suffix or _next_time()}"
    seeds.create_thought_seed(
        seed_id=seed_id,
        stream_id="g1",
        seed_type="opinion",
        event=f"事件{suffix}",
        intensity=5,
        confidence=70,
        evidence_json="[]",
        reasoning="推理",
        potential_impact_json="{}",
    )
    if status != "pending":
        assert seeds.update_seed_status(seed_id, status, expected_status="pending")
    if created_at:
        conn = seeds._get_conn()
        conn.execute(
            "UPDATE soul_thought_seeds SET created_at = ? WHERE seed_id = ?",
            (created_at, seed_id),
        )
        conn.commit()
    return seed_id


_T = 0


def _next_time() -> str:
    """递增时间戳，避免同秒创建的排序歧义。"""
    global _T
    _T += 1
    return f"2026-09-01T00:00:{_T:02d}"


# ─── 计数只算终态 ───────────────────────────────────────────────────


def test_count_reviewed_excludes_fermenting(soul_db: Any) -> None:
    """发酵中种子不计入「已审核」——它不是可清理的历史记录。"""
    seeds = _seeds()
    _mk(seeds, status="approved", created_at=_next_time())
    _mk(seeds, status="fermenting", created_at=_next_time())

    assert seeds.count_reviewed_seeds() == 1


def test_count_reviewed_counts_all_terminal_states(soul_db: Any) -> None:
    """approved / rejected / expired / internalized 都算终态。"""
    seeds = _seeds()
    for status in ("approved", "rejected", "expired", "internalized"):
        _mk(seeds, status=status, created_at=_next_time())

    assert seeds.count_reviewed_seeds() == 4


# ─── 清理不得删除在途种子 ───────────────────────────────────────────


def test_cleanup_never_deletes_fermenting_seed(soul_db: Any) -> None:
    """超过保留上限时，发酵中种子必须保住（哪怕它最旧）。"""
    seeds = _seeds()
    fermenting = _mk(seeds, status="fermenting", created_at="2026-09-01T00:00:00")
    for _ in range(5):
        _mk(seeds, status="approved", created_at=_next_time())

    # 只保留 2 条终态 → 3 条 approved 应被删，fermenting 必须留下
    deleted = seeds.delete_oldest_reviewed_seeds(keep_count=2)

    assert deleted == 3
    assert seeds.get_thought_seed_by_id(fermenting) is not None


def test_cleanup_preserves_fermentation_inputs(soul_db: Any) -> None:
    """发酵中种子的发酵输入不能被连带删除。"""
    seeds = _seeds()
    fermenting = _mk(seeds, status="fermenting", created_at="2026-09-01T00:00:00")
    seeds.add_fermentation_input(fermenting, "g1", "群里又提了一次这个观点", 0.8)
    for _ in range(5):
        _mk(seeds, status="rejected", created_at=_next_time())

    seeds.delete_oldest_reviewed_seeds(keep_count=1)

    assert len(seeds.get_fermentation_inputs(fermenting)) == 1


def test_cleanup_keeps_pending_untouched(soul_db: Any) -> None:
    """pending（待审）不属于历史记录，不得被保留策略删除。"""
    seeds = _seeds()
    pending = _mk(seeds, status="pending", created_at="2026-09-01T00:00:00")
    for _ in range(4):
        _mk(seeds, status="approved", created_at=_next_time())

    seeds.delete_oldest_reviewed_seeds(keep_count=1)

    assert seeds.get_thought_seed_by_id(pending) is not None


def test_cleanup_returns_zero_when_under_cap(soul_db: Any) -> None:
    """未超上限 → 不删任何记录。"""
    seeds = _seeds()
    _mk(seeds, status="approved", created_at=_next_time())
    assert seeds.delete_oldest_reviewed_seeds(keep_count=10) == 0


def test_cleanup_zero_keep_count_is_noop(soul_db: Any) -> None:
    """keep_count<=0 视为不启用清理（不得把全部历史删空）。"""
    seeds = _seeds()
    _mk(seeds, status="approved", created_at=_next_time())
    assert seeds.delete_oldest_reviewed_seeds(keep_count=0) == 0
