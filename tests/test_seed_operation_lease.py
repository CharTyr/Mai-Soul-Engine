"""种子内化的幂等与单赢家认领（T09）。

原流程：先跑内化（LLM + 光谱/trait 写入），成功后才标种子终态，且**忽略返回值**。
两个后果：

- 并发批准同一颗种子 → 两次内化都跑完 → 光谱影响施加两次
- 内化成功后进程中断 / 标状态失败 → 种子仍是 pending → 管理员再批一次 → 再施加一次

修复方向：LLM 调用**之前**用带租约的操作记账认领种子（CAS 单赢家）；
成功时在**同一事务**里同提「操作结果 + 种子终态」；失败释放租约允许重试。
LLM 调用本身不能放进事务，所以正确性靠「单赢家租约 + 幂等终结」而不是长事务。
"""

from __future__ import annotations

from typing import Any

from .conftest import _import_soul_submodule


def _ops() -> Any:
    return _import_soul_submodule("models.operations")


def _seeds() -> Any:
    return _import_soul_submodule("models.seeds")


def _mk_seed(seeds: Any, seed_id: str) -> str:
    seeds.create_thought_seed(
        seed_id=seed_id,
        stream_id="g1",
        seed_type="opinion",
        event="事件",
        intensity=5,
        confidence=70,
        evidence_json="[]",
        reasoning="推理",
        potential_impact_json="{}",
    )
    return seed_id


# ─── 单赢家认领 ─────────────────────────────────────────────────────


def test_claim_returns_operation_id(soul_db: Any) -> None:
    """首次认领成功并返回 operation_id。"""
    ops = _ops()
    op_id = ops.claim_seed_operation("seed-1", lease_seconds=300)
    assert op_id


def test_second_claim_while_running_is_rejected(soul_db: Any) -> None:
    """同一颗种子在租约内只允许一个内化在跑 —— 并发批准不得双开。"""
    ops = _ops()
    first = ops.claim_seed_operation("seed-1", lease_seconds=300)
    second = ops.claim_seed_operation("seed-1", lease_seconds=300)

    assert first
    assert second is None


def test_concurrent_claims_have_exactly_one_winner(soul_db: Any) -> None:
    """并发认领只有一条能拿到（唯一索引兜底）。"""
    ops = _ops()
    results = [ops.claim_seed_operation("seed-2", lease_seconds=300) for _ in range(5)]
    winners = [r for r in results if r]
    assert len(winners) == 1


def test_different_seeds_claim_independently(soul_db: Any) -> None:
    """不同种子互不影响。"""
    ops = _ops()
    assert ops.claim_seed_operation("seed-a", lease_seconds=300)
    assert ops.claim_seed_operation("seed-b", lease_seconds=300)


# ─── 租约过期可恢复（崩溃后重试） ───────────────────────────────────


def test_expired_lease_can_be_reclaimed(soul_db: Any) -> None:
    """租约过期（进程崩溃）后可被重新认领，不会永久卡死。"""
    ops = _ops()
    first = ops.claim_seed_operation("seed-3", lease_seconds=0)
    assert first

    second = ops.claim_seed_operation("seed-3", lease_seconds=300)

    assert second
    assert second != first


def test_running_lease_not_reclaimable_early(soul_db: Any) -> None:
    """租约未过期时不得被抢走（避免同一颗种子两个内化并行）。"""
    ops = _ops()
    assert ops.claim_seed_operation("seed-4", lease_seconds=600)
    assert ops.claim_seed_operation("seed-4", lease_seconds=600) is None


# ─── 终结：同事务写操作结果 + 种子终态 ──────────────────────────────


def test_finish_commits_operation_and_seed_status_together(soul_db: Any) -> None:
    """终结时操作结果与种子终态在同一事务落地。"""
    ops = _ops()
    seeds = _seeds()
    _mk_seed(seeds, "seed-5")
    op_id = ops.claim_seed_operation("seed-5", lease_seconds=300)

    assert ops.finish_seed_operation(op_id, seed_status="approved", result_json='{"trait_id":"t1"}')

    assert ops.get_seed_operation(op_id).status == "done"
    assert seeds.get_thought_seed_by_id("seed-5").status == "approved"


def test_finish_is_idempotent(soul_db: Any) -> None:
    """重复终结同一个 operation 只生效一次（幂等），第二次返回 False。"""
    ops = _ops()
    seeds = _seeds()
    _mk_seed(seeds, "seed-6")
    op_id = ops.claim_seed_operation("seed-6", lease_seconds=300)

    assert ops.finish_seed_operation(op_id, seed_status="internalized", result_json="{}") is True
    assert ops.finish_seed_operation(op_id, seed_status="internalized", result_json="{}") is False


def test_settled_seed_cannot_be_reclaimed(soul_db: Any) -> None:
    """已终结的种子不能再被认领（防管理员重复批准双开）。"""
    ops = _ops()
    seeds = _seeds()
    _mk_seed(seeds, "seed-7")
    op_id = ops.claim_seed_operation("seed-7", lease_seconds=300)
    ops.finish_seed_operation(op_id, seed_status="approved", result_json="{}")

    assert ops.claim_seed_operation("seed-7", lease_seconds=300) is None


def test_failed_operation_allows_retry(soul_db: Any) -> None:
    """内化失败 → 释放租约，允许重试（种子保持非终态）。"""
    ops = _ops()
    seeds = _seeds()
    _mk_seed(seeds, "seed-8")
    op_id = ops.claim_seed_operation("seed-8", lease_seconds=300)

    assert ops.release_seed_operation(op_id, error="LLM 超时")
    assert ops.get_seed_operation(op_id).status == "failed"
    assert seeds.get_thought_seed_by_id("seed-8").status == "pending"

    retry = ops.claim_seed_operation("seed-8", lease_seconds=300)
    assert retry and retry != op_id


def test_finish_unknown_operation_returns_false(soul_db: Any) -> None:
    """未知 operation_id 不得写任何状态。"""
    ops = _ops()
    assert ops.finish_seed_operation("no-such-op", seed_status="approved", result_json="{}") is False


def test_finish_rejects_unknown_seed_status(soul_db: Any) -> None:
    """只允许写入合法种子终态，防止把种子写成任意字符串。"""
    ops = _ops()
    seeds = _seeds()
    _mk_seed(seeds, "seed-9")
    op_id = ops.claim_seed_operation("seed-9", lease_seconds=300)

    assert ops.finish_seed_operation(op_id, seed_status="made_up", result_json="{}") is False
    assert seeds.get_thought_seed_by_id("seed-9").status == "pending"
    assert ops.get_seed_operation(op_id).status == "running"


# ─── 端到端：重复批准同一颗种子只内化一次 ───────────────────────────


def _approve_kwargs() -> dict:
    return {
        "platform": "qq",
        "user_id": "admin123",
        "text": "/soul_approve seed_dup",
        "message": {
            "platform": "qq",
            "user_info": {"user_id": "admin123"},
            "processed_plain_text": "/soul_approve seed_dup",
        },
    }


def _approve_plugin() -> Any:
    from types import SimpleNamespace

    class _Ctx:
        class _Send:
            async def text(self, text: str, stream_id: str = "") -> None:
                pass

        send = _Send()

    return SimpleNamespace(
        config=SimpleNamespace(
            admin=SimpleNamespace(admin_user_id="qq:admin123"),
            thought_cabinet=SimpleNamespace(
                enabled=True,
                fermentation_enabled=False,
                auto_dedup_enabled=True,
                auto_dedup_threshold=0.78,
            ),
        ),
        ctx=_Ctx(),
        _plugin_dir="/tmp",
    )


def test_double_approve_internalizes_only_once(soul_db: Any) -> None:
    """同一颗种子批准两次：内化只跑一次（旧实现会施加两遍光谱影响）。"""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    tc = _import_soul_submodule("components.thought_commands")
    seeds = _seeds()
    _mk_seed(seeds, "seed_dup")

    mock_seed = {
        "seed_id": "seed_dup",
        "stream_id": "global",
        "type": "观点",
        "event": "事件",
        "intensity": 0.8,
        "confidence": 0.75,
        "evidence": [],
        "context": [],
        "reasoning": "推理",
        "status": "pending",
        "created_at": None,
    }

    manager = MagicMock()
    manager.get_seed_by_id = AsyncMock(return_value=mock_seed)

    engine = MagicMock()
    engine.internalize_seed = AsyncMock(
        return_value={
            "success": True,
            "spectrum_impact": {"sincerity": 3},
            "trait_id": "trait-x",
            "merged": False,
            "thought": "结论",
        }
    )

    plugin = _approve_plugin()
    seed_manager_mod = _import_soul_submodule("thought.seed_manager")
    engine_mod = _import_soul_submodule("thought.internalization_engine")
    with patch.object(seed_manager_mod, "ThoughtSeedManager") as mgr_cls, \
         patch.object(engine_mod, "InternalizationEngine", return_value=engine):
        mgr_cls.from_plugin_config.return_value = manager
        asyncio.run(tc.handle_seed_approve(plugin, "g", **_approve_kwargs()))
        asyncio.run(tc.handle_seed_approve(plugin, "g", **_approve_kwargs()))

    assert engine.internalize_seed.await_count == 1, "重复批准不得重复内化"
    assert seeds.get_thought_seed_by_id("seed_dup").status == "approved"
