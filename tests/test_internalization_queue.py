"""内化操作队列：命令快速返回，后台执行。

**要解决的**：`/soul_approve` 以前在命令里直接调 LLM 内化。插件给 LLM 的超时是
120s，宿主给命令的 RPC 超时是 60s（`src/plugin_runtime/host/component_timeout.py`），
于是慢一点就出现"命令报超时/失败，但内化其实已经成功、光谱影响已经写入"——
管理员的认知与实际状态相反。

现在：命令只认领 + 入队，立刻回 `operation_id`；后台任务按预算执行。
租约保证同一颗种子只有一个内化在跑，终结时操作结果与种子终态同事务提交。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from .conftest import _import_soul_submodule


def _queue() -> Any:
    return _import_soul_submodule("thought.internalization_queue")


def _ops() -> Any:
    return _import_soul_submodule("models.operations")


def _seeds() -> Any:
    return _import_soul_submodule("models.seeds")


def _make_seed(seed_id: str = "seed_q1", status: str = "pending") -> None:
    seeds = _seeds()
    seeds.create_thought_seed(
        seed_id=seed_id,
        stream_id="qq-123-group",
        seed_type="价值观冲突",
        event="讨论真诚",
        intensity=0.8,
        confidence=0.7,
        evidence_json="[]",
        reasoning="观察",
        potential_impact_json="{}",
    )
    if status != "pending":
        seeds.update_seed_status(seed_id, status, expected_status="pending")


def _plugin(*, llm_result: dict | None = None, raise_exc: Exception | None = None) -> Any:
    """构造插件：内化引擎被替换为可控 mock。"""
    engine = MagicMock()
    if raise_exc is not None:
        engine.internalize_seed = AsyncMock(side_effect=raise_exc)
    else:
        engine.internalize_seed = AsyncMock(
            return_value=llm_result
            if llm_result is not None
            else {"success": True, "trait_id": "trait_q1", "spectrum_impact": {"sincerity": 3}}
        )

    sent: list[str] = []

    class _Ctx:
        class _Send:
            async def text(self, text: str, stream_id: str = "") -> dict:
                sent.append(text)
                return {"success": True}

        class _Chat:
            async def get_stream_by_user_id(self, **kwargs: Any) -> str:
                return "qq:private:admin"

        send = _Send()
        chat = _Chat()

    class _Manager:
        @classmethod
        def from_plugin_config(cls, plugin: Any) -> "_Manager":
            return cls()

        async def get_seed_by_id(self, seed_id: str) -> dict | None:
            seed = _seeds().get_thought_seed_by_id(seed_id)
            if seed is None:
                return None
            return {
                "seed_id": seed.seed_id,
                "id": seed.seed_id,
                "stream_id": seed.stream_id,
                "type": seed.seed_type,
                "event": seed.event,
                "reasoning": seed.reasoning,
                "intensity": seed.intensity,
                "confidence": seed.confidence,
                "status": seed.status,
                "evidence": [],
                "context": [],
                "created_at": None,
            }

    plugin = SimpleNamespace(
        ctx=_Ctx(),
        config=SimpleNamespace(
            plugin=SimpleNamespace(enabled=True, mode="apply"),
            admin=SimpleNamespace(admin_user_id="qq:admin123"),
            thought_cabinet=SimpleNamespace(
                enabled=True,
                fermentation_enabled=False,
                auto_dedup_enabled=True,
                auto_dedup_threshold=0.78,
                max_internalize_delta=10,
            ),
            worldview=SimpleNamespace(p1_enabled=False),
        ),
        _sent=sent,
    )
    plugin._engine = engine
    plugin._manager_cls = _Manager
    return plugin


def _patch_engine(plugin: Any) -> Any:
    """把队列模块里用到的引擎与管理器换成可控实现。"""
    engine_mod = _import_soul_submodule("thought.internalization_engine")
    manager_mod = _import_soul_submodule("thought.seed_manager")
    return (
        patch.object(engine_mod, "InternalizationEngine", return_value=plugin._engine),
        patch.object(manager_mod, "ThoughtSeedManager", plugin._manager_cls),
    )


# ─── 入队 ───────────────────────────────────────────────────────────


def test_enqueue_creates_running_operation(soul_db: Any) -> None:
    """入队产生一条 running 操作。"""
    _make_seed("seed_q1")
    q, ops = _queue(), _ops()

    operation_id = asyncio.run(q.enqueue_internalization(_plugin(), "seed_q1"))

    assert operation_id
    assert ops.get_seed_operation(operation_id).status == "running"


def test_second_enqueue_is_refused_while_running(soul_db: Any) -> None:
    """已有进行中的操作 → 第二次入队被拒（并发批准只跑一次）。"""
    _make_seed("seed_q1")
    q = _queue()
    plugin = _plugin()

    first = asyncio.run(q.enqueue_internalization(plugin, "seed_q1"))
    second = asyncio.run(q.enqueue_internalization(plugin, "seed_q1"))

    assert first is not None
    assert second is None


def test_enqueue_refused_after_settled(soul_db: Any) -> None:
    """已经内化过 → 不再入队（防止二次施加光谱影响）。"""
    _make_seed("seed_q1")
    q = _queue()
    ops = _ops()
    op_id = asyncio.run(q.enqueue_internalization(_plugin(), "seed_q1"))
    ops.finish_seed_operation(op_id, seed_status="approved")

    assert asyncio.run(q.enqueue_internalization(_plugin(), "seed_q1")) is None


# ─── 消费 ───────────────────────────────────────────────────────────


def test_run_queue_internalizes_and_settles(soul_db: Any) -> None:
    """消费一轮：执行内化 + 操作 done + 种子 approved。"""
    _make_seed("seed_q1")
    q, ops, seeds = _queue(), _ops(), _seeds()
    plugin = _plugin()
    op_id = asyncio.run(q.enqueue_internalization(plugin, "seed_q1"))

    with _patch_engine(plugin)[0], _patch_engine(plugin)[1]:
        stats = asyncio.run(q.run_queue_once(plugin))

    assert stats == {"done": 1, "retry": 0, "skipped": 0}
    assert ops.get_seed_operation(op_id).status == "done"
    assert seeds.get_thought_seed_by_id("seed_q1").status == "approved"


def test_run_queue_releases_on_engine_failure(soul_db: Any) -> None:
    """内化抛异常 → 释放租约、种子保持 pending（可重试），不写终态。"""
    _make_seed("seed_q1")
    q, ops, seeds = _queue(), _ops(), _seeds()
    plugin = _plugin(raise_exc=RuntimeError("LLM 挂了"))
    op_id = asyncio.run(q.enqueue_internalization(plugin, "seed_q1"))

    with _patch_engine(plugin)[0], _patch_engine(plugin)[1]:
        stats = asyncio.run(q.run_queue_once(plugin))

    assert stats["retry"] == 1
    assert ops.get_seed_operation(op_id).status == "failed"
    assert seeds.get_thought_seed_by_id("seed_q1").status == "pending", "失败不得推进终态"
    # 释放后可重新入队重试
    assert asyncio.run(q.enqueue_internalization(plugin, "seed_q1")) is not None


def test_run_queue_releases_on_unsuccessful_result(soul_db: Any) -> None:
    """内化返回 success=False（如候选被拒）→ 释放租约且不改种子状态。"""
    _make_seed("seed_q1")
    q, ops, seeds = _queue(), _ops(), _seeds()
    plugin = _plugin(llm_result={"success": False, "error": "候选被拒: empty_thought"})
    op_id = asyncio.run(q.enqueue_internalization(plugin, "seed_q1"))

    with _patch_engine(plugin)[0], _patch_engine(plugin)[1]:
        asyncio.run(q.run_queue_once(plugin))

    record = ops.get_seed_operation(op_id)
    assert record.status == "failed"
    assert "候选被拒" in record.error
    assert seeds.get_thought_seed_by_id("seed_q1").status == "pending"


def test_run_queue_skips_fermenting_seed(soul_db: Any) -> None:
    """发酵中的种子归发酵循环处理，队列不得抢（避免双重内化）。"""
    _make_seed("seed_q1")
    seeds = _seeds()
    seeds.mark_seed_fermenting("seed_q1")
    q, ops = _queue(), _ops()
    plugin = _plugin()
    op_id = asyncio.run(q.enqueue_internalization(plugin, "seed_q1"))

    with _patch_engine(plugin)[0], _patch_engine(plugin)[1]:
        stats = asyncio.run(q.run_queue_once(plugin))

    assert stats["skipped"] == 1
    assert plugin._engine.internalize_seed.await_count == 0, "队列不得内化发酵中的种子"
    assert ops.get_seed_operation(op_id).status == "failed"


def test_run_queue_handles_missing_seed(soul_db: Any) -> None:
    """种子已被删除 → 释放操作，不抛异常。"""
    _make_seed("seed_q1")
    q = _queue()
    plugin = _plugin()
    asyncio.run(q.enqueue_internalization(plugin, "seed_q1"))
    _seeds().delete_thought_seed("seed_q1")

    with _patch_engine(plugin)[0], _patch_engine(plugin)[1]:
        stats = asyncio.run(q.run_queue_once(plugin))

    assert stats["skipped"] == 1


def test_run_queue_respects_limit(soul_db: Any) -> None:
    """一轮最多处理 limit 条（按预算执行）。"""
    for i in range(3):
        _make_seed(f"seed_q{i}")
    q = _queue()
    plugin = _plugin()
    for i in range(3):
        asyncio.run(q.enqueue_internalization(plugin, f"seed_q{i}"))

    with _patch_engine(plugin)[0], _patch_engine(plugin)[1]:
        stats = asyncio.run(q.run_queue_once(plugin, limit=2))

    assert stats["done"] == 2
    assert plugin._engine.internalize_seed.await_count == 2


# ─── 命令层：快速返回，不内联调用 LLM ───────────────────────────────


def test_approve_returns_immediately_without_calling_llm(soul_db: Any) -> None:
    """`/soul_approve` 不再在命令里调 LLM：立刻回 operation_id。"""
    _make_seed("seed_q1")
    tc = _import_soul_submodule("components.thought_commands")
    plugin = _plugin()

    with _patch_engine(plugin)[0], _patch_engine(plugin)[1]:
        asyncio.run(
            tc.handle_seed_approve(
                plugin, "g",
                platform="qq", user_id="admin123",
                text="/soul_approve seed_q1",
                message={"platform": "qq", "user_info": {"user_id": "admin123"},
                         "processed_plain_text": "/soul_approve seed_q1"},
            )
        )
        assert plugin._engine.internalize_seed.await_count == 0, "命令内不得内联调用 LLM"

    assert any("operation_id" in m for m in plugin._sent)
    assert _ops().list_running_operations(), "操作应已入队"


def test_double_approve_only_enqueues_once(soul_db: Any) -> None:
    """连续批准两次：只有一次真正进入内化（第二次被拒）。"""
    _make_seed("seed_q1")
    tc = _import_soul_submodule("components.thought_commands")
    q = _queue()
    plugin = _plugin()
    kwargs = {
        "platform": "qq", "user_id": "admin123",
        "text": "/soul_approve seed_q1",
        "message": {"platform": "qq", "user_info": {"user_id": "admin123"},
                    "processed_plain_text": "/soul_approve seed_q1"},
    }

    with _patch_engine(plugin)[0], _patch_engine(plugin)[1]:
        asyncio.run(tc.handle_seed_approve(plugin, "g", **kwargs))
        asyncio.run(tc.handle_seed_approve(plugin, "g", **kwargs))
        assert len(_ops().list_running_operations(limit=10)) == 1, "只允许一条进行中的操作"

        stats = asyncio.run(q.run_queue_once(plugin))

    assert stats["done"] == 1
    assert plugin._engine.internalize_seed.await_count == 1, "同一颗种子只内化一次"


def test_run_queue_survives_item_level_config_error(soul_db: Any) -> None:
    """单条条目的配置读取异常不得中断整轮（逐条隔离）。

    队列跑在后台循环里，一条卡住就等于整队停摆。
    """
    _make_seed("seed_q1")
    _make_seed("seed_q2")
    q = _queue()
    plugin = _plugin()
    asyncio.run(q.enqueue_internalization(plugin, "seed_q1"))
    asyncio.run(q.enqueue_internalization(plugin, "seed_q2"))

    # 让第一条的 dedup 配置读取炸掉
    real_cabinet = plugin.config.thought_cabinet

    class _BoomOnce:
        """只炸第一次的配置视图：第一条失败、第二条照常。"""

        def __init__(self) -> None:
            self._fired = False

        def __getattr__(self, name: str) -> Any:
            if name == "auto_dedup_enabled" and not self._fired:
                self._fired = True
                raise RuntimeError("配置读取炸了")
            return getattr(real_cabinet, name)

    plugin.config.thought_cabinet = _BoomOnce()

    with _patch_engine(plugin)[0], _patch_engine(plugin)[1]:
        stats = asyncio.run(q.run_queue_once(plugin))

    assert stats["retry"] == 1, "炸掉的那条计入重试"
    assert stats["done"] == 1, "其余条目仍应被处理"
