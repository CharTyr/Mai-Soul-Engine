"""T14：任务异常 / 热更 / 卸载 / 取消后——**无重复任务、无未说明残留**。

方案原文：`T14：任务异常、热更、卸载、取消后无重复任务和未说明残留。`

这里测的是**生命周期管理**（`_reconcile_background_tasks` / `_supervise_background_tasks`
/ `_stop_all_background_tasks`），所以五个循环协程一律换成"长睡"桩：
被测对象是"起了几个任务、谁还活着、有没有残留"，不是循环内部逻辑。

为什么必须断言"还是同一个 Task 对象"：`_task_action` 的 restart 分支会重建任务，
热更时若判定错误就会 cancel 掉活着的任务再起一个新的——表面看"有一个任务在跑"，
实际每次热更都丢一次工作现场。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from .conftest import _import_soul_submodule
from .test_feature_supervisor import _build_config


def _plugin_instance(cfg: Any) -> Any:
    """真插件实例 + 注入配置。

    `config` 是只读属性，真身是 `_plugin_config_instance`（SDK 注入）。
    学习类任务还受**运行模式闸门**控制（`off` 全停），所以配置必须显式
    给 `plugin.mode = "apply"`——否则测试会因为"任务没起"而假绿。
    """
    plugin_mod = _import_soul_submodule("plugin")
    inst = plugin_mod.MaiSoulEnginePlugin()
    inst._plugin_config_instance = cfg
    return inst


def _cfg(*, evolution: bool = False, mode: str = "apply") -> Any:
    return _build_config(**{
        "plugin": {"enabled": True, "mode": mode},
        "evolution": {"evolution_enabled": evolution},
    })


async def _sleep_forever() -> None:
    await asyncio.sleep(3600)


def _stub_all_loops(inst: Any) -> None:
    """把五个循环协程换成"长睡"桩，避免真循环在测试里到处 IO。"""
    for _key, _attr, loop_attr in inst._TASK_ENTRIES:
        setattr(inst, loop_attr, _sleep_forever)


def _alive(inst: Any) -> dict[str, Any]:
    """当前活着的任务（key → Task）。"""
    out: dict[str, Any] = {}
    for key, attr, _loop in inst._TASK_ENTRIES:
        task = getattr(inst, attr)
        if task is not None and not task.done():
            out[key] = task
    return out


async def _cleanup(inst: Any) -> None:
    inst._unloading = True
    await inst._stop_all_background_tasks()


# ─── 1. 热更：反复启停不产生重复任务 ─────────────────────────────


@pytest.mark.asyncio
async def test_hot_update_never_duplicates_tasks() -> None:
    """同配置反复热更 → keep（同一个 Task）；关→开 → 新的单个任务；全程无重复。"""
    inst = _plugin_instance(_cfg(evolution=True))
    _stub_all_loops(inst)
    try:
        await inst._reconcile_background_tasks()
        first = inst._evolution_task
        assert first is not None and not first.done()
        assert len(_alive(inst)) == 1

        # 配置没变的热更：必须是 keep，不能重建
        for _ in range(3):
            await inst._reconcile_background_tasks()
        assert inst._evolution_task is first, "热更重建了活着的任务（工作现场丢失）"
        assert len(_alive(inst)) == 1, "热更产生了重复任务"

        # 关：任务取消且属性清空
        inst._plugin_config_instance = _cfg(evolution=False)
        await inst._reconcile_background_tasks()
        assert inst._evolution_task is None
        assert first.done(), "关闭后旧任务还活着"
        assert len(_alive(inst)) == 0

        # 再开：恰好一个新任务
        inst._plugin_config_instance = _cfg(evolution=True)
        await inst._reconcile_background_tasks()
        second = inst._evolution_task
        assert second is not None and not second.done() and second is not first
        assert len(_alive(inst)) == 1, "重新开启后出现重复任务"
    finally:
        await _cleanup(inst)


# ─── 2. 任务异常：巡检重启后仍是单实例 ───────────────────────────


@pytest.mark.asyncio
async def test_crashed_task_is_replaced_by_single_instance() -> None:
    """任务崩溃 → 巡检记录死亡 → 退避到期后重启，且**恰好一个**实例。"""
    inst = _plugin_instance(_cfg(evolution=True))
    _stub_all_loops(inst)
    try:
        await inst._reconcile_background_tasks()
        alive = inst._evolution_task
        assert alive is not None

        # 模拟崩溃：取消掉（done() 为真）
        alive.cancel()
        with pytest.raises(asyncio.CancelledError):
            await alive

        # 第一次巡检：只记死亡 + 退避，不立刻重启
        await inst._supervise_background_tasks()
        state = inst._task_supervisor.state_of("evolution")
        assert state.status == "backoff", f"崩溃后状态应为 backoff，实际 {state.status}"
        assert inst._evolution_task is None
        assert len(_alive(inst)) == 0

        # 让退避立即到期 → 巡检重启
        inst._task_supervisor.note_started  # noqa: B018 — 仅表明 API 存在
        state.next_retry_at = 0.0
        await inst._supervise_background_tasks()
        new_task = inst._evolution_task
        assert new_task is not None and not new_task.done()
        assert new_task is not alive, "重启后仍指向已死任务"
        assert len(_alive(inst)) == 1, "巡检重启产生了重复任务"
    finally:
        await _cleanup(inst)


# ─── 3. 卸载 / 取消：无残留，且卸载期不得重新拉起 ────────────────


@pytest.mark.asyncio
async def test_unload_leaves_no_residue_and_blocks_restart() -> None:
    """卸载后：任务属性清空、任务全部结束；且卸载期巡检不得重新拉起。"""
    inst = _plugin_instance(_cfg(evolution=True))
    _stub_all_loops(inst)
    try:
        await inst._reconcile_background_tasks()
        task = inst._evolution_task
        assert task is not None

        inst._unloading = True
        await inst._supervise_background_tasks()  # 卸载期巡检：必须什么都不做
        assert inst._evolution_task is task, "卸载期巡检把任务动过了"

        await inst._stop_all_background_tasks()
        assert inst._evolution_task is None, "卸载后任务属性未清空（残留引用）"
        assert task.done(), "卸载后任务没被取消"
        assert len(_alive(inst)) == 0, "卸载后仍有活着任务"

        # 卸载期即使巡检也不得重新拉起
        await inst._supervise_background_tasks()
        assert inst._evolution_task is None, "卸载期重新拉起了任务"
        assert len(_alive(inst)) == 0
    finally:
        await _cleanup(inst)


@pytest.mark.asyncio
async def test_cancelled_task_is_detected_not_left_as_zombie() -> None:
    """被取消的任务不能留下"看着还在跑"的僵尸引用。"""
    inst = _plugin_instance(_cfg(evolution=True))
    _stub_all_loops(inst)
    try:
        await inst._reconcile_background_tasks()
        task = inst._evolution_task
        assert task is not None

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await inst._supervise_background_tasks()
        state = inst._task_supervisor.state_of("evolution")
        assert state.status == "backoff", "被取消的任务没有被识别为异常结束"
        assert not _alive(inst), "僵尸任务引用没被清理"
    finally:
        await _cleanup(inst)
