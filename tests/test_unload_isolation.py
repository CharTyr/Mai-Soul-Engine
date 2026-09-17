"""卸载清理的逐项隔离。

`on_unload` 是一串清理步骤：取消四个后台任务 → 清模块级可变状态 → 关库。
旧实现是一条直线，任何一步抛异常都会**中断后续所有步骤**，结果是：

- 数据库没关（连接泄漏，下次启动可能撞锁）
- 模块级状态没清（插件重载时旧冷却/旧缓存串到新实例）

而且这几个模块级状态是不同模块的私有对象，一个 clear 失败通常与其它无关。
所以每一步独立兜底：失败记警告、继续往下走。
"""

from __future__ import annotations

import asyncio
from typing import Any

from .conftest import _import_soul_submodule


def _plugin_instance() -> Any:
    plugin_mod = _import_soul_submodule("plugin")
    return plugin_mod.MaiSoulEnginePlugin()


class _BoomDict(dict):
    """clear() 会炸的 dict —— 模拟其中一个清理步骤失败。"""

    def clear(self) -> None:  # type: ignore[override]
        raise RuntimeError("模拟清理失败")


class _BoomSet(set):
    def clear(self) -> None:  # type: ignore[override]
        raise RuntimeError("模拟清理失败")


def test_unload_continues_after_a_cleanup_step_fails(monkeypatch: Any) -> None:
    """第一个清理步骤炸掉，后面的步骤仍必须执行。"""
    inst = _plugin_instance()
    injector = _import_soul_submodule("components.ideology_injector")
    capture = _import_soul_submodule("components.reflection_capture")

    capture._context_cache["sess-x"] = (["某条消息"], 0.0)

    monkeypatch.setattr(injector, "_RECENT_TRAIT_INJECTION", _BoomDict({"t": 1.0}))

    asyncio.run(inst.on_unload())

    assert capture._context_cache == {}, "前面的步骤失败不得阻断后续清理"


def test_unload_still_closes_db_after_module_state_failure(monkeypatch: Any) -> None:
    """模块级状态清理失败，数据库仍必须被关闭。"""
    inst = _plugin_instance()
    evo = _import_soul_submodule("components.evolution_task")
    ideology = _import_soul_submodule("models.ideology_model")

    closed: list[bool] = []
    monkeypatch.setattr(ideology, "close_db", lambda: closed.append(True))

    def boom() -> None:
        raise RuntimeError("模拟聚合状态重置失败")

    monkeypatch.setattr(evo, "reset_aggregation_state", boom)

    asyncio.run(inst.on_unload())

    assert closed == [True], "关库必须执行，否则连接泄漏"


def test_unload_normal_path_clears_everything(monkeypatch: Any) -> None:
    """正常路径：状态清空 + 关库 + 任务引用复位。"""
    inst = _plugin_instance()
    injector = _import_soul_submodule("components.ideology_injector")
    capture = _import_soul_submodule("components.reflection_capture")
    evo = _import_soul_submodule("components.evolution_task")
    ideology = _import_soul_submodule("models.ideology_model")

    injector._RECENT_TRAIT_INJECTION["t1"] = 1.0
    capture._context_cache["s1"] = (["a"], 0.0)
    evo._bot_filter_warned.add("qq-1")

    closed: list[bool] = []
    monkeypatch.setattr(ideology, "close_db", lambda: closed.append(True))

    asyncio.run(inst.on_unload())

    assert dict(injector._RECENT_TRAIT_INJECTION) == {}
    assert capture._context_cache == {}
    assert not evo._bot_filter_warned
    assert closed == [True]
    assert inst._evolution_task is None
    assert inst._fermentation_task is None


def test_unload_survives_task_cancel_failure(monkeypatch: Any) -> None:
    """某个后台任务取消失败，其它任务与后续清理照常。"""
    inst = _plugin_instance()
    capture = _import_soul_submodule("components.reflection_capture")
    ideology = _import_soul_submodule("models.ideology_model")

    class _BadTask:
        def cancel(self) -> None:
            raise RuntimeError("模拟取消失败")

    inst._evolution_task = _BadTask()
    capture._context_cache["s2"] = (["b"], 0.0)
    closed: list[bool] = []
    monkeypatch.setattr(ideology, "close_db", lambda: closed.append(True))

    asyncio.run(inst.on_unload())

    assert capture._context_cache == {}
    assert closed == [True]
    assert inst._evolution_task is None


def test_unload_reports_failure_in_log(monkeypatch: Any, caplog: Any) -> None:
    """失败必须留下可查的日志（否则卸载不完整会完全无声）。"""
    import logging

    inst = _plugin_instance()
    injector = _import_soul_submodule("components.ideology_injector")

    monkeypatch.setattr(injector, "_RECENT_TRAIT_INJECTION", _BoomDict())

    def boom() -> None:
        raise RuntimeError("炸")

    monkeypatch.setattr(capture := _import_soul_submodule("components.reflection_capture"), "_context_cache", _BoomDict())

    with caplog.at_level(logging.WARNING):
        asyncio.run(inst.on_unload())

    assert any("炸" in r.message or "清理" in r.message for r in caplog.records)
