"""后台任务监督：崩溃可发现、可重启、超限转 failed（T14）。

确认的缺陷：`_reconcile_background_tasks` 只判断 `current is None`，
不判断 `current.done()`。任务一旦抛异常退出，Task 对象仍在、不为 None，
于是：

- 不会被重启 → 该学的一直不学了
- `/soul_health` 仍显示"运行中" → 假绿

本模块把「任务死了」变成可观察、可恢复、且不会无限重启刷屏的状态机。
"""

from __future__ import annotations

from typing import Any

from .conftest import _import_soul_submodule


def _sup() -> Any:
    return _import_soul_submodule("utils.task_supervisor")


def _plugin() -> Any:
    return _import_soul_submodule("plugin")


# ─── 纯决策：何时 start / restart / keep / stop ─────────────────────


def test_action_start_when_not_running() -> None:
    """需要跑且当前没有任务 → 启动。"""
    p = _plugin()
    assert p.MaiSoulEnginePlugin._task_action(None, True) == "start"


def test_action_stop_when_not_desired() -> None:
    """不需要跑且当前有任务 → 停止。"""
    p = _plugin()

    class _T:
        def done(self) -> bool:
            return False

    assert p.MaiSoulEnginePlugin._task_action(_T(), False) == "stop"


def test_action_restart_when_finished_unexpectedly() -> None:
    """需要跑但任务已结束 → 重启（旧实现漏掉这一支，导致静默停摆）。"""
    p = _plugin()

    class _T:
        def done(self) -> bool:
            return True

    assert p.MaiSoulEnginePlugin._task_action(_T(), True) == "restart"


def test_action_keep_when_healthy() -> None:
    """需要跑且任务在跑 → 不动。"""
    p = _plugin()

    class _T:
        def done(self) -> bool:
            return False

    assert p.MaiSoulEnginePlugin._task_action(_T(), True) == "keep"


def test_action_keep_when_nothing_to_do() -> None:
    """不需要跑且没有任务 → 不动。"""
    p = _plugin()
    assert p.MaiSoulEnginePlugin._task_action(None, False) == "keep"


# ─── 监督器状态机 ───────────────────────────────────────────────────


def test_initial_state_is_stopped() -> None:
    """未启动的任务状态是 stopped。"""
    s = _sup().TaskSupervisor()
    assert s.state_of("evolution").status == "stopped"


def test_note_started_marks_running() -> None:
    """启动后是 running。"""
    s = _sup().TaskSupervisor()
    s.note_started("evolution")
    assert s.state_of("evolution").status == "running"


def test_note_death_marks_backoff_and_counts() -> None:
    """异常结束 → 记为 **backoff**（不是笼统的 restarting）并累加重启次数。

    backoff 与 restarting 的区别是要紧的：backoff 表示「正在退避窗口里等」，
    看板据此能显示下次重试时刻；笼统的 restarting 看不出还要等多久。
    """
    s = _sup().TaskSupervisor()
    s.note_started("evolution")
    s.note_death("evolution", reason="RuntimeError: boom")

    state = s.state_of("evolution")
    assert state.status == "backoff"
    assert state.restart_count == 1
    assert "boom" in state.last_death_reason


def test_intentional_stop_is_not_counted_as_death() -> None:
    """主动停止（配置关闭）不计入异常重启。"""
    s = _sup().TaskSupervisor()
    s.note_started("evolution")
    s.note_stopped("evolution")

    state = s.state_of("evolution")
    assert state.status == "stopped"
    assert state.restart_count == 0


def test_restart_allowed_until_limit() -> None:
    """未超上限 → 允许重启。"""
    s = _sup().TaskSupervisor(max_restarts=3)
    s.note_started("evolution")
    s.note_death("evolution", reason="boom")
    assert s.should_restart("evolution") is True


def test_restart_refused_after_limit() -> None:
    """超过上限 → 拒绝继续重启（避免无限重启刷屏），转为 failed。"""
    s = _sup().TaskSupervisor(max_restarts=2)
    s.note_started("evolution")
    for _ in range(3):
        s.note_death("evolution", reason="boom")
        if s.should_restart("evolution"):
            s.note_started("evolution")

    state = s.state_of("evolution")
    assert state.status == "failed"
    assert s.should_restart("evolution") is False


def test_note_started_resets_running_after_backoff() -> None:
    """重启成功后状态回到 running。"""
    s = _sup().TaskSupervisor()
    s.note_started("evolution")
    s.note_death("evolution", reason="boom")
    assert s.should_restart("evolution") is True
    s.note_started("evolution")
    assert s.state_of("evolution").status == "running"


def test_failed_state_is_visible_in_summary() -> None:
    """汇总必须能让看板区分「在跑」与「已失败」。"""
    s = _sup().TaskSupervisor(max_restarts=1)
    s.note_started("evolution")
    s.note_started("fermentation")
    s.note_death("evolution", reason="boom")
    s.note_death("evolution", reason="boom")

    summary = s.describe()
    assert "evolution" in summary
    assert "failed" in summary
    assert "fermentation" in summary


def test_healthy_predicate() -> None:
    """有 failed 任务 → 整体不健康(供 health 报 degraded)。"""
    s = _sup().TaskSupervisor(max_restarts=1)
    assert s.is_healthy() is True
    s.note_started("evolution")
    s.note_death("evolution", reason="boom")
    s.note_death("evolution", reason="boom")
    assert s.is_healthy() is False


def test_states_cover_all_declared_names() -> None:
    """四个任务的初始状态都能查到（看板不会 KeyError）。"""
    s = _sup().TaskSupervisor()
    for name in ("evolution", "notion", "reflection", "fermentation"):
        assert s.state_of(name).status == "stopped"


# ─── failed 之后不得被下一次 reconcile 重新拉起 ─────────────────────


def test_failed_task_is_not_restarted_by_start_path() -> None:
    """超限判定 failed 后，即便任务对象是 None，也不得再自动启动。

    否则"达到上限转 failed 等人工介入"会被下一次 reconcile 立刻推翻。
    """
    s = _sup().TaskSupervisor(max_restarts=1)
    s.note_started("evolution")
    s.note_death("evolution", reason="boom")

    assert s.state_of("evolution").status == "failed"
    assert s.should_restart("evolution") is False

    # 此时任务对象会被置为 None；监督器必须拦住 start
    assert s.start_allowed("evolution") is False


def test_start_allowed_when_healthy_or_restarting() -> None:
    """正常状态下允许启动（不误伤）。"""
    s = _sup().TaskSupervisor(max_restarts=3)
    assert s.start_allowed("evolution") is True
    s.note_started("evolution")
    s.note_death("evolution", reason="boom")
    assert s.start_allowed("evolution") is True


def test_reset_clears_failed_state() -> None:
    """操作者处理后可显式复位（例如配置变更后重新启用）。"""
    s = _sup().TaskSupervisor(max_restarts=1)
    s.note_started("evolution")
    s.note_death("evolution", reason="boom")
    assert s.start_allowed("evolution") is False

    s.reset("evolution")

    assert s.start_allowed("evolution") is True
    assert s.state_of("evolution").restart_count == 0


# ─── 长时间稳定运行后的偶发崩溃不应累积 ─────────────────────────────


def test_healthy_run_resets_failure_streak() -> None:
    """任务稳定跑了很久才崩一次 → 视为新的偶发事故，不累积重启次数。

    否则一个每几天崩一次的任务，几个月后会被永久判 failed——
    这跟"连续崩溃"是两码事，不能混为一谈。
    """
    s = _sup().TaskSupervisor(max_restarts=3)

    # 第一次崩溃后重启，随后"稳定运行"了 2 小时再崩
    s.note_started("evolution")
    s.note_death("evolution", reason="偶发1")
    assert s.state_of("evolution").restart_count == 1

    s.note_started("evolution", started_at=1000.0)
    s.note_death("evolution", reason="偶发2", now=1000.0 + 7200)

    assert s.state_of("evolution").restart_count == 1, "稳定运行后的崩溃应重新计数"
    assert s.state_of("evolution").status in ("backoff", "restarting")


def test_rapid_repeated_crashes_still_accumulate() -> None:
    """短时间内连续崩溃 → 照旧累积，直到判 failed。"""
    s = _sup().TaskSupervisor(max_restarts=2)

    s.note_started("evolution", started_at=0.0)
    s.note_death("evolution", reason="快速崩1", now=5.0)
    s.note_started("evolution", started_at=5.0)
    s.note_death("evolution", reason="快速崩2", now=10.0)

    assert s.state_of("evolution").restart_count == 2
    assert s.state_of("evolution").status == "failed"
    assert s.start_allowed("evolution") is False


def test_all_five_plan_states_are_reachable() -> None:
    """方案要求监督器识别 running / waiting / backoff / failed / stopped。

    五态都必须**可达**且互不混同——特别是 waiting（该跑但还没跑起来）不能
    被显示成 running（在跑），否则看板会在任务实际停摆时显示正常。
    """
    from .conftest import _import_soul_submodule

    ts = _import_soul_submodule("utils.task_supervisor")
    s = ts.TaskSupervisor(max_restarts=2)

    seen: set[str] = set()

    s.note_started("evolution")
    seen.add(s.state_of("evolution").status)

    s.note_waiting("evolution", reason="等待下一轮间隔")
    seen.add(s.state_of("evolution").status)

    s.note_death("evolution")
    seen.add(s.state_of("evolution").status)

    s.note_stopped("evolution")
    seen.add(s.state_of("evolution").status)

    # failed：连续异常超限
    for _ in range(3):
        s.note_started("evolution")
        s.note_death("evolution")
    seen.add(s.state_of("evolution").status)

    assert seen == {"running", "waiting", "backoff", "stopped", "failed"}, (
        f"五态未全部可达: {sorted(seen)}"
    )


def test_waiting_is_reported_by_real_loops() -> None:
    """真实循环必须调用 waiting 打点（否则该状态只是装饰）。

    直接检查源码调用点：模块级循环拿不到 supervisor，须经薄封装打点。
    """
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
    expected = {
        "components/evolution_task.py": "evolution",
        "thought/fermentation_engine.py": "fermentation",
        "components/reflection_evaluator.py": "reflection",
        "components/notion_sync.py": "notion",
        "plugin.py": "internalization",
    }
    missing = [
        f"{rel} → {key}"
        for rel, key in expected.items()
        if f'"{key}"' not in (root / rel).read_text(encoding="utf-8")
        or "note_task_waiting" not in (root / rel).read_text(encoding="utf-8")
        and 'note_waiting("internalization"' not in (root / rel).read_text(encoding="utf-8")
    ]
    assert not missing, f"这些循环没有 waiting 打点: {missing}"
