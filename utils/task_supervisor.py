"""后台任务监督：状态可见、崩溃可恢复、超限转 failed。

**解决的问题**：`_reconcile_background_tasks` 原先只判断任务对象是否为 None。
后台协程一旦抛异常退出，Task 对象仍在（不为 None），于是既不会被重启
（该学的永久停摆），`/soul_health` 又仍显示「运行中」（假绿）。

**边界**：这里不试图给每个后台循环加逐轮埋点（那需要改 4 个循环模块），
只盯着真正会静默出事的信号——**任务意外结束**。重启有上限，超限转 failed
并要求人工介入，避免"崩了重启、重启又崩"刷屏。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

__all__ = [
    "TASK_NAMES",
    "TaskState",
    "TaskSupervisor",
]

# 与 plugin.py 的 task_entries 对齐
TASK_NAMES: tuple[str, ...] = ("evolution", "notion", "reflection", "fermentation")

STATUS_STOPPED = "stopped"
STATUS_RUNNING = "running"
STATUS_RESTARTING = "restarting"
STATUS_FAILED = "failed"

# 默认重启上限：连续异常结束超过该次数即判 failed，等人工介入。
DEFAULT_MAX_RESTARTS = 5


@dataclass
class TaskState:
    """单个后台任务的监督状态。"""

    name: str
    status: str = STATUS_STOPPED
    restart_count: int = 0
    last_death_reason: str = ""
    last_change_at: float = field(default_factory=time.time)


class TaskSupervisor:
    """后台任务状态机（进程内，不持久化）。"""

    def __init__(self, max_restarts: int = DEFAULT_MAX_RESTARTS) -> None:
        self._max_restarts = max(1, int(max_restarts))
        self._states: dict[str, TaskState] = {
            name: TaskState(name=name) for name in TASK_NAMES
        }

    # ── 查询 ───────────────────────────────────────────────────────

    def state_of(self, name: str) -> TaskState:
        """取任务状态；未登记的名字给一个临时 stopped 状态（不抛异常）。"""
        state = self._states.get(name)
        if state is None:
            state = TaskState(name=name)
            self._states[name] = state
        return state

    def should_restart(self, name: str) -> bool:
        """是否还允许自动重启。"""
        state = self.state_of(name)
        if state.restart_count >= self._max_restarts:
            state.status = STATUS_FAILED
            return False
        return True

    def start_allowed(self, name: str) -> bool:
        """是否允许（重新）启动该任务。

        与 ``should_restart`` 的区别：这里也拦住「任务对象已是 None 的新一轮
        start」。否则判 failed 之后，下一次 reconcile 看到 None 又会把它拉起来，
        「达到上限等人工介入」就成了空话。
        """
        state = self.state_of(name)
        return state.status != STATUS_FAILED and state.restart_count < self._max_restarts

    def is_healthy(self) -> bool:
        """没有任务处于 failed → 健康（供 /soul_health 判 degraded）。"""
        return all(s.status != STATUS_FAILED for s in self._states.values())

    def describe(self) -> str:
        """给看板用的一行汇总。"""
        parts = [
            f"{s.name}={s.status}" + (f"(重启{s.restart_count})" if s.restart_count else "")
            for s in self._states.values()
        ]
        return "后台任务: " + " ".join(parts)

    # ── 状态迁移 ───────────────────────────────────────────────────

    def note_started(self, name: str) -> None:
        """任务已启动（含重启成功）。"""
        state = self.state_of(name)
        state.status = STATUS_RUNNING
        state.last_change_at = time.time()

    def note_stopped(self, name: str) -> None:
        """任务被**主动**停止（配置关闭/卸载）——不计入异常重启。"""
        state = self.state_of(name)
        state.status = STATUS_STOPPED
        state.last_change_at = time.time()

    def note_death(self, name: str, *, reason: str = "") -> None:
        """任务**意外**结束（异常或提前 return）。"""
        state = self.state_of(name)
        state.restart_count += 1
        state.last_death_reason = str(reason)[:300]
        state.status = (
            STATUS_FAILED
            if state.restart_count >= self._max_restarts
            else STATUS_RESTARTING
        )
        state.last_change_at = time.time()

    def reset(self, name: str) -> None:
        """清空某任务的重启计数（例如配置变更后由操作者重新启用）。"""
        state = self.state_of(name)
        state.restart_count = 0
        state.last_death_reason = ""
        state.status = STATUS_STOPPED
        state.last_change_at = time.time()
