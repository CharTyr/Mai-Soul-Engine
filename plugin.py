"""Mai-Soul-Engine 插件入口 — maibot_sdk 2.x 版本。

通过聊天塑造 MaiBot 三观的人格底座：
- 意识形态光谱（四维）初始化 + 自动演化
- 思维阁（种子审核 → 内化 → 固化 trait）
- 回复注入（通过 maisaka.planner.before_request Hook）
- Soul 数据 @API 组件（供 WebUI / 其他插件调用）
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Iterable

from maibot_sdk import API, Command, HookHandler, MaiBotPlugin
from maibot_sdk.types import HookMode, HookOrder, ErrorPolicy

from .plugin_ui_schema import CONFIG_VERSION, MaiSoulEngineConfig
from .worldview.service import WorldviewConfigView, WorldviewService, config_from_plugin

logger = logging.getLogger(__name__)

# ─── 配置模型见 plugin_ui_schema.py（WebUI label/hint）──────────────


# ─── 插件类 ─────────────────────────────────────────────────────────


class MaiSoulEnginePlugin(MaiBotPlugin):
    """Mai-Soul-Engine 插件 — 通过聊天塑造 MaiBot 三观。"""

    config_model = MaiSoulEngineConfig

    def normalize_plugin_config(
        self, config_data: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], bool]:
        """补齐 [plugin].config_version，兼容旧版无该节的 config.toml。"""
        raw_config = deepcopy(dict(config_data)) if isinstance(config_data, Mapping) else {}
        plugin_section = raw_config.get("plugin")
        changed = False
        if not isinstance(plugin_section, dict):
            admin_section = raw_config.get("admin")
            soul_enabled = True
            if isinstance(admin_section, dict) and "enabled" in admin_section:
                soul_enabled = bool(admin_section.pop("enabled"))
                changed = True
            raw_config["plugin"] = {"enabled": soul_enabled, "config_version": CONFIG_VERSION}
            changed = True
        else:
            if not str(plugin_section.get("config_version", "") or "").strip():
                plugin_section["config_version"] = CONFIG_VERSION
                changed = True
        base_normalized, base_changed = super().normalize_plugin_config(raw_config)
        return base_normalized, changed or base_changed or base_normalized != raw_config

    def __init__(self) -> None:
        super().__init__()
        self._plugin_dir: Path = Path(__file__).parent
        self._data_dir: Path = self._plugin_dir / "data"
        self._data_dir_source: str = "plugin_dir"
        self._data_dir_info: dict | None = None  # P1.4: 数据目录解析/迁移详情
        # 后台任务监督器：区分「在跑」与「已死」，防止任务静默停摆
        from .utils.task_supervisor import TaskSupervisor

        self._task_supervisor = TaskSupervisor()
        self._evolution_task: asyncio.Task | None = None
        self._notion_sync_task: asyncio.Task | None = None
        self._self_reflection_task: asyncio.Task | None = None
        self._fermentation_task: asyncio.Task | None = None  # v2.4.0 发酵循环
        # 内化队列消费者：命令只入队，实际内化在这里跑（避免命令 RPC 超时）
        self._internalization_task: asyncio.Task | None = None
        # 问卷会话状态：{session_key: {current, answers, started_at}}
        self._questionnaire_sessions: dict[str, dict[str, Any]] = {}
        # P1 缓存：避免每条消息重建 WorldviewConfigView 和 WorldviewService
        self._wv_config_view: WorldviewConfigView | None = None
        self._wv_service: WorldviewService | None = None
        # /soul_reset 二次确认：{session_id: timestamp}
        self._reset_confirm_ts: dict[str, float] = {}
        # 任务巡检：崩溃不必等配置热更就能被发现（见 _supervise_background_tasks）
        self._supervisor_task: asyncio.Task | None = None
        self._unloading = False

    # ===== 后台任务管理 =====

    @staticmethod
    def _compute_desired_tasks(config: MaiSoulEngineConfig) -> dict[str, bool]:
        """纯函数：根据配置计算四个后台任务的期望状态（便于单测）。"""
        # 学习类任务由运行模式的 learning_enabled 闸门控制：
        # off 全停；observe/apply 才按各自开关启动（observe 只学习、不改人格）。
        from .utils.runtime_mode import resolve_runtime_mode

        master = resolve_runtime_mode(config).learning_enabled
        return {
            "evolution": master and bool(config.evolution.evolution_enabled),
            "notion": master and bool(config.notion.enabled),
            "reflection": master and bool(config.self_reflection.enabled),
            "fermentation": master
            and bool(config.thought_cabinet.enabled)
            and bool(getattr(config.thought_cabinet, "fermentation_enabled", False)),
            # 队列消费者：思维阁开着就需要（管理员批准后要靠它执行内化）
            "internalization": master and bool(config.thought_cabinet.enabled),
        }

    @staticmethod
    def _task_action(current: Any, should_run: bool) -> str:
        """后台任务的期望动作：start / restart / stop / keep（纯函数，便于单测）。

        关键：`current is not None` **不等于**任务还活着。协程抛异常退出后
        Task 对象仍在，旧实现据此认为"运行中"，导致任务静默停摆且不重启。
        """
        if should_run:
            if current is None:
                return "start"
            if current.done():
                return "restart"
            return "keep"
        return "stop" if current is not None else "keep"

    async def _reconcile_background_tasks(self) -> None:
        """统一管理后台任务生命周期（on_load / on_config_update 共用）。

        根据 _compute_desired_tasks 的结果，启动缺失的任务、重启已死亡的任务、
        停止多余的任务，并把状态记入监督器供 /soul_health 展示。
        """
        desired = self._compute_desired_tasks(self.config)

        for key, attr_name, loop_attr in self._TASK_ENTRIES:
            current = getattr(self, attr_name)
            action = self._task_action(current, desired[key])

            if action == "keep":
                continue

            if action == "stop":
                current.cancel()
                try:
                    await current
                except (asyncio.CancelledError, Exception):
                    pass
                setattr(self, attr_name, None)
                self._task_supervisor.note_stopped(key)
                logger.info("[Mai-Soul-Engine] %s 任务已停止", attr_name)
                continue

            if action == "restart":
                # 任务意外结束：记录原因，判断是否还允许重启
                reason = ""
                try:
                    exc = current.exception()
                    reason = f"{type(exc).__name__}: {exc}" if exc else "任务提前结束"
                except (asyncio.CancelledError, asyncio.InvalidStateError):
                    reason = "任务被取消"
                self._task_supervisor.note_death(key, reason=reason)
                setattr(self, attr_name, None)
                if not self._task_supervisor.should_restart(key):
                    logger.error(
                        "[Mai-Soul-Engine] %s 连续异常结束已达上限，转为 failed，"
                        "请人工检查后重启插件（原因: %s）",
                        attr_name, reason,
                    )
                    continue
                logger.warning(
                    "[Mai-Soul-Engine] %s 意外结束（%s），正在重启", attr_name, reason,
                )

            loop_fn = getattr(self, loop_attr)
            if not self._task_supervisor.start_allowed(key):
                logger.error(
                    "[Mai-Soul-Engine] %s 已因连续异常停止重试，跳过启动；"
                    "请人工检查后重启插件", attr_name,
                )
                continue
            setattr(self, attr_name, asyncio.create_task(loop_fn()))
            self._task_supervisor.note_started(key)
            logger.info("[Mai-Soul-Engine] %s 任务已启动", attr_name)

    # 巡检间隔（秒）：任务崩溃后最多这么久被发现
    SUPERVISOR_INTERVAL_SECONDS = 30.0

    _TASK_ENTRIES: list[tuple[str, str, str]] = [
        ("evolution", "_evolution_task", "_evolution_loop"),
        ("notion", "_notion_sync_task", "_notion_sync_loop"),
        ("reflection", "_self_reflection_task", "_self_reflection_loop"),
        ("fermentation", "_fermentation_task", "_fermentation_loop"),
        ("internalization", "_internalization_task", "_internalization_loop"),
    ]

    async def _supervise_background_tasks(self) -> None:
        """**一次巡检**：发现意外结束的任务 → 记死亡 → 退避到期后重启。

        与 `_reconcile_background_tasks` 的分工：那个是「配置驱动的对账」
        （只在加载/热更时跑），这个是「时间驱动的巡检」——任务崩溃不会等
        配置热更才发生，没有巡检就会一直假绿。

        卸载期间（``_unloading``）**禁止重新拉起任务**。
        """
        if self._unloading:
            return

        desired = self._compute_desired_tasks(self.config)

        for key, attr_name, loop_attr in self._TASK_ENTRIES:
            current = getattr(self, attr_name)

            if current is not None and not current.done():
                continue  # 还活着

            if current is not None and current.done():
                # 意外结束（非我们主动 cancel）→ 记录死亡，退避后由后续巡检重启
                reason = "任务提前结束"
                if current.cancelled():
                    reason = "任务被取消"
                else:
                    try:
                        exc = current.exception()
                        if exc is not None:
                            reason = f"{type(exc).__name__}: {exc}"
                    except asyncio.InvalidStateError:
                        pass
                self._task_supervisor.note_death(key, reason=reason)
                setattr(self, attr_name, None)
                logger.warning(
                    "[Mai-Soul-Engine] 巡检发现 %s 意外结束（%s），将按退避重试",
                    attr_name, reason,
                )
                continue

            if not desired[key]:
                continue
            if not self._task_supervisor.start_allowed(key):
                continue
            if not self._task_supervisor.ready_to_restart(key):
                # 退避未到期：状态在**发现死亡那一轮**已经记为 backoff，
                # 这里只等待。再调一次 note_death 会把一次死亡记成两次、
                # 重启计数虚高，最后把任务误判成 failed。
                continue

            loop_fn = getattr(self, loop_attr)
            setattr(self, attr_name, asyncio.create_task(loop_fn()))
            self._task_supervisor.note_started(key)
            logger.info("[Mai-Soul-Engine] 巡检已拉起 %s", attr_name)

    async def _supervisor_loop(self) -> None:
        """周期巡检循环（自身崩溃不拖垮插件）。"""
        while True:
            try:
                await asyncio.sleep(self.SUPERVISOR_INTERVAL_SECONDS)
                await self._supervise_background_tasks()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — 巡检自身异常必须吞掉后继续
                logger.exception("[Mai-Soul-Engine] 任务巡检异常（继续下一轮）")

    async def _stop_all_background_tasks(self) -> None:
        """停止所有后台任务（卸载用；测试也用它收尾）。"""
        for _key, attr_name, _loop in self._TASK_ENTRIES:
            task = getattr(self, attr_name)
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 — 收尾时不应再抛
                logger.debug("[Mai-Soul-Engine] %s 停止时异常（已忽略）", attr_name)
            setattr(self, attr_name, None)
        if self._supervisor_task is not None:
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                logger.debug("[Mai-Soul-Engine] 巡检任务停止时异常（已忽略）")
            self._supervisor_task = None

    async def on_load(self) -> None:
        """插件加载：初始化数据库、执行旧版迁移、启动周期任务。"""
        from .models.ideology_model import init_db
        from .utils.audit_log import init_audit_log
        from .migration.legacy_import import run_legacy_import

        # ── 数据目录解析与迁移（在 init_db 之前，此时 soul.db 未打开）─
        from .utils.data_dir import resolve_and_prepare_data_dir

        dir_info = resolve_and_prepare_data_dir(self)
        self._data_dir = dir_info["data_dir"]
        self._data_dir_source = dir_info.get("source", "plugin_dir")
        self._data_dir_info = dir_info  # P1.4: 整包存，供健康命令读取
        logger.info(
            "[Mai-Soul-Engine] 数据目录: %s (source=%s migrated=%s)",
            self._data_dir,
            self._data_dir_source,
            dir_info.get("migrated", False),
        )

        self._data_dir.mkdir(parents=True, exist_ok=True)

        # 审计日志开关（从配置读取，在 init_db 之前确保 degraded 模式也读 config）
        from .utils.audit_log import set_audit_enabled
        set_audit_enabled(getattr(self.config.admin, "audit_enabled", True))

        # P1 缓存（不依赖 DB，在 init_db 之前构造确保 degraded 模式也可用）
        self._wv_config_view = config_from_plugin(self)
        self._wv_service = WorldviewService(self._wv_config_view)

        # 初始化插件自有 SQLite
        soul_db_path = self._data_dir / "soul.db"
        try:
            init_db(soul_db_path)
            logger.info("[Mai-Soul-Engine] 数据库已初始化: %s", soul_db_path)
        except Exception as e:
            logger.error("[Mai-Soul-Engine] 数据库初始化失败，插件将以降级模式运行: %s", e, exc_info=True)
            return  # 不启动后台任务，但不让 SDK 崩

        # 初始化审计日志
        try:
            init_audit_log(self._plugin_dir)
        except Exception as e:
            logger.error("[Mai-Soul-Engine] 审计日志初始化失败: %s", e, exc_info=True)

        # 旧版数据迁移（带超时，防宿主 DB 锁住时卡 on_load）
        project_root = self._plugin_dir.parent.parent
        try:
            await asyncio.wait_for(
                asyncio.to_thread(run_legacy_import, self._data_dir, project_root),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.error("[Mai-Soul-Engine] 旧版数据迁移超时（30s），跳过")
        except Exception as e:
            logger.error("[Mai-Soul-Engine] 旧版数据迁移失败: %s", e, exc_info=True)

        # 启动周期任务（统一管理）
        await self._reconcile_background_tasks()

        # 重启后补发上次未送出的管理员通知
        try:
            from .utils.notify import drain_notifications

            stats = await drain_notifications(self)
            if stats["sent"] or stats["failed"]:
                logger.info(
                    "[Mai-Soul-Engine] 启动补发通知: 发出 %s / 待重试 %s / 放弃 %s",
                    stats["sent"], stats["retry"], stats["failed"],
                )
        except Exception as e:  # noqa: BLE001 — 补发失败不影响启动
            logger.warning("[Mai-Soul-Engine] 启动补发通知失败: %s: %s", type(e).__name__, e)

        # 任务巡检：崩溃不必等配置热更才被发现（这里是唯一的自动恢复入口）
        self._unloading = False
        if self._supervisor_task is None or self._supervisor_task.done():
            self._supervisor_task = asyncio.create_task(self._supervisor_loop())

    async def on_unload(self) -> None:
        """插件卸载：取消周期任务、清模块级状态、关闭数据库。

        **每一步独立兜底**：单个清理步骤失败不得中断其余步骤。卸载不完整是
        「重启后状态诡异」的常见来源（库没关 → 连接泄漏；状态没清 → 旧冷却
        串到新实例），而失败原因必须留在日志里，不能无声跳过。
        """
        # 先挡住巡检：否则这里刚 cancel，巡检又把任务拉起来（卸载期最典型的竞态）
        self._unloading = True

        async def _step(label: str, fn: Any) -> None:
            try:
                result = fn()
                if inspect.isawaitable(result):
                    await result
            except Exception as e:  # noqa: BLE001 — 清理步骤的失败必须被隔离
                logger.warning(
                    "[Mai-Soul-Engine] 卸载清理步骤「%s」失败，继续其余步骤: %s: %s",
                    label, type(e).__name__, e,
                )

        async def _cancel_task(attr_name: str) -> None:
            task = getattr(self, attr_name, None)
            if task is None:
                return
            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001 — 任务自身异常不该阻断卸载
                logger.warning(
                    "[Mai-Soul-Engine] 取消 %s 时任务报错: %s: %s",
                    attr_name, type(e).__name__, e,
                )
            finally:
                setattr(self, attr_name, None)

        # 巡检先停：它是唯一会重新拉起任务的地方
        await _step("停止任务巡检", lambda: _cancel_task("_supervisor_task"))

        for attr_name in (
            "_evolution_task",
            "_notion_sync_task",
            "_self_reflection_task",
            "_fermentation_task",
            "_internalization_task",
        ):
            await _step(f"取消 {attr_name}", lambda a=attr_name: _cancel_task(a))

        # 清模块级可变状态，防插件重载间泄漏
        async def _clear_module_state() -> None:
            from .components.ideology_injector import _RECENT_TRAIT_INJECTION

            _RECENT_TRAIT_INJECTION.clear()

        async def _clear_context_cache() -> None:
            from .components.reflection_capture import _context_cache

            _context_cache.clear()

        async def _clear_reply_tail_cache() -> None:
            from .components.reflection_capture import _reply_tail_cache

            _reply_tail_cache.clear()

        async def _reset_evolution_state() -> None:
            from .components.evolution_task import _bot_filter_warned, reset_aggregation_state

            _bot_filter_warned.clear()
            reset_aggregation_state()

        await _step("清注入冷却表", _clear_module_state)
        await _step("清自评上下文缓存", _clear_context_cache)
        await _step("清回复配对证据缓存", _clear_reply_tail_cache)
        await _step("清演化聚合状态", _reset_evolution_state)

        async def _close_database() -> None:
            from .models.ideology_model import close_db

            close_db()

        await _step("关闭数据库", _close_database)

        logger.info("[Mai-Soul-Engine] 插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        """处理配置热重载。"""
        if scope == "self":
            # 配置已自动注入到 self.config，这里只需处理需要即时响应的变更
            logger.info("[Mai-Soul-Engine] 配置已更新 (version=%s)", version)
            # 刷新 P1 缓存
            self._wv_config_view = config_from_plugin(self)
            self._wv_service = WorldviewService(self._wv_config_view)
            # 审计日志开关同步
            from .utils.audit_log import set_audit_enabled
            set_audit_enabled(getattr(self.config.admin, "audit_enabled", True))
            # 清注入冷却表（配置热更后旧冷却状态可能与新 max_traits/cooldown_seconds 不匹配）
            from .components.ideology_injector import _RECENT_TRAIT_INJECTION
            _RECENT_TRAIT_INJECTION.clear()
            # 统一管理后台任务启停（含发酵补上）
            await self._reconcile_background_tasks()

    # ===== 周期任务 =====

    async def _evolution_loop(self) -> None:
        """演化循环 — 委托到 evolution_task 模块。"""
        from .components.evolution_task import run_evolution_loop

        await run_evolution_loop(self)

    async def _notion_sync_loop(self) -> None:
        """Notion 同步循环 — 委托到 notion_sync 模块。"""
        from .components.notion_sync import run_notion_sync_loop

        await run_notion_sync_loop(self)

    async def _self_reflection_loop(self) -> None:
        """自我评价循环 — 委托到 reflection_evaluator 模块。"""
        from .components.reflection_evaluator import run_reflection_loop

        await run_reflection_loop(self)

    async def _internalization_loop(self) -> None:
        """内化队列消费者 — 按间隔执行排队中的内化操作。"""
        from .thought.internalization_queue import run_queue_once

        interval = 15.0
        try:
            interval = max(
                5.0,
                float(getattr(self.config.thought_cabinet, "internalization_check_interval_seconds", 15)),
            )
        except (AttributeError, TypeError, ValueError):
            pass
        logger.info("[Mai-Soul-Engine] 内化队列消费者已启动，间隔 %.0f 秒", interval)

        while True:
            try:
                self._task_supervisor.note_waiting("internalization", reason="等待下一步队列消费")
                await asyncio.sleep(interval)
                stats = await run_queue_once(self)
                if stats["done"] or stats["retry"]:
                    logger.info(
                        "[Mai-Soul-Engine] 内化队列: 完成 %s / 待重试 %s / 跳过 %s",
                        stats["done"], stats["retry"], stats["skipped"],
                    )
            except asyncio.CancelledError:
                logger.info("[Mai-Soul-Engine] 内化队列消费者已停止")
                raise
            except Exception:
                logger.exception("[Mai-Soul-Engine] 内化队列消费异常，下轮重试")

    async def _fermentation_loop(self) -> None:
        """v2.4.0 发酵循环 — 委托到 fermentation_engine 模块。"""
        from .thought.fermentation_engine import run_fermentation_loop

        await run_fermentation_loop(self)

    # ===== HookHandler：意识形态注入 =====

    @HookHandler(
        "maisaka.replyer.before_model_request",
        name="soul_replyer_injector",
        description="replyer 请求前注入「本次观点 + 表达倾向」视图（分用途投递）",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=3000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def soul_replyer_injector(self, **kwargs: Any) -> dict[str, Any]:
        """分用途投递（方案 §4.1）：replyer 只收观点与表达倾向。

        与 planner 视图的差异：
        - 内容：不含分层摘要/图谱/自评自查（那是决策材料）
        - 副作用：**不落快照、不打冷却**（快照锚点与冷却都属于 planner 的
          before_request；replyer 也写会让「同会话多快照」恒真、把歧义判定打满）

        宿主每次重试都会调用本 hook，且每次传入重建的 items，因此
        「每次调用各注入一次」是正确的；`append_block_to_first_system`
        自身按标记幂等，重入不会叠加。
        """
        from .components.ideology_injector import inject_ideology

        # 配对内容证据：自评开着时，先记下本轮回复的触发消息（尾行）。
        # 放在开关判断**之前**——即使管理员只想配对不要 replyer 注入，
        # 这条证据也该记；只依赖 self_reflection.enabled。
        if self.config.self_reflection.enabled:
            try:
                from .components.reflection_capture import cache_reply_tail

                cache_reply_tail(
                    str(kwargs.get("session_id", "") or ""),
                    str(kwargs.get("reply_message_id", "") or ""),
                    kwargs.get("items") or [],
                )
            except Exception:  # noqa: BLE001 — 证据记录失败不得影响注入
                logger.debug("[Soul] 记录回复配对证据失败（忽略）")

        if not self.config.injection.replyer_injection_enabled:
            return {"success": True, "action": "continue"}
        return await inject_ideology(self, _purpose="replyer", **kwargs)

    @HookHandler(
        "maisaka.planner.before_request",
        name="soul_ideology_injector",
        description="在 planner 请求前注入意识形态光谱提示词与相关 trait",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=3000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def hook_ideology_inject(self, **kwargs: Any) -> dict[str, Any]:
        """在 planner 发起 LLM 请求前注入意识形态提示词。"""
        from .components.ideology_injector import inject_ideology

        return await inject_ideology(self, **kwargs)

    # ===== HookHandler：自我评价捕获（OBSERVE，不改写输出）=====

    @HookHandler(
        "maisaka.replyer.after_response",
        name="soul_reflection_replyer_capture",
        description="捕获 replyer 最终回复供自我评价（OBSERVE，不改写）",
        mode=HookMode.OBSERVE,
        order=HookOrder.NORMAL,
        error_policy=ErrorPolicy.SKIP,
    )
    async def hook_reflection_replyer_after(self, **kwargs: Any) -> dict[str, Any]:
        """replyer 回复后捕获最终文本入待评队列。"""
        from .components.reflection_capture import capture_after_response

        return await capture_after_response(self, "replyer", **kwargs)

    # ===== Command：问卷初始化 =====

    @Command("soul_setup", description="初始化灵魂光谱问卷（管理员私聊）", pattern=r"^/soul_setup(?:\s+(?P<flags>--\w+(?:\s+--\w+)*))?\s*$")
    async def cmd_soul_setup(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """初始化灵魂光谱问卷。"""
        from .components.setup_command import handle_setup

        return await handle_setup(self, stream_id, **kwargs)

    @Command("soul_answer", description="问卷答题：/soul_answer <1-5>", pattern=r"^/soul_answer\s+(?P<answer>[1-5])\s*$")
    async def cmd_soul_answer(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """处理问卷回答。"""
        from .components.setup_command import handle_answer

        return await handle_answer(self, stream_id, **kwargs)

    # ===== Command：状态查看 =====

    @Command("soul_status", description="查看当前意识形态光谱状态", pattern=r"^/soul_status\s*$")
    async def cmd_soul_status(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """查看当前灵魂光谱。"""
        from .components.status_command import handle_status

        return await handle_status(self, stream_id, **kwargs)

    @Command("soul_dashboard", description="查看 Soul 引擎全状态可视化卡片", pattern=r"^/soul_dashboard\s*$")
    async def cmd_soul_dashboard(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """Soul 引擎全状态卡片。"""
        from .components.dashboard_command import handle_dashboard

        return await handle_dashboard(self, stream_id, **kwargs)

    @Command("soul_inspect", description="预览文本会命中哪些 trait（管理员，不实际注入）", pattern=r"^/soul_inspect\s+(.+)\s*$")
    async def cmd_soul_inspect(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """注入命中预览。"""
        from .components.inspect_command import handle_inspect

        return await handle_inspect(self, stream_id, **kwargs)

    # ===== Command：重置 =====

    @Command("soul_reset", description="重置意识形态光谱（需二次确认）", pattern=r"^/soul_reset(?:\s+confirm)?\s*$")
    async def cmd_soul_reset(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """重置灵魂光谱。"""
        from .components.reset_command import handle_reset

        return await handle_reset(self, stream_id, **kwargs)

    # ===== Command：思维阁种子管理 =====

    @Command("soul_seeds", description="查看待审核的思维种子（管理员）", pattern=r"^/soul_seeds\s*$")
    async def cmd_soul_seeds(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """查看待审核种子。"""
        from .components.thought_commands import handle_seeds_list

        return await handle_seeds_list(self, stream_id, **kwargs)

    @Command(
        "soul_op",
        description="查看内化队列/操作状态（管理员）",
        pattern=r"^/soul_op(?:\s+([\w-]{8,}))?\s*$",
    )
    async def cmd_soul_op(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """查看内化操作状态。"""
        from .components.thought_commands import handle_op_status

        return await handle_op_status(self, stream_id, **kwargs)

    @Command("soul_seed", description="查看单个思维种子详情（管理员）", pattern=r"^/soul_seed\s+([\w-]{8,})\s*$")
    async def cmd_soul_seed(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """查看单个种子详情。"""
        from .components.thought_commands import handle_seed_detail

        return await handle_seed_detail(self, stream_id, **kwargs)

    @Command("soul_approve", description="批准思维种子内化（管理员）", pattern=r"^/soul_approve\s+([\w-]{8,})\s*$")
    async def cmd_soul_approve(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """批准种子内化。"""
        from .components.thought_commands import handle_seed_approve

        return await handle_seed_approve(self, stream_id, **kwargs)

    @Command("soul_reject", description="拒绝并删除思维种子（管理员）", pattern=r"^/soul_reject\s+([\w-]{8,})\s*$")
    async def cmd_soul_reject(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """拒绝种子。"""
        from .components.thought_commands import handle_seed_reject

        return await handle_seed_reject(self, stream_id, **kwargs)

    @Command("soul_reject_all", description="批量拒绝所有待审核思维种子（管理员）", pattern=r"^/soul_reject_all\s*$")
    async def cmd_soul_reject_all(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """批量拒绝所有待审核种子。"""
        from .components.thought_commands import handle_seed_reject_all

        return await handle_seed_reject_all(self, stream_id, **kwargs)

    # ===== Command：trait 管理 =====

    @Command("soul_traits", description="查看已固化的 traits（管理员，可按群过滤）", pattern=r"^/soul_traits(?:\s+(\S+))?\s*$")
    async def cmd_soul_traits(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """查看 traits 列表。"""
        from .components.thought_commands import handle_traits_list

        return await handle_traits_list(self, stream_id, **kwargs)

    @Command("soul_trait", description="查看单个 trait 详情（管理员）", pattern=r"^/soul_trait\s+([\w-]{8,})\s*$")
    async def cmd_soul_trait(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """查看单个 trait 详情。"""
        from .components.thought_commands import handle_trait_detail

        return await handle_trait_detail(self, stream_id, **kwargs)

    @Command("soul_trait_set_tags", description="设置 trait 的 tags（管理员）", pattern=r"^/soul_trait_set_tags\s+([\w-]{8,})\s+(.+?)\s*$")
    async def cmd_soul_trait_set_tags(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """设置 trait tags。"""
        from .components.thought_commands import handle_trait_set_tags

        return await handle_trait_set_tags(self, stream_id, **kwargs)

    @Command("soul_trait_merge", description="合并两个 trait（管理员）", pattern=r"^/soul_trait_merge\s+([\w-]{8,})\s+([\w-]{8,})\s*$")
    async def cmd_soul_trait_merge(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """合并 traits。"""
        from .components.thought_commands import handle_trait_merge

        return await handle_trait_merge(self, stream_id, **kwargs)

    @Command("soul_trait_disable", description="禁用指定 trait（管理员）", pattern=r"^/soul_trait_disable\s+([\w-]{8,})\s*$")
    async def cmd_soul_trait_disable(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """禁用 trait。"""
        from .components.thought_commands import handle_trait_disable

        return await handle_trait_disable(self, stream_id, **kwargs)

    @Command("soul_trait_enable", description="启用指定 trait（管理员）", pattern=r"^/soul_trait_enable\s+([\w-]{8,})\s*$")
    async def cmd_soul_trait_enable(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """启用 trait。"""
        from .components.thought_commands import handle_trait_enable

        return await handle_trait_enable(self, stream_id, **kwargs)

    @Command("soul_trait_delete", description="删除指定 trait（管理员，软删除）", pattern=r"^/soul_trait_delete\s+([\w-]{8,})\s*$")
    async def cmd_soul_trait_delete(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """删除 trait。"""
        from .components.thought_commands import handle_trait_delete

        return await handle_trait_delete(self, stream_id, **kwargs)

    # ===== Command：trait 槽位管理 =====

    @Command("soul_slot", description="设置 trait 思维阁槽位 1-12（管理员）",
             pattern=r"^/soul_slot(?:\s+([\w-]{8,})\s+(\d+|clear))?\s*$")
    async def cmd_soul_slot(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """管理 trait 槽位。"""
        from .components.thought_commands import handle_trait_slot

        return await handle_trait_slot(self, stream_id, **kwargs)

    @Command("soul_promote_global", description="将群锁 trait 提升为全局作用域（管理员）",
             pattern=r"^/soul_promote_global\s+([\w-]{8,})\s*$")
    async def cmd_soul_promote_global(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """提升 trait 为全局作用域。"""
        from .components.thought_commands import handle_promote_global

        return await handle_promote_global(self, stream_id, **kwargs)

    # ===== Command：自我评价 =====

    @Command("soul_reflect", description="查看近期自我评价记录（管理员）", pattern=r"^/soul_reflect(?:\s+(?P<count>\d+))?\s*$")
    async def cmd_soul_reflect(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """/soul_reflect [N] — 查看近期自我评价。"""
        from .components.reflection_command import handle_reflect

        return await handle_reflect(self, stream_id, **kwargs)

    @Command("soul_observe", description="Soul 运行观察摘要（管理员/开发）", pattern=r"^/soul_observe\s*$")
    async def cmd_soul_observe(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        from .components.observe_command import handle_observe

        return await handle_observe(self, stream_id, **kwargs)

    # ===== Command：帮助 =====

    @Command("soul_help", description="查看可用命令列表", pattern=r"^/soul_help\s*$")
    async def cmd_soul_help(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        from .components.help_command import handle_help

        return await handle_help(self, stream_id, **kwargs)

    # ===== Command：健康状态 =====

    @Command("soul_health", description="查看插件健康状态（管理员）", pattern=r"^/soul_health\s*$")
    async def cmd_soul_health(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        from .components.health_command import handle_health

        return await handle_health(self, stream_id, **kwargs)

    # ===== @API 组件：Soul 数据接口 =====
    #
    # 安全模型（以下 7 个 @API 组件）：
    # - 均为 SDK 级 @API(public=False) 组件，无网络暴露面（无 HTTP server/路由），
    #   仅 Runner 内可信组件（如 WebUI 或其他插件）可调用。
    # - 双层访问控制：@API(public=False)（SDK 层）+ api.enabled 配置守卫（默认关闭）。
    # - 唯一写接口 api_set_spectrum 同步记录审计日志（data/audit.jsonl）。
    # - 不自行实现网络级认证（无网络面）。

    @API("soul.get_spectrum", description="获取当前意识形态光谱", version="1", public=False)
    async def api_get_spectrum(self, **kwargs: Any) -> dict[str, Any]:
        """获取当前光谱状态。"""
        if not self.config.api.enabled:
            return {"success": False, "error": "Soul API 未启用（api.enabled=false）"}
        from .models.ideology_model import get_or_create_spectrum

        spectrum = get_or_create_spectrum("global")
        return {
            "success": True,
            "spectrum": {
                "sincerity": spectrum.sincerity,
                "engagement": spectrum.engagement,
                "closeness": spectrum.closeness,
                "directness": spectrum.directness,
                "initialized": spectrum.initialized,
                "last_evolution": spectrum.last_evolution.isoformat() if spectrum.last_evolution else None,
                "updated_at": spectrum.updated_at.isoformat() if spectrum.updated_at else None,
            },
        }

    @API("soul.get_evolution_history", description="获取演化历史", version="1", public=False)
    async def api_get_evolution_history(self, limit: int = 100, **kwargs: Any) -> dict[str, Any]:
        """获取演化历史记录。"""
        if not self.config.api.enabled:
            return {"success": False, "error": "Soul API 未启用（api.enabled=false）"}
        from .models.ideology_model import get_evolution_history

        records = get_evolution_history(limit=limit)
        return {
            "success": True,
            "history": [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "group_id": r.group_id,
                    "deltas": {
                        "sincerity": r.sincerity_delta,
                        "engagement": r.engagement_delta,
                        "closeness": r.closeness_delta,
                        "directness": r.directness_delta,
                    },
                    "reason": r.reason,
                }
                for r in records
            ],
        }

    @API("soul.get_traits", description="获取已固化的 traits 列表", version="1", public=False)
    async def api_get_traits(self, stream_id: str = "", limit: int = 50, **kwargs: Any) -> dict[str, Any]:
        """获取 traits 列表。"""
        if not self.config.api.enabled:
            return {"success": False, "error": "Soul API 未启用（api.enabled=false）"}
        from .models.ideology_model import query_crystallized_traits
        from .utils.trait_tags import parse_tags_json
        from .utils.trait_evidence import parse_trait_evidence_json

        traits = query_crystallized_traits(
            deleted=False,
            stream_id=stream_id or None,
            limit=limit,
        )
        return {
            "success": True,
            "traits": [
                {
                    "trait_id": t.trait_id,
                    "stream_id": t.stream_id,
                    "name": t.name,
                    "question": t.question,
                    "thought": t.thought,
                    "tags": parse_tags_json(t.tags_json),
                    "confidence": t.confidence,
                    "evidence_count": len(parse_trait_evidence_json(t.evidence_json)),
                    "enabled": t.enabled,
                    "ideology_layer": getattr(t, "ideology_layer", "conduct"),
                    "lifecycle_state": getattr(t, "lifecycle_state", "active"),
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
                for t in traits
            ],
        }

    @API("soul.get_worldview", description="获取 P1 三观分层/情绪/群切片摘要", version="1", public=False)
    async def api_get_worldview(self, stream_id: str = "", **kwargs: Any) -> dict[str, Any]:
        """P1 三观生长状态（dev）。"""
        if not self.config.api.enabled:
            return {"success": False, "error": "Soul API 未启用（api.enabled=false）"}
        from .worldview.service import WorldviewService, config_from_plugin

        payload = WorldviewService(config_from_plugin(self)).api_worldview_payload(stream_id=stream_id or "")
        return {"success": True, "worldview": payload}

    @API("soul.get_seeds", description="获取待审核的思维种子", version="1", public=False)
    async def api_get_seeds(self, stream_id: str = "", **kwargs: Any) -> dict[str, Any]:
        """获取待审核种子列表。"""
        if not self.config.api.enabled:
            return {"success": False, "error": "Soul API 未启用（api.enabled=false）"}
        from .models.ideology_model import get_pending_thought_seeds
        from .utils.evidence_utils import parse_evidence_json
        import json as _json

        seeds = get_pending_thought_seeds(stream_id=stream_id or None)
        return {
            "success": True,
            "seeds": [
                {
                    "seed_id": s.seed_id,
                    "stream_id": s.stream_id,
                    "type": s.seed_type,
                    "event": s.event,
                    "intensity": s.intensity / 100.0,
                    "confidence": s.confidence / 100.0,
                    "evidence": parse_evidence_json(s.evidence_json),
                    "reasoning": s.reasoning,
                    "potential_impact": _json.loads(s.potential_impact_json or "{}"),
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "status": s.status,
                }
                for s in seeds
            ],
        }

    @API("soul.set_spectrum", description="手动设置光谱数值", version="1", public=False)
    async def api_set_spectrum(
        self,
        economic: int | None = None,
        social: int | None = None,
        diplomatic: int | None = None,
        progressive: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """手动设置光谱数值。

        参数名映射（向后兼容别名 → 实际社交轴）：
        - ``economic`` → ``sincerity``（真诚）
        - ``social`` → ``engagement``（投入）
        - ``diplomatic`` → ``closeness``（亲近）
        - ``progressive`` → ``directness``（直率）

        新调用方建议直接使用新轴名（暂未开放，仍用旧别名）。
        """
        if not self.config.api.enabled:
            return {"success": False, "error": "Soul API 未启用（api.enabled=false）"}
        from .models.ideology_model import get_or_create_spectrum
        from .utils.audit_log import log_api_set_spectrum

        spectrum = get_or_create_spectrum("global")
        before = {
            "sincerity": spectrum.sincerity,
            "engagement": spectrum.engagement,
            "closeness": spectrum.closeness,
            "directness": spectrum.directness,
        }
        if economic is not None:
            spectrum.sincerity = max(0, min(100, economic))
        if social is not None:
            spectrum.engagement = max(0, min(100, social))
        if diplomatic is not None:
            spectrum.closeness = max(0, min(100, diplomatic))
        if progressive is not None:
            spectrum.directness = max(0, min(100, progressive))
        spectrum.updated_at = datetime.now()
        spectrum.save()

        after = {
            "sincerity": spectrum.sincerity,
            "engagement": spectrum.engagement,
            "closeness": spectrum.closeness,
            "directness": spectrum.directness,
        }
        await log_api_set_spectrum(before, after)

        return {
            "success": True,
            "spectrum": after,
        }

    @API("soul.health", description="Soul 引擎健康检查", version="1", public=False)
    async def api_health(self, **kwargs: Any) -> dict[str, Any]:
        """健康检查。"""
        if not self.config.api.enabled:
            return {"success": False, "error": "Soul API 未启用（api.enabled=false）"}
        from .models.ideology_model import get_or_create_spectrum, count_pending_thought_seeds
        from .components.ideology_injector import _injection_metrics
        from .utils.audit_log import read_recent_audit

        spectrum = get_or_create_spectrum("global")

        # ── 补充指标（try/except 包裹，失败返回 None） ──────────────
        pending_seeds_count: int | None = None
        try:
            pending_seeds_count = count_pending_thought_seeds()
        except Exception:
            pass

        db_size_bytes: int | None = None
        try:
            db_path = self._data_dir / "soul.db"
            if db_path.exists():
                db_size_bytes = db_path.stat().st_size
        except Exception:
            pass

        evolution_last_run: str | None = None
        try:
            recent = read_recent_audit(limit=50)
            for entry in recent:
                if entry.get("type") == "evolution_cycle":
                    evolution_last_run = entry.get("ts")
                    break
        except Exception:
            pass

        return {
            "success": True,
            "status": "ok",
            "spectrum_initialized": spectrum.initialized,
            "pending_seeds": pending_seeds_count,
            "db_size_bytes": db_size_bytes,
            "evolution_last_run": evolution_last_run,
            "injection_metrics": dict(_injection_metrics),
            "evolution_running": self._evolution_task is not None and not self._evolution_task.done(),
            "notion_sync_running": self._notion_sync_task is not None and not self._notion_sync_task.done(),
            "self_reflection_running": self._self_reflection_task is not None and not self._self_reflection_task.done(),
        }


# ─── 工厂函数 ───────────────────────────────────────────────────────


def create_plugin() -> MaiSoulEnginePlugin:
    """创建 Mai-Soul-Engine 插件实例。"""
    return MaiSoulEnginePlugin()
