"""Mai-Soul-Engine 插件入口 — maibot_sdk 2.x 版本。

通过聊天塑造 MaiBot 三观的人格底座：
- 意识形态光谱（四维）初始化 + 自动演化
- 思维阁（种子审核 → 内化 → 固化 trait）
- 回复注入（通过 maisaka.planner.before_request Hook）
- Soul 数据 @API 组件（供 WebUI / 其他插件调用）
"""

from __future__ import annotations

import asyncio
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
        self._evolution_task: asyncio.Task | None = None
        self._notion_sync_task: asyncio.Task | None = None
        # 问卷会话状态：{session_key: {current, answers, started_at}}
        self._questionnaire_sessions: dict[str, dict[str, Any]] = {}
        # P1 缓存：避免每条消息重建 WorldviewConfigView 和 WorldviewService
        self._wv_config_view: WorldviewConfigView | None = None
        self._wv_service: WorldviewService | None = None

    # ===== 生命周期 =====

    async def on_load(self) -> None:
        """插件加载：初始化数据库、执行旧版迁移、启动周期任务。"""
        from .models.ideology_model import init_db
        from .utils.audit_log import init_audit_log
        from .migration.legacy_import import run_legacy_import

        self._data_dir.mkdir(parents=True, exist_ok=True)

        # 初始化插件自有 SQLite
        soul_db_path = self._data_dir / "soul.db"
        init_db(soul_db_path)
        logger.info("[Mai-Soul-Engine] 数据库已初始化: %s", soul_db_path)

        # 初始化审计日志
        init_audit_log(self._plugin_dir)

        # 旧版数据迁移
        project_root = self._plugin_dir.parent.parent
        try:
            run_legacy_import(self._data_dir, project_root)
        except Exception as e:
            logger.error("[Mai-Soul-Engine] 旧版数据迁移失败: %s", e, exc_info=True)

        # 启动演化任务
        if self.config.evolution.evolution_enabled:
            self._evolution_task = asyncio.create_task(self._evolution_loop())
            logger.info("[Mai-Soul-Engine] 演化任务已启动")

        # 启动 Notion 同步任务
        if self.config.notion.enabled:
            self._notion_sync_task = asyncio.create_task(self._notion_sync_loop())
            logger.info("[Mai-Soul-Engine] Notion 同步任务已启动")

        # 初始化 P1 缓存
        self._wv_config_view = config_from_plugin(self)
        self._wv_service = WorldviewService(self._wv_config_view)

    async def on_unload(self) -> None:
        """插件卸载：取消周期任务、关闭数据库。"""
        from .models.ideology_model import close_db

        if self._evolution_task is not None:
            self._evolution_task.cancel()
            try:
                await self._evolution_task
            except asyncio.CancelledError:
                pass
            self._evolution_task = None

        if self._notion_sync_task is not None:
            self._notion_sync_task.cancel()
            try:
                await self._notion_sync_task
            except asyncio.CancelledError:
                pass
            self._notion_sync_task = None

        close_db()
        logger.info("[Mai-Soul-Engine] 插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        """处理配置热重载。"""
        if scope == "self":
            # 配置已自动注入到 self.config，这里只需处理需要即时响应的变更
            logger.info("[Mai-Soul-Engine] 配置已更新 (version=%s)", version)
            # 刷新 P1 缓存
            self._wv_config_view = config_from_plugin(self)
            self._wv_service = WorldviewService(self._wv_config_view)
            # 演化任务启停
            if self.config.evolution.evolution_enabled and self._evolution_task is None:
                self._evolution_task = asyncio.create_task(self._evolution_loop())
            elif not self.config.evolution.evolution_enabled and self._evolution_task is not None:
                self._evolution_task.cancel()
                self._evolution_task = None
            # Notion 同步任务启停
            if self.config.notion.enabled and self._notion_sync_task is None:
                self._notion_sync_task = asyncio.create_task(self._notion_sync_loop())
            elif not self.config.notion.enabled and self._notion_sync_task is not None:
                self._notion_sync_task.cancel()
                self._notion_sync_task = None

    # ===== 周期任务 =====

    async def _evolution_loop(self) -> None:
        """演化循环 — 委托到 evolution_task 模块。"""
        from .components.evolution_task import run_evolution_loop

        await run_evolution_loop(self)

    async def _notion_sync_loop(self) -> None:
        """Notion 同步循环 — 委托到 notion_sync 模块。"""
        from .components.notion_sync import run_notion_sync_loop

        await run_notion_sync_loop(self)

    # ===== HookHandler：意识形态注入 =====

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

    # ===== Command：问卷初始化 =====

    @Command("soul_setup", description="初始化灵魂光谱问卷（管理员私聊）", pattern=r"^/soul_setup\s*$")
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

    @Command("soul_reset", description="重置意识形态光谱", pattern=r"^/soul_reset\s*$")
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

    @Command("soul_seed", description="查看单个思维种子详情（管理员）", pattern=r"^/soul_seed\s+(\w+)\s*$")
    async def cmd_soul_seed(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """查看单个种子详情。"""
        from .components.thought_commands import handle_seed_detail

        return await handle_seed_detail(self, stream_id, **kwargs)

    @Command("soul_approve", description="批准思维种子内化（管理员）", pattern=r"^/soul_approve\s+(\w+)\s*$")
    async def cmd_soul_approve(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """批准种子内化。"""
        from .components.thought_commands import handle_seed_approve

        return await handle_seed_approve(self, stream_id, **kwargs)

    @Command("soul_reject", description="拒绝并删除思维种子（管理员）", pattern=r"^/soul_reject\s+(\w+)\s*$")
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

    @Command("soul_trait", description="查看单个 trait 详情（管理员）", pattern=r"^/soul_trait\s+(\w+)\s*$")
    async def cmd_soul_trait(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """查看单个 trait 详情。"""
        from .components.thought_commands import handle_trait_detail

        return await handle_trait_detail(self, stream_id, **kwargs)

    @Command("soul_trait_set_tags", description="设置 trait 的 tags（管理员）", pattern=r"^/soul_trait_set_tags\s+(\w+)\s+(.+?)\s*$")
    async def cmd_soul_trait_set_tags(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """设置 trait tags。"""
        from .components.thought_commands import handle_trait_set_tags

        return await handle_trait_set_tags(self, stream_id, **kwargs)

    @Command("soul_trait_merge", description="合并两个 trait（管理员）", pattern=r"^/soul_trait_merge\s+(\w+)\s+(\w+)\s*$")
    async def cmd_soul_trait_merge(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """合并 traits。"""
        from .components.thought_commands import handle_trait_merge

        return await handle_trait_merge(self, stream_id, **kwargs)

    @Command("soul_trait_disable", description="禁用指定 trait（管理员）", pattern=r"^/soul_trait_disable\s+(\w+)\s*$")
    async def cmd_soul_trait_disable(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """禁用 trait。"""
        from .components.thought_commands import handle_trait_disable

        return await handle_trait_disable(self, stream_id, **kwargs)

    @Command("soul_trait_enable", description="启用指定 trait（管理员）", pattern=r"^/soul_trait_enable\s+(\w+)\s*$")
    async def cmd_soul_trait_enable(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """启用 trait。"""
        from .components.thought_commands import handle_trait_enable

        return await handle_trait_enable(self, stream_id, **kwargs)

    @Command("soul_trait_delete", description="删除指定 trait（管理员，软删除）", pattern=r"^/soul_trait_delete\s+(\w+)\s*$")
    async def cmd_soul_trait_delete(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """删除 trait。"""
        from .components.thought_commands import handle_trait_delete

        return await handle_trait_delete(self, stream_id, **kwargs)

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
        from .utils.trait_evidence import parse_evidence_json

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
                    "evidence_count": len(parse_evidence_json(t.evidence_json)),
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
        """手动设置光谱数值。"""
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

        spectrum = get_or_create_spectrum("global")
        return {
            "success": True,
            "status": "ok",
            "spectrum_initialized": spectrum.initialized,
            "pending_seeds": count_pending_thought_seeds(),
            "evolution_running": self._evolution_task is not None and not self._evolution_task.done(),
            "notion_sync_running": self._notion_sync_task is not None and not self._notion_sync_task.done(),
        }


# ─── 工厂函数 ───────────────────────────────────────────────────────


def create_plugin() -> MaiSoulEnginePlugin:
    """创建 Mai-Soul-Engine 插件实例。"""
    return MaiSoulEnginePlugin()
