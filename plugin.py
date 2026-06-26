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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Iterable

from maibot_sdk import API, Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase
from maibot_sdk.types import HookMode, HookOrder, ErrorPolicy

logger = logging.getLogger(__name__)

# ─── 配置模型 ───────────────────────────────────────────────────────


class AdminConfig(PluginConfigBase):
    """管理员配置。"""

    __ui_label__ = "管理员"
    __ui_icon__ = "shield"
    __ui_order__ = 0

    admin_user_id: str = Field(default="", description="管理员用户ID（必填，格式：平台:ID，如 qq:12345678）")
    enabled: bool = Field(default=True, description="启用插件")


class EvolutionConfig(PluginConfigBase):
    """演化设置。"""

    __ui_label__ = "演化"
    __ui_icon__ = "trending-up"
    __ui_order__ = 1

    evolution_enabled: bool = Field(default=True, description="启用自动演化")
    evolution_interval_hours: float = Field(default=1.0, description="演化周期(小时)")
    evolution_rate: int = Field(default=5, description="每次演化最大变化值")
    ema_alpha: float = Field(default=0.3, description="EMA平滑系数(0-1，越大变化越快)")
    direction_resistance: float = Field(default=0.5, description="反向变动阻力(0-1，越大阻力越强)")
    max_messages_per_analysis: int = Field(default=200, description="每次分析的最大消息数")
    max_chars_per_message: int = Field(default=200, description="每条消息的最大字符数")


class MonitorConfig(PluginConfigBase):
    """监控范围。"""

    __ui_label__ = "监控"
    __ui_icon__ = "eye"
    __ui_order__ = 2

    monitored_groups: list[str] = Field(default_factory=list, description="监控的群ID列表(空=不分析)，格式：平台:群号:group")
    excluded_groups: list[str] = Field(default_factory=list, description="排除的群ID列表")
    monitored_users: list[str] = Field(default_factory=list, description="监控的用户ID列表(空=全部)")
    excluded_users: list[str] = Field(default_factory=list, description="排除的用户ID列表")


class ThresholdConfig(PluginConfigBase):
    """档位阈值。"""

    __ui_label__ = "阈值"
    __ui_icon__ = "gauge"
    __ui_order__ = 3

    enable_extreme: bool = Field(default=False, description="启用极端档位(98-100触发)")
    custom_prompts: dict[str, Any] = Field(default_factory=dict, description="自定义提示词模板(覆盖默认)")


class InjectionConfig(PluginConfigBase):
    """注入设置。"""

    __ui_label__ = "注入"
    __ui_icon__ = "syringe"
    __ui_order__ = 4

    scope: str = Field(default="global", description="群聊注入范围：global=所有群，monitored_only=仅 monitored_groups")
    inject_private: bool = Field(default=True, description="是否允许私聊注入")
    max_traits: int = Field(default=3, description="每次注入最多携带的 trait 数量")
    fallback_recent_impact: bool = Field(default=True, description="无 tags 命中时是否 fallback 注入最近影响最大的 traits")
    trait_cooldown_seconds: int = Field(default=180, description="trait 冷却时间（秒）")


class ThoughtCabinetConfig(PluginConfigBase):
    """思维阁设置。"""

    __ui_label__ = "思维阁"
    __ui_icon__ = "brain"
    __ui_order__ = 5

    enabled: bool = Field(default=False, description="启用思维阁系统（默认关闭）")
    max_seeds: int = Field(default=20, description="思维种子上限")
    min_trigger_intensity: float = Field(default=0.7, description="最小触发强度")
    admin_notification_enabled: bool = Field(default=True, description="启用管理员审核通知")
    auto_dedup_enabled: bool = Field(default=True, description="自动合并相似 trait")
    auto_dedup_threshold: float = Field(default=0.78, description="自动去重阈值(0-1)")


class ApiConfig(PluginConfigBase):
    """API 设置。"""

    __ui_label__ = "API"
    __ui_icon__ = "key"
    __ui_order__ = 6

    enabled: bool = Field(default=True, description="启用 Soul @API 组件")
    token: str = Field(default="", description="访问令牌（可选），也可用环境变量 SOUL_API_TOKEN 覆盖")
    public_mode: bool = Field(default=False, description="公共展示模式：脱敏敏感字段")


class NotionConfig(PluginConfigBase):
    """Notion 前端展示（可选）。"""

    __ui_label__ = "Notion"
    __ui_icon__ = "book-open"
    __ui_order__ = 7

    enabled: bool = Field(default=False, description="启用 Notion 数据库同步")
    token: str = Field(default="", description="Notion Integration Token，也可用环境变量 MAIBOT_SOUL_NOTION_TOKEN")
    database_id: str = Field(default="", description="Notion traits 数据库 ID")
    sync_spectrum: bool = Field(default=True, description="是否同步光谱到 Notion")
    spectrum_database_id: str = Field(default="", description="Notion 光谱数据库 ID")
    spectrum_scope_id: str = Field(default="global", description="光谱记录 scope_id")
    spectrum_mode: str = Field(default="dimension_rows", description="光谱同步模式：dimension_rows / single_row")
    sync_interval_seconds: int = Field(default=600, description="同步间隔（秒，最小 60）")
    first_delay_seconds: int = Field(default=5, description="启动后首次同步延迟（秒）")
    max_traits: int = Field(default=200, description="单次最多同步的 trait 数量")
    visibility_default: str = Field(default="Public", description="新建 trait 时默认 Visibility 值")
    never_overwrite_user_fields: bool = Field(default=True, description="永不覆盖用户可编辑字段")
    max_rich_text_chars: int = Field(default=1800, description="写入 Notion 的长文本最大长度")
    property_title: str = Field(default="Name", description="数据库 Title 字段名")
    property_trait_id: str = Field(default="TraitId", description="trait_id 字段名")
    property_tags: str = Field(default="Tags", description="tags 字段名")
    property_question: str = Field(default="Question", description="question 字段名")
    property_thought: str = Field(default="Thought", description="thought 字段名")
    property_confidence: str = Field(default="Confidence", description="confidence 字段名")
    property_impact_score: str = Field(default="ImpactScore", description="impact_score 字段名")
    property_status: str = Field(default="Status", description="status 字段名")
    property_visibility: str = Field(default="Visibility", description="visibility 字段名")
    property_updated_at: str = Field(default="UpdatedAt", description="updated_at 字段名")
    spectrum_property_title: str = Field(default="Name", description="光谱数据库 Title 字段名")
    spectrum_property_scope_id: str = Field(default="ScopeId", description="光谱 scope_id 字段名")
    spectrum_property_economic: str = Field(default="Economic", description="光谱 economic 字段名")
    spectrum_property_social: str = Field(default="Social", description="光谱 social 字段名")
    spectrum_property_diplomatic: str = Field(default="Diplomatic", description="光谱 diplomatic 字段名")
    spectrum_property_progressive: str = Field(default="Progressive", description="光谱 progressive 字段名")
    spectrum_property_value: str = Field(default="Value", description="光谱 Value 字段名（dimension_rows 模式）")
    spectrum_property_initialized: str = Field(default="Initialized", description="光谱 initialized 字段名")
    spectrum_property_last_evolution: str = Field(default="LastEvolution", description="光谱 last_evolution 字段名")
    spectrum_property_updated_at: str = Field(default="UpdatedAt", description="光谱 updated_at 字段名")


class MaiSoulEngineConfig(PluginConfigBase):
    """Mai-Soul-Engine 插件配置。"""

    admin: AdminConfig = Field(default_factory=AdminConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    thought_cabinet: ThoughtCabinetConfig = Field(default_factory=ThoughtCabinetConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    notion: NotionConfig = Field(default_factory=NotionConfig)


# ─── 插件类 ─────────────────────────────────────────────────────────


class MaiSoulEnginePlugin(MaiBotPlugin):
    """Mai-Soul-Engine 插件 — 通过聊天塑造 MaiBot 三观。"""

    config_model = MaiSoulEngineConfig

    def __init__(self) -> None:
        super().__init__()
        self._plugin_dir: Path = Path(__file__).parent
        self._data_dir: Path = self._plugin_dir / "data"
        self._evolution_task: asyncio.Task | None = None
        self._notion_sync_task: asyncio.Task | None = None
        # 问卷会话状态：{session_key: {current, answers, started_at}}
        self._questionnaire_sessions: dict[str, dict[str, Any]] = {}

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

    # ===== Command：trait 管理 =====

    @Command("soul_traits", description="查看已固化的 traits（管理员，可按群过滤）", pattern=r"^/soul_traits(?:\s+(\S+))?\s*$")
    async def cmd_soul_traits(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, bool]:
        """查看 traits 列表。"""
        from .components.thought_commands import handle_traits_list

        return await handle_traits_list(self, stream_id, **kwargs)

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
                "economic": spectrum.economic,
                "social": spectrum.social,
                "diplomatic": spectrum.diplomatic,
                "progressive": spectrum.progressive,
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
                        "economic": r.economic_delta,
                        "social": r.social_delta,
                        "diplomatic": r.diplomatic_delta,
                        "progressive": r.progressive_delta,
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
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
                for t in traits
            ],
        }

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

        spectrum = get_or_create_spectrum("global")
        if economic is not None:
            spectrum.economic = max(0, min(100, economic))
        if social is not None:
            spectrum.social = max(0, min(100, social))
        if diplomatic is not None:
            spectrum.diplomatic = max(0, min(100, diplomatic))
        if progressive is not None:
            spectrum.progressive = max(0, min(100, progressive))
        spectrum.updated_at = datetime.now()
        spectrum.save()

        return {
            "success": True,
            "spectrum": {
                "economic": spectrum.economic,
                "social": spectrum.social,
                "diplomatic": spectrum.diplomatic,
                "progressive": spectrum.progressive,
            },
        }

    @API("soul.health", description="Soul 引擎健康检查", version="1", public=False)
    async def api_health(self, **kwargs: Any) -> dict[str, Any]:
        """健康检查。"""
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
