from typing import List, Tuple, Type
import logging
from src.plugin_system import (
    BasePlugin,
    register_plugin,
    ComponentInfo,
    ConfigField,
)

logger = logging.getLogger(__name__)


@register_plugin
class MaiSoulEngine(BasePlugin):
    plugin_name: str = "Mai-Soul-Engine"
    enable_plugin: bool = True
    dependencies: List[str] = []
    python_dependencies: List[str] = []
    config_file_name: str = "config.toml"

    config_section_descriptions = {
        "admin": "管理员配置",
        "evolution": "演化设置",
        "monitor": "监控范围",
        "threshold": "档位阈值",
        "injection": "注入设置",
        "thought_cabinet": "思维阁设置",
        "api": "API 设置",
        "notion": "Notion 前端展示（可选）",
    }

    config_schema: dict = {
        "admin": {
            "admin_user_id": ConfigField(
                type=str, default="", description="管理员用户ID（必填，格式：平台:ID，如qq:768295235）"
            ),
            "enabled": ConfigField(type=bool, default=True, description="启用插件"),
        },
        "evolution": {
            "evolution_enabled": ConfigField(type=bool, default=True, description="启用自动演化"),
            "evolution_interval_hours": ConfigField(type=float, default=1.0, description="演化周期(小时)"),
            "evolution_rate": ConfigField(type=int, default=5, description="每次演化最大变化值"),
            "ema_alpha": ConfigField(type=float, default=0.3, description="EMA平滑系数(0-1，越大变化越快)"),
            "direction_resistance": ConfigField(type=float, default=0.5, description="反向变动阻力(0-1，越大阻力越强)"),
            "max_messages_per_analysis": ConfigField(type=int, default=200, description="每次分析的最大消息数"),
            "max_chars_per_message": ConfigField(type=int, default=200, description="每条消息的最大字符数"),
        },
        "monitor": {
            "monitored_groups": ConfigField(
                type=list,
                default=[],
                description="监控的群ID列表(空=不分析任何群)，格式：平台:群号:group（或直接填stream_id）",
            ),
            "excluded_groups": ConfigField(
                type=list, default=[], description="排除的群ID列表，格式：平台:群号:group（或直接填stream_id）"
            ),
            "monitored_users": ConfigField(
                type=list, default=[], description="监控的用户ID列表(空=全部)，格式：平台:ID"
            ),
            "excluded_users": ConfigField(type=list, default=[], description="排除的用户ID列表，格式：平台:ID"),
        },
        "threshold": {
            "enable_extreme": ConfigField(type=bool, default=False, description="启用极端档位(98-100触发)"),
            "custom_prompts": ConfigField(type=dict, default={}, description="自定义提示词模板(覆盖默认)"),
        },
        "injection": {
            "scope": ConfigField(
                type=str,
                default="global",
                description="群聊注入范围：global=所有群，monitored_only=仅 monitored_groups（仍受 excluded_groups 影响）",
            ),
            "inject_private": ConfigField(type=bool, default=True, description="是否允许私聊注入（默认开启）"),
            "max_traits": ConfigField(type=int, default=3, description="每次注入最多携带的 trait 数量"),
            "fallback_recent_impact": ConfigField(
                type=bool,
                default=True,
                description="当无 tags 命中时，是否 fallback 注入“最近影响最大”的 traits（避免完全空窗）",
            ),
            "trait_cooldown_seconds": ConfigField(
                type=int,
                default=180,
                description="trait 冷却时间（秒）。冷却期间同一 trait 不会被重复注入（避免刷屏）",
            ),
        },
        "thought_cabinet": {
            "enabled": ConfigField(type=bool, default=False, description="启用思维阁系统（默认关闭）"),
            "max_seeds": ConfigField(type=int, default=20, description="思维种子上限"),
            "min_trigger_intensity": ConfigField(type=float, default=0.7, description="最小触发强度"),
            "admin_notification_enabled": ConfigField(type=bool, default=True, description="启用管理员审核通知"),
            "auto_dedup_enabled": ConfigField(
                type=bool, default=True, description="自动合并相似 trait（去重，避免同义 trait 越积越多）"
            ),
            "auto_dedup_threshold": ConfigField(
                type=float, default=0.78, description="自动去重阈值（0-1，越高越严格）"
            ),
        },
        "api": {
            "enabled": ConfigField(type=bool, default=True, description="启用 Soul HTTP API（/api/v1/soul/*）"),
            "token": ConfigField(
                type=str,
                default="",
                description="访问令牌（可选）。前端请求头：X-Soul-Token；也可用环境变量 SOUL_API_TOKEN 覆盖",
            ),
            "public_mode": ConfigField(
                type=bool,
                default=False,
                description="公共展示模式：对外展示时减少/脱敏敏感字段（targets、evidence、注入细节等）",
            ),
        },
        "notion": {
            "enabled": ConfigField(type=bool, default=False, description="启用 Notion 数据库同步（公共展示前端，可选）"),
            "token": ConfigField(
                type=str,
                default="",
                description="Notion Integration Token（可选）。也可用环境变量 MAIBOT_SOUL_NOTION_TOKEN 覆盖",
            ),
            "database_id": ConfigField(type=str, default="", description="Notion 数据库 ID（复制链接取 32 位 ID）"),
            "sync_interval_seconds": ConfigField(type=int, default=600, description="同步间隔（秒，最小 60）"),
            "first_delay_seconds": ConfigField(type=int, default=5, description="启动后首次同步延迟（秒）"),
            "max_traits": ConfigField(type=int, default=200, description="单次最多同步的 trait 数量（按创建时间倒序）"),
            "visibility_default": ConfigField(type=str, default="Public", description="新建 trait 时默认 Visibility 值"),
            "never_overwrite_user_fields": ConfigField(
                type=bool,
                default=True,
                description="永不覆盖用户可编辑字段（Name/Question/Thought/Visibility）。仅在新建时写入",
            ),
            "max_rich_text_chars": ConfigField(type=int, default=1800, description="写入 Notion 的长文本最大长度（字符）"),
            "property_title": ConfigField(type=str, default="Name", description="数据库 Title 字段名"),
            "property_trait_id": ConfigField(type=str, default="TraitId", description="trait_id 字段名（rich_text）"),
            "property_tags": ConfigField(type=str, default="Tags", description="tags 字段名（multi_select）"),
            "property_question": ConfigField(type=str, default="Question", description="question 字段名（rich_text）"),
            "property_thought": ConfigField(type=str, default="Thought", description="thought 字段名（rich_text）"),
            "property_confidence": ConfigField(type=str, default="Confidence", description="confidence 字段名（number）"),
            "property_impact_score": ConfigField(type=str, default="ImpactScore", description="impact_score 字段名（number）"),
            "property_status": ConfigField(type=str, default="Status", description="status 字段名（select）"),
            "property_visibility": ConfigField(type=str, default="Visibility", description="visibility 字段名（select）"),
            "property_updated_at": ConfigField(type=str, default="UpdatedAt", description="updated_at 字段名（date）"),
        },
    }

    def register_plugin(self) -> bool:
        ok = super().register_plugin()
        if not ok:
            return False

        try:
            from .webui.http_api import create_soul_api_router
            from src.common.server import get_global_server

            router = create_soul_api_router(self)

            server = get_global_server()
            core_app = server.get_app()
            if not getattr(core_app.state, "soul_engine_api_registered", False):
                server.register_router(router)
                core_app.state.soul_engine_api_registered = True
                logger.info("[Mai-Soul-Engine] 已注册 Soul API 到 Core Server")

        except Exception as e:
            logger.error("[Mai-Soul-Engine] Soul API 注册失败: %s", e, exc_info=True)

        return True

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        from .components.setup_command import SetupCommand, SetupAnswerHandler
        from .components.status_command import StatusCommand
        from .components.reset_command import ResetCommand
        from .components.ideology_injector import IdeologyInjector
        from .components.evolution_task import EvolutionTaskHandler
        from .components.notion_frontend_sync import NotionFrontendSyncTask
        from .components.thought_commands import (
            SeedListCommand,
            SeedApproveCommand,
            SeedRejectCommand,
            TraitListCommand,
            TraitSetTagsCommand,
            TraitMergeCommand,
            TraitDisableCommand,
            TraitEnableCommand,
            TraitDeleteCommand,
        )

        return [
            (SetupCommand.get_command_info(), SetupCommand),
            (SetupAnswerHandler.get_command_info(), SetupAnswerHandler),
            (StatusCommand.get_command_info(), StatusCommand),
            (ResetCommand.get_command_info(), ResetCommand),
            (IdeologyInjector.get_handler_info(), IdeologyInjector),
            (EvolutionTaskHandler.get_handler_info(), EvolutionTaskHandler),
            (NotionFrontendSyncTask.get_handler_info(), NotionFrontendSyncTask),
            (SeedListCommand.get_command_info(), SeedListCommand),
            (SeedApproveCommand.get_command_info(), SeedApproveCommand),
            (SeedRejectCommand.get_command_info(), SeedRejectCommand),
            (TraitListCommand.get_command_info(), TraitListCommand),
            (TraitSetTagsCommand.get_command_info(), TraitSetTagsCommand),
            (TraitMergeCommand.get_command_info(), TraitMergeCommand),
            (TraitDisableCommand.get_command_info(), TraitDisableCommand),
            (TraitEnableCommand.get_command_info(), TraitEnableCommand),
            (TraitDeleteCommand.get_command_info(), TraitDeleteCommand),
        ]
