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
        },
        "thought_cabinet": {
            "enabled": ConfigField(type=bool, default=False, description="启用思维阁系统（默认关闭）"),
            "max_seeds": ConfigField(type=int, default=20, description="思维种子上限"),
            "min_trigger_intensity": ConfigField(type=float, default=0.7, description="最小触发强度"),
            "admin_notification_enabled": ConfigField(type=bool, default=True, description="启用管理员审核通知"),
        },
        "api": {
            "enabled": ConfigField(type=bool, default=True, description="启用 Soul HTTP API（/api/v1/soul/*）"),
            "token": ConfigField(
                type=str,
                default="",
                description="访问令牌（可选）。前端请求头：X-Soul-Token；也可用环境变量 SOUL_API_TOKEN 覆盖",
            ),
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
        from .components.thought_commands import (
            SeedListCommand,
            SeedApproveCommand,
            SeedRejectCommand,
            TraitListCommand,
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
            (SeedListCommand.get_command_info(), SeedListCommand),
            (SeedApproveCommand.get_command_info(), SeedApproveCommand),
            (SeedRejectCommand.get_command_info(), SeedRejectCommand),
            (TraitListCommand.get_command_info(), TraitListCommand),
            (TraitDisableCommand.get_command_info(), TraitDisableCommand),
            (TraitEnableCommand.get_command_info(), TraitEnableCommand),
            (TraitDeleteCommand.get_command_info(), TraitDeleteCommand),
        ]
