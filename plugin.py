from typing import List, Tuple, Type
from src.plugin_system import (
    BasePlugin,
    register_plugin,
    ComponentInfo,
    ConfigField,
)


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
        "thought_cabinet": "思维阁设置",
    }

    config_schema: dict = {
        "admin": {
            "admin_user_id": ConfigField(
                type=str, default="", description="管理员用户ID（必填，格式：平台:ID，如qq:768295235）"
            ),
            "enabled": ConfigField(type=bool, default=True, description="启用插件"),
            "initialized": ConfigField(type=bool, default=False, description="是否已完成初始化问卷"),
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
                type=list, default=[], description="监控的群ID列表(空=全部)，格式：平台:ID:private/group"
            ),
            "excluded_groups": ConfigField(
                type=list, default=[], description="排除的群ID列表，格式：平台:ID:private/group"
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
        "thought_cabinet": {
            "enabled": ConfigField(type=bool, default=False, description="启用思维阁系统（默认关闭）"),
            "max_seeds": ConfigField(type=int, default=20, description="思维种子上限"),
            "min_trigger_intensity": ConfigField(type=float, default=0.7, description="最小触发强度"),
            "admin_notification_enabled": ConfigField(type=bool, default=True, description="启用管理员审核通知"),
        },
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        from .components.setup_command import SetupCommand, SetupAnswerHandler
        from .components.status_command import StatusCommand
        from .components.reset_command import ResetCommand
        from .components.ideology_injector import IdeologyInjector
        from .components.evolution_task import EvolutionTaskHandler
        from .components.thought_commands import SeedListCommand, SeedApproveCommand, SeedRejectCommand

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
        ]
