from typing import List, Tuple, Type
from src.plugin_system import (
    BasePlugin,
    register_plugin,
    ComponentInfo,
    ConfigField,
)


@register_plugin
class WorldviewPlugin(BasePlugin):
    plugin_name: str = "MaiBot_Soul_Engine"
    enable_plugin: bool = True
    dependencies: List[str] = []
    python_dependencies: List[str] = []
    config_file_name: str = "config.toml"

    config_section_descriptions = {
        "admin": "管理员配置",
        "evolution": "演化设置",
        "monitor": "监控范围",
        "threshold": "档位阈值",
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
            "threshold_mild": ConfigField(type=int, default=25, description="轻微倾向阈值"),
            "threshold_moderate": ConfigField(type=int, default=50, description="明显倾向阈值"),
            "threshold_extreme": ConfigField(type=int, default=75, description="极端倾向阈值"),
            "custom_prompts": ConfigField(type=dict, default={}, description="自定义提示词模板(覆盖默认)"),
        },
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        from .components.setup_command import SetupCommand
        from .components.status_command import StatusCommand
        from .components.reset_command import ResetCommand
        from .components.ideology_injector import IdeologyInjector
        from .components.evolution_task import EvolutionTaskHandler

        return [
            (SetupCommand.get_command_info(), SetupCommand),
            (StatusCommand.get_command_info(), StatusCommand),
            (ResetCommand.get_command_info(), ResetCommand),
            (IdeologyInjector.get_handler_info(), IdeologyInjector),
            (EvolutionTaskHandler.get_handler_info(), EvolutionTaskHandler),
        ]
