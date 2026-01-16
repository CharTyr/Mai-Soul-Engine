from typing import Optional, Tuple
from src.plugin_system import BaseCommand


class StatusCommand(BaseCommand):
    command_name = "soul_status"
    command_description = "查看当前意识形态光谱状态"
    command_pattern = r"^/soul_status$"

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..models.ideology_model import get_or_create_spectrum, init_tables
        from ..utils.spectrum_utils import format_spectrum_display

        init_tables()
        spectrum = get_or_create_spectrum("global")

        if not spectrum.initialized:
            return True, "灵魂光谱尚未初始化，请管理员使用 /soul_setup 进行初始化", 2

        spectrum_dict = {
            "economic": spectrum.economic,
            "social": spectrum.social,
            "diplomatic": spectrum.diplomatic,
            "progressive": spectrum.progressive,
        }

        display = format_spectrum_display(spectrum_dict)
        last_update = spectrum.updated_at.strftime("%Y-%m-%d %H:%M:%S") if spectrum.updated_at else "未知"

        return True, f"当前灵魂光谱：\n\n{display}\n\n上次更新: {last_update}", 2
