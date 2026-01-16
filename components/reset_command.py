from typing import Optional, Tuple
from datetime import datetime
from src.plugin_system import BaseCommand


class ResetCommand(BaseCommand):
    command_name = "worldview_reset"
    command_description = "重置意识形态光谱"
    command_pattern = r"^/worldview_reset$"

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..models.ideology_model import get_or_create_spectrum, init_tables
        from ..utils.spectrum_utils import match_user

        admin_user_id = self.get_config("admin_user_id", "")
        platform = getattr(self.message, "platform", "")
        user_id = str(getattr(self.message, "user_id", ""))

        if not match_user(platform, user_id, admin_user_id):
            return True, "只有管理员可以重置灵魂光谱", 2

        init_tables()
        spectrum = get_or_create_spectrum("global")
        spectrum.economic = 50
        spectrum.social = 50
        spectrum.diplomatic = 50
        spectrum.progressive = 50
        spectrum.initialized = False
        spectrum.updated_at = datetime.now()
        spectrum.save()

        return True, "灵魂光谱已重置为中立状态，请使用 /worldview_setup 重新初始化", 2
