from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path
from src.plugin_system import BaseCommand
from src.plugin_system.apis import send_api


class ResetCommand(BaseCommand):
    command_name = "soul_reset"
    command_description = "重置灵魂光谱"
    command_pattern = r"^/soul_reset\s*$"

    async def _send_response(self, text: str):
        """发送响应消息到聊天"""
        if self.message.chat_stream:
            await send_api.text_to_stream(text, self.message.chat_stream.stream_id, typing=False, storage_message=False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..models.ideology_model import get_or_create_spectrum, init_tables
        from ..utils.spectrum_utils import match_user
        from ..utils.audit_log import log_reset, init_audit_log

        plugin_dir = Path(__file__).parent.parent
        init_audit_log(plugin_dir)

        admin_user_id = self.get_config("admin.admin_user_id", "")
        # 从 message_info 中正确获取平台和用户信息
        platform = self.message.message_info.platform if self.message.message_info else ""
        user_id = str(self.message.message_info.user_info.user_id) if self.message.message_info and self.message.message_info.user_info else ""

        if not match_user(platform, user_id, admin_user_id):
            msg = "只有管理员可以重置灵魂光谱"
            await self._send_response(msg)
            return True, msg, 2

        init_tables()
        spectrum = get_or_create_spectrum("global")
        spectrum.economic = 50
        spectrum.social = 50
        spectrum.diplomatic = 50
        spectrum.progressive = 50
        spectrum.initialized = False
        spectrum.updated_at = datetime.now()
        spectrum.save()

        await log_reset(f"{platform}:{user_id}")

        msg = "灵魂光谱已重置为中立状态，请使用 /soul_setup 重新初始化"
        await self._send_response(msg)
        return True, msg, 2
