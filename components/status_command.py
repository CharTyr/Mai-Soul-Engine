from typing import Optional, Tuple
from src.plugin_system import BaseCommand
from src.plugin_system.apis import send_api


class StatusCommand(BaseCommand):
    command_name = "soul_status"
    command_description = "查看当前意识形态光谱状态"
    command_pattern = r"^/soul_status\s*$"

    async def _send_response(self, text: str):
        """发送响应消息到聊天"""
        if self.message.chat_stream:
            await send_api.text_to_stream(text, self.message.chat_stream.stream_id, typing=False, storage_message=False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..models.ideology_model import get_or_create_spectrum, init_tables
        from ..utils.spectrum_utils import format_spectrum_display

        init_tables()
        spectrum = get_or_create_spectrum("global")

        if not spectrum.initialized:
            msg = "灵魂光谱尚未初始化，请管理员使用 /soul_setup 进行初始化"
            await self._send_response(msg)
            return True, msg, 2

        spectrum_dict = {
            "economic": spectrum.economic,
            "social": spectrum.social,
            "diplomatic": spectrum.diplomatic,
            "progressive": spectrum.progressive,
        }

        display = format_spectrum_display(spectrum_dict)
        last_update = spectrum.updated_at.strftime("%Y-%m-%d %H:%M:%S") if spectrum.updated_at else "未知"

        msg = f"当前灵魂光谱：\n\n{display}\n\n上次更新: {last_update}"
        await self._send_response(msg)
        return True, msg, 2
