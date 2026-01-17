from typing import Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from src.plugin_system import BaseCommand
from src.plugin_system.apis import send_api

questionnaire_sessions: dict = {}
SESSION_TIMEOUT_MINUTES = 30


def cleanup_expired_sessions():
    now = datetime.now()
    expired = [
        k
        for k, v in questionnaire_sessions.items()
        if now - v["started_at"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    ]
    for k in expired:
        del questionnaire_sessions[k]


class SetupCommand(BaseCommand):
    command_name = "soul_setup"
    command_description = "初始化灵魂光谱问卷（管理员私聊）"
    command_pattern = r"^/soul_setup\s*$"

    async def _send_response(self, text: str):
        """发送响应消息到聊天"""
        if self.message.chat_stream:
            await send_api.text_to_stream(text, self.message.chat_stream.stream_id, typing=False, storage_message=False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..questions.setup_questions import QUESTIONS
        from ..models.ideology_model import init_tables
        from ..utils.spectrum_utils import match_user
        from ..utils.audit_log import init_audit_log

        cleanup_expired_sessions()

        plugin_dir = Path(__file__).parent.parent
        init_audit_log(plugin_dir)

        admin_user_id = self.get_config("admin.admin_user_id", "")
        if not admin_user_id:
            msg = "请先在配置文件中设置 admin_user_id（格式：平台:ID，如qq:768295235）"
            await self._send_response(msg)
            return True, msg, 2

        # 从 message_info 中正确获取平台和用户信息
        platform = self.message.message_info.platform if self.message.message_info else ""
        user_id = str(self.message.message_info.user_info.user_id) if self.message.message_info and self.message.message_info.user_info else ""

        if not match_user(platform, user_id, admin_user_id):
            msg = "只有管理员可以执行此命令"
            await self._send_response(msg)
            return True, msg, 2

        init_tables()

        session_key = f"{platform}:{user_id}"
        if session_key in questionnaire_sessions:
            session = questionnaire_sessions[session_key]
            msg = f"问卷进行中，当前第{session['current'] + 1}题，请回复1-5"
            await self._send_response(msg)
            return True, msg, 2

        questionnaire_sessions[session_key] = {
            "current": 0,
            "answers": [],
            "started_at": datetime.now(),
        }

        q = QUESTIONS[0]
        msg = f"灵魂光谱问卷开始！共20题，请回复1-5分。\n\n第1题：{q['text']}"
        await self._send_response(msg)
        return True, msg, 2


class SetupAnswerHandler(BaseCommand):
    command_name = "soul_answer"
    command_description = "处理问卷回答"
    command_pattern = r"^[1-5]\s*$"

    async def _send_response(self, text: str):
        """发送响应消息到聊天"""
        if self.message.chat_stream:
            await send_api.text_to_stream(text, self.message.chat_stream.stream_id, typing=False, storage_message=False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..questions.setup_questions import QUESTIONS, calculate_initial_spectrum
        from ..models.ideology_model import get_or_create_spectrum, init_tables
        from ..utils.spectrum_utils import format_spectrum_display, match_user
        from ..utils.audit_log import log_init

        cleanup_expired_sessions()

        admin_user_id = self.get_config("admin.admin_user_id", "")
        # 从 message_info 中正确获取平台和用户信息
        platform = self.message.message_info.platform if self.message.message_info else ""
        user_id = str(self.message.message_info.user_info.user_id) if self.message.message_info and self.message.message_info.user_info else ""
        session_key = f"{platform}:{user_id}"

        if not match_user(platform, user_id, admin_user_id) or session_key not in questionnaire_sessions:
            return False, None, 0

        session = questionnaire_sessions[session_key]
        # 从 processed_plain_text 获取消息内容
        content = self.message.processed_plain_text if hasattr(self.message, "processed_plain_text") else ""
        answer = int(str(content).strip())
        session["answers"].append(answer)
        session["current"] += 1

        if session["current"] >= len(QUESTIONS):
            spectrum_values = calculate_initial_spectrum(session["answers"])
            init_tables()
            spectrum = get_or_create_spectrum("global")
            spectrum.economic = spectrum_values["economic"]
            spectrum.social = spectrum_values["social"]
            spectrum.diplomatic = spectrum_values["diplomatic"]
            spectrum.progressive = spectrum_values["progressive"]
            spectrum.initialized = True
            spectrum.updated_at = datetime.now()
            spectrum.save()

            del questionnaire_sessions[session_key]

            await log_init(session_key, spectrum_values)

            display = format_spectrum_display(spectrum_values)
            msg = f"问卷完成！初始灵魂光谱：\n\n{display}"
            await self._send_response(msg)
            return True, msg, 2

        q = QUESTIONS[session["current"]]
        msg = f"第{session['current'] + 1}题：{q['text']}"
        await self._send_response(msg)
        return True, msg, 2
