from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path
from src.plugin_system import BaseCommand

questionnaire_sessions: dict = {}


class SetupCommand(BaseCommand):
    command_name = "worldview_setup"
    command_description = "初始化灵魂光谱问卷（管理员私聊）"
    command_pattern = r"^/worldview_setup$"

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..questions.setup_questions import QUESTIONS, calculate_initial_spectrum
        from ..models.ideology_model import get_or_create_spectrum, init_tables
        from ..utils.spectrum_utils import format_spectrum_display, match_user
        from ..utils.audit_log import init_audit_log

        plugin_dir = Path(__file__).parent.parent
        init_audit_log(plugin_dir)

        admin_user_id = self.get_config("admin_user_id", "")
        if not admin_user_id:
            return True, "请先在配置文件中设置 admin_user_id（格式：平台:ID，如qq:768295235）", 2

        platform = getattr(self.message, "platform", "")
        user_id = str(getattr(self.message, "user_id", ""))

        if not match_user(platform, user_id, admin_user_id):
            return True, "只有管理员可以执行此命令", 2

        init_tables()

        session_key = f"{platform}:{user_id}"
        if session_key in questionnaire_sessions:
            session = questionnaire_sessions[session_key]
            return True, f"问卷进行中，当前第{session['current'] + 1}题，请回复1-5", 2

        questionnaire_sessions[session_key] = {
            "current": 0,
            "answers": [],
            "started_at": datetime.now(),
        }

        q = QUESTIONS[0]
        return True, f"灵魂光谱问卷开始！共20题，请回复1-5分。\n\n第1题：{q['text']}", 2


class SetupAnswerHandler(BaseCommand):
    command_name = "worldview_answer"
    command_description = "处理问卷回答"
    command_pattern = r"^[1-5]$"

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..questions.setup_questions import QUESTIONS, calculate_initial_spectrum
        from ..models.ideology_model import get_or_create_spectrum, init_tables
        from ..utils.spectrum_utils import format_spectrum_display, match_user
        from ..utils.audit_log import log_init

        admin_user_id = self.get_config("admin_user_id", "")
        platform = getattr(self.message, "platform", "")
        user_id = str(getattr(self.message, "user_id", ""))
        session_key = f"{platform}:{user_id}"

        if not match_user(platform, user_id, admin_user_id) or session_key not in questionnaire_sessions:
            return False, None, 0

        session = questionnaire_sessions[session_key]
        content = getattr(self.message, "content", "")
        if hasattr(content, "get_plain_text"):
            content = content.get_plain_text()
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
            return True, f"问卷完成！初始灵魂光谱：\n\n{display}", 2

        q = QUESTIONS[session["current"]]
        return True, f"第{session['current'] + 1}题：{q['text']}", 2
