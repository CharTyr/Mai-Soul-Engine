"""问卷初始化命令模块 — maibot_sdk 2.x 版本。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

SESSION_TIMEOUT_MINUTES = 30


def cleanup_expired_sessions(plugin: Any) -> None:
    """清理过期的问卷会话。"""
    now = datetime.now()
    expired = [
        k
        for k, v in plugin._questionnaire_sessions.items()
        if now - v["started_at"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    ]
    for k in expired:
        del plugin._questionnaire_sessions[k]


async def handle_setup(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """初始化灵魂光谱问卷（管理员私聊）。

    从 kwargs 中提取 message 信息，不走旧 BaseCommand 的 self.message。
    """
    from ..questions.setup_questions import QUESTIONS
    from ..utils.spectrum_utils import match_user

    cleanup_expired_sessions(plugin)

    admin_user_id = plugin.config.admin.admin_user_id
    if not admin_user_id:
        msg = "请先在配置文件中设置 admin_user_id（格式：平台:ID，如 qq:12345678）"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # 从 kwargs 中获取平台和用户信息
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以执行此命令"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    session_key = f"{platform}:{user_id}"
    if session_key in plugin._questionnaire_sessions:
        session = plugin._questionnaire_sessions[session_key]
        msg = f"问卷进行中，当前第{session['current'] + 1}题，请使用 /soul_answer <1-5> 作答"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    plugin._questionnaire_sessions[session_key] = {
        "current": 0,
        "answers": [],
        "started_at": datetime.now(),
    }

    q = QUESTIONS[0]
    msg = f"灵魂光谱问卷开始！共20题，请使用 /soul_answer <1-5> 作答。\n\n第1题：{q['text']}"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


async def handle_answer(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """处理问卷回答。

    使用 kwargs 中的 matched_groups 获取答案，不使用旧 self.matched_groups。
    """
    from ..questions.setup_questions import QUESTIONS, calculate_initial_spectrum
    from ..utils.spectrum_utils import format_spectrum_display, match_user
    from ..utils.audit_log import log_init

    cleanup_expired_sessions(plugin)

    admin_user_id = plugin.config.admin.admin_user_id
    # 从 kwargs 中获取平台和用户信息
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))
    session_key = f"{platform}:{user_id}"

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以进行问卷答题"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if session_key not in plugin._questionnaire_sessions:
        msg = "当前没有进行中的问卷，请先使用 /soul_setup 开始"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    session = plugin._questionnaire_sessions[session_key]
    matched_groups = kwargs.get("matched_groups", {})
    answer = int(matched_groups.get("answer", "0") or 0)
    session["answers"].append(answer)
    session["current"] += 1

    if session["current"] >= len(QUESTIONS):
        spectrum_values = calculate_initial_spectrum(session["answers"])
        from ..models.ideology_model import get_or_create_spectrum

        spectrum = get_or_create_spectrum("global")
        spectrum.sincerity = spectrum_values["sincerity"]
        spectrum.engagement = spectrum_values["engagement"]
        spectrum.closeness = spectrum_values["closeness"]
        spectrum.directness = spectrum_values["directness"]
        spectrum.initialized = True
        spectrum.updated_at = datetime.now()
        spectrum.save()

        del plugin._questionnaire_sessions[session_key]

        await log_init(session_key, spectrum_values)

        display = format_spectrum_display(spectrum_values)
        msg = f"问卷完成！初始灵魂光谱：\n\n{display}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    q = QUESTIONS[session["current"]]
    msg = f"第{session['current'] + 1}题：{q['text']}"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
