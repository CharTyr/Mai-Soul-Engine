"""问卷初始化命令模块 — maibot_sdk 2.x 版本（v2.3.0 断点续答交互优化）。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

SESSION_TIMEOUT_MINUTES = 30
TOTAL_QUESTIONS = 26

# 答案描述映射
ANSWER_LABELS = {
    1: "非常偏向左边",
    2: "稍微偏向左边",
    3: "看情况/不确定",
    4: "稍微偏向右边",
    5: "非常偏向右边",
}


def _progress_bar(answered: int, total: int = TOTAL_QUESTIONS) -> str:
    """生成进度条，如 9/26 → [█████████░░░░░░░░░░░░░░░░░]"""
    filled = answered
    empty = total - answered
    bar = "█" * filled + "░" * empty
    return f"[{bar}]"


def _format_question_text(q: dict, index: int) -> str:
    """格式化题目文本，去掉 direction 等元信息。"""
    return q["text"]


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
    支持 --restart 和 --restart --yes 参数。
    """
    from ..questions.setup_questions import QUESTIONS
    from ..utils.spectrum_utils import check_admin_permission

    cleanup_expired_sessions(plugin)

    admin_user_id = plugin.config.admin.admin_user_id
    if not admin_user_id:
        msg = "请先在配置文件中设置 admin_user_id（格式：平台:ID，如 qq:12345678）"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    ok, err = check_admin_permission(plugin, kwargs, "初始化灵魂光谱")
    if not ok:
        await plugin.ctx.send.text(err, stream_id)
        return True, err, True

    from ..utils.spectrum_utils import extract_command_actor
    platform, user_id = extract_command_actor(kwargs)
    session_key = f"{platform}:{user_id}"

    # 解析参数（从 matched_groups 获取 flags）
    matched_groups = kwargs.get("matched_groups", {}) or {}
    flags = (matched_groups.get("flags") or "").strip()
    has_restart = "--restart" in flags
    has_yes = "--yes" in flags

    # 检查是否有进行中的会话
    if session_key in plugin._questionnaire_sessions:
        session = plugin._questionnaire_sessions[session_key]
        answered = session["current"]

        if has_restart:
            if has_yes:
                # 确认重启：清空并开始新问卷
                del plugin._questionnaire_sessions[session_key]
                return await _start_new_questionnaire(plugin, stream_id, session_key)
            else:
                # 二次确认
                msg = (
                    "确定要重新开始吗？当前进度会被清空，无法恢复。\n\n"
                    "确认：/soul_setup --restart --yes\n"
                    "取消：/soul_setup --continue 或直接继续答题"
                )
                await plugin.ctx.send.text(msg, stream_id)
                return True, msg, True

        # 已有进行中会话，提示继续
        bar = _progress_bar(answered)
        msg = (
            f"你还有一份没答完的问卷。\n\n"
            f"{bar} {answered}/{TOTAL_QUESTIONS}\n\n"
            f"要继续就回答下一题，或者选择：\n"
            f"/soul_answer <1-5>  继续答题\n"
            f"/soul_setup --restart  重新开始（会清空当前进度）"
        )
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # 无进行中会话，开始新问卷
    return await _start_new_questionnaire(plugin, stream_id, session_key)


async def _start_new_questionnaire(plugin: Any, stream_id: str, session_key: str) -> tuple[bool, str, bool]:
    """开始新问卷。"""
    from ..questions.setup_questions import QUESTIONS

    plugin._questionnaire_sessions[session_key] = {
        "current": 0,
        "answers": [],
        "started_at": datetime.now(),
    }

    bar = _progress_bar(0)
    q = QUESTIONS[0]
    msg = (
        f"新问卷已开始，一共 {TOTAL_QUESTIONS} 题。\n"
        f"请用 /soul_answer <1-5> 作答，3 = 看情况/不确定，没有标准答案。\n"
        f"如果 30 分钟内没有继续，进度会自动清空。\n\n"
        f"{bar} 0/{TOTAL_QUESTIONS}\n"
        f"第 1 题：{_format_question_text(q, 0)}"
    )
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


async def handle_answer(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """处理问卷回答。

    使用 kwargs 中的 matched_groups 获取答案，不使用旧 self.matched_groups。
    """
    from ..questions.setup_questions import QUESTIONS, calculate_initial_spectrum
    from ..utils.spectrum_utils import check_admin_permission, format_spectrum_display, extract_command_actor
    from ..utils.audit_log import log_init

    cleanup_expired_sessions(plugin)

    ok, err = check_admin_permission(plugin, kwargs, "进行问卷答题")
    if not ok:
        await plugin.ctx.send.text(err, stream_id)
        return True, err, True

    platform, user_id = extract_command_actor(kwargs)
    session_key = f"{platform}:{user_id}"

    if session_key not in plugin._questionnaire_sessions:
        msg = (
            "这份问卷已经超时了（超过 30 分钟没动静），之前的进度已自动清空。\n"
            "想重新来就再发一次 /soul_setup。"
        )
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    session = plugin._questionnaire_sessions[session_key]
    matched_groups = kwargs.get("matched_groups", {})
    answer = int(matched_groups.get("answer", "0") or 0)
    session["answers"].append(answer)
    session["current"] += 1
    answered = session["current"]

    if answered >= len(QUESTIONS):
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

        bar = _progress_bar(TOTAL_QUESTIONS)
        display = format_spectrum_display(spectrum_values)
        msg = (
            f"问卷完成 {bar} {TOTAL_QUESTIONS}/{TOTAL_QUESTIONS}\n\n"
            f"光谱结果：\n\n"
            f"• 真诚度（sincerity）：{spectrum_values['sincerity']}\n"
            f"• 投入度（engagement）：{spectrum_values['engagement']}\n"
            f"• 亲密度（closeness）：{spectrum_values['closeness']}\n"
            f"• 直率度（directness）：{spectrum_values['directness']}\n\n"
            f"已保存，bot 会按这个光谱来调整群聊风格。"
        )
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # 答题后反馈
    label = ANSWER_LABELS.get(answer, f"{answer}")
    bar = _progress_bar(answered)
    q = QUESTIONS[answered]
    msg = (
        f"已记录：{answer}（{label}）\n\n"
        f"{bar} {answered}/{TOTAL_QUESTIONS}\n"
        f"第 {answered + 1} 题：{_format_question_text(q, answered)}"
    )
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
