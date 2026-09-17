"""重置命令模块 — maibot_sdk 2.x 版本。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

_RESET_CONFIRM_TIMEOUT = 300  # 5 分钟


def parse_reset_command(message_text: str) -> tuple[bool, bool]:
    """解析重置命令，返回 ``(是否重置命令, 是否确认)``。

    **必须整串精确匹配**。此前用 ``"confirm" in text`` 做子串判断，
    结果 ``/soul_reset disconfirm`` 会被当成确认并**真的重置人格**——
    含糊输入不该产生破坏性副作用。

    未知参数一律按「非确认」处理：重新提示，不执行。
    """
    tokens = (message_text or "").strip().split()
    if not tokens:
        return False, False
    if tokens[0].casefold() != "/soul_reset":
        return False, False
    rest = [t.casefold() for t in tokens[1:]]
    return True, rest == ["confirm"]


def reset_confirm_key(platform: str, user_id: str, stream_id: str) -> str:
    """确认状态按 **操作者 + 会话** 绑定。

    只按 stream_id 绑定时，同群的任何管理员（或任何能发命令的人）都能
    把别人发起的确认补上——确认应归属于发起人。
    """
    return f"{platform}:{user_id}:{stream_id}"


async def handle_reset(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """重置灵魂光谱为中立状态（管理员）。

    两步确认机制：
    1. 第一次 /soul_reset → 提示确认，并**明确写出重置范围**
    2. /soul_reset confirm → 5 分钟内、同一操作者、同一会话有效

    作用域说明：当前只会重置**全局**光谱；提示文案必须如实写出，
    不能让操作者以为只影响当前群。
    """
    from ..utils.spectrum_utils import check_admin_permission
    from ..utils.audit_log import log_reset
    from ..models.ideology_model import get_or_create_spectrum

    ok, err = check_admin_permission(plugin, kwargs, "重置灵魂光谱")
    if not ok:
        await plugin.ctx.send.text(err, stream_id)
        return True, err, True

    from ..utils.spectrum_utils import check_mutation_mode

    allowed, mode_err = check_mutation_mode(plugin, "重置灵魂光谱")
    if not allowed:
        await plugin.ctx.send.text(mode_err, stream_id)
        return True, mode_err, True

    from ..utils.spectrum_utils import extract_command_actor, extract_command_text

    platform, user_id = extract_command_actor(kwargs)
    message_text = extract_command_text(kwargs)
    _is_reset, is_confirm = parse_reset_command(message_text)
    key = reset_confirm_key(platform, user_id, stream_id)

    if is_confirm:
        # 先清理过期项（>5 分钟），防内存泄漏
        now = time.time()
        expired = [k for k, ts in plugin._reset_confirm_ts.items() if now - ts > _RESET_CONFIRM_TIMEOUT]
        for k in expired:
            plugin._reset_confirm_ts.pop(k, None)

        last_ts = plugin._reset_confirm_ts.get(key, 0.0)
        if now - last_ts > _RESET_CONFIRM_TIMEOUT:
            msg = "确认超时，请重新发送 /soul_reset 开始重置流程"
            await plugin.ctx.send.text(msg, stream_id)
            return True, msg, True

        # 执行前重新校验（鉴权已在入口做过；此处防「确认窗口内被降权/切模式」）
        ok2, err2 = check_admin_permission(plugin, kwargs, "重置灵魂光谱")
        if not ok2:
            await plugin.ctx.send.text(err2, stream_id)
            return True, err2, True
        allowed2, mode_err2 = check_mutation_mode(plugin, "重置灵魂光谱")
        if not allowed2:
            await plugin.ctx.send.text(mode_err2, stream_id)
            return True, mode_err2, True

        # 执行重置（作用域：全局）
        spectrum = get_or_create_spectrum("global")
        spectrum.sincerity = 50
        spectrum.engagement = 50
        spectrum.closeness = 50
        spectrum.directness = 50
        spectrum.initialized = False
        spectrum.updated_at = datetime.now()
        spectrum.save()

        await log_reset(f"{platform}:{user_id}")

        plugin._reset_confirm_ts.pop(key, None)

        msg = "灵魂光谱已重置为中立状态（范围：全局），请使用 /soul_setup 重新初始化"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # 第一次请求：记录确认状态，提示二次确认
    plugin._reset_confirm_ts[key] = time.time()
    msg = (
        "⚠️ 确认重置？**范围：全局光谱**（所有群共享的人格数值），其他数据不受影响。\n"
        "回复 /soul_reset confirm 确认，5 分钟内有效（仅你本人、本会话内有效）。"
    )
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
