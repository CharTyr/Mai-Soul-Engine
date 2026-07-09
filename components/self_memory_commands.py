"""自我事件记忆管理命令（管理员）。"""

from __future__ import annotations

import re
from typing import Any

from ..models.self_memory import (
    create_self_memory,
    format_memories_for_tool,
    get_self_memory,
    list_self_memories,
    search_self_memories,
    soft_delete_self_memory,
)
from ..utils.spectrum_utils import extract_command_actor, match_user


def _require_admin(plugin: Any, kwargs: dict[str, Any]) -> tuple[bool, str, str]:
    admin_user_id = plugin.config.admin.admin_user_id
    platform, user_id = extract_command_actor(kwargs)
    if not admin_user_id:
        return False, platform, user_id
    if not match_user(platform, user_id, admin_user_id):
        return False, platform, user_id
    return True, platform, user_id


def _feature_enabled(plugin: Any) -> bool:
    cfg = getattr(plugin.config, "self_memory", None)
    if cfg is None:
        return True
    return bool(getattr(cfg, "enabled", True))


async def handle_note(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """/soul_note <text> — 写入 bot 自我记忆。"""
    if not _feature_enabled(plugin):
        msg = "自我记忆功能已关闭（[self_memory].enabled=false）"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    ok, _, _ = _require_admin(plugin, kwargs)
    if not ok:
        msg = "只有管理员可以写入自我记忆"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    text = str(kwargs.get("text") or "").strip()
    # strip command prefix
    m = re.match(r"^/soul_note(?:\s+|$)(.*)$", text, flags=re.S)
    body = (m.group(1) if m else text).strip()
    if not body:
        msg = "用法: /soul_note <要记住的事件或约定>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # optional leading tags: #a #b rest...
    tags: list[str] = []
    rest = body
    while True:
        tm = re.match(r"^#([^\s#]+)\s+(.*)$", rest, flags=re.S)
        if not tm:
            break
        tags.append(tm.group(1))
        rest = tm.group(2).strip()
    content = rest or body

    mem = create_self_memory(
        content,
        tags=tags,
        source="manual",
        stream_id=stream_id or "",
        importance=3,
    )
    if not mem:
        msg = "写入失败：内容为空"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    tag_s = " ".join(f"#{t}" for t in mem.tags) if mem.tags else "(无标签)"
    msg = f"已记住 [{mem.memory_id}]\n{mem.content}\n{tag_s}"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


async def handle_memories(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """/soul_memories [query] — 检索/列出自我记忆。"""
    if not _feature_enabled(plugin):
        msg = "自我记忆功能已关闭（[self_memory].enabled=false）"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    ok, _, _ = _require_admin(plugin, kwargs)
    if not ok:
        msg = "只有管理员可以查看自我记忆"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    text = str(kwargs.get("text") or "").strip()
    m = re.match(r"^/soul_memories(?:\s+|$)(.*)$", text, flags=re.S)
    query = (m.group(1) if m else "").strip()

    max_results = 8
    cfg = getattr(plugin.config, "self_memory", None)
    if cfg is not None:
        try:
            max_results = int(getattr(cfg, "max_results", 8) or 8)
        except (TypeError, ValueError):
            max_results = 8

    if query:
        memories = search_self_memories(query, limit=max_results)
        head = f"自我记忆检索「{query}」："
    else:
        memories = list_self_memories(limit=max_results)
        head = "最近自我记忆："

    body = format_memories_for_tool(memories, max_chars=1800)
    msg = f"{head}\n{body}"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


async def handle_memory_get(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """/soul_memory <id> — 查看单条。"""
    if not _feature_enabled(plugin):
        msg = "自我记忆功能已关闭"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    ok, _, _ = _require_admin(plugin, kwargs)
    if not ok:
        msg = "只有管理员可以查看自我记忆"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    text = str(kwargs.get("text") or "").strip()
    m = re.match(r"^/soul_memory\s+(\w+)\s*$", text)
    if not m:
        msg = "用法: /soul_memory <memory_id>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    mem = get_self_memory(m.group(1))
    if not mem:
        msg = "未找到该记忆（或不存在/已删除）"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    tag_s = ", ".join(mem.tags) if mem.tags else "-"
    msg = (
        f"[{mem.memory_id}] importance={mem.importance}\n"
        f"{mem.content}\n"
        f"tags: {tag_s}\n"
        f"source: {mem.source}\n"
        f"time: {mem.event_time or mem.created_at}"
    )
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


async def handle_memory_del(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """/soul_memory_del <id> — 软删除。"""
    if not _feature_enabled(plugin):
        msg = "自我记忆功能已关闭"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    ok, _, _ = _require_admin(plugin, kwargs)
    if not ok:
        msg = "只有管理员可以删除自我记忆"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    text = str(kwargs.get("text") or "").strip()
    m = re.match(r"^/soul_memory_del\s+(\w+)\s*$", text)
    if not m:
        msg = "用法: /soul_memory_del <memory_id>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    mid = m.group(1)
    if soft_delete_self_memory(mid):
        msg = f"已删除自我记忆 [{mid}]"
    else:
        msg = f"删除失败：找不到 [{mid}]"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
