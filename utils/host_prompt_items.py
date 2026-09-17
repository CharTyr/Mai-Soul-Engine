"""宿主 Prompt Item 适配层（SDK 2.x Context Item 契约）。

宿主 `maisaka.planner.before_request` 传入/回读的是 **Context Item 列表**（`items`），
不是旧式的 `messages`（`role`/`content`）：

- 调用方：`src/maisaka/chat_loop_service.py` → `items=serialize_prompt_items(...)`，
  回读 `before_request_result.kwargs["items"]`。
- Item 形状（`src/llm_models/request_snapshot.py:274`）::

      {"item_type": "SystemMessageItem",
       "meta": {"item_id": ..., "logical_turn_id": ..., "timestamp": ...},
       "parts": [{"type": "text", "text": ...}, ...]}

本模块把「读取 / 提取用户文本 / 回写」三种形状差异收在一个 seam 后面，
调用方只关心「把一段文本追加进宿主 system 项」。旧 `messages` 形状保留只读兼容，
因为同一 hook 在历史宿主上曾以该形状传参。
"""

from __future__ import annotations

import copy
from typing import Any

PROMPT_ITEMS_KEY = "items"
LEGACY_MESSAGES_KEY = "messages"

# 动态层标记：声明该段受宿主固定人设约束，不得覆盖身份事实与表达风格。
DYNAMIC_LAYER_MARKER = (
    "\n\n---\n"
    "[Mai-Soul 动态层 | 受上方固定人设与表达风格约束，不得覆盖身份事实与 reply_style]\n"
)

STRATEGY_APPENDED = "append_host_system"
STRATEGY_NO_SYSTEM = "skip_no_host_system"


def read_prompt_items(kwargs: dict[str, Any]) -> list[dict]:
    """读取 hook 载荷中的提示项列表（items 优先，旧 messages 作兼容）。"""
    if not isinstance(kwargs, dict):
        return []
    items = kwargs.get(PROMPT_ITEMS_KEY)
    if isinstance(items, list):
        return items
    legacy = kwargs.get(LEGACY_MESSAGES_KEY)
    if isinstance(legacy, list):
        return legacy
    return []


def _item_kind(item: Any) -> str:
    """判定提示项形态：'item' / 'message' / 'unknown'。"""
    if not isinstance(item, dict):
        return "unknown"
    item_type = item.get("item_type")
    if isinstance(item_type, str) and item_type.strip():
        return "item"
    if isinstance(item.get("role"), str):
        return "message"
    return "unknown"


def _is_system(item: dict, kind: str) -> bool:
    """是否为宿主 system 项（大小写不敏感）。"""
    if kind == "item":
        return "system" in str(item.get("item_type", "")).lower()
    if kind == "message":
        return str(item.get("role", "")).lower() == "system"
    return False


def _is_user(item: dict, kind: str) -> bool:
    """是否为用户项。"""
    if kind == "item":
        return "user" in str(item.get("item_type", "")).lower()
    if kind == "message":
        return str(item.get("role", "")).lower() == "user"
    return False


def _item_text(item: dict, kind: str) -> str:
    """提取单项文本；无文本（如纯图片项）返回空串。"""
    if kind == "message":
        content = item.get("content")
        return content if isinstance(content, str) else ""
    parts = item.get("parts")
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if str(part.get("type", "")).lower() != "text":
            continue
        text = part.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def extract_user_texts(prompt_items: list[dict], limit: int) -> list[str]:
    """取最近 limit 条用户文本，按时间正序返回（自评上下文缓存用）。"""
    if limit <= 0:
        return []
    collected: list[str] = []
    for item in reversed(prompt_items):
        if len(collected) >= limit:
            break
        kind = _item_kind(item)
        if not _is_user(item, kind):
            continue
        text = _item_text(item, kind)
        if text.strip():
            collected.append(text)
    collected.reverse()
    return collected


def extract_latest_user_text(kwargs: dict[str, Any]) -> str:
    """取最后一条用户项的文本（用于 trait tag / 关键词匹配）。"""
    texts = extract_user_texts(read_prompt_items(kwargs), 1)
    return texts[0] if texts else ""


def _append_to_item(item: dict, block: str) -> dict:
    """把 block 追加到 item 的最后一个文本 part 上（无文本 part 时新增一个）。

    **只复制必要结构**（顶层 dict + parts 列表 + 被改的那一个 part），不做深拷贝：
    这里的 item 可能带着历史消息甚至 base64 图片，而注入跑在 planner 热路径上，
    深拷贝整个提示项等于每次请求都白复制一大块数据。
    """
    new_item = dict(item)
    parts = item.get("parts")
    if not isinstance(parts, list):
        new_item["parts"] = [
            {"type": "text", "text": DYNAMIC_LAYER_MARKER.lstrip() + block.lstrip()}
        ]
        return new_item

    new_parts = list(parts)
    for index in range(len(new_parts) - 1, -1, -1):
        part = new_parts[index]
        if isinstance(part, dict) and str(part.get("type", "")).lower() == "text":
            existing = part.get("text")
            new_parts[index] = {
                **part,
                "text": (existing if isinstance(existing, str) else "")
                + DYNAMIC_LAYER_MARKER
                + block.lstrip(),
            }
            new_item["parts"] = new_parts
            return new_item

    new_parts.append({"type": "text", "text": DYNAMIC_LAYER_MARKER.lstrip() + block.lstrip()})
    new_item["parts"] = new_parts
    return new_item


def _append_to_message(message: dict, block: str) -> dict:
    """旧 messages 形状：追加到 content 末尾（同样只复制顶层）。"""
    new_message = dict(message)
    content = message.get("content")
    new_message["content"] = (
        (content if isinstance(content, str) else "") + DYNAMIC_LAYER_MARKER + block.lstrip()
    )
    return new_message


def append_block_to_first_system(
    kwargs: dict[str, Any],
    block: str,
) -> tuple[dict[str, Any] | None, str]:
    """把注入块合并进宿主首个 system 项，返回 (modified_kwargs, strategy)。

    - 命中 system 项 → (新的完整 kwargs, "append_host_system")
    - 无 system 项   → (None, "skip_no_host_system")，调用方应 fail-open 放弃注入

    不原地修改入参；回写键跟随实际形状（items / messages）。
    """
    if not isinstance(kwargs, dict) or not isinstance(block, str) or not block.strip():
        return None, STRATEGY_NO_SYSTEM

    items = read_prompt_items(kwargs)
    if not items:
        return None, STRATEGY_NO_SYSTEM

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        kind = _item_kind(item)
        if not _is_system(item, kind):
            continue

        new_items = list(items)
        if kind == "item":
            new_items[index] = _append_to_item(item, block)
            write_key = PROMPT_ITEMS_KEY
        else:
            new_items[index] = _append_to_message(item, block)
            write_key = LEGACY_MESSAGES_KEY

        merged = dict(kwargs)
        merged[write_key] = new_items
        return merged, STRATEGY_APPENDED

    return None, STRATEGY_NO_SYSTEM
