"""宿主 Prompt Item 契约测试（真实 SDK 2.x 形状，非旧 messages mock）。

契约来源（宿主只读参考）：
- `src/maisaka/chat_loop_service.py:1062-1082`：`maisaka.planner.before_request`
  以 `items=serialize_prompt_items(built_messages)` 调用，回读 `kwargs["items"]`。
- `src/plugin_runtime/hook_payloads.py:123`：`serialize_prompt_items` →
  `[serialize_context_item_snapshot(item) for item in items]`（dict 列表）。
- `src/llm_models/request_snapshot.py:274`：item 形状为
  `{"item_type": <类名>, "meta": {...}, "parts": [{"type": "text", "text": ...}]}`。
- hook spec `allow_kwargs_mutation=True`，故返回 `modified_kwargs` 会被整体替换。
"""

from __future__ import annotations

import copy

from .conftest import _import_soul_submodule

BLOCK = "- 倾向：真诚+70 投入+60\n请综合上述倾向与固化观点来组织回复。\n"
HOST_SYSTEM = "你是 Mai，一个友善的群聊助手。回复风格要活泼可爱。\n"

_TS = "2026-09-17T10:00:00+00:00"


def _item(item_type: str, item_id: str, *texts: str) -> dict:
    """构造宿主序列化后的 Context Item（与 hook_payloads 输出同形）。"""
    return {
        "item_type": item_type,
        "meta": {"item_id": item_id, "logical_turn_id": None, "timestamp": _TS},
        "parts": [{"type": "text", "text": t} for t in texts],
    }


def _host_kwargs(*items: dict, **extra) -> dict:
    """模拟宿主 invoke_hook 传入的完整 kwargs。"""
    payload = {
        "items": list(items),
        "item_schema_version": 3,
        "tool_definitions": [{"name": "send_message"}],
        "selected_history_count": 5,
        "built_message_count": 6,
        "selection_reason": "recent",
        "session_id": "qq-123-group",
    }
    payload.update(extra)
    return payload


def _texts_of(item: dict) -> str:
    return "".join(p.get("text", "") for p in item.get("parts", []) if p.get("type") == "text")


# ─── 读取：items 形状 ───────────────────────────────────────────────


def test_reads_items_key_not_messages() -> None:
    """宿主传 items → 适配器必须读到 items（旧代码只读 messages）。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(_item("SystemMessageItem", "i1", HOST_SYSTEM))
    assert adapter.read_prompt_items(kwargs) == kwargs["items"]


def test_reads_legacy_messages_shape() -> None:
    """旧 messages 形状仍可读取（兼容层）。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    legacy = [{"role": "system", "content": HOST_SYSTEM}]
    assert adapter.read_prompt_items({"messages": legacy}) == legacy


# ─── 合并：items 形状 ───────────────────────────────────────────────


def test_appends_block_to_first_system_item() -> None:
    """注入块追加到首个 SystemMessageItem 的文本，条数与其它 item 不变。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(
        _item("SystemMessageItem", "sys1", HOST_SYSTEM),
        _item("UserMessageItem", "u1", "你好"),
    )
    merged, strategy = adapter.append_block_to_first_system(kwargs, BLOCK)

    assert strategy == "append_host_system"
    assert merged is not None
    items = merged["items"]
    assert len(items) == 2
    sys_text = _texts_of(items[0])
    assert HOST_SYSTEM in sys_text
    assert "[Mai-Soul 动态层" in sys_text
    assert "倾向：真诚+70" in sys_text
    assert items[0]["item_type"] == "SystemMessageItem"
    # 非 system item 不得被触碰
    assert items[1] == kwargs["items"][1]


def test_modified_kwargs_preserves_other_keys() -> None:
    """回写必须保留 items 之外的 kwargs（schema 版本/工具定义/会话）。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(_item("SystemMessageItem", "sys1", HOST_SYSTEM))
    merged, _ = adapter.append_block_to_first_system(kwargs, BLOCK)

    assert merged is not None
    assert merged["item_schema_version"] == kwargs["item_schema_version"]
    assert merged["tool_definitions"] == kwargs["tool_definitions"]
    assert merged["session_id"] == kwargs["session_id"]
    assert merged["selection_reason"] == kwargs["selection_reason"]
    assert set(merged.keys()) == set(kwargs.keys())


def test_writes_back_to_items_not_messages() -> None:
    """回写键必须是 items —— 写成 messages 宿主会整份忽略。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(_item("SystemMessageItem", "sys1", HOST_SYSTEM))
    merged, _ = adapter.append_block_to_first_system(kwargs, BLOCK)

    assert merged is not None
    assert "items" in merged
    assert "messages" not in merged


def test_only_first_system_item_modified() -> None:
    """多条 system item 时只改第一条。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(
        _item("SystemMessageItem", "sys1", "人设 A"),
        _item("SystemMessageItem", "sys2", "人设 B"),
    )
    merged, strategy = adapter.append_block_to_first_system(kwargs, BLOCK)

    assert strategy == "append_host_system"
    assert merged is not None
    assert "[Mai-Soul 动态层" in _texts_of(merged["items"][0])
    assert merged["items"][1] == kwargs["items"][1]


def test_appends_to_last_text_part_when_multiple() -> None:
    """一个 system item 含多个 text part → 追加到最后一个，其余 part 不变。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    item = _item("SystemMessageItem", "sys1", "第一段", "第二段")
    kwargs = _host_kwargs(item)
    merged, _ = adapter.append_block_to_first_system(kwargs, BLOCK)

    assert merged is not None
    parts = merged["items"][0]["parts"]
    assert len(parts) == 2
    assert parts[0] == item["parts"][0]
    assert "第二段" in parts[1]["text"]
    assert "[Mai-Soul 动态层" in parts[1]["text"]


def test_skip_when_no_system_item() -> None:
    """无 system item → (None, skip_no_host_system)，不得凭空 prepend。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(_item("UserMessageItem", "u1", "你好"))
    merged, strategy = adapter.append_block_to_first_system(kwargs, BLOCK)

    assert merged is None
    assert strategy == "skip_no_host_system"


def test_skip_when_no_items_at_all() -> None:
    """items 为空 → 跳过。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    merged, strategy = adapter.append_block_to_first_system({"items": []}, BLOCK)
    assert merged is None
    assert strategy == "skip_no_host_system"


def test_input_kwargs_not_mutated() -> None:
    """不得原地修改宿主导入的 kwargs / items。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(_item("SystemMessageItem", "sys1", HOST_SYSTEM))
    snapshot = copy.deepcopy(kwargs)
    adapter.append_block_to_first_system(kwargs, BLOCK)

    assert kwargs == snapshot


def test_unknown_item_type_not_treated_as_system() -> None:
    """未知 item 类型不得被误认成 system。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(_item("ReasoningItem", "r1", "思考中"))
    merged, strategy = adapter.append_block_to_first_system(kwargs, BLOCK)

    assert merged is None
    assert strategy == "skip_no_host_system"


# ─── 用户文本提取（tag 匹配输入） ───────────────────────────────────


def test_extract_latest_user_text_from_items() -> None:
    """从 items 取最后一条 UserMessageItem 的文本。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(
        _item("SystemMessageItem", "sys1", HOST_SYSTEM),
        _item("UserMessageItem", "u1", "第一条"),
        _item("AssistantMessageItem", "a1", "回复"),
        _item("UserMessageItem", "u2", "最后一条用户消息"),
    )
    assert adapter.extract_latest_user_text(kwargs) == "最后一条用户消息"


def test_extract_user_text_falls_back_to_legacy_messages() -> None:
    """旧 messages 形状仍能提取用户文本。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = {"messages": [{"role": "user", "content": "旧形状提问"}]}
    assert adapter.extract_latest_user_text(kwargs) == "旧形状提问"


def test_extract_user_text_empty_when_none() -> None:
    """无用户 item → 空串（不抛异常）。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(_item("SystemMessageItem", "sys1", HOST_SYSTEM))
    assert adapter.extract_latest_user_text(kwargs) == ""


def test_extract_user_text_ignores_image_only_item() -> None:
    """只有图片 part 的用户 item → 不产生文本，不崩。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    item = {
        "item_type": "UserMessageItem",
        "meta": {"item_id": "u1", "logical_turn_id": None, "timestamp": _TS},
        "parts": [{"type": "image", "image_base64": "AAAA", "image_format": "png"}],
    }
    kwargs = _host_kwargs(item)
    assert adapter.extract_latest_user_text(kwargs) == ""


# ─── 多条用户文本（自评上下文缓存用） ───────────────────────────────


def test_extract_user_texts_returns_chronological_tail() -> None:
    """取最后 N 条用户文本，按时间正序返回。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(
        _item("UserMessageItem", "u1", "最早"),
        _item("AssistantMessageItem", "a1", "回复"),
        _item("UserMessageItem", "u2", "中间"),
        _item("UserMessageItem", "u3", "最新"),
    )
    assert adapter.extract_user_texts(adapter.read_prompt_items(kwargs), 6) == [
        "最早",
        "中间",
        "最新",
    ]


def test_extract_user_texts_respects_limit() -> None:
    """limit 只保留最近若干条。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    kwargs = _host_kwargs(
        _item("UserMessageItem", "u1", "一"),
        _item("UserMessageItem", "u2", "二"),
        _item("UserMessageItem", "u3", "三"),
    )
    assert adapter.extract_user_texts(adapter.read_prompt_items(kwargs), 2) == ["二", "三"]


def test_extract_user_texts_supports_legacy_messages() -> None:
    """旧 messages 列表同样支持多条提取。"""
    adapter = _import_soul_submodule("utils.host_prompt_items")
    legacy = [
        {"role": "user", "content": "旧一"},
        {"role": "assistant", "content": "旧回"},
        {"role": "user", "content": "旧二"},
    ]
    assert adapter.extract_user_texts(legacy, 6) == ["旧一", "旧二"]


# ─── 热路径：只复制被改的部分，不做深拷贝 ────────────────────────────


def test_untouched_parts_are_not_deep_copied() -> None:
    """未改动的 part 必须保持**同一对象**（证明没有深拷贝整个提示项）。

    planner 的 items 里可能有历史消息甚至 base64 图片；注入在热路径上，
    深拷贝整个提示项等于每次请求白复制一大块数据。
    """
    hpi = _import_soul_submodule("utils.host_prompt_items")

    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    text_part = {"type": "text", "text": "宿主原始 system"}
    items = [
        {"item_type": "system", "meta": {}, "parts": [image_part, text_part]},
    ]
    original_system = items[0]
    original_parts = original_system["parts"]

    merged, strategy = hpi.append_block_to_first_system({"items": items}, "注入块")

    assert strategy == hpi.STRATEGY_APPENDED
    out_parts = merged["items"][0]["parts"]
    # 图片 part 原样复用（同一对象）
    assert out_parts[0] is image_part
    # 被改的文本 part 是新对象，原对象不被污染
    assert out_parts[1] is not text_part
    assert text_part["text"] == "宿主原始 system"
    # 宿主的 items / parts 容器也没被改动
    assert original_system["parts"] is original_parts
    assert len(original_parts) == 2


def test_input_kwargs_are_not_mutated() -> None:
    """入参 kwargs 与其中的 items 列表都不得被原地修改。"""
    hpi = _import_soul_submodule("utils.host_prompt_items")

    items = [{"item_type": "system", "meta": {}, "parts": [{"type": "text", "text": "S"}]}]
    kwargs = {"items": items, "item_schema_version": 1}

    merged, _ = hpi.append_block_to_first_system(kwargs, "块")

    assert kwargs["items"] is items
    assert items[0]["parts"][0]["text"] == "S"
    assert merged is not kwargs
    assert merged["item_schema_version"] == 1
