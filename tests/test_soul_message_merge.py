"""测试 `_apply_soul_injection_to_messages` 消息合并函数。

验证：
1. 有 host system → content 含原人设 + 动态层标记 + injection；messages 长度不变；首条仍是 system；无额外 prepend
2. 无 system 仅 user → 返回 skip，不改
3. 多条 system → 只改第一条
4. role 大小写 System 也能识别
"""

from __future__ import annotations

from .conftest import _import_soul_submodule

INJECTION_BLOCK = (
    "- 倾向：真诚+70 投入+60 亲近+50 直率+40\n"
    "- 固化观点：be honest (confidence 0.85)\n"
    "请综合上述倾向与固化观点来组织回复。\n"
)
HOST_SYSTEM_CONTENT = "你是 Mai，一个友善的群聊助手。回复风格要活泼可爱。\n"


def _make_msgs(*msgs: tuple[str, str]) -> list[dict]:
    """从 (role, content) 对构建消息列表。"""
    return [{"role": r, "content": c} for r, c in msgs]


# ─── 测试 1：有 host system → 追加注入块 ────────────────────────────


def test_append_to_host_system() -> None:
    """有 host system → content 末尾追加动态层标记 + injection_block。"""
    injector = _import_soul_submodule("components.ideology_injector")
    msgs = _make_msgs(
        ("system", HOST_SYSTEM_CONTENT),
        ("user", "你好"),
    )
    new_msgs, strategy = injector._apply_soul_injection_to_messages(msgs, INJECTION_BLOCK)

    assert strategy == "append_host_system"
    assert new_msgs is not None
    # 消息条数不变
    assert len(new_msgs) == len(msgs)
    # 首条仍是 system
    assert new_msgs[0]["role"] == "system"
    # 首条 content 包含原人设
    assert HOST_SYSTEM_CONTENT in new_msgs[0]["content"]
    # 首条 content 包含动态层标记
    assert "[Mai-Soul 动态层" in new_msgs[0]["content"]
    assert "受上方固定人设与表达风格约束" in new_msgs[0]["content"]
    assert "不得覆盖身份事实与 reply_style" in new_msgs[0]["content"]
    # 首条 content 包含 injection_block 内容
    assert "倾向：真诚+70" in new_msgs[0]["content"]
    # 首条 content 以 injection_block 内容结尾（去掉 trailing 空白）
    assert new_msgs[0]["content"].endswith("组织回复。\n")
    # 首条 role 仍然是 system（dict 未被篡改 key）
    assert new_msgs[0]["role"] == "system"
    # 第二条（user）不受影响
    assert new_msgs[1] == msgs[1]


# ─── 测试 2：无 system → skip_no_host_system ─────────────────────────


def test_skip_no_system() -> None:
    """无 system 仅 user → 返回 (None, 'skip_no_host_system')，原始列表不改。"""
    injector = _import_soul_submodule("components.ideology_injector")
    msgs = _make_msgs(
        ("user", "你好"),
        ("assistant", "嗨！"),
    )
    new_msgs, strategy = injector._apply_soul_injection_to_messages(msgs, INJECTION_BLOCK)

    assert strategy == "skip_no_host_system"
    assert new_msgs is None


def test_skip_empty_list() -> None:
    """空列表 → skip_no_host_system。"""
    injector = _import_soul_submodule("components.ideology_injector")
    new_msgs, strategy = injector._apply_soul_injection_to_messages([], INJECTION_BLOCK)

    assert strategy == "skip_no_host_system"
    assert new_msgs is None


# ─── 测试 3：多条 system → 只改第一条 ────────────────────────────────


def test_only_first_system_modified() -> None:
    """多条 system 消息时，只修改第一条。"""
    injector = _import_soul_submodule("components.ideology_injector")
    msgs = _make_msgs(
        ("system", "第一条 system：我是人设 A"),
        ("system", "第二条 system：我是人设 B"),
        ("user", "测试"),
    )
    new_msgs, strategy = injector._apply_soul_injection_to_messages(msgs, INJECTION_BLOCK)

    assert strategy == "append_host_system"
    assert new_msgs is not None
    # 首条被修改
    assert "[Mai-Soul 动态层" in new_msgs[0]["content"]
    # 第二条 system 完全不变
    assert new_msgs[1] == msgs[1]
    # user 完全不变
    assert new_msgs[2] == msgs[2]


# ─── 测试 4：role 大小写不敏感 ──────────────────────────────────────


def test_case_insensitive_system_role() -> None:
    """role 为 'System'（大写 S）也能识别。"""
    injector = _import_soul_submodule("components.ideology_injector")
    msgs = _make_msgs(
        ("System", HOST_SYSTEM_CONTENT),
        ("user", "你好"),
    )
    new_msgs, strategy = injector._apply_soul_injection_to_messages(msgs, INJECTION_BLOCK)

    assert strategy == "append_host_system"
    assert new_msgs is not None
    assert "[Mai-Soul 动态层" in new_msgs[0]["content"]


def test_case_mixed_system_role() -> None:
    """role 为 'SYSTEM'（全大写）也能识别。"""
    injector = _import_soul_submodule("components.ideology_injector")
    msgs = _make_msgs(
        ("SYSTEM", HOST_SYSTEM_CONTENT),
        ("user", "你好"),
    )
    new_msgs, strategy = injector._apply_soul_injection_to_messages(msgs, INJECTION_BLOCK)

    assert strategy == "append_host_system"
    assert new_msgs is not None
    assert "[Mai-Soul 动态层" in new_msgs[0]["content"]


# ─── 测试 5：无 prepend，不修改原始列表 ─────────────────────────────


def test_no_prepend_new_message() -> None:
    """不 prepend 新 system message，原始列表 length 不变。"""
    injector = _import_soul_submodule("components.ideology_injector")
    msgs = _make_msgs(
        ("system", HOST_SYSTEM_CONTENT),
        ("user", "你好"),
        ("assistant", "嗨！"),
    )
    new_msgs, strategy = injector._apply_soul_injection_to_messages(msgs, INJECTION_BLOCK)

    assert strategy == "append_host_system"
    assert new_msgs is not None
    assert len(new_msgs) == len(msgs)
    # 没有额外 prepend 的 system
    assert new_msgs[0]["role"] == "system"
    # 没有新的 system 被插入在最前（列表长度不变已经隐含了）


def test_original_messages_unchanged() -> None:
    """原始 messages 列表不应被修改（不可变性）。"""
    injector = _import_soul_submodule("components.ideology_injector")
    msgs = _make_msgs(
        ("system", HOST_SYSTEM_CONTENT),
        ("user", "你好"),
    )
    original_content = msgs[0]["content"]
    _new_msgs, _strategy = injector._apply_soul_injection_to_messages(msgs, INJECTION_BLOCK)

    # 原始列表的 system content 未被修改
    assert msgs[0]["content"] == original_content
    # 原始列表结构不变
    assert len(msgs) == 2
    assert msgs[1]["role"] == "user"
