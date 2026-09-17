"""SDK2 Command 入参形状测试。

契约来源（宿主只读参考）：
- `src/plugin_runtime/component_query.py:522-534` 构造 `invoke_args`：
  顶层 `text` = `message.processed_plain_text`，另有 `stream_id` / `group_id` /
  `platform` / `user_id` / `message`（完整会话消息字典）。
- `PluginMessageUtils._session_message_to_dict`（`message_utils.py:412`）输出的键为
  `message_id/timestamp/platform/message_info/raw_message/is_*/session_id`，
  可选 `processed_plain_text`/`reply_to` —— **没有 `text` 键**。

因此 `(kwargs["message"]).get("text")` 恒为空；确认类命令（如 /soul_reset confirm）
会永远等不到确认。
"""

from __future__ import annotations

from typing import Any

from .conftest import _import_soul_submodule

REAL_TEXT = "/soul_reset confirm"


def _real_command_kwargs(text: str = REAL_TEXT) -> dict:
    """模拟宿主 invoke_args 的真实形状。"""
    return {
        "text": text,
        "stream_id": "qq-123-group",
        "group_id": "123",
        "platform": "qq",
        "user_id": "3659592968",
        "is_local_operator": False,
        "matched_groups": {},
        "message": {
            "message_id": "m1",
            "platform": "qq",
            "message_info": {
                "user_info": {"user_id": "3659592968", "user_nickname": "n"},
                "group_info": {"group_id": "123", "group_name": "g"},
            },
            "raw_message": [],
            "session_id": "qq-123-group",
            "processed_plain_text": text,
        },
    }


def test_extract_command_text_uses_top_level_text() -> None:
    """顶层 text 是权威来源（= processed_plain_text）。"""
    su = _import_soul_submodule("utils.spectrum_utils")
    assert su.extract_command_text(_real_command_kwargs()) == REAL_TEXT


def test_extract_command_text_falls_back_to_message_processed_plain_text() -> None:
    """顶层 text 缺失时退回 message.processed_plain_text。"""
    su = _import_soul_submodule("utils.spectrum_utils")
    kwargs = _real_command_kwargs()
    kwargs.pop("text")
    assert su.extract_command_text(kwargs) == REAL_TEXT


def test_extract_command_text_never_reads_message_text_key() -> None:
    """message 字典里没有 text 键，不得依赖它。"""
    su = _import_soul_submodule("utils.spectrum_utils")
    kwargs = _real_command_kwargs()
    kwargs["message"]["text"] = "伪造值"
    assert su.extract_command_text(kwargs) == REAL_TEXT


def test_extract_command_text_empty_payload() -> None:
    """空 / 畸形载荷 → 空串，不抛异常。"""
    su = _import_soul_submodule("utils.spectrum_utils")
    assert su.extract_command_text(None) == ""
    assert su.extract_command_text({}) == ""
    assert su.extract_command_text({"message": "not-a-dict"}) == ""


def test_extract_command_text_strips_whitespace() -> None:
    """两端空白剥离。"""
    su = _import_soul_submodule("utils.spectrum_utils")
    assert su.extract_command_text({"text": "  /soul_status  "}) == "/soul_status"


# ─── /soul_reset 两步确认在真实载荷下的行为 ──────────────────────────


def _reset_plugin(sent: list[str]) -> Any:
    from types import SimpleNamespace

    class _Ctx:
        class _Send:
            async def text(self, text: str, stream_id: str = "") -> None:
                sent.append(text)

        send = _Send()

    return SimpleNamespace(
        config=SimpleNamespace(
            admin=SimpleNamespace(admin_user_id="qq:3659592968"),
        ),
        ctx=_Ctx(),
        _reset_confirm_ts={},
    )


def test_reset_confirm_reaches_execution_with_real_payload(soul_db) -> None:
    """真实 Command 载荷下 /soul_reset confirm 必须走确认分支（旧代码恒为空串）。"""
    import asyncio
    from unittest.mock import AsyncMock, patch

    rc = _import_soul_submodule("components.reset_command")
    audit = _import_soul_submodule("utils.audit_log")
    sent: list[str] = []
    plugin = _reset_plugin(sent)

    with patch.object(audit, "log_reset", AsyncMock()):
        # 第一次：请求重置 → 记录确认状态
        asyncio.run(rc.handle_reset(plugin, "qq-123-group", **{
            k: v for k, v in _real_command_kwargs("/soul_reset").items() if k != "stream_id"
        }))
        assert "qq-123-group" in plugin._reset_confirm_ts
        assert "确认" in sent[-1]

        # 第二次：确认 → 必须真正执行重置（而不是再次提示确认）
        asyncio.run(rc.handle_reset(plugin, "qq-123-group", **{
            k: v for k, v in _real_command_kwargs("/soul_reset confirm").items() if k != "stream_id"
        }))

    assert "已重置为中立状态" in sent[-1]
    assert "qq-123-group" not in plugin._reset_confirm_ts


def test_reset_without_confirm_only_asks(soul_db) -> None:
    """只发 /soul_reset → 仅提示确认，不执行重置。"""
    import asyncio

    rc = _import_soul_submodule("components.reset_command")
    sent: list[str] = []
    plugin = _reset_plugin(sent)

    asyncio.run(rc.handle_reset(plugin, "qq-123-group", **{
        k: v for k, v in _real_command_kwargs("/soul_reset").items() if k != "stream_id"
    }))

    assert "已重置为中立状态" not in sent[-1]
    assert "确认" in sent[-1]
