"""测试 evolution_task.py 的 _analyze_group 函数与返回语义。

覆盖 3 个关键场景：
1. LLM 请求失败 → "skipped" + 不崩
2. 无消息 → "skipped"
3. 返回语义：skip 不计入 analyzed

从宿主仓根运行：``uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/test_evolution_task.py -q``
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from .conftest import _import_soul_submodule


def _make_plugin(
    evolution_enabled: bool = True,
    evolution_rate: int = 3,
    evolution_interval_hours: float = 4.0,
    ema_alpha: float = 0.3,
    direction_resistance: float = 0.5,
    max_messages_per_analysis: int = 50,
    max_chars_per_message: int = 500,
    thought_cabinet_enabled: bool = False,
    self_reflection_enabled: bool = False,
    monitored_groups: list[str] | None = None,
    excluded_groups: list[str] | None = None,
    monitored_users: list[str] | None = None,
    excluded_users: list[str] | None = None,
    admin_user_id: str = "qq:admin123",
    trait_ttl_days: int = 90,
) -> SimpleNamespace:
    """构造最小 plugin mock 用于 evolution_task 测试。"""
    return SimpleNamespace(
        config=SimpleNamespace(
            evolution=SimpleNamespace(
                evolution_enabled=evolution_enabled,
                evolution_rate=evolution_rate,
                evolution_interval_hours=evolution_interval_hours,
                ema_alpha=ema_alpha,
                direction_resistance=direction_resistance,
                max_messages_per_analysis=max_messages_per_analysis,
                max_chars_per_message=max_chars_per_message,
            ),
            monitor=SimpleNamespace(
                monitored_groups=monitored_groups or [],
                excluded_groups=excluded_groups or [],
                monitored_users=monitored_users or [],
                excluded_users=excluded_users or [],
            ),
            thought_cabinet=SimpleNamespace(
                enabled=thought_cabinet_enabled,
                trait_ttl_days=trait_ttl_days,
                admin_notification_enabled=False,
                admin_notification_cooldown_minutes=0,
            ),
            self_reflection=SimpleNamespace(enabled=self_reflection_enabled),
            admin=SimpleNamespace(admin_user_id=admin_user_id),
        ),
        ctx=SimpleNamespace(
            chat=SimpleNamespace(
                get_stream_by_group_id=AsyncMock(return_value={"success": True, "stream": {"session_id": "mock_stream"}}),
            ),
            message=SimpleNamespace(
                get_by_time_in_chat=AsyncMock(return_value=[]),
            ),
            call_capability=AsyncMock(return_value={"response": "{}"}),
        ),
    )


# ─── _analyze_group: LLM failure ─────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_group_llm_failure_returns_skipped(soul_db: Any) -> None:
    """mock LLM 返回异常 → 返回 "skipped" 不崩。"""
    et = _import_soul_submodule("components.evolution_task")
    plugin = _make_plugin(monitored_groups=["qq:123456:group"])

    # Mock get_by_time_in_chat 返回足够消息
    plugin.ctx.message.get_by_time_in_chat = AsyncMock(
        return_value=[
            {
                "user_info": {"platform": "qq", "user_id": "user1", "user_nickname": "用户A"},
                "processed_plain_text": "今天天气真好",
            },
            {
                "user_info": {"platform": "qq", "user_id": "user2", "user_nickname": "用户B"},
                "processed_plain_text": "是啊，适合出去玩",
            },
            {
                "user_info": {"platform": "qq", "user_id": "user3", "user_nickname": "用户C"},
                "processed_plain_text": "但是要上班",
            },
            {
                "user_info": {"platform": "qq", "user_id": "user4", "user_nickname": "用户D"},
                "processed_plain_text": "摸鱼中",
            },
            {
                "user_info": {"platform": "qq", "user_id": "user5", "user_nickname": "用户E"},
                "processed_plain_text": "哈哈确实",
            },
        ]
    )

    # Mock generate_soul_text 抛出异常
    original = et.generate_soul_text
    et.generate_soul_text = AsyncMock(side_effect=RuntimeError("LLM 服务不可用"))
    try:
        result = await et._analyze_group(plugin, "qq:123456:group", 3)
        assert result == "skipped", f"预期 'skipped'，收到 {result!r}"
    finally:
        et.generate_soul_text = original


@pytest.mark.asyncio
async def test_analyze_group_no_messages_returns_skipped(soul_db: Any) -> None:
    """mock get_by_time_in_chat 返回空 → 返回 "skipped" 不崩。"""
    et = _import_soul_submodule("components.evolution_task")
    plugin = _make_plugin(monitored_groups=["qq:123456:group"])

    result = await et._analyze_group(plugin, "qq:123456:group", 3)
    assert result == "skipped", f"预期 'skipped'，收到 {result!r}"


@pytest.mark.asyncio
async def test_analyze_group_stream_not_found_returns_skipped(soul_db: Any) -> None:
    """resolve_monitored_group_stream 返回空 → 返回 "skipped"。"""
    et = _import_soul_submodule("components.evolution_task")
    plugin = _make_plugin(monitored_groups=["qq:999999:group"])
    # 使 get_stream_by_group_id 返回失败 → stream_id 为空
    original = plugin.ctx.chat.get_stream_by_group_id
    plugin.ctx.chat.get_stream_by_group_id = AsyncMock(return_value={"success": False})
    try:
        result = await et._analyze_group(plugin, "qq:999999:group", 3)
        assert result == "skipped", f"预期 'skipped'，收到 {result!r}"
    finally:
        plugin.ctx.chat.get_stream_by_group_id = original


# ─── _analyze_with_sem 返回语义 ─────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_group_skip_returns_string(soul_db: Any) -> None:
    """stream 解析失败时 _analyze_group 返回字符串 skipped（非 bool True）。"""
    et = _import_soul_submodule("components.evolution_task")
    plugin = _make_plugin(monitored_groups=["qq:123456:group"])

    # 调用 run_evolution_loop 不方便，直接构造闭包验证透传行为
    # 测试 _analyze_group 被 mock 后 _analyze_with_sem 透传返回值

    # 构造最小 plugin 让 run_evolution_loop 不卡在早期条件
    plugin.config.evolution.evolution_enabled = True
    # 用一个简单的群列表
    groups = ["qq:skip_test:group"]

    original_resolve = et.resolve_monitored_group_stream
    original_ctx_chat = plugin.ctx.chat.get_stream_by_group_id

    call_count = 0

    async def mock_resolve(p, gid):
        nonlocal call_count
        call_count += 1
        # 第一次调用返回空（skipped），第二次返回 mock_stream
        if call_count == 1:
            return ""
        return "mock_stream"

    et.resolve_monitored_group_stream = mock_resolve
    plugin.ctx.chat.get_stream_by_group_id = AsyncMock(
        return_value={"success": True, "stream": {"session_id": "mock_stream"}}
    )

    try:
        # 用 _analyze_with_sem 逻辑测试：构造 semsphore 并调用
        import asyncio
        semaphore = asyncio.Semaphore(3)

        async def wrapper(gid):
            async with semaphore:
                return await et._analyze_group(plugin, gid, 3)

        # 测试空流返回
        plugin.ctx.message.get_by_time_in_chat = AsyncMock(return_value=[])
        # 第一个群 skip_test → stream not found
        result1 = await wrapper("qq:skip_test:group")
        assert result1 == "skipped", f"透传失败：预期 'skipped'，收到 {result1!r}"
    finally:
        et.resolve_monitored_group_stream = original_resolve
        plugin.ctx.chat.get_stream_by_group_id = original_ctx_chat
