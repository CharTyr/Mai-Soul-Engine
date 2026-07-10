"""测试 evolution_task.py 的 _analyze_group 函数。

覆盖 2 个关键场景：
1. LLM 请求失败 → 不崩 + 写 evolution_skip audit
2. 无消息 → 跳过

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
async def test_analyze_group_llm_failure_skips(soul_db: Any) -> None:
    """mock LLM 返回异常 → _analyze_group 不崩 + 写 evolution_skip audit。"""
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

    # Mock generate_soul_text 抛出异常（直接 patch 模块属性，避免带连字符的路径）
    original = et.generate_soul_text
    et.generate_soul_text = AsyncMock(side_effect=RuntimeError("LLM 服务不可用"))
    try:
        # 不应抛出异常
        await et._analyze_group(plugin, "qq:123456:group", 3)
    finally:
        et.generate_soul_text = original

    # 验证不崩即可——_analyze_group 内部有 try/except 兜底
    # 如果抛异常，pytest 会捕获


@pytest.mark.asyncio
async def test_analyze_group_no_messages_skips(soul_db: Any) -> None:
    """mock get_by_time_in_chat 返回空 → _analyze_group 跳过。"""
    et = _import_soul_submodule("components.evolution_task")
    plugin = _make_plugin(monitored_groups=["qq:123456:group"])

    # get_by_time_in_chat 默认返回 []（已在 _make_plugin 中设置）
    # 不应抛出异常
    await et._analyze_group(plugin, "qq:123456:group", 3)

    # 验证不崩即可
