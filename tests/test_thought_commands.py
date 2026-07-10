"""测试 thought_commands.py 的关键命令 handler。

覆盖 4 个关键场景：
1. 非管理员调用 handle_seed_approve → 拒绝
2. 管理员调用 handle_seed_approve + mock ThoughtSeedManager → 成功
3. handle_trait_detail 不存在的 trait_id → 未找到
4. handle_traits_list 无 trait → 空列表

从宿主仓根运行：``uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/test_thought_commands.py -q``
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .conftest import _import_soul_submodule


def _make_plugin(
    admin_user_id: str = "qq:admin123",
    thought_cabinet_enabled: bool = True,
    card_enabled: bool = False,
) -> SimpleNamespace:
    """构造最小 plugin mock。"""
    sent_messages: list[str] = []

    class _Ctx:
        class _Send:
            async def text(self, text: str, stream_id: str = "") -> None:
                sent_messages.append(text)

        send = _Send()

    return SimpleNamespace(
        config=SimpleNamespace(
            admin=SimpleNamespace(admin_user_id=admin_user_id),
            thought_cabinet=SimpleNamespace(
                enabled=thought_cabinet_enabled,
                auto_dedup_enabled=True,
                auto_dedup_threshold=0.78,
            ),
            render=SimpleNamespace(card_enabled=card_enabled),
        ),
        ctx=_Ctx(),
        _sent_messages=sent_messages,
    )


def _get_sent(plugin: Any) -> list[str]:
    return plugin._sent_messages


# ─── handle_seed_approve ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_seed_approve_non_admin_rejected() -> None:
    """非管理员调用 handle_seed_approve → 返回拒绝消息。"""
    tc = _import_soul_submodule("components.thought_commands")
    plugin = _make_plugin(admin_user_id="qq:real_admin")
    kwargs = {
        "platform": "qq",
        "user_id": "not_admin",
        "message": {"platform": "qq", "user_info": {"user_id": "not_admin"}},
    }

    result = await tc.handle_seed_approve(plugin, "g", **kwargs)

    assert result[0] is True  # handled
    sent = _get_sent(plugin)
    assert len(sent) == 1
    assert "只有管理员" in sent[0] or "管理员" in sent[0]


@pytest.mark.asyncio
async def test_seed_approve_admin_success(soul_db: Any) -> None:
    """管理员调用 + mock ThoughtSeedManager → 返回成功消息。"""
    tc = _import_soul_submodule("components.thought_commands")
    plugin = _make_plugin(admin_user_id="qq:admin123")
    kwargs = {
        "platform": "qq",
        "user_id": "admin123",
        "text": "/soul_approve seed_abc",
        "message": {
            "platform": "qq",
            "user_info": {"user_id": "admin123"},
            "processed_plain_text": "/soul_approve seed_abc",
        },
    }

    # Mock ThoughtSeedManager.get_seed_by_id 返回一个 pending 种子
    mock_seed = {
        "seed_id": "seed_abc",
        "stream_id": "global",
        "type": "真诚与虚伪的冲突",
        "event": "讨论场面话",
        "intensity": 0.85,
        "confidence": 0.75,
        "evidence": ["A: 我觉得对不熟的人客气点好"],
        "context": ["A: 我觉得对不熟的人客气点好"],
        "reasoning": "观察到群友讨论",
        "potential_impact": {"sincerity": 3},
        "created_at": None,
        "status": "pending",
    }

    # Mock InternalizationEngine.internalize_seed 返回成功
    mock_internalize_result = {
        "success": True,
        "spectrum_impact": {"sincerity": 3, "engagement": 0, "closeness": 0, "directness": 0},
        "thought": "我认为真诚比场面话更重要",
        "trait_id": "trait_new",
        "merged": False,
        "merged_into": None,
        "dedup_similarity": None,
    }

    with patch.object(
        _import_soul_submodule("thought.seed_manager"), "ThoughtSeedManager"
    ) as MockManager:
        manager_instance = MagicMock()
        manager_instance.get_seed_by_id = AsyncMock(return_value=mock_seed)
        manager_instance.mark_seed_status = MagicMock(return_value=True)
        MockManager.from_plugin_config.return_value = manager_instance

        with patch.object(
            _import_soul_submodule("thought.internalization_engine"), "InternalizationEngine"
        ) as MockEngine:
            engine_instance = MagicMock()
            engine_instance.internalize_seed = AsyncMock(return_value=mock_internalize_result)
            MockEngine.return_value = engine_instance

            result = await tc.handle_seed_approve(plugin, "g", **kwargs)

    assert result[0] is True
    sent = _get_sent(plugin)
    assert len(sent) == 1
    assert "已批准" in sent[0] or "seed_abc" in sent[0]


# ─── handle_trait_detail ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_trait_detail_not_found(soul_db: Any) -> None:
    """管理员调用 + 不存在的 trait_id → 返回未找到消息。"""
    tc = _import_soul_submodule("components.thought_commands")
    plugin = _make_plugin(admin_user_id="qq:admin123")
    kwargs = {
        "platform": "qq",
        "user_id": "admin123",
        "text": "/soul_trait nonexistent_trait",
        "message": {
            "platform": "qq",
            "user_info": {"user_id": "admin123"},
            "processed_plain_text": "/soul_trait nonexistent_trait",
        },
    }

    result = await tc.handle_trait_detail(plugin, "g", **kwargs)

    assert result[0] is True
    sent = _get_sent(plugin)
    assert len(sent) == 1
    assert "未找到" in sent[0] or "nonexistent" in sent[0]


# ─── handle_traits_list ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_traits_list_empty(soul_db: Any) -> None:
    """管理员调用 + 无 trait → 返回空列表消息。"""
    tc = _import_soul_submodule("components.thought_commands")
    plugin = _make_plugin(admin_user_id="qq:admin123")
    kwargs = {
        "platform": "qq",
        "user_id": "admin123",
        "text": "/soul_traits",
        "message": {
            "platform": "qq",
            "user_info": {"user_id": "admin123"},
            "processed_plain_text": "/soul_traits",
        },
    }

    result = await tc.handle_traits_list(plugin, "g", **kwargs)

    assert result[0] is True
    sent = _get_sent(plugin)
    assert len(sent) == 1
    assert "没有" in sent[0] or "空" in sent[0]
