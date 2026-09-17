"""T06（后半）：低流量消息不得因「样本不足」被永久跳过。

方案原文：`T06：相同批次重复运行不重复影响，低流量消息不会因不足样本永久丢失。`

机制：演化循环以 `soul_group_evolution.last_analyzed` 为游标，只分析游标之后的消息。
如果「消息 < 5 条 → 跳过」时**顺手推进了游标**，窗口里的消息就被永久跳过——
安静群永远凑不满 5 条，于是永远丢。所以正确行为是：**样本不足只跳过、不动游标**，
消息留在窗口里继续累积，凑够了再一起分析。

（前半「相同批次重复运行不重复影响」由 `test_evolution_task.py` 的批次幂等组覆盖。）
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from .conftest import _import_soul_submodule
from .test_evolution_task import _make_plugin


def _msg(nickname: str, text: str, uid: str) -> dict[str, Any]:
    return {
        "user_info": {"platform": "qq", "user_id": uid, "user_nickname": nickname},
        "processed_plain_text": text,
    }


_LOW_FLOW_MSGS = [
    _msg("用户A", "低流量期第一条：这家店的面不错", "u1"),
    _msg("用户B", "低流量期第二条：下次一起去", "u2"),
    _msg("用户A", "低流量期第三条：说定了", "u1"),
]


def _cursor(stream_id: str = "mock_stream") -> Any:
    """读演化游标（last_analyzed）。"""
    return _import_soul_submodule("models.spectrum").get_or_create_group_evolution(
        group_id=stream_id
    ).last_analyzed


# ─── 1. 样本不足：跳过但不推进游标 ──────────────────────────────


@pytest.mark.asyncio
async def test_insufficient_sample_does_not_advance_cursor(soul_db: Any) -> None:
    """消息 < 5 条 → 跳过，**游标必须不动**（否则这批消息永久丢失）。"""
    et = _import_soul_submodule("components.evolution_task")
    plugin = _make_plugin(monitored_groups=["qq:123456:group"])
    plugin.ctx.message.get_by_time_in_chat = AsyncMock(return_value=list(_LOW_FLOW_MSGS))

    before = _cursor()
    result = await et._analyze_group(plugin, "qq:123456:group", 3)
    after = _cursor()

    assert result == "skipped"
    assert after == before, (
        "样本不足时推进了游标——窗口里的消息会被永久跳过，安静群永远丢消息"
    )


@pytest.mark.asyncio
async def test_filtered_insufficient_sample_does_not_advance_cursor(soul_db: Any) -> None:
    """过滤后不足 5 条（如全是 bot 自己/命令）同样不得推进游标。"""
    et = _import_soul_submodule("components.evolution_task")
    plugin = _make_plugin(monitored_groups=["qq:123456:group"])
    # 6 条，但其中 4 条是命令消息（会被过滤）→ 过滤后不足 5
    plugin.ctx.message.get_by_time_in_chat = AsyncMock(
        return_value=[
            _msg("用户A", "/soul_status", "u1"),
            _msg("用户B", "/soul_health", "u2"),
            _msg("用户C", "/soul_dashboard", "u3"),
            _msg("用户D", "/soul_inspect 测试", "u4"),
            _msg("用户A", "正常消息一", "u1"),
            _msg("用户B", "正常消息二", "u2"),
        ]
    )

    before = _cursor()
    result = await et._analyze_group(plugin, "qq:123456:group", 3)
    after = _cursor()

    assert result == "skipped"
    assert after == before, "过滤后样本不足却推进了游标"


# ─── 2. 低流量消息在窗口里存活到凑够样本 ────────────────────────


@pytest.mark.asyncio
async def test_low_flow_messages_survive_until_enough_sample(soul_db: Any) -> None:
    """第二轮必须以**同一取数起点**再取一次，且第一轮那 3 条要进分析输入。

    「不会永久丢失」的可验证形态有两层：
      1. 取数窗口起点不变（游标没动）——窗口里还是那批消息，宿主仍能返回它们；
      2. 这些消息的原话真的进了送给 LLM 的 prompt。
    第 1 层是承重的：取数起点一旦变晚，第一轮的消息就落在新窗口之外，
    宿主再也取不回来（第 2 层在 mock 下看不出来，故必须两条都断言）。
    """
    et = _import_soul_submodule("components.evolution_task")
    plugin = _make_plugin(monitored_groups=["qq:123456:group"])

    starts: list[str] = []

    async def _fetch(**kwargs: Any) -> list[dict[str, Any]]:
        starts.append(str(kwargs.get("start_time", "")))
        # 第一轮 3 条；第二轮给 6 条（模拟新消息到达，窗口不变则老消息仍在）
        return list(_LOW_FLOW_MSGS) if len(starts) == 1 else [
            *_LOW_FLOW_MSGS,
            _msg("用户C", "新消息一", "u3"),
            _msg("用户B", "新消息二", "u2"),
            _msg("用户D", "新消息三", "u4"),
        ]

    plugin.ctx.message.get_by_time_in_chat = AsyncMock(side_effect=_fetch)

    assert await et._analyze_group(plugin, "qq:123456:group", 3) == "skipped"

    captured: list[str] = []
    original = et.generate_soul_text

    async def _capture(_plugin: Any, prompt: str) -> dict[str, str]:
        captured.append(prompt)
        return {"response": ""}

    et.generate_soul_text = _capture
    try:
        await et._analyze_group(plugin, "qq:123456:group", 3)
    finally:
        et.generate_soul_text = original

    assert len(starts) == 2
    assert starts[1] == starts[0], (
        f"第二轮取数起点变晚了（{starts[0]} → {starts[1]}）——"
        "第一轮的低流量消息落在新窗口之外，会被永久丢失"
    )
    assert captured, "没有走到 LLM（测试前提不成立）"
    for m in _LOW_FLOW_MSGS:
        text = m["processed_plain_text"]
        assert text in captured[0], f"低流量期消息 {text!r} 没进第二轮分析输入"
