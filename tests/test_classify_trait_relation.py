"""_classify_trait_relation（内省矛盾检测）的 mock LLM 测试。

矛盾检测依赖一次 LLM 调用，本测试用 AsyncMock 注入预置 JSON 响应，
锁住不依赖真实 LLM 的代码层逻辑：
- strengthened 候选的代码层兜底（即使 LLM 判矛盾也强制降级 none）；
- target_trait_id 不在候选集时返回 None；
- 正常 active 候选的矛盾关系可透传（阈值降级在 _upsert 层，不在此函数）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from .conftest import _import_soul_submodule


def _make_engine(llm_response: dict) -> Any:
    """构造一个注入了 mock LLM 的 InternalizationEngine。

    LLM.generate 是 async，返回 {"response": <json 字符串>}。
    """
    engine_mod = _import_soul_submodule("thought.internalization_engine")

    async def _fake_generate(prompt: str) -> dict:
        return {"response": json.dumps(llm_response, ensure_ascii=False)}

    fake_plugin = SimpleNamespace(
        ctx=SimpleNamespace(llm=SimpleNamespace(generate=_fake_generate))
    )
    return engine_mod.InternalizationEngine(fake_plugin)


def _create_trait(im: Any, trait_id: str, *, lifecycle: str = "active") -> None:
    im.create_crystallized_trait(
        trait_id=trait_id,
        stream_id="global",
        seed_id="",
        name="边界感",
        question="如何看待边界感",
        thought="我重视群聊里的边界感与真诚",
        tags_json='["边界", "真诚"]',
        confidence=80,
        evidence_json="[]",
        spectrum_impact_json='{"sincerity": 3}',
        ideology_layer="values",
        lifecycle_state=lifecycle,
    )


@pytest.mark.asyncio
async def test_strengthened_candidate_forced_to_none(soul_db: Any) -> None:
    """LLM 把 strengthened 候选判为 contradicted 时，代码层强制降级为 none。"""
    _create_trait(soul_db, "strong_trait", lifecycle="strengthened")
    engine = _make_engine(
        {"target_trait_id": "strong_trait", "similarity": 0.9, "relation": "contradicted", "reason": "x"}
    )

    result = await engine._classify_trait_relation(
        stream_id="global",
        new_name="边界感",
        new_question="如何看待边界感",
        new_thought="我认为群聊不需要边界感，越界才热闹",
        new_tags=["边界", "真诚"],
        threshold=0.78,
    )

    assert result is not None
    assert result["relation"] == "none", "strengthened trait 不可被判矛盾，应降级 none"


@pytest.mark.asyncio
async def test_active_candidate_contradiction_passes_through(soul_db: Any) -> None:
    """active 候选的 contradicted 关系正常透传（阈值降级在 _upsert 层处理）。"""
    _create_trait(soul_db, "active_trait", lifecycle="active")
    engine = _make_engine(
        {"target_trait_id": "active_trait", "similarity": 0.85, "relation": "contradicted", "reason": "对立"}
    )

    result = await engine._classify_trait_relation(
        stream_id="global",
        new_name="边界感",
        new_question="如何看待边界感",
        new_thought="我认为群聊不需要边界感，越界才热闹",
        new_tags=["边界", "真诚"],
        threshold=0.78,
    )

    assert result is not None
    assert result["relation"] == "contradicted"
    assert result["target_trait_id"] == "active_trait"


@pytest.mark.asyncio
async def test_unknown_target_returns_none(soul_db: Any) -> None:
    """LLM 返回的 target_trait_id 不在候选集时返回 None。"""
    _create_trait(soul_db, "real_trait", lifecycle="active")
    engine = _make_engine(
        {"target_trait_id": "ghost_trait", "similarity": 0.9, "relation": "contradicted", "reason": "x"}
    )

    result = await engine._classify_trait_relation(
        stream_id="global",
        new_name="边界感",
        new_question="如何看待边界感",
        new_thought="我认为群聊不需要边界感",
        new_tags=["边界"],
        threshold=0.78,
    )

    assert result is None, "target 不在候选集应返回 None"
