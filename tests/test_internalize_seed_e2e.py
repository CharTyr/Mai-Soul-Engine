"""端到端测试 internalize_seed 的三种关系路径。

mock LLM 返回预置 JSON 响应，验证：
- duplicate → 旧 trait 被 strengthened + evidence 合并
- contradicted → 新 trait 创建 + 旧 trait disabled + contradicted_by 边
- none → 新 trait 创建

从宿主仓根运行：``uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/test_internalize_seed_e2e.py -q``
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from .conftest import _import_soul_submodule


def _make_engine(llm_responses: list[dict]) -> Any:
    """构造一个注入了 mock LLM 的 InternalizationEngine。

    llm_responses: 按调用顺序返回的 LLM 响应列表。
    第一个响应是 internalize_seed 的主 prompt，
    第二个响应（如果有）是 _classify_trait_relation 的 prompt。
    """
    engine_mod = _import_soul_submodule("thought.internalization_engine")

    class _Context:
        async def call_capability(self, capability: str, timeout_ms: int, **kwargs: Any) -> dict:
            assert capability == "llm.generate"
            assert timeout_ms == 120_000
            assert kwargs.get("model") == "planner"
            if not llm_responses:
                return {"response": ""}
            return {"response": json.dumps(llm_responses.pop(0), ensure_ascii=False)}

    fake_plugin = SimpleNamespace(
        ctx=_Context(),
        config=SimpleNamespace(
            worldview=SimpleNamespace(
                p1_enabled=True,
                values_max_delta=2,
                worldview_max_delta=4,
                conduct_max_delta=6,
                local_influence_ratio=0.35,
                mood_enabled=True,
                mood_decay_hours=8.0,
                mood_inject=True,
                graph_inject=True,
            ),
        ),
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


def _make_seed(seed_id: str = "seed_test") -> dict:
    return {
        "seed_id": seed_id,
        "stream_id": "global",
        "type": "真诚与虚伪的冲突",
        "event": "群友在讨论是否应该对不熟的人说场面话",
        "intensity": 0.85,
        "confidence": 0.75,
        "evidence": ["A: 我觉得对不熟的人客气点好", "B: 但是太假了"],
        "context": ["A: 我觉得对不熟的人客气点好", "B: 但是太假了"],
        "created_at": None,
        "reasoning": "观察到群友对真诚与场面话的冲突讨论",
        "potential_impact": {"sincerity": 3, "directness": 2},
    }


_INTERNALIZE_RESPONSE = {
    "thought": "我认为在群聊中，真诚比场面话更重要，但也要看场合。",
    "ideology_layer": "values",
    "spectrum_impact": {"sincerity": 3, "engagement": 0, "closeness": 0, "directness": 2},
    "reasoning": "基于对话中体现的价值观冲突",
    "confidence": 0.85,
    "tags": ["真诚", "边界"],
}


@pytest.mark.asyncio
async def test_internalize_duplicate_merges_trait(soul_db: Any) -> None:
    """mock LLM 返回 relation=duplicate，断言 target trait 被 strengthened + evidence 合并。"""
    _create_trait(soul_db, "existing_trait", lifecycle="active")

    engine = _make_engine([
        _INTERNALIZE_RESPONSE,
        {"target_trait_id": "existing_trait", "similarity": 0.85, "relation": "duplicate", "reason": "高度重复"},
    ])

    result = await engine.internalize_seed(_make_seed())

    assert result["success"] is True
    assert result["merged"] is True
    assert result["merged_into"] == "existing_trait"

    # 验证旧 trait 被 strengthened
    trait = soul_db.get_crystallized_trait_by_id("existing_trait")
    assert trait is not None
    assert trait.lifecycle_state == "strengthened"
    # 验证 evidence 被合并（evidence_json 不再是空数组）
    ev = json.loads(trait.evidence_json or "[]")
    assert len(ev) >= 1


@pytest.mark.asyncio
async def test_internalize_contradicted_disables_old(soul_db: Any) -> None:
    """mock LLM 返回 relation=contradicted，断言新 trait 创建 + 旧 trait enabled=0 + contradicted_by 边。"""
    _create_trait(soul_db, "old_trait", lifecycle="active")

    engine = _make_engine([
        _INTERNALIZE_RESPONSE,
        {"target_trait_id": "old_trait", "similarity": 0.85, "relation": "contradicted", "reason": "立场相反"},
    ])

    result = await engine.internalize_seed(_make_seed())

    assert result["success"] is True
    assert result["merged"] is False
    new_trait_id = result["trait_id"]
    assert new_trait_id != "old_trait"

    # 验证旧 trait 被禁用
    old_trait = soul_db.get_crystallized_trait_by_id("old_trait")
    assert old_trait is not None
    assert old_trait.enabled is False
    assert old_trait.lifecycle_state == "contradicted"

    # 验证新 trait 创建
    new_trait = soul_db.get_crystallized_trait_by_id(new_trait_id)
    assert new_trait is not None
    assert new_trait.enabled is True

    # 验证 contradicted_by 边写入
    edges = soul_db.list_thought_edges_for_trait("old_trait")
    assert len(edges) >= 1
    assert edges[0].relation_type == "contradicted_by"
    assert edges[0].to_trait_id == new_trait_id


@pytest.mark.asyncio
async def test_internalize_none_creates_new(soul_db: Any) -> None:
    """mock LLM 返回 relation=none，断言新 trait 创建。"""
    engine = _make_engine([
        _INTERNALIZE_RESPONSE,
        {"target_trait_id": "", "similarity": 0.0, "relation": "none", "reason": ""},
    ])

    result = await engine.internalize_seed(_make_seed())

    assert result["success"] is True
    assert result["merged"] is False
    new_trait_id = result["trait_id"]
    assert new_trait_id

    # 验证新 trait 创建
    new_trait = soul_db.get_crystallized_trait_by_id(new_trait_id)
    assert new_trait is not None
    assert new_trait.enabled is True
    assert new_trait.lifecycle_state == "active"
