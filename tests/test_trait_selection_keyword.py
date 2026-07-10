"""文本关键词相关召回 + activation_reason 单元测试。

验证 `_text_relevance_score` 与 `_select_traits` 的关键词补位阶段：
- tag 不命中但文本含 trait.name/tag/question/thought 子串 → 仍选中
- tag 命中优先于关键词
- 无关文本不误选
- activation_reason 正确记录
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from .conftest import _import_soul_submodule


# ─── mock trait 工厂 ──────────────────────────────────────────────


def _make_trait(
    trait_id: str = "t1",
    name: str = "边界意识",
    tags_json: str = '["边界","尊重"]',
    question: str = "你觉得人与人之间需要保持距离吗？",
    thought: str = "我认为适当的边界感是健康关系的基础。",
    confidence: int = 80,
    lifecycle_state: str = "active",
    spectrum_impact_json: str = '{"sincerity": 5, "engagement": 3}',
    created_at: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        trait_id=trait_id,
        stream_id="global",
        name=name,
        tags_json=tags_json,
        question=question,
        thought=thought,
        confidence=confidence,
        lifecycle_state=lifecycle_state,
        spectrum_impact_json=spectrum_impact_json,
        evidence_json="[]",
        enabled=True,
        deleted=False,
        created_at=created_at,
        ideology_layer="conduct",
        origin_stream_id="",
    )


# ─── _text_relevance_score 单元测试 ──────────────────────────────


def test_relevance_name_match() -> None:
    """trait.name 子串匹配到文本。"""
    injector = _import_soul_submodule("components.ideology_injector")
    t = _make_trait(name="边界意识")
    score, terms = injector._text_relevance_score(t, "我们需要建立边界意识")
    assert score >= 1.0
    assert "边界意识" in terms


def test_relevance_tag_match() -> None:
    """trait.tag 子串匹配到文本。"""
    injector = _import_soul_submodule("components.ideology_injector")
    t = _make_trait(tags_json='["真诚","信任"]')
    score, terms = injector._text_relevance_score(t, "这需要真诚的沟通")
    assert score >= 1.0
    assert "真诚" in terms


def test_relevance_question_match() -> None:
    """trait.question 中分词 token 匹配到文本。"""
    injector = _import_soul_submodule("components.ideology_injector")
    t = _make_trait(question="你觉得信任在关系中重要吗")
    score, terms = injector._text_relevance_score(t, "我觉得信任很重要")
    assert score >= 1.0
    assert "信任" in terms


def test_relevance_thought_match() -> None:
    """trait.thought 前 80 字中 2-gram token 匹配。"""
    injector = _import_soul_submodule("components.ideology_injector")
    t = _make_trait(
        name="简",  # 短 name 减少 term 干扰
        tags_json='["虚"]',  # 单标签减少 term 数量
        question="",
        thought="真诚是建立信任的基础，没有真诚的关系是不稳固的。",
    )
    score, terms = injector._text_relevance_score(t, "你们之间有没有真诚")
    assert score >= 1.0
    assert "真诚" in terms


def test_relevance_no_match_returns_zero() -> None:
    """无关文本返回 score=0。"""
    injector = _import_soul_submodule("components.ideology_injector")
    t = _make_trait(name="边界意识", tags_json='["边界"]', question="距离感", thought="适可而止")
    score, terms = injector._text_relevance_score(t, "今天天气不错我们去吃饭吧")
    assert score == 0.0
    assert terms == []


def test_relevance_short_terms_ignored() -> None:
    """长度 <2 字的 token 不被提取。"""
    injector = _import_soul_submodule("components.ideology_injector")
    t = _make_trait(name="A", tags_json='["x"]', question="是", thought="好")
    score, terms = injector._text_relevance_score(t, "A x 是 好")
    assert score == 0.0
    assert terms == []


def test_relevance_max_terms_limit() -> None:
    """最多提取 12 个 term。"""
    injector = _import_soul_submodule("components.ideology_injector")
    many_tags = ",".join(f'"tag{i}"' for i in range(20))
    t = _make_trait(
        name="长名称",
        tags_json=f"[{many_tags}]",
        question="问题一 问题二 问题三 问题四 问题五 问题六",
        thought="想法一 想法二 想法三 想法四 想法五 想法六",
    )
    score, terms = injector._text_relevance_score(t, "长名称 tag1 tag2 问题一 想法一")
    # 至少 name 和 tag1/tag2 应该被匹配
    assert score >= 1.0
    assert len(injector._extract_terms(t)) <= 12


# ─── _select_traits 关键词补位测试 ──────────────────────────────


def test_keyword_fill_when_tag_miss() -> None:
    """tag 不命中但 name/question/thought 命中 → 通过关键词补位选中。"""
    injector = _import_soul_submodule("components.ideology_injector")
    # trait 有 tag 但 tag 不匹配文本，但 name "边界意识" 匹配
    t1 = _make_trait(trait_id="t1", name="边界意识", tags_json='["尊重"]')
    selected, mode, picked = injector._select_traits(
        [t1], "我们需要有边界意识", "global",
        max_traits=5, fallback_recent_impact=False, now_ts=0,
    )
    assert len(selected) == 1
    assert selected[0].trait_id == "t1"
    assert mode == "keyword_fill"
    assert picked[0]["activation_reason"].startswith("keyword:")


def test_tag_hit_priority_over_keyword() -> None:
    """tag 命中的 trait 优先于关键词补位的。"""
    injector = _import_soul_submodule("components.ideology_injector")
    # t1: tag "边界" 命中 "边界很重要"
    t1 = _make_trait(trait_id="t1", name="杂项", tags_json='["边界"]')
    # t2: tag 不命中，但 name "边界" 作为关键词命中
    t2 = _make_trait(trait_id="t2", name="边界", tags_json='["文化"]', created_at="2024-01-01")
    selected, mode, picked = injector._select_traits(
        [t1, t2], "边界很重要", "global",
        max_traits=5, fallback_recent_impact=False, now_ts=0,
    )
    # t1 应为第一个（tag 命中），t2 应为第二个（关键词补位，name"边界"在"边界很重要"中）
    assert len(selected) == 2
    assert selected[0].trait_id == "t1"
    assert selected[1].trait_id == "t2"
    assert mode == "tag_hit+keyword"
    assert picked[0]["activation_reason"].startswith("tag_hit:")
    assert picked[1]["activation_reason"].startswith("keyword:")


def test_noise_text_not_matched() -> None:
    """无关短文本不触发关键词误选。"""
    injector = _import_soul_submodule("components.ideology_injector")
    t1 = _make_trait(trait_id="t1", name="边界意识", tags_json='["边界"]')
    # 短文本不含任何关键词子串
    selected, mode, picked = injector._select_traits(
        [t1], "哈哈好的", "global",
        max_traits=5, fallback_recent_impact=False, now_ts=0,
    )
    # 没有 tag 命中也没有关键词命中 → spectrum_only
    assert len(selected) == 0
    assert mode == "spectrum_only"


def test_keyword_fill_respects_max_traits() -> None:
    """关键词补位不超出 max_traits 上限。"""
    injector = _import_soul_submodule("components.ideology_injector")
    traits = [
        _make_trait(trait_id=f"t{i}", name=f"特质{i}", tags_json='["虚标签"]')
        for i in range(5)
    ]
    # 所有 name 都匹配
    selected, mode, picked = injector._select_traits(
        traits, "特质1 特质2 特质3", "global",
        max_traits=2, fallback_recent_impact=False, now_ts=0,
    )
    assert len(selected) == 2


def test_keyword_fill_with_tagless_fallback() -> None:
    """关键词补位后，无 tag trait 通过 tagless 补位。"""
    injector = _import_soul_submodule("components.ideology_injector")
    # 有 tag 但关键词命中的 trait
    t1 = _make_trait(trait_id="t1", name="直率表达", tags_json='["诚实"]')
    # 无 tag 高 impact trait
    t2 = _make_trait(
        trait_id="t2", name="社交投入", tags_json="[]",
        spectrum_impact_json='{"engagement": 10}',
    )
    selected, mode, picked = injector._select_traits(
        [t1, t2], "需要直率表达", "global",
        max_traits=5, fallback_recent_impact=False, now_ts=0,
    )
    assert len(selected) == 2
    assert mode in ("keyword+tagless", "tag_hit+keyword+tagless")
    assert any("keyword:" in p["activation_reason"] for p in picked)
    assert any(p["activation_reason"] == "tagless_impact" for p in picked)


def test_fallback_still_works() -> None:
    """无任何匹配时 fallback_recent_impact 仍生效。"""
    injector = _import_soul_submodule("components.ideology_injector")
    # 有 tag 但 tag 不匹配，关键词不匹配 → keyword 不选
    # 无 tag 零 impact → tagless 不选
    # 结果 selected 空 → fallback 选中
    t1 = _make_trait(
        trait_id="t1", name="无关特质", tags_json='["虚构标签"]',
        spectrum_impact_json='{"sincerity": 8}',
    )
    t2 = _make_trait(
        trait_id="t2", name="零影响", tags_json="[]",
        spectrum_impact_json="{}",
    )
    selected, mode, picked = injector._select_traits(
        [t1, t2], "完全无关的文字内容", "global",
        max_traits=5, fallback_recent_impact=True, now_ts=0,
    )
    assert len(selected) == 1
    assert mode == "fallback_recent_impact"
    assert picked[0]["activation_reason"] == "fallback_recent_impact"
