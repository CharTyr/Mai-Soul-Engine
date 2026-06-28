"""P1.4: 评价层测试。

验证：
- _parse_evaluation_json：代码围栏/无效/非数组
- _normalize_batch_scores：均值减法 + clamp + 仅 evaluated 项
- _evaluate_cycle：mock LLM 完整周期（self_reflection 落库 + pending 状态更新 + 相关性门槛跳过）

从宿主仓根运行：``uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/test_reflection_evaluator.py -q``
"""

from __future__ import annotations

import asyncio
import json as _json
from types import SimpleNamespace
from typing import Any

from .conftest import _import_soul_submodule


# ─── _parse_evaluation_json ───────────────────────────────────────


def test_parse_plain_json_array() -> None:
    ev = _import_soul_submodule("components.reflection_evaluator")
    out = ev._parse_evaluation_json('[{"index": 1, "evaluated": 1}]')
    assert len(out) == 1
    assert out[0]["index"] == 1


def test_parse_fenced_json() -> None:
    ev = _import_soul_submodule("components.reflection_evaluator")
    raw = '```json\n[{"index": 1}]\n```'
    assert len(ev._parse_evaluation_json(raw)) == 1


def test_parse_invalid_returns_empty() -> None:
    ev = _import_soul_submodule("components.reflection_evaluator")
    assert ev._parse_evaluation_json("not json") == []
    assert ev._parse_evaluation_json("") == []


def test_parse_non_array_returns_empty() -> None:
    ev = _import_soul_submodule("components.reflection_evaluator")
    assert ev._parse_evaluation_json('{"index": 1}') == []


# ─── _normalize_batch_scores ──────────────────────────────────────


def test_normalize_subtracts_mean_and_recenters() -> None:
    ev = _import_soul_submodule("components.reflection_evaluator")
    results = [
        {"index": 1, "evaluated": 1, "consistency_score": 80},
        {"index": 2, "evaluated": 1, "consistency_score": 60},
        {"index": 3, "evaluated": 0, "consistency_score": 0},  # 跳过项不参与
    ]
    out = ev._normalize_batch_scores(results)
    # 均值=70，重新中心化到 50：80→60，60→40
    assert out[0]["consistency_score"] == 60
    assert out[1]["consistency_score"] == 40
    # 跳过项不变
    assert out[2]["consistency_score"] == 0


def test_normalize_clamps_to_0_100() -> None:
    ev = _import_soul_submodule("components.reflection_evaluator")
    results = [
        {"index": 1, "evaluated": 1, "consistency_score": 100},
        {"index": 2, "evaluated": 1, "consistency_score": 0},
    ]
    out = ev._normalize_batch_scores(results)
    # 均值=50，100→100，0→0
    assert out[0]["consistency_score"] == 100
    assert out[1]["consistency_score"] == 0


def test_normalize_single_evaluated_is_noop() -> None:
    ev = _import_soul_submodule("components.reflection_evaluator")
    results = [{"index": 1, "evaluated": 1, "consistency_score": 75}]
    out = ev._normalize_batch_scores(results)
    # 单项均值=75，中心化到 50
    assert out[0]["consistency_score"] == 50


# ─── _evaluate_cycle（mock LLM 完整周期）──────────────────────────


def _self_reflection_config() -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        evaluation_interval_hours=6.0,
        max_replies_per_cycle=20,
        self_reflection_weight=0.5,
        auto_internalize_threshold=0,
        user_reaction_signal_enabled=False,
        relevance_gate_enabled=True,
        normalize_across_batch=False,
        pending_max_age_hours=48,
        pending_max_rows=5000,
    )


def _mock_plugin(llm_response: str) -> SimpleNamespace:
    """构造带 mock ctx.llm 的 plugin。"""

    class _FakeLLM:
        async def generate(self, prompt: str) -> str:
            return llm_response

    return SimpleNamespace(
        config=SimpleNamespace(self_reflection=_self_reflection_config()),
        ctx=SimpleNamespace(llm=_FakeLLM()),
    )


def test_evaluate_cycle_creates_reflections_and_updates_pending(soul_db: Any) -> None:
    """mock LLM 返回 2 条评价（1 social_glue 跳过 + 1 substantive 已评），验证落库与状态。"""
    ev = _import_soul_submodule("components.reflection_evaluator")
    # 先初始化光谱（get_or_create_spectrum 需要）
    soul_db.get_or_create_spectrum("global")
    # 入队 2 条 pending
    pid1 = soul_db.create_pending_reflection("g", "s1", "r1", "", "planner", "哈哈")
    pid2 = soul_db.create_pending_reflection("g", "s2", "r2", "", "replyer", "我觉得应该直接说")
    llm_resp = _json.dumps(
        [
            {"index": 1, "reply_type": "social_glue", "evaluated": 0, "consistency_score": 0,
             "deviating_axis": "", "deviating_direction": "", "reason": "纯闲聊", "self_observation_trait": None},
            {"index": 2, "reply_type": "substantive", "evaluated": 1, "consistency_score": 55,
             "deviating_axis": "directness", "deviating_direction": "low",
             "reason": "该直说时绕弯了", "self_observation_trait": None},
        ],
        ensure_ascii=False,
    )
    plugin = _mock_plugin(llm_resp)
    asyncio.run(ev._evaluate_cycle(plugin))
    # 验证 self_reflections 落库
    refs = soul_db.list_recent_reflections("g", limit=10)
    assert len(refs) == 2
    # 验证 pending 状态：social_glue→skipped, substantive→done
    counts = soul_db.count_pending_reflections()
    assert counts.get("skipped", 0) == 1
    assert counts.get("done", 0) == 1
    # 验证 substantive 那条的分值与偏离轴
    substantive = [r for r in refs if r.reply_type == "substantive"][0]
    assert substantive.consistency_score == 55
    assert substantive.deviating_axis == "directness"
    assert substantive.evaluated == 1


def test_evaluate_cycle_no_pending_is_noop(soul_db: Any) -> None:
    ev = _import_soul_submodule("components.reflection_evaluator")
    soul_db.get_or_create_spectrum("global")
    plugin = _mock_plugin("[]")
    # 无 pending，不应抛异常
    asyncio.run(ev._evaluate_cycle(plugin))
    assert soul_db.count_self_reflections()["total"] == 0


def test_evaluate_cycle_invalid_llm_response_no_crash(soul_db: Any) -> None:
    """LLM 返回乱码时不崩，pending 仍保留 pending 状态（未被错误标 done）。"""
    ev = _import_soul_submodule("components.reflection_evaluator")
    soul_db.get_or_create_spectrum("global")
    soul_db.create_pending_reflection("g", "s", "r", "", "planner", "测试")
    plugin = _mock_plugin("这不是JSON")
    asyncio.run(ev._evaluate_cycle(plugin))
    # 无 self_reflection 落库
    assert soul_db.count_self_reflections()["total"] == 0
    # pending 仍为 pending（未被错误标记）
    counts = soul_db.count_pending_reflections()
    assert counts.get("pending", 0) == 1
