"""候选纪律（Phase D「候选优先」）。

LLM 的输出是**候选**，不是人格。写入之前必须结构化校验：
边界越界、类型不对、无法解析的值 → **判为无效候选并给出原因**，不得静默 clamp
或静默归零——静默处理会把「模型吐了垃圾」变成「人格悄悄变了」。

同时修一个真实崩溃点：旧代码 `int(impact.get("sincerity", 0) or 0)` 遇到
LLM 返回非数字（如 `"很真诚"`）会直接抛 ValueError。
"""

from __future__ import annotations

from typing import Any

from .conftest import _import_soul_submodule


def _cand() -> Any:
    return _import_soul_submodule("thought.candidate")


def _valid_raw(**over: Any) -> dict:
    raw = {
        "thought": "我认为真诚比圆滑更重要",
        "ideology_layer": "values",
        "spectrum_deltas": {"sincerity": 5, "engagement": 0, "closeness": 0, "directness": 0},
        "reasoning": "多次观察到",
        "confidence": 0.85,
        "tags": ["真诚", "圆滑"],
    }
    raw.update(over)
    return raw


# ─── 正常候选 ───────────────────────────────────────────────────────


def test_accepts_well_formed_candidate() -> None:
    """结构完整的候选通过，字段被规整。"""
    c = _cand().build_trait_candidate(_valid_raw(), max_delta=10)

    assert c.valid is True
    assert c.rejection_reason == ""
    assert c.thought == "我认为真诚比圆滑更重要"
    assert c.layer == "values"
    assert c.confidence == 0.85
    assert c.spectrum_deltas == {
        "sincerity": 5, "engagement": 0, "closeness": 0, "directness": 0,
    }


def test_missing_axes_default_to_zero() -> None:
    """未给出的轴按 0 处理（缺省不等于非法）。"""
    c = _cand().build_trait_candidate(
        _valid_raw(spectrum_deltas={"sincerity": 3}), max_delta=10,
    )
    assert c.valid is True
    assert c.spectrum_deltas == {
        "sincerity": 3, "engagement": 0, "closeness": 0, "directness": 0,
    }


def test_numeric_strings_are_coerced() -> None:
    """数字字符串（LLM 常见）可解析 → 接受。"""
    c = _cand().build_trait_candidate(
        _valid_raw(spectrum_deltas={"sincerity": "5"}, confidence="0.6"), max_delta=10,
    )
    assert c.valid is True
    assert c.spectrum_deltas["sincerity"] == 5
    assert abs(c.confidence - 0.6) < 1e-9


def test_missing_spectrum_deltas_is_allowed() -> None:
    """完全没有 deltas → 全 0，不构成拒绝（只形成观点、不改光谱）。"""
    raw = _valid_raw()
    raw.pop("spectrum_deltas")
    c = _cand().build_trait_candidate(raw, max_delta=10)

    assert c.valid is True
    assert all(v == 0 for v in c.spectrum_deltas.values())


# ─── 拒绝：会污染人格或会崩溃的输入 ─────────────────────────────────


def test_rejects_empty_thought() -> None:
    """空观点 → 拒绝（否则写出一条没有内容的人格）。"""
    c = _cand().build_trait_candidate(_valid_raw(thought="   "), max_delta=10)
    assert c.valid is False
    assert c.rejection_reason == "empty_thought"


def test_rejects_non_mapping() -> None:
    """不是 dict → 拒绝。"""
    c = _cand().build_trait_candidate("不是 JSON 对象", max_delta=10)
    assert c.valid is False
    assert c.rejection_reason == "not_a_mapping"


def test_rejects_non_numeric_delta() -> None:
    """delta 无法解析为数字 → 拒绝（旧代码会在这里抛 ValueError）。"""
    c = _cand().build_trait_candidate(
        _valid_raw(spectrum_deltas={"sincerity": "很真诚"}), max_delta=10,
    )
    assert c.valid is False
    assert "delta_not_numeric" in c.rejection_reason
    assert "sincerity" in c.rejection_reason


def test_rejects_delta_exceeding_max() -> None:
    """delta 越界 → 拒绝，而不是静默 clamp。"""
    c = _cand().build_trait_candidate(
        _valid_raw(spectrum_deltas={"sincerity": 999}), max_delta=10,
    )
    assert c.valid is False
    assert "delta_exceeds_max" in c.rejection_reason


def test_rejects_confidence_out_of_range() -> None:
    """置信度越界 → 拒绝，而不是静默夹到 [0,1]。"""
    c = _cand().build_trait_candidate(_valid_raw(confidence=1.8), max_delta=10)
    assert c.valid is False
    assert "confidence_out_of_range" in c.rejection_reason


def test_rejects_confidence_wrong_type() -> None:
    """置信度类型不对 → 拒绝。"""
    c = _cand().build_trait_candidate(_valid_raw(confidence="很高"), max_delta=10)
    assert c.valid is False
    assert "confidence_not_numeric" in c.rejection_reason


def test_negative_confidence_rejected() -> None:
    """负数置信度 → 拒绝。"""
    c = _cand().build_trait_candidate(_valid_raw(confidence=-0.2), max_delta=10)
    assert c.valid is False


def test_bool_is_not_a_number() -> None:
    """bool 是 int 的子类，但不得被当作合法数值。"""
    c = _cand().build_trait_candidate(_valid_raw(confidence=True), max_delta=10)
    assert c.valid is False
    assert "confidence_not_numeric" in c.rejection_reason


# ─── 噪声处理（记录而非拒绝） ───────────────────────────────────────


def test_unknown_axes_are_dropped_with_warning() -> None:
    """未知轴名丢弃并记录告警——不拒绝整条候选，也不让它静默消失。"""
    c = _cand().build_trait_candidate(
        _valid_raw(spectrum_deltas={"sincerity": 2, "honesty": 9}), max_delta=10,
    )
    assert c.valid is True
    assert "honesty" not in c.spectrum_deltas
    assert any("honesty" in w for w in c.warnings)


def test_non_list_tags_are_dropped_with_warning() -> None:
    """tags 不是列表 → 清空并告警，不拒绝候选。"""
    c = _cand().build_trait_candidate(_valid_raw(tags="真诚"), max_delta=10)
    assert c.valid is True
    assert c.tags == ()
    assert any("tags" in w for w in c.warnings)


def test_non_string_tags_are_filtered() -> None:
    """tags 里的非字符串元素被过滤。"""
    c = _cand().build_trait_candidate(
        _valid_raw(tags=["真诚", 123, "", None, "圆滑"]), max_delta=10,
    )
    assert c.valid is True
    assert c.tags == ("真诚", "圆滑")


def test_unknown_layer_falls_back_with_warning() -> None:
    """层名无法识别 → 回落到给定默认层并告警。"""
    c = _cand().build_trait_candidate(
        _valid_raw(ideology_layer="超能力"), max_delta=10, default_layer="conduct",
    )
    assert c.valid is True
    assert c.layer == "conduct"
    assert any("layer" in w.lower() for w in c.warnings)


def test_rejection_reason_is_machine_readable_prefix() -> None:
    """拒绝原因以稳定的机器可读前缀开头（便于统计与告警）。"""
    c = _cand().build_trait_candidate(_valid_raw(thought=""), max_delta=10)
    assert c.rejection_reason.split(":")[0] == "empty_thought"


def test_spectrum_impact_alias_is_honoured() -> None:
    """`spectrum_impact` 是历史别名，必须与 `spectrum_deltas` 同等对待。

    否则旧 prompt / 旧模型输出会被静默当成「零光谱影响」。
    """
    raw = _valid_raw()
    raw.pop("spectrum_deltas")
    raw["spectrum_impact"] = {"sincerity": 3, "directness": 2}

    c = _cand().build_trait_candidate(raw, max_delta=10)

    assert c.valid is True
    assert c.spectrum_deltas["sincerity"] == 3
    assert c.spectrum_deltas["directness"] == 2


def test_spectrum_deltas_takes_precedence_over_alias() -> None:
    """两个键同时存在时以 `spectrum_deltas` 为准。"""
    raw = _valid_raw(spectrum_deltas={"sincerity": 4})
    raw["spectrum_impact"] = {"sincerity": 99}

    c = _cand().build_trait_candidate(raw, max_delta=10)

    assert c.valid is True
    assert c.spectrum_deltas["sincerity"] == 4


# ─── 集成：无效候选不得写入任何人格状态 ─────────────────────────────


def _engine(llm_response: dict) -> Any:
    """构造注入了 mock LLM 的 InternalizationEngine（独立于 e2e 文件）。"""
    import json as _json

    engine_mod = _import_soul_submodule("thought.internalization_engine")
    responses = [llm_response, {"target_trait_id": "", "similarity": 0.0, "relation": "none"}]

    class _Context:
        async def call_capability(self, capability: str, timeout_ms: int, **kwargs: Any) -> dict:
            payload = responses.pop(0) if responses else {}
            return {"response": _json.dumps(payload, ensure_ascii=False)}

    from types import SimpleNamespace

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
            thought_cabinet=SimpleNamespace(max_internalize_delta=10, fermented_max_internalize_delta=15),
        ),
    )
    return engine_mod.InternalizationEngine(fake_plugin)


def _seed() -> dict:
    return {
        "seed_id": "seed_cand",
        "id": "seed_cand",
        "stream_id": "global",
        "type": "价值观冲突",
        "event": "群友讨论真诚与圆滑",
        "reasoning": "观察",
        "intensity": 0.8,
        "confidence": 0.7,
        "evidence": [],
        "context": [],
        "created_at": None,
    }


def test_invalid_candidate_writes_nothing(soul_db: Any) -> None:
    """非数字 delta 的候选 → 拒绝、返回结构化原因、光谱与 trait 都不动。"""
    import asyncio

    before = soul_db.get_or_create_spectrum("global").sincerity
    traits_before = len(soul_db.query_crystallized_traits(deleted=False, limit=50))

    engine = _engine(_valid_raw(spectrum_deltas={"sincerity": "很真诚"}))
    result = asyncio.run(engine.internalize_seed(_seed()))

    assert result["success"] is False
    assert result["candidate_rejected"] is True
    assert result["rejection_reason"].startswith("delta_not_numeric")
    assert soul_db.get_or_create_spectrum("global").sincerity == before
    assert len(soul_db.query_crystallized_traits(deleted=False, limit=50)) == traits_before


def test_empty_thought_candidate_writes_nothing(soul_db: Any) -> None:
    """空观点候选 → 拒绝且不写入。"""
    import asyncio

    traits_before = len(soul_db.query_crystallized_traits(deleted=False, limit=50))
    engine = _engine(_valid_raw(thought="   "))
    result = asyncio.run(engine.internalize_seed(_seed()))

    assert result["success"] is False
    assert result["rejection_reason"] == "empty_thought"
    assert len(soul_db.query_crystallized_traits(deleted=False, limit=50)) == traits_before


def test_out_of_range_delta_rejected_not_clamped(soul_db: Any) -> None:
    """越界 delta → 拒绝而不是被 clamp 成上限（静默 clamp 会伪装成合法影响）。"""
    import asyncio

    before = soul_db.get_or_create_spectrum("global").sincerity
    engine = _engine(_valid_raw(spectrum_deltas={"sincerity": 999}))
    result = asyncio.run(engine.internalize_seed(_seed()))

    assert result["success"] is False
    assert "delta_exceeds_max" in result["rejection_reason"]
    assert soul_db.get_or_create_spectrum("global").sincerity == before


def test_valid_candidate_still_writes(soul_db: Any) -> None:
    """正常候选照常写入（校验不得把正常路径挡掉）。"""
    import asyncio

    engine = _engine(_valid_raw())
    result = asyncio.run(engine.internalize_seed(_seed()))

    assert result["success"] is True
    assert soul_db.get_crystallized_trait_by_id(result["trait_id"]) is not None


# ─── 局部优先演化（默认保持全局，需显式开启） ────────────────────────


def _engine_with_scope(local_first: bool) -> Any:
    engine = _engine(_valid_raw())
    engine._plugin.config.worldview.local_first_evolution = local_first
    return engine


def test_default_scope_is_global(soul_db: Any) -> None:
    """显式关闭局部优先：观点写全局，来源群只作溯源（旧行为，仍可切回）。"""
    engine = _engine_with_scope(False)
    stream_id, origin = engine._trait_scope_for_seed({"stream_id": "qq-123-group"})

    assert stream_id == "global"
    assert origin == "qq-123-group"


def test_local_first_scope_writes_group(soul_db: Any) -> None:
    """开启后：观点写入来源群，只影响该群。"""
    engine = _engine_with_scope(True)
    stream_id, origin = engine._trait_scope_for_seed({"stream_id": "qq-123-group"})

    assert stream_id == "qq-123-group"
    assert origin == "qq-123-group"


def test_local_first_without_origin_still_global(soul_db: Any) -> None:
    """来源群未知（空/global）→ 仍写全局，不产生无主的局部 trait。"""
    engine = _engine_with_scope(True)

    assert engine._trait_scope_for_seed({"stream_id": ""})[0] == "global"
    assert engine._trait_scope_for_seed({"stream_id": "global"})[0] == "global"


def test_local_first_is_the_default_scope() -> None:
    """默认局部优先：单群输入不足以改写 bot 的全局人格。

    要恢复旧行为（观点直接写全局）把 ``local_first_evolution`` 设为 false。
    """
    schema = _import_soul_submodule("plugin_ui_schema")
    assert schema.WorldviewConfig().local_first_evolution is True


def test_local_first_end_to_end_writes_group_scoped_trait(soul_db: Any) -> None:
    """端到端：开启开关后，内化出的 trait 落在来源群。"""
    import asyncio

    engine = _engine_with_scope(True)
    seed = _seed()
    seed["stream_id"] = "qq-123-group"

    result = asyncio.run(engine.internalize_seed(seed))

    assert result["success"] is True
    trait = soul_db.get_crystallized_trait_by_id(result["trait_id"])
    assert trait.stream_id == "qq-123-group"
    assert trait.origin_stream_id == "qq-123-group"
