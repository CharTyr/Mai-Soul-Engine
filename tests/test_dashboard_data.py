"""dashboard_data 数据聚合测试。

验证 `collect_dashboard_data` 在空库、满状态、无 stream_id 三种场景下
返回精确的结构化字段值。
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from .conftest import _import_soul_submodule


def _make_plugin(
    *,
    p1_enabled: bool = True,
    mood_enabled: bool = True,
    graph_inject: bool = True,
    thought_cabinet_enabled: bool = True,
    notion_enabled: bool = False,
    api_enabled: bool = True,
    card_enabled: bool = True,
    self_reflection_enabled: bool = False,
) -> Any:
    """构造模拟的插件实例（仅含 collect_dashboard_data 需要的 config 属性）。"""
    return SimpleNamespace(
        config=SimpleNamespace(
            worldview=SimpleNamespace(
                p1_enabled=p1_enabled,
                mood_enabled=mood_enabled,
                graph_inject=graph_inject,
            ),
            thought_cabinet=SimpleNamespace(enabled=thought_cabinet_enabled),
            notion=SimpleNamespace(enabled=notion_enabled),
            api=SimpleNamespace(enabled=api_enabled),
            render=SimpleNamespace(card_enabled=card_enabled),
            self_reflection=SimpleNamespace(enabled=self_reflection_enabled),
        ),
    )


def _create_trait(
    im: Any,
    trait_id: str,
    *,
    layer: str = "conduct",
    lifecycle: str = "active",
    enabled: bool = True,
    deleted: bool = False,
    stream_id: str = "global",
) -> None:
    """辅助：插入一个 trait 到 soul_db。"""
    im.create_crystallized_trait(
        trait_id=trait_id,
        stream_id=stream_id,
        seed_id="",
        name=f"测试_{trait_id}",
        question="测试问题",
        thought="测试观点内容",
        tags_json='["测试"]',
        confidence=75,
        evidence_json="[]",
        spectrum_impact_json='{"sincerity": 1}',
        ideology_layer=layer,
        lifecycle_state=lifecycle,
        enabled=enabled,
        deleted=deleted,
    )


# ─── 1. 空库聚合 ─────────────────────────────────────────────────────


def test_empty_db_aggregation(soul_db: Any) -> None:
    """空库：所有聚合值应为零值/默认值，feature_flags 与 config 一致。"""
    # 空库时 get_or_create_spectrum 创建默认行（轴=50，initialized=False）
    plugin = _make_plugin(mood_enabled=False, thought_cabinet_enabled=False)

    data_mod = _import_soul_submodule("components.dashboard_data")
    result = data_mod.collect_dashboard_data(plugin, stream_id="")

    # ── 顶层旗帜 ──
    assert result["initialized"] is False
    assert result["stream_id"] == ""
    assert result["p1_enabled"] is True

    # ── 社交光谱 ──
    spec = result["spectrum"]
    assert spec["sincerity"] == 50
    assert spec["engagement"] == 50
    assert spec["closeness"] == 50
    assert spec["directness"] == 50
    assert spec["updated_at"] is not None  # 刚创建时有时间
    assert spec["last_evolution"] is not None

    # ── 三观分层 ──
    assert result["trait_counts_by_layer"] == {"values": 0, "worldview": 0, "conduct": 0}

    # ── 生命周期分布与总数 ──
    assert result["lifecycle_distribution"] == {
        "active": 0,
        "strengthened": 0,
        "expired": 0,
        "contradicted": 0,
        "weakened": 0,
        "revised": 0,
    }
    assert result["trait_total"] == 0

    # ── 短期情绪（mood_enabled=False → 仍返回 mood 行，只是 enabled=False） ──
    assert result["mood"]["enabled"] is False
    assert result["mood"]["valence"] == 0
    assert result["mood"]["arousal"] == 0
    assert result["mood"]["energy"] == 0

    # ── 群切片 ──
    assert result["group_slice"] is None

    # ── 思维阁（thought_cabinet_enabled=False） ──
    assert result["thought_cabinet"]["enabled"] is False
    assert result["thought_cabinet"]["pending_seeds"] == 0

    # ── 最近演化 ──
    assert result["recent_evolutions"] == []

    # ── 图谱边 ──
    assert result["graph_edge_total"] == 0

    # ── 功能开关 ──
    assert result["feature_flags"] == {
        "p1_enabled": True,
        "mood_enabled": False,
        "graph_inject": True,
        "thought_cabinet": False,
        "notion": False,
        "api": True,
        "card_render": True,
        "self_reflection": False,
    }

    # ── 自我评价（未启用时仅 enabled）──
    assert result["self_reflection"]["enabled"] is False


# ─── 2. 满状态聚合 ───────────────────────────────────────────────────


def test_full_state_aggregation(soul_db: Any) -> None:
    """满状态：验证各计数精确正确（生命周期分布、去重边数、演化上限、切片）。"""
    plugin = _make_plugin(mood_enabled=True, thought_cabinet_enabled=True)

    # ── ① 光谱初始化为已初始化 ──
    spectrum = soul_db.get_or_create_spectrum("global")
    spectrum.initialized = True
    spectrum.sincerity = 72
    spectrum.engagement = 45
    spectrum.closeness = 62
    spectrum.directness = 14
    spectrum.updated_at = datetime(2025, 6, 20, 10, 0, 0)
    spectrum.last_evolution = datetime(2025, 6, 20, 9, 0, 0)
    soul_db.save_spectrum(spectrum)

    # ── ② 建 8 个 trait + 1 个已删除（不计入统计） ──
    # trait_counts_by_layer 只计 deleted=0 AND enabled=1
    # trait_total 计所有 deleted=0 的（含 enabled=0）
    _create_trait(soul_db, "t1", layer="values", lifecycle="active")                    # values, active, 入 both
    _create_trait(soul_db, "t2", layer="values", lifecycle="active")                    # values, active, 入 both
    _create_trait(soul_db, "t3", layer="worldview", lifecycle="active")                 # worldview, active
    _create_trait(soul_db, "t4", layer="worldview", lifecycle="strengthened")            # worldview, strengthened
    _create_trait(soul_db, "t5", layer="conduct", lifecycle="expired", enabled=False)    # conduct, expired, enabled=0
    _create_trait(soul_db, "t6", layer="conduct", lifecycle="contradicted", enabled=False)  # conduct, contradicted, enabled=0
    _create_trait(soul_db, "t7", layer="conduct", lifecycle="weakened")                  # conduct, weakened
    _create_trait(soul_db, "t8", layer="conduct", lifecycle="revised")                   # conduct, revised
    _create_trait(soul_db, "t9", layer="conduct", lifecycle="active", deleted=True)      # 不计入任何统计

    # ── ③ 情绪 ──
    mood = soul_db.get_or_create_mood("global")
    mood.valence = 30
    mood.arousal = -10
    mood.energy = 15
    soul_db.save_mood(mood)

    # ── ④ 演化记录（7 条，仅返回最近 5 条） ──
    from datetime import timedelta

    base = datetime(2025, 6, 26, 12, 0, 0)
    for i in range(7):
        soul_db.create_evolution_history(
            timestamp=base - timedelta(hours=i),
            group_id=f"群{i}",
            sincerity_delta=i,
            engagement_delta=0,
            closeness_delta=0,
            directness_delta=0,
            reason=f"演化记录{i}",
        )

    # ── ⑤ 待审种子 ──
    soul_db.create_thought_seed(
        seed_id="s1", stream_id="global", seed_type="opinion", event="群聊讨论",
        intensity=8, confidence=75, evidence_json="[]", reasoning="觉得真诚",
        potential_impact_json='{"sincerity": 2}',
    )
    soul_db.create_thought_seed(
        seed_id="s2", stream_id="global", seed_type="opinion", event="群聊讨论",
        intensity=6, confidence=65, evidence_json="[]", reasoning="觉得投入",
        potential_impact_json='{"engagement": 1}',
    )

    # ── ⑥ 图谱边（3 条唯一边, 去重后 graph_edge_total=3） ──
    soul_db.create_thought_edge(from_trait_id="t1", to_trait_id="t2", relation_type="supports")
    soul_db.create_thought_edge(from_trait_id="t2", to_trait_id="t3", relation_type="conflicts")
    soul_db.create_thought_edge(from_trait_id="t1", to_trait_id="t3", relation_type="extends")

    # ── ⑦ 群切片 ──
    soul_db.upsert_context_slice("group", "test-stream", 3, -2, 5, 1, 10)

    # ── ⑧ 聚合 ──
    data_mod = _import_soul_submodule("components.dashboard_data")

    # 带 stream_id → group_slice 应有值
    result = data_mod.collect_dashboard_data(plugin, stream_id="test-stream")

    # ── 光谱 ──
    assert result["initialized"] is True
    assert result["spectrum"]["sincerity"] == 72
    assert result["spectrum"]["engagement"] == 45
    assert result["spectrum"]["closeness"] == 62
    assert result["spectrum"]["directness"] == 14

    # ── trait 分层计数：仅 enabled=1（t5/t6 被排除） ──
    assert result["trait_counts_by_layer"] == {"values": 2, "worldview": 2, "conduct": 2}

    # ── lifecycle 分布：所有 deleted=0（含 enabled=0） ──
    assert result["lifecycle_distribution"] == {
        "active": 3,      # t1, t2, t3
        "strengthened": 1, # t4
        "expired": 1,     # t5
        "contradicted": 1, # t6
        "weakened": 1,    # t7
        "revised": 1,     # t8
    }
    assert result["trait_total"] == 8  # 排除 t9 (deleted)

    # ── 情绪 ──
    assert result["mood"]["enabled"] is True
    assert result["mood"]["valence"] == 30
    assert result["mood"]["arousal"] == -10
    assert result["mood"]["energy"] == 15

    # ── 群切片 ──
    assert result["group_slice"] == {
        "sincerity_offset": 3,
        "engagement_offset": -2,
        "closeness_offset": 5,
        "directness_offset": 1,
        "sample_count": 10,
    }

    # ── 待审种子 ──
    assert result["thought_cabinet"]["pending_seeds"] == 2

    # ── 最近演化（最多 5 条，最新在前） ──
    # 按 id DESC 排序，最新创建（i=6 群6）排最前
    assert len(result["recent_evolutions"]) == 5
    expected_order = [("群6", 6), ("群5", 5), ("群4", 4), ("群3", 3), ("群2", 2)]
    for i, (gid, sid) in enumerate(expected_order):
        assert result["recent_evolutions"][i]["group_id"] == gid
        assert result["recent_evolutions"][i]["deltas"]["sincerity"] == sid

    # ── 图谱边去重 ──
    # 3 条边，每条被 from/to 各引用一次，但去重后为 3
    assert result["graph_edge_total"] == 3

    # ── 功能开关 ──
    assert result["feature_flags"]["p1_enabled"] is True
    assert result["feature_flags"]["mood_enabled"] is True
    assert result["feature_flags"]["graph_inject"] is True
    assert result["feature_flags"]["thought_cabinet"] is True
    assert result["feature_flags"]["notion"] is False
    assert result["feature_flags"]["api"] is True
    assert result["feature_flags"]["card_render"] is True


# ─── 3. stream_id 为空时 group_slice 为 None ─────────────────────────


def test_stream_id_empty_group_slice_none(soul_db: Any) -> None:
    """stream_id 传空串时 group_slice 应为 None（即使有切片数据也不查）。"""
    # 先插入切片数据
    soul_db.upsert_context_slice("group", "some-group", 3, -2, 5, 1, 10)

    plugin = _make_plugin()
    data_mod = _import_soul_submodule("components.dashboard_data")

    # 传空 stream_id → group_slice=None（流 ID 为空时不查询切片）
    result = data_mod.collect_dashboard_data(plugin, stream_id="")
    assert result["group_slice"] is None

    # 传有效 stream_id → group_slice=dict（对应切片存在）
    result2 = data_mod.collect_dashboard_data(plugin, stream_id="some-group")
    assert isinstance(result2["group_slice"], dict)
    assert result2["group_slice"]["sample_count"] == 10
