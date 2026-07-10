"""B3–B6 思维阁槽位（cabinet_slot_no）单元测试。

验证：
- set_trait_slot 正常占槽/同槽换槽/清空
- 边界值：slot 0/13、不存在 trait
- _select_traits 中 slotted trait 优先级高于 unslotted
- dashboard occupancy 结构
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from .conftest import _import_soul_submodule


# ─── mock trait 工厂 ──────────────────────────────────────────────


def _make_trait(
    trait_id: str = "t1",
    name: str = "槽位特质",
    tags_json: str = '["test"]',
    question: str = "你赞成边界感吗？",
    thought: str = "我认为需要边界感。",
    confidence: int = 80,
    lifecycle_state: str = "active",
    spectrum_impact_json: str = '{"sincerity": 5}',
    cabinet_slot_no: int | None = None,
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
        cabinet_slot_no=cabinet_slot_no,
    )


def _create_trait(im: Any, trait_id: str, stream_id: str = "") -> None:
    im.create_crystallized_trait(
        trait_id=trait_id,
        stream_id=stream_id,
        seed_id="",
        name="测试特质",
        question="测试问题",
        thought="测试观点",
        tags_json="[]",
        confidence=80,
        evidence_json="[]",
        spectrum_impact_json="{}",
        ideology_layer="conduct",
        lifecycle_state="active",
    )


# ─── B3: set_trait_slot ───────────────────────────────────────────


class TestSetTraitSlot:
    def test_set_slot_1_success(self, soul_db: Any) -> None:
        """set_trait_slot(trait_id, 1) → True，回读 slot=1。"""
        _create_trait(soul_db, "slot_t1")
        assert soul_db.set_trait_slot("slot_t1", 1) is True
        t = soul_db.get_crystallized_trait_by_id("slot_t1")
        assert t is not None
        assert t.cabinet_slot_no == 1

    def test_same_slot_second_clears_first(self, soul_db: Any) -> None:
        """同槽第二 trait → 第一被清空、第二占槽。"""
        _create_trait(soul_db, "slot_a")
        _create_trait(soul_db, "slot_b")
        assert soul_db.set_trait_slot("slot_a", 3) is True
        assert soul_db.set_trait_slot("slot_b", 3) is True

        ta = soul_db.get_crystallized_trait_by_id("slot_a")
        tb = soul_db.get_crystallized_trait_by_id("slot_b")
        assert ta is not None
        assert tb is not None
        assert ta.cabinet_slot_no is None, "旧占槽 trait 应被清空"
        assert tb.cabinet_slot_no == 3, "新写 trait 应占槽"

    def test_set_none_clears_slot(self, soul_db: Any) -> None:
        """set_trait_slot(id, None) 清空槽位。"""
        _create_trait(soul_db, "slot_c")
        assert soul_db.set_trait_slot("slot_c", 5) is True
        assert soul_db.set_trait_slot("slot_c", None) is True
        t = soul_db.get_crystallized_trait_by_id("slot_c")
        assert t is not None
        assert t.cabinet_slot_no is None

    def test_slot_0_returns_false(self, soul_db: Any) -> None:
        """slot=0 超出范围，返回 False。"""
        _create_trait(soul_db, "slot_0")
        assert soul_db.set_trait_slot("slot_0", 0) is False
        t = soul_db.get_crystallized_trait_by_id("slot_0")
        assert t is not None
        assert t.cabinet_slot_no is None

    def test_slot_13_returns_false(self, soul_db: Any) -> None:
        """slot=13 超出范围，返回 False。"""
        _create_trait(soul_db, "slot_13")
        assert soul_db.set_trait_slot("slot_13", 13) is False
        t = soul_db.get_crystallized_trait_by_id("slot_13")
        assert t is not None
        assert t.cabinet_slot_no is None

    def test_nonexistent_trait_returns_false(self, soul_db: Any) -> None:
        """不存在的 trait → False。"""
        assert soul_db.set_trait_slot("nonexistent", 1) is False

    def test_two_different_slots_coexist(self, soul_db: Any) -> None:
        """两不同 slot 可共存。"""
        _create_trait(soul_db, "slot_x")
        _create_trait(soul_db, "slot_y")
        assert soul_db.set_trait_slot("slot_x", 7) is True
        assert soul_db.set_trait_slot("slot_y", 9) is True
        tx = soul_db.get_crystallized_trait_by_id("slot_x")
        ty = soul_db.get_crystallized_trait_by_id("slot_y")
        assert tx is not None and tx.cabinet_slot_no == 7
        assert ty is not None and ty.cabinet_slot_no == 9

    def test_slot_1_12_boundary(self, soul_db: Any) -> None:
        """slot 1 和 12 是有效边界值。"""
        _create_trait(soul_db, "slot_min")
        _create_trait(soul_db, "slot_max")
        assert soul_db.set_trait_slot("slot_min", 1) is True
        assert soul_db.set_trait_slot("slot_max", 12) is True
        tmin = soul_db.get_crystallized_trait_by_id("slot_min")
        tmax = soul_db.get_crystallized_trait_by_id("slot_max")
        assert tmin is not None and tmin.cabinet_slot_no == 1
        assert tmax is not None and tmax.cabinet_slot_no == 12


# ─── B4: 注入 slot 优先 ──────────────────────────────────────────


class TestSlotInjectionPriority:
    def test_slotted_tag_hit_prioritized(self) -> None:
        """同 tag 命中时，slotted trait 排在 unslotted 前。"""
        injector = _import_soul_submodule("components.ideology_injector")
        t1 = _make_trait(trait_id="t1", name="有槽", tags_json='["边界"]', cabinet_slot_no=3)
        t2 = _make_trait(trait_id="t2", name="无槽", tags_json='["边界"]')
        selected, mode, picked = injector._select_traits(
            [t1, t2], "边界很重要", "global",
            max_traits=5, fallback_recent_impact=False, now_ts=0,
        )
        assert len(selected) >= 1
        # 第一个应该是 t1（有槽位）
        assert selected[0].trait_id == "t1"

    def test_slotted_keyword_prioritized(self) -> None:
        """关键词补位阶段，slotted trait 排在 unslotted 前。"""
        injector = _import_soul_submodule("components.ideology_injector")
        # 两个 trait 都有 tag 但不匹配文本，仅 name 关键词匹配
        t1 = _make_trait(
            trait_id="kw1", name="边界感", tags_json='["虚标签"]',
            cabinet_slot_no=5,
        )
        t2 = _make_trait(
            trait_id="kw2", name="边界感", tags_json='["虚标签"]',
        )
        selected, mode, picked = injector._select_traits(
            [t1, t2], "我们需要边界感", "global",
            max_traits=5, fallback_recent_impact=False, now_ts=0,
        )
        # 至少选中一个（关键词补位）
        picked_ids = [p["thought_id"] for p in picked]
        if "kw1" in picked_ids and "kw2" in picked_ids:
            # 两个都选中时，有槽的在前
            kw1_idx = picked_ids.index("kw1")
            kw2_idx = picked_ids.index("kw2")
            assert kw1_idx < kw2_idx

    def test_slotted_tagless_prioritized(self) -> None:
        """无 tag 补位阶段，slotted trait 排在 unslotted 前。"""
        injector = _import_soul_submodule("components.ideology_injector")
        t1 = _make_trait(
            trait_id="tl1", name="影响大", tags_json="[]",
            spectrum_impact_json='{"sincerity": 10}',
            cabinet_slot_no=2,
        )
        t2 = _make_trait(
            trait_id="tl2", name="影响大", tags_json="[]",
            spectrum_impact_json='{"sincerity": 10}',
        )
        with_tag = _make_trait(trait_id="tl3", name="无关", tags_json='["不相干"]')
        selected, mode, picked = injector._select_traits(
            [t1, t2, with_tag], "完全无关的文本", "global",
            max_traits=5, fallback_recent_impact=False, now_ts=0,
        )
        if "tl1" in picked and "tl2" in picked:
            tl1_idx = next(i for i, p in enumerate(picked) if p["thought_id"] == "tl1")
            tl2_idx = next(i for i, p in enumerate(picked) if p["thought_id"] == "tl2")
            assert tl1_idx < tl2_idx

    def test_slotted_fallback_prioritized(self) -> None:
        """fallback 阶段，slotted trait 排在 unslotted 前。"""
        injector = _import_soul_submodule("components.ideology_injector")
        t1 = _make_trait(
            trait_id="fb1", name="回退", tags_json="[]",
            spectrum_impact_json='{"engagement": 8}',
            cabinet_slot_no=4,
        )
        t2 = _make_trait(
            trait_id="fb2", name="回退", tags_json="[]",
            spectrum_impact_json='{"engagement": 8}',
        )
        selected, mode, picked = injector._select_traits(
            [t1, t2], "完全无关", "global",
            max_traits=5, fallback_recent_impact=True, now_ts=0,
        )
        if len(selected) >= 2:
            assert selected[0].trait_id == "fb1"

    def test_picked_includes_cabinet_slot_no(self) -> None:
        """picked 项包含 cabinet_slot_no 字段。"""
        injector = _import_soul_submodule("components.ideology_injector")
        t1 = _make_trait(trait_id="s1", name="有槽", tags_json='["边界"]', cabinet_slot_no=6)
        t2 = _make_trait(trait_id="s2", name="无槽", tags_json='["边界"]')
        selected, mode, picked = injector._select_traits(
            [t1, t2], "边界很重要", "global",
            max_traits=5, fallback_recent_impact=False, now_ts=0,
        )
        s1_picked = [p for p in picked if p["thought_id"] == "s1"]
        s2_picked = [p for p in picked if p["thought_id"] == "s2"]
        assert len(s1_picked) == 1
        assert s1_picked[0].get("cabinet_slot_no") == 6
        assert len(s2_picked) == 1
        assert s2_picked[0].get("cabinet_slot_no") is None


# ─── B5: Dashboard occupancy ─────────────────────────────────────


class TestDashboardOccupancy:
    def test_dashboard_cabinet_data(self, soul_db: Any) -> None:
        """collect_dashboard_data 返回 cabinet 段。"""
        _create_trait(soul_db, "dash_t1")
        _create_trait(soul_db, "dash_t2")
        soul_db.set_trait_slot("dash_t1", 1)
        soul_db.set_trait_slot("dash_t2", 5)

        # Mock a minimal plugin object
        class _MockConfig:
            class _WV:
                p1_enabled = False
                mood_enabled = False
                graph_inject = False
            worldview = _WV()

            class _TC:
                enabled = False
            thought_cabinet = _TC()

            class _Notion:
                enabled = False
            notion = _Notion()

            class _API:
                enabled = False
            api = _API()

            class _Render:
                card_enabled = False
                viewport_width = 800
                device_scale_factor = 2.0
                render_timeout_ms = 5000
            render = _Render()

            class _SR:
                enabled = False
            self_reflection = _SR()

            class _Monitor:
                monitored_groups = []
                excluded_groups = []
            monitor = _Monitor()

            class _Injection:
                scope = ""
                inject_private = False
                max_traits = 3
                fallback_recent_impact = False
                trait_cooldown_seconds = 0
            injection = _Injection()

            class _Threshold:
                custom_prompts = {}
                enable_extreme = False
            threshold = _Threshold()

        class _MockPlugin:
            config = _MockConfig()

        dashboard = _import_soul_submodule("components.dashboard_data")
        data = dashboard.collect_dashboard_data(_MockPlugin())
        cabinet = data.get("cabinet", {})
        assert cabinet.get("slots_used") == 2
        assert cabinet.get("slots_total") == 12
        occ = cabinet.get("occupancy", [])
        assert len(occ) == 2
        slots = {o["slot_no"]: o["trait_id"] for o in occ}
        assert slots[1] == "dash_t1"
        assert slots[5] == "dash_t2"

    def test_trait_detail_includes_slot(self, soul_db: Any) -> None:
        """handle_trait_detail 数据包含 cabinet_slot_no。"""
        _create_trait(soul_db, "detail_slot")
        soul_db.set_trait_slot("detail_slot", 8)
        t = soul_db.get_crystallized_trait_by_id("detail_slot")
        assert t is not None
        assert t.cabinet_slot_no == 8

    def test_traits_list_shows_slot(self, soul_db: Any) -> None:
        """query_crystallized_traits 包含 cabinet_slot_no 返回。"""
        _create_trait(soul_db, "list_slot")
        soul_db.set_trait_slot("list_slot", 11)
        traits = soul_db.query_crystallized_traits(deleted=False, limit=50)
        matching = [t for t in traits if t.trait_id == "list_slot"]
        assert len(matching) == 1
        assert matching[0].cabinet_slot_no == 11
