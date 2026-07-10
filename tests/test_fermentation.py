"""发酵引擎测试 — v2.4.0。

覆盖：
- 发酵输入 CRUD
- 种子状态机（pending → fermenting → internalized）
- 发酵窗口延长
- 每群每天种子上限
- 关键词 L1 过滤

从宿主仓根运行：``uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/test_fermentation.py -q``
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from .conftest import _import_soul_submodule


@pytest.fixture
def ferm_db(tmp_path: Path):
    """专用发酵测试 fixture。"""
    im = _import_soul_submodule("models.ideology_model")
    im.init_db(tmp_path / "soul.db")
    yield im
    im.close_db()


def _create_test_seed(im, seed_id: str = "seed_test1", stream_id: str = "test_group", status: str = "pending") -> str:
    """创建测试种子。"""
    im.create_thought_seed(
        seed_id=seed_id,
        stream_id=stream_id,
        seed_type="真诚与虚伪的冲突",
        event="测试事件",
        intensity=85,
        confidence=75,
        evidence_json='["A: 测试证据"]',
        reasoning="测试原因",
        potential_impact_json='{"sincerity": 2}',
        context_json='["A: 上下文"]',
        status=status,
    )
    return seed_id


class TestFermentationInputCRUD:
    """发酵输入 CRUD 测试。"""

    def test_add_and_get_fermentation_input(self, ferm_db):
        seed_id = _create_test_seed(ferm_db)
        input_id = ferm_db.add_fermentation_input(seed_id, "test_group", "B: 这段对话确实涉及真诚", 0.85)
        assert input_id.startswith("fi_")

        inputs = ferm_db.get_fermentation_inputs(seed_id)
        assert len(inputs) == 1
        assert inputs[0].seed_id == seed_id
        assert inputs[0].message_text == "B: 这段对话确实涉及真诚"
        assert abs(inputs[0].relevance_score - 0.85) < 0.01

    def test_count_fermentation_inputs(self, ferm_db):
        seed_id = _create_test_seed(ferm_db)
        assert ferm_db.count_fermentation_inputs(seed_id) == 0

        ferm_db.add_fermentation_input(seed_id, "g", "msg1", 0.8)
        ferm_db.add_fermentation_input(seed_id, "g", "msg2", 0.6)
        ferm_db.add_fermentation_input(seed_id, "g", "msg3", 0.3)
        assert ferm_db.count_fermentation_inputs(seed_id) == 3

    def test_delete_fermentation_inputs(self, ferm_db):
        seed_id = _create_test_seed(ferm_db)
        ferm_db.add_fermentation_input(seed_id, "g", "msg1", 0.8)
        ferm_db.add_fermentation_input(seed_id, "g", "msg2", 0.6)

        deleted = ferm_db.delete_fermentation_inputs(seed_id)
        assert deleted == 2
        assert ferm_db.count_fermentation_inputs(seed_id) == 0

    def test_fermentation_inputs_ordered_by_time(self, ferm_db):
        seed_id = _create_test_seed(ferm_db)
        ferm_db.add_fermentation_input(seed_id, "g", "first", 0.8)
        ferm_db.add_fermentation_input(seed_id, "g", "second", 0.7)
        ferm_db.add_fermentation_input(seed_id, "g", "third", 0.6)

        inputs = ferm_db.get_fermentation_inputs(seed_id)
        assert len(inputs) == 3
        assert inputs[0].message_text == "first"
        assert inputs[2].message_text == "third"


class TestSeedStateMachine:
    """种子状态机测试：pending → fermenting → internalized。"""

    def test_mark_seed_fermenting_from_pending(self, ferm_db):
        seed_id = _create_test_seed(ferm_db)
        ok = ferm_db.mark_seed_fermenting(seed_id)
        assert ok

        seed = ferm_db.get_thought_seed_by_id(seed_id)
        assert seed.status == "fermenting"
        assert seed.fermentation_started_at is not None
        assert seed.fermentation_checked_at is not None

    def test_mark_seed_fermenting_atomic_guard(self, ferm_db):
        """非 pending 状态不能转发酵。"""
        seed_id = _create_test_seed(ferm_db)
        ferm_db.update_seed_status(seed_id, "approved", expected_status="pending")
        ok = ferm_db.mark_seed_fermenting(seed_id)
        assert not ok

    def test_mark_seed_internalized_from_fermenting(self, ferm_db):
        seed_id = _create_test_seed(ferm_db)
        ferm_db.mark_seed_fermenting(seed_id)
        ok = ferm_db.mark_seed_internalized(seed_id)
        assert ok

        seed = ferm_db.get_thought_seed_by_id(seed_id)
        assert seed.status == "internalized"

    def test_mark_seed_internalized_atomic_guard(self, ferm_db):
        """非 fermenting 状态不能转 internalized。"""
        seed_id = _create_test_seed(ferm_db)
        ok = ferm_db.mark_seed_internalized(seed_id)
        assert not ok

    def test_get_fermenting_seeds(self, ferm_db):
        _create_test_seed(ferm_db, "seed_a", "group_a")
        _create_test_seed(ferm_db, "seed_b", "group_b")
        _create_test_seed(ferm_db, "seed_c", "group_c")

        ferm_db.mark_seed_fermenting("seed_a")
        ferm_db.mark_seed_fermenting("seed_b")

        fermenting = ferm_db.get_fermenting_seeds()
        assert len(fermenting) == 2
        ids = {s.seed_id for s in fermenting}
        assert ids == {"seed_a", "seed_b"}

    def test_update_fermentation_checked(self, ferm_db):
        seed_id = _create_test_seed(ferm_db)
        ferm_db.mark_seed_fermenting(seed_id)

        _conn_mod = _import_soul_submodule("models._conn")
        new_time = _conn_mod._dt_to_str(datetime.now() + timedelta(minutes=30))
        ok = ferm_db.update_fermentation_checked(seed_id, new_time)
        assert ok

        updated = ferm_db.get_thought_seed_by_id(seed_id)
        assert updated.fermentation_checked_at is not None

    def test_extend_fermentation_window(self, ferm_db):
        seed_id = _create_test_seed(ferm_db)
        ferm_db.mark_seed_fermenting(seed_id)

        original = ferm_db.get_thought_seed_by_id(seed_id)
        ok = ferm_db.extend_fermentation_window(seed_id)
        assert ok

        updated = ferm_db.get_thought_seed_by_id(seed_id)
        assert updated.fermentation_extension_count == original.fermentation_extension_count + 1


class TestSeedDailyCap:
    """每群每天种子上限测试。"""

    def test_count_seeds_created_today_empty(self, ferm_db):
        assert ferm_db.count_seeds_created_today("group_a") == 0

    def test_count_seeds_created_today_with_seeds(self, ferm_db):
        _create_test_seed(ferm_db, "seed_1", "group_a")
        _create_test_seed(ferm_db, "seed_2", "group_a")
        _create_test_seed(ferm_db, "seed_3", "group_b")

        assert ferm_db.count_seeds_created_today("group_a") == 2
        assert ferm_db.count_seeds_created_today("group_b") == 1

    def test_count_seeds_created_today_all_statuses(self, ferm_db):
        """所有状态的种子都应被计数。"""
        _create_test_seed(ferm_db, "seed_1", "group_a", status="pending")
        _create_test_seed(ferm_db, "seed_2", "group_a", status="approved")
        _create_test_seed(ferm_db, "seed_3", "group_a", status="rejected")

        assert ferm_db.count_seeds_created_today("group_a") == 3


class TestKeywordFilter:
    """L1 关键词过滤测试。"""

    def test_extract_keywords(self):
        fe = _import_soul_submodule("thought.fermentation_engine")
        kw = fe._extract_keywords("真诚与虚伪的冲突 这是一个测试")
        assert len(kw) > 0

    def test_keyword_overlap_score_high(self):
        fe = _import_soul_submodule("thought.fermentation_engine")
        seed_kw = fe._extract_keywords("真诚 虚伪 冲突 表演")
        score = fe._keyword_overlap_score(seed_kw, "这段对话讨论了真诚和虚伪的表演")
        assert score > 0

    def test_keyword_overlap_score_zero(self):
        fe = _import_soul_submodule("thought.fermentation_engine")
        seed_kw = fe._extract_keywords("真诚 虚伪 冲突")
        score = fe._keyword_overlap_score(seed_kw, "今天天气不错适合出去玩")
        assert score == 0.0

    def test_keyword_overlap_empty(self):
        fe = _import_soul_submodule("thought.fermentation_engine")
        assert fe._keyword_overlap_score(set(), "any message") == 0.0
        assert fe._keyword_overlap_score({"a", "b"}, "") == 0.0
