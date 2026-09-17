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


class TestF1LLMFailureNoCheckedAt:
    """F1: LLM 失败不推进 checked_at。"""

    @pytest.mark.asyncio
    async def test_llm_judge_relevance_returns_empty_on_failure(self):
        """LLM 调用异常应返回 [] 而非 [0.0]*N。"""
        fe = _import_soul_submodule("thought.fermentation_engine")
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        # 构造 mock plugin：LLM.generate 抛异常
        async def _raise(*a, **kw):
            raise RuntimeError("LLM 模拟失败")

        ctx = SimpleNamespace(call_capability=_raise)
        plugin = SimpleNamespace(ctx=ctx)

        seed = SimpleNamespace(seed_type="t1", event="e1", reasoning="r1")
        messages = ["A: 测试消息1", "B: 测试消息2"]

        result = await fe._llm_judge_relevance(plugin, seed, messages)
        assert result == [], "LLM 异常时应返回 []"

    @pytest.mark.asyncio
    async def test_llm_judge_relevance_returns_empty_on_bad_json(self):
        """非 JSON 响应应返回 []。"""
        fe = _import_soul_submodule("thought.fermentation_engine")
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        async def _bad_response(*a, **kw):
            return {"response": "not json at all"}

        ctx = SimpleNamespace(call_capability=_bad_response)
        plugin = SimpleNamespace(ctx=ctx)

        seed = SimpleNamespace(seed_type="t1", event="e1", reasoning="r1")
        messages = ["A: 测试消息"]

        result = await fe._llm_judge_relevance(plugin, seed, messages)
        assert result == [], "非 JSON 响应时应返回 []"

    @pytest.mark.asyncio
    async def test_llm_judge_relevance_returns_empty_on_non_list(self):
        """解析结果为非 list 应返回 []。"""
        fe = _import_soul_submodule("thought.fermentation_engine")
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        async def _not_list(*a, **kw):
            return {"response": '{"key": "value"}'}

        ctx = SimpleNamespace(call_capability=_not_list)
        plugin = SimpleNamespace(ctx=ctx)

        seed = SimpleNamespace(seed_type="t1", event="e1", reasoning="r1")
        messages = ["A: 测试消息"]

        result = await fe._llm_judge_relevance(plugin, seed, messages)
        assert result == [], "非 list 解析结果时应返回 []"

    @pytest.mark.asyncio
    async def test_process_fermenting_skips_checked_at_on_empty_scores(self, ferm_db):
        """_process_fermenting_seed: LLM 返回 [] 时不更新 checked_at。"""
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        fe = _import_soul_submodule("thought.fermentation_engine")
        _conn_mod = _import_soul_submodule("models._conn")

        # 准备：创建 seed 并标记为 fermenting
        seed_id = "f1_test_checked"
        _create_test_seed(ferm_db, seed_id, "test_group_checked")
        ferm_db.mark_seed_fermenting(seed_id)
        original_seed = ferm_db.get_thought_seed_by_id(seed_id)
        original_checked = original_seed.fermentation_checked_at

        # 构造 mock 配置 — 使用长窗口确保 _check_completion 不会触发延长（避免 checked_at 变化）
        tc_cfg = SimpleNamespace(
            fermentation_max_inputs=10,
            fermentation_relevance_threshold=0.5,
            fermentation_window_hours=9999,  # 窗口远未到期
            fermentation_min_inputs=3,
            fermentation_max_extensions=3,
            auto_dedup_enabled=False,
            auto_dedup_threshold=0.8,
        )
        evo_cfg = SimpleNamespace(max_chars_per_message=200)
        mon_cfg = SimpleNamespace(monitored_users=[], excluded_users=[])
        admin_cfg = SimpleNamespace(admin_user_id="")
        cfg = SimpleNamespace(
            thought_cabinet=tc_cfg, evolution=evo_cfg, monitor=mon_cfg, admin=admin_cfg,
        )

        # mock ctx.message.get_by_time_in_chat 返回一条含有关键词的消息（必须包含 seed 关键词子串）
        fake_msg = {
            "processed_plain_text": "我觉得真诚与虚伪的冲突很有意思",
            "user_info": {"user_id": "u1", "user_nickname": "Tester"},
        }
        ctx = SimpleNamespace(
            message=SimpleNamespace(get_by_time_in_chat=AsyncMock(return_value=[fake_msg])),
            chat=SimpleNamespace(get_stream_by_user_id=AsyncMock(return_value=None)),
            send=SimpleNamespace(text=AsyncMock()),
        )
        plugin = SimpleNamespace(ctx=ctx, config=cfg)

        # 替换 _llm_judge_relevance 为返回空列表
        original_llm = fe._llm_judge_relevance
        fe._llm_judge_relevance = AsyncMock(return_value=[])

        # mock resolve_host_bot_self_ids（函数内 from ..utils.runtime_resolution import，需 patch 原模块）
        rr_mod = _import_soul_submodule("utils.runtime_resolution")
        with patch.object(rr_mod, "resolve_host_bot_self_ids", AsyncMock(return_value=[])):
            try:
                await fe._process_fermenting_seed(plugin, original_seed)
            finally:
                fe._llm_judge_relevance = original_llm

        # 验证 checked_at 未更新（用 DB 字符串比较，避免 datetime 精度差异）
        conn = _conn_mod._get_conn()
        raw_row = conn.execute(
            "SELECT fermentation_checked_at FROM soul_thought_seeds WHERE seed_id = ?",
            (seed_id,),
        ).fetchone()
        raw_checked_after = str(raw_row["fermentation_checked_at"]) if raw_row else ""
        raw_original_checked = _conn_mod._dt_to_str(original_checked) if original_checked else ""
        assert raw_checked_after == raw_original_checked, (
            f"LLM 失败后 checked_at 不应推进: {raw_checked_after} != {raw_original_checked}"
        )
        after_seed = ferm_db.get_thought_seed_by_id(seed_id)
        assert after_seed.status == "fermenting"


class TestF2NoForceInternalizeWithoutEvidence:
    """F2: 无证据不强制内化。"""

    @pytest.mark.asyncio
    async def test_no_inputs_max_extensions_keeps_fermenting(self, ferm_db):
        """0 输入 + 已达 max_extensions → 不 internalize。"""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        fe = _import_soul_submodule("thought.fermentation_engine")
        _conn_mod = _import_soul_submodule("models._conn")

        seed_id = "f2_no_input"
        _create_test_seed(ferm_db, seed_id, "test_group_f2")
        ferm_db.mark_seed_fermenting(seed_id)

        # 直接把 started_at 设到过去、extension_count 设到 max
        conn = _conn_mod._get_conn()
        conn.execute(
            "UPDATE soul_thought_seeds SET fermentation_started_at = '2020-01-01T00:00:00', "
            "fermentation_extension_count = 3 WHERE seed_id = ?",
            (seed_id,),
        )
        conn.commit()

        seed = ferm_db.get_thought_seed_by_id(seed_id)

        tc_cfg = SimpleNamespace(
            fermentation_window_hours=1,
            fermentation_min_inputs=3,
            fermentation_max_extensions=3,
            auto_dedup_enabled=False,
            auto_dedup_threshold=0.8,
        )
        admin_cfg = SimpleNamespace(admin_user_id="")
        cfg = SimpleNamespace(thought_cabinet=tc_cfg, admin=admin_cfg)
        ctx = SimpleNamespace(
            chat=SimpleNamespace(get_stream_by_user_id=AsyncMock(return_value=None)),
            send=SimpleNamespace(text=AsyncMock()),
        )
        plugin = SimpleNamespace(config=cfg, ctx=ctx)

        await fe._check_completion(plugin, seed)

        # 验证种子仍为 fermenting
        after = ferm_db.get_thought_seed_by_id(seed_id)
        assert after.status == "fermenting", (
            f"输入不足+max_extensions 时应保持 fermenting，实际为 {after.status}"
        )

        # 验证没有创建 trait
        traits = ferm_db.query_crystallized_traits(stream_id="test_group_f2")
        assert len(traits) == 0

    @pytest.mark.asyncio
    async def test_sufficient_inputs_finalizes(self, ferm_db):
        """有足够输入时正常 finalize（验证 _check_completion 分叉正确）。"""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        fe = _import_soul_submodule("thought.fermentation_engine")
        _conn_mod = _import_soul_submodule("models._conn")

        seed_id = "f2_sufficient"
        _create_test_seed(ferm_db, seed_id, "test_group_f2_suff")
        ferm_db.mark_seed_fermenting(seed_id)

        # 添加足够输入
        for i in range(3):
            ferm_db.add_fermentation_input(seed_id, "test_group_f2_suff", f"msg{i}", 0.8)

        # 设 started_at 到过去、extension_count=0（确保走 finalize 而非延长）
        conn = _conn_mod._get_conn()
        conn.execute(
            "UPDATE soul_thought_seeds SET fermentation_started_at = '2020-01-01T00:00:00', "
            "fermentation_extension_count = 0 WHERE seed_id = ?",
            (seed_id,),
        )
        conn.commit()

        seed = ferm_db.get_thought_seed_by_id(seed_id)

        tc_cfg = SimpleNamespace(
            fermentation_window_hours=1,
            fermentation_min_inputs=3,
            fermentation_max_extensions=3,
            auto_dedup_enabled=False,
            auto_dedup_threshold=0.8,
        )
        admin_cfg = SimpleNamespace(admin_user_id="")
        cfg = SimpleNamespace(thought_cabinet=tc_cfg, admin=admin_cfg)
        ctx = SimpleNamespace(
            chat=SimpleNamespace(get_stream_by_user_id=AsyncMock(return_value=None)),
            send=SimpleNamespace(text=AsyncMock()),
        )
        plugin = SimpleNamespace(config=cfg, ctx=ctx)

        with patch.object(fe, "_finalize_fermentation", new=AsyncMock()) as mock_finalize:
            await fe._check_completion(plugin, seed)
            mock_finalize.assert_awaited_once()
