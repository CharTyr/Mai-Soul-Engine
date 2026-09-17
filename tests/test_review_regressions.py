"""审查复现转回归测试（P0 写入边界 / 队列 / 租约 / 隔离 / 演化 / 通知）。

来源：2026-09-17 两路独立审查的**已复现**问题。本文件每个测试断言「应有行为」，
修复前应失败（先红后修；不许删断言或改预期让它变绿）。

约束：不连生产、不发消息、不调用真实 LLM（全部 stub）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import pytest

from .conftest import _import_soul_submodule as imp

im = imp("models.ideology_model")
connmod = imp("models._conn")
ops = imp("models.operations")
seeds = imp("models.seeds")
q = imp("thought.internalization_queue")
eng = imp("thought.internalization_engine")
fer = imp("thought.fermentation_engine")
ev = imp("components.evolution_task")
sr = imp("models.self_reflection")
notify = imp("utils.notify")
ntf = imp("models.notifications")
hp = imp("utils.host_persona")
Schema = imp("plugin_ui_schema").MaiSoulEngineConfig
Plugin = imp("plugin").MaiSoulEnginePlugin

GLOBAL_BASE = 50


# ---------------------------------------------------------------- harness


@pytest.fixture
def db(tmp_path):
    im.close_db()
    im.init_db(tmp_path / "soul.db")
    im.get_or_create_spectrum("global")
    yield
    im.close_db()


def make_plugin(mode: str = "apply", **overrides) -> NS:
    cfg = Schema()
    cfg.plugin.mode = mode
    cfg.thought_cabinet.enabled = True
    cfg.thought_cabinet.auto_dedup_enabled = False
    cfg.evolution.evolution_enabled = False
    cfg.admin.admin_user_id = ""
    for path, value in overrides.items():
        section, _, field = path.partition("__")
        setattr(getattr(cfg, section), field, value)
    ctx = NS(
        send=NS(text=AsyncMock(return_value={"success": True})),
        chat=NS(
            get_stream_by_user_id=AsyncMock(return_value="admin_stream"),
            get_group_streams=AsyncMock(return_value=[]),
            get_private_streams=AsyncMock(return_value=[]),
        ),
    )
    return NS(config=cfg, ctx=ctx)


def make_seed(seed_id: str = "seed_test0001", stream: str = "group-A", *, ferment: bool = False):
    seeds.create_thought_seed(
        seed_id=seed_id,
        stream_id=stream,
        seed_type="互动偏好",
        event="合成测试",
        intensity=80,
        confidence=80,
        evidence_json="[]",
        reasoning="fixture",
        potential_impact_json="{}",
    )
    if ferment:
        seeds.mark_seed_fermenting(seed_id)
    return seeds.get_thought_seed_by_id(seed_id)


OPINION = {"thought": "倾向直接交流", "confidence": 0.8, "spectrum_deltas": {"sincerity": 5}}
RELATION_CONTRADICTED = {"relation": "contradicted", "target_trait_id": "global-trait", "similarity": 0.9}


async def llm_ok(*a, **k):
    return {"success": True, "response": json.dumps(OPINION, ensure_ascii=False)}


@contextlib.contextmanager
def llm_patched(behavior=None):
    with patch.object(eng, "generate_soul_text", behavior or llm_ok), patch.object(
        hp, "fetch_host_persona", AsyncMock(return_value=None)
    ), patch.object(hp, "format_persona_for_prompt", return_value=""):
        yield


def state() -> dict:
    c = connmod._get_conn()
    return {
        "sincerity": im.get_or_create_spectrum("global").sincerity,
        "traits": c.execute("SELECT COUNT(*) FROM soul_crystallized_traits").fetchone()[0],
        "seed": c.execute("SELECT status FROM soul_thought_seeds").fetchone()[0],
        "ops": [dict(x) for x in c.execute("SELECT operation_id,status FROM soul_seed_operations")],
    }


@contextlib.contextmanager
def evolution_patched(plugin):
    """演化路径的宿主桩（消息、群流、LLM、日志）。

    `get_by_time_in_chat` **尊重时间窗**（真实宿主就是这样）：游标推进后
    同一批消息不会再被返回——否则测出来的「重复分析」是 mock 的假象。
    """
    import time as _time

    msg_time = _time.time() - 30
    msgs = [
        {
            "message_id": f"m{i}",
            "platform": "qq",
            "user_info": {"platform": "qq", "user_id": "fixture-user"},
            "processed_plain_text": f"这是外部证据观点{i}",
        }
        for i in range(5)
    ]

    async def _windowed(*, chat_id="", start_time="", end_time="", **kw):
        try:
            since = float(start_time)
        except (TypeError, ValueError):
            since = 0.0
        return msgs if since < msg_time else []

    plugin.ctx.message = NS(get_by_time_in_chat=_windowed)
    patchers = [
        patch.object(ev, "resolve_monitored_group_stream", AsyncMock(return_value="group-A")),
        patch.object(ev, "resolve_host_bot_self_ids", AsyncMock(return_value=["qq:fixture-bot"])),
        patch.object(
            ev,
            "generate_soul_text",
            AsyncMock(return_value={"response": json.dumps({"spectrum_deltas": {"sincerity": 5}})}),
        ),
        patch.object(ev, "log_evolution", AsyncMock()),
        patch.object(ev, "log_evolution_skip", AsyncMock()),
    ]
    for p in patchers:
        p.start()
    try:
        yield
    finally:
        for p in patchers:
            p.stop()


# ------------------------------------------------- P0: 运行模式写入边界


@pytest.mark.asyncio
async def test_observe_mode_queue_does_not_mutate_personality(db):
    """observe 下队列消费不得写正式人格（光谱 / 观点 / 种子终态）。"""
    p = make_plugin("apply")
    make_seed()
    await q.enqueue_internalization(p, "seed_test0001")

    p.config.plugin.mode = "observe"
    with llm_patched():
        await q.run_queue_once(p)

    st = state()
    assert st["sincerity"] == GLOBAL_BASE, f"observe 改了光谱: {st}"
    assert st["traits"] == 0, f"observe 建了观点: {st}"
    assert st["seed"] == "pending", f"observe 把种子改成终态: {st}"


@pytest.mark.asyncio
async def test_off_mode_queue_does_not_mutate_personality(db):
    """off 下队列消费不得写正式人格。"""
    p = make_plugin("apply")
    make_seed()
    await q.enqueue_internalization(p, "seed_test0001")

    p.config.plugin.mode = "off"
    with llm_patched():
        await q.run_queue_once(p)

    st = state()
    assert st["sincerity"] == GLOBAL_BASE
    assert st["traits"] == 0
    assert st["seed"] == "pending"


@pytest.mark.asyncio
async def test_mode_switch_during_llm_blocks_commit(db):
    """LLM 在途时从 apply 切到 observe：返回后不得提交人格修改。"""
    p = make_plugin("apply")
    make_seed()
    await q.enqueue_internalization(p, "seed_test0001")

    entered, proceed = asyncio.Event(), asyncio.Event()

    async def blocked(*a, **k):
        entered.set()
        await proceed.wait()
        return await llm_ok()

    with llm_patched(blocked):
        task = asyncio.create_task(q.run_queue_once(p))
        await entered.wait()
        p.config.plugin.mode = "observe"
        proceed.set()
        await task

    st = state()
    assert st["sincerity"] == GLOBAL_BASE, f"在途切模式后仍提交了人格: {st}"
    assert st["traits"] == 0
    assert st["seed"] == "pending"


@pytest.mark.asyncio
async def test_observe_mode_evolution_does_not_write_personality(db):
    """observe 下演化不得写全局光谱。"""
    p = make_plugin("observe")
    p.config.evolution.ema_alpha = 0.0
    p.config.evolution.direction_resistance = 0.0
    s = im.get_or_create_spectrum("global")
    s.initialized = True
    s.save()

    with evolution_patched(p):
        await ev._analyze_group(p, "qq:fixture-group:group", 5)
        await ev._analyze_group(p, "qq:fixture-group:group", 5)

    assert im.get_or_create_spectrum("global").sincerity == GLOBAL_BASE, "observe 下演化写了全局光谱"


# ------------------------------------------- P0: 一次生效 / 租约 / 覆盖


@pytest.mark.asyncio
async def test_finish_failure_reports_failure_and_never_double_applies(db):
    """操作终结失败：不得报 done，且重试后人格影响只能有一次。"""
    p = make_plugin("apply", worldview__local_first_evolution=False)
    make_seed()
    await q.enqueue_internalization(p, "seed_test0001")

    c = connmod._get_conn()
    c.execute(
        "CREATE TRIGGER reject_finish BEFORE UPDATE OF status ON soul_seed_operations "
        "WHEN NEW.status='done' BEGIN SELECT RAISE(ABORT, 'injected finish failure'); END"
    )
    c.commit()

    with llm_patched():
        s1 = await q.run_queue_once(p)
    first = state()
    assert s1["done"] == 0, f"终结失败却报告 done: {s1}"
    assert first["traits"] == 0 and first["sincerity"] == GLOBAL_BASE, f"终结失败仍提交了人格（半成品）: {first}"

    c.execute("DROP TRIGGER reject_finish")
    c.commit()
    with llm_patched():
        await q.run_queue_once(p)
    final = state()
    assert final["sincerity"] == GLOBAL_BASE + 5 and final["traits"] == 1, f"重试重复施加人格影响: {final}"


@pytest.mark.asyncio
async def test_queue_must_not_steal_fermentation_lease(db):
    """队列不得撤销正在发酵的执行租约（否则发酵完成后会重复内化）。"""
    p = make_plugin("apply", worldview__local_first_evolution=False, thought_cabinet__fermentation_enabled=True)
    s = make_seed(ferment=True)

    entered, proceed = asyncio.Event(), asyncio.Event()

    async def blocked(*a, **k):
        entered.set()
        await proceed.wait()
        return await llm_ok()

    with llm_patched(blocked):
        task = asyncio.create_task(fer._finalize_fermentation(p, s))
        await entered.wait()
        await q.run_queue_once(p)
        mid = state()
        assert all(op["status"] != "failed" for op in mid["ops"]), f"队列把发酵执行中的租约置为 failed: {mid}"
        proceed.set()
        await task

    first = state()
    assert first["traits"] == 1 and first["sincerity"] == GLOBAL_BASE + 5, f"发酵完成应恰好生效一次: {first}"

    # 发酵循环下一轮会再尝试一次（正常时序）；不得因此再施加一次影响
    await fer._finalize_fermentation(p, seeds.get_thought_seed_by_id(s.seed_id))
    final = state()
    assert final["traits"] == 1 and final["sincerity"] == GLOBAL_BASE + 5, f"发酵被重复内化: {final}"


@pytest.mark.asyncio
async def test_expired_lease_owner_cannot_commit_after_takeover(db):
    """执行租约被接管后，旧执行者不得提交。"""
    p = make_plugin("apply", worldview__local_first_evolution=False)
    make_seed()
    await q.enqueue_internalization(p, "seed_test0001")
    c = connmod._get_conn()
    c.execute("UPDATE soul_seed_operations SET lease_expires_at='2000-01-01'")
    c.commit()

    entered, proceed = asyncio.Event(), asyncio.Event()

    async def blocked(*a, **k):
        entered.set()
        await proceed.wait()
        return await llm_ok()

    with llm_patched(blocked):
        task = asyncio.create_task(q.run_queue_once(p))
        await entered.wait()
        await q.enqueue_internalization(p, "seed_test0001")
        proceed.set()
        await task

    with llm_patched():
        await q.run_queue_once(p)

    final = state()
    assert final["traits"] == 1, f"租约接管后旧执行者仍提交: {final}"
    assert final["sincerity"] == GLOBAL_BASE + 5, f"同一操作生效多次: {final}"


@pytest.mark.asyncio
async def test_reject_during_inflight_is_not_overwritten(db):
    """管理员在途拒绝的种子，完成逻辑不得改回 approved，也不得写人格。"""
    p = make_plugin("apply", worldview__local_first_evolution=False)
    make_seed()
    await q.enqueue_internalization(p, "seed_test0001")

    entered, proceed = asyncio.Event(), asyncio.Event()

    async def blocked(*a, **k):
        entered.set()
        await proceed.wait()
        return await llm_ok()

    with llm_patched(blocked):
        task = asyncio.create_task(q.run_queue_once(p))
        await entered.wait()
        seeds.update_seed_status("seed_test0001", "rejected", expected_status="pending")
        proceed.set()
        await task

    final = state()
    assert final["seed"] == "rejected", f"在途拒绝被覆盖: {final}"
    assert final["traits"] == 0 and final["sincerity"] == GLOBAL_BASE, f"已拒绝的种子仍写了人格: {final}"


# ------------------------------------------------------- P0: 局部优先隔离


@pytest.mark.asyncio
async def test_local_first_internalization_does_not_touch_global_spectrum(db):
    """局部优先：来源群内化不得修改全局光谱。"""
    p = make_plugin("apply", worldview__local_first_evolution=True)
    make_seed(stream="group-A")
    await q.enqueue_internalization(p, "seed_test0001")

    with llm_patched():
        await q.run_queue_once(p)

    st = state()
    c = connmod._get_conn()
    scopes = [dict(x) for x in c.execute("SELECT DISTINCT stream_id FROM soul_crystallized_traits")]
    assert any(s["stream_id"] == "group-A" for s in scopes), f"新观点未写来源群: {scopes}"
    assert st["sincerity"] == GLOBAL_BASE, f"局部内化污染了全局光谱: {st}"


@pytest.mark.asyncio
async def test_local_internalization_cannot_disable_global_trait(db):
    """局部证据不得直接禁用/改写全局观点。"""
    p = make_plugin("apply", worldview__local_first_evolution=True)
    im.create_crystallized_trait(
        trait_id="global-trait",
        stream_id="global",
        seed_id="old",
        name="旧全局观点",
        question="交流方式",
        thought="偏好直接交流",
        tags_json="[]",
        confidence=75,
        evidence_json="[]",
        spectrum_impact_json="{}",
    )
    make_seed(stream="group-A")

    fake_llm = AsyncMock(
        side_effect=[
            {"response": json.dumps(OPINION, ensure_ascii=False)},
            {"response": json.dumps(RELATION_CONTRADICTED, ensure_ascii=False)},
        ]
    )
    with llm_patched(fake_llm):
        await eng.InternalizationEngine(p).internalize_seed(
            {
                "seed_id": "seed_test0001",
                "stream_id": "group-A",
                "type": "互动偏好",
                "event": "合成测试",
                "reasoning": "fixture",
                "intensity": 0.8,
                "confidence": 0.7,
                "evidence": [],
                "context": [],
                "created_at": None,
            },
            dedup={"enabled": True, "threshold": 0.78},
        )

    old = im.get_crystallized_trait_by_id("global-trait")
    assert old is not None, "全局观点丢失"
    assert old.enabled and old.lifecycle_state == "active", (
        f"局部证据改写了全局观点: enabled={old.enabled} state={old.lifecycle_state}"
    )


# --------------------------------------------------- P0: 演化批次幂等


@pytest.mark.asyncio
async def test_evolution_cursor_failure_retry_applies_once(db):
    """游标写入失败后重试同一窗口，人格影响只能生效一次。"""
    p = make_plugin("apply")
    p.config.evolution.ema_alpha = 0.0
    p.config.evolution.direction_resistance = 0.0
    s = im.get_or_create_spectrum("global")
    s.initialized = True
    s.save()
    record = im.get_or_create_group_evolution("group-A")
    # 制造真实窗口：游标退到 1 小时前（否则窗口为空，什么都分析不到）
    from datetime import datetime as _dt, timedelta as _td

    record.last_analyzed = _dt.now() - _td(hours=1)
    record.save()
    old_cursor = record.last_analyzed

    with evolution_patched(p):
        with patch.object(im.GroupEvolutionRecord, "save", side_effect=RuntimeError("injected cursor write failure")):
            r1 = await ev._analyze_group(p, "qq:fixture-group:group", 5)

    after_first = im.get_or_create_spectrum("global").sincerity
    cursor_unchanged = im.get_or_create_group_evolution("group-A").last_analyzed == old_cursor
    assert r1 != "success" or cursor_unchanged, "游标写失败却报告成功"
    assert after_first == GLOBAL_BASE, f"游标失败却已提交人格影响: {after_first}"

    with evolution_patched(p):
        await ev._analyze_group(p, "qq:fixture-group:group", 5)

    applied_once = im.get_or_create_spectrum("global").sincerity
    assert applied_once != GLOBAL_BASE, "重试后应恰好施加一次影响（不该什么都没写）"

    # 同一窗口再跑：游标已推进，不得再加一次
    with evolution_patched(p):
        await ev._analyze_group(p, "qq:fixture-group:group", 5)
    final = im.get_or_create_spectrum("global").sincerity
    assert final == applied_once, (
        f"同一窗口被重复施加影响: {final}（一次后为 {applied_once}）"
    )


# --------------------------------------------- P0: 重置确认 / P1: 关联


@pytest.mark.asyncio
async def test_reset_requires_exact_confirmation(db):
    """`/soul_reset disconfirm` 不得被当成确认。"""
    p = make_plugin("apply")
    p.config.admin.admin_user_id = "qq:fixture-admin"
    p._reset_confirm_ts = {}
    s = im.get_or_create_spectrum("global")
    s.sincerity = 77
    s.save()

    reset = imp("components.reset_command")
    audit = imp("utils.audit_log")
    with patch.object(audit, "log_reset", AsyncMock()):
        await reset.handle_reset(p, "group-A", text="/soul_reset", platform="qq", user_id="fixture-admin")
        await reset.handle_reset(p, "group-A", text="/soul_reset disconfirm", platform="qq", user_id="fixture-admin")

    assert im.get_or_create_spectrum("global").sincerity == 77, "含糊输入被解释成确认并重置了人格"


@pytest.mark.asyncio
async def test_snapshot_pairing_marks_ambiguous_instead_of_guessing(db):
    """回复乱序时不得用 FIFO 猜配对；必须标 ambiguous（不得据此改写人格）。"""
    a = sr.create_injection_snapshot("s", "s", '["A"]', "{}", "{}", "fixture")
    b = sr.create_injection_snapshot("s", "s", '["B"]', "{}", "{}", "fixture")
    claimed = sr.claim_snapshot_for_response("s", "reply-B")

    ambiguous = claimed is None or bool(getattr(claimed, "pairing_ambiguous", False))
    assert ambiguous, (
        f"乱序回复被静默配对到 snapshot={getattr(claimed, 'snapshot_id', None)}"
        f"（A={a.snapshot_id} B={b.snapshot_id}），未标记 ambiguous"
    )


# ------------------------------------------------- P1: 通知 / 候选校验


def test_candidate_validation_rejects_fabricated_evidence_and_unknown_axis(db):
    """候选校验必须拒绝伪造证据与未知光谱轴。"""
    cand = imp("thought.candidate")
    c = cand.build_trait_candidate(
        {
            "thought": "内容",
            "scope": "global",
            "evidence_refs": ["does-not-exist"],
            "spectrum_deltas": {"imaginary_axis": 999},
        },
        max_delta=10,
    )
    assert not c.valid, f"伪造证据/未知轴仍判 valid: valid={c.valid} warnings={c.warnings}"


@pytest.mark.asyncio
async def test_aggregated_notification_handles_float_time(db):
    """聚合通知的时间是浮点时间戳，不得对其直接 strftime（会崩、且内容进不了 outbox）。"""
    p = make_plugin("apply")
    p.config.admin.admin_user_id = "qq:fixture-admin"
    ev._pending_seed_notifications.append(("fixture-seed", "type", "event"))
    p.ctx.send.text = AsyncMock(return_value={"success": True})

    await ev._send_aggregated_seed_notification(p)  # 修前抛 AttributeError

    assert p.ctx.send.text.await_count == 1, "聚合通知没有真正发出去"


@pytest.mark.asyncio
async def test_aggregated_notification_not_lost_on_early_return(db):
    """早退路径（未配置管理员）不得清空待发内容——否则种子通知被静默丢弃。"""
    p = make_plugin("apply")
    p.config.admin.admin_user_id = ""
    ev._pending_seed_notifications.append(("fixture-seed", "type", "event"))

    await ev._send_aggregated_seed_notification(p)

    assert list(ev._pending_seed_notifications), "未交付的内容被清空了"


@pytest.mark.asyncio
async def test_outbox_concurrent_drain_claims_once(db):
    """并发重放不得把同一条待发通知发两次（认领必须是原子 CAS）。"""
    p = make_plugin("apply")
    ntf.enqueue_notification("test", "admin", "notice")

    started, release = asyncio.Event(), asyncio.Event()
    calls: list[dict] = []

    async def send(**kwargs):
        calls.append(kwargs)
        started.set()
        await release.wait()
        return {"success": True}

    p.ctx.send.text = send
    first = asyncio.create_task(notify.drain_notifications(p))
    await started.wait()  # 第一条已在发送中（已被认领）
    second = asyncio.create_task(notify.drain_notifications(p))
    await asyncio.sleep(0.05)  # 给第二个消费者足够时间「抢」同一条
    release.set()
    await asyncio.gather(first, second)

    assert len(calls) == 1, f"同一条通知被发送 {len(calls)} 次"


@pytest.mark.asyncio
async def test_outbox_direct_success_clears_pending(db):
    """同去重键直发成功后，不得残留待发项（否则会再发一遍）。"""
    p = make_plugin("apply")
    p.ctx.send.text = AsyncMock(side_effect=[{"success": False}, {"success": True}])

    await notify.send_or_queue(p, "notice", "admin", dedupe_key="same")
    await notify.send_or_queue(p, "notice", "admin", dedupe_key="same")

    p.ctx.send.text = AsyncMock(return_value={"success": True})
    await notify.drain_notifications(p)
    assert p.ctx.send.text.await_count == 0, "直发成功后仍重放了历史待发项"


@pytest.mark.asyncio
async def test_ambiguous_pairing_never_drives_personality_feedback(db):
    """配对歧义的回复：评估必须记为「未评」，光谱修正（会改人格）不得消费它。

    链路：同会话 2 条未认领快照 → 认领必歧义 → 评估层强制 evaluated=0
    → list_unconsumed_reflections_for_correction 只取 evaluated=1 → 无光谱副作用。
    """
    from .test_reflection_evaluator import _mock_plugin

    sr.create_injection_snapshot("g", "g", '["trait-A"]', "{}", "{}", "fixture")
    sr.create_injection_snapshot("g", "g", '["trait-B"]', "{}", "{}", "fixture")
    claimed = sr.claim_snapshot_for_response("g", "reply-1")
    assert claimed is not None and claimed.pairing_ambiguous, "同会话多快照应判为歧义"

    sr.create_pending_reflection(
        "g", "g", "reply-1", claimed.snapshot_id, "replyer", "我觉得应该直接说"
    )
    # LLM 会给一条「已评 + 显著偏离」——若没有歧义拦截，它会驱动光谱修正
    llm_resp = json.dumps(
        [
            {
                "index": 1,
                "reply_type": "substantive",
                "evaluated": 1,
                "consistency_score": 20,
                "deviating_axis": "sincerity",
                "deviating_direction": "low",
                "reason": "很敷衍",
                "self_observation_trait": None,
            }
        ],
        ensure_ascii=False,
    )

    ev_mod = imp("components.reflection_evaluator")
    await ev_mod._evaluate_cycle(_mock_plugin(llm_resp))

    refs = sr.list_recent_reflections("g", limit=10)
    assert len(refs) == 1
    assert refs[0].evaluated == 0, f"歧义配对仍被记为已评: {refs[0]}"
    assert "歧义" in (refs[0].reason or ""), f"未说明跳过原因: {refs[0].reason}"
    assert sr.list_unconsumed_reflections_for_correction(limit=10) == [], (
        "歧义配对的自评仍会驱动人格修正"
    )


@pytest.mark.asyncio
async def test_supervisor_detects_crash_without_config_update(db):
    """任务崩溃后必须有自动巡检发现它（无需配置热更），健康状态不得假绿。"""
    actual = Plugin()
    actual._evolution_loop = _crash_once
    actual.set_plugin_config(_cfg_dict(mode="apply", evolution=True))
    await actual._reconcile_background_tasks()
    await asyncio.sleep(0.3)  # 让任务真的崩掉

    sweep = getattr(actual, "_supervise_background_tasks", None)
    assert sweep is not None, "缺少自动巡检入口——任务崩溃后无人发现（假绿）"
    await sweep()
    await asyncio.sleep(0.1)

    sup = actual._task_supervisor
    st = sup.state_of("evolution")
    assert st.restart_count >= 1 or st.status == "failed", (
        f"崩溃任务未被发现/重启: status={st.status} restart_count={st.restart_count} healthy={sup.is_healthy()}"
    )
    await actual._stop_all_background_tasks()


async def _crash_once() -> None:
    raise RuntimeError("review crash")


def _cfg_dict(mode: str = "apply", **flags) -> dict:
    cfg = Schema()
    cfg.plugin.mode = mode
    cfg.evolution.evolution_enabled = bool(flags.get("evolution", False))
    cfg.thought_cabinet.enabled = bool(flags.get("thought_cabinet", False))
    return cfg.model_dump()


def test_snapshot_records_bot_identity_scope_field(db):
    """作用域字段：快照必须记录**机器人身份**，且取不到时留空而不是编造。

    来源必须是宿主 ``bot.qq_account``（权威）。同一条 session_id 在换机器人后
    可能指向不同人格数据——不带身份的快照无法回答「这条记录属于谁」。
    """
    snap_id = sr.create_injection_snapshot(
        "group-A", "sess-1", '["trait-x"]', "{}", "{}", "tag_hit",
        bot_identity="qq:12345678",
    )
    stored = sr.get_injection_snapshot(snap_id)
    assert stored is not None
    assert stored.bot_identity == "qq:12345678", "快照没有记录机器人身份"

    # 取不到身份时留空——空串表示「未知」，不是「无身份」
    snap2 = sr.create_injection_snapshot(
        "group-A", "sess-2", "[]", "{}", "{}", "spectrum_only",
    )
    stored2 = sr.get_injection_snapshot(snap2)
    assert stored2 is not None and stored2.bot_identity == ""
