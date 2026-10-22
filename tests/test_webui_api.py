"""WebUI 审核使用真实 SQLite，覆盖权限、队列、状态竞争和注册。"""

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from .conftest import _import_soul_submodule


@pytest.fixture
def webui(soul_db):
    module = _import_soul_submodule("plugin")
    plugin = module.create_plugin()
    config = _import_soul_submodule("plugin_ui_schema").MaiSoulEngineConfig()
    config.api.enabled = True
    config.plugin.mode = "apply"
    config.thought_cabinet.enabled = True
    plugin._plugin_config_instance = config
    return plugin


def create_seed(seed_id: str = "seed_test") -> None:
    seeds = _import_soul_submodule("models.seeds")
    seeds.create_thought_seed(
        seed_id=seed_id,
        stream_id="test-group",
        seed_type="真诚",
        event="讨论边界",
        intensity=80,
        confidence=75,
        evidence_json='["证据"]',
        reasoning="测试推理",
        potential_impact_json='{"sincerity": 1}',
        context_json='["原始讨论"]',
    )


@pytest.mark.asyncio
async def test_private_apis_registered_from_mixin(webui):
    components = {c["name"]: c for c in webui.get_components()}
    for name in ("overview", "seeds", "approve", "reject"):
        assert components[f"soul.webui_{name}"]["metadata"]["public"] is False


@pytest.mark.asyncio
async def test_disabled_api_blocks_read_and_write(webui):
    webui.config.api.enabled = False
    for call in (webui.webui_overview(), webui.webui_seeds(), webui.webui_approve("seed_test")):
        with pytest.raises(PermissionError, match="api.enabled"):
            await call


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "observe"])
async def test_mode_blocks_both_decisions(webui, mode: str):
    create_seed()
    webui.config.plugin.mode = mode
    blocked = _import_soul_submodule("utils.runtime_mode").MutationBlocked
    for action in (webui.webui_approve, webui.webui_reject):
        with pytest.raises(blocked):
            await action("seed_test")
    assert (await webui.webui_seeds())["rows"][0]["reviewable"] is False


@pytest.mark.asyncio
async def test_approve_queues_once_without_llm_or_chat(webui):
    create_seed()
    result = await webui.webui_approve("seed_test")
    assert result["operation_id"].startswith("op_")
    for action in (webui.webui_approve, webui.webui_reject):
        with pytest.raises(PermissionError, match="队列"):
            await action("seed_test")
    rows = (await webui.webui_seeds())["rows"]
    assert rows[0]["status"] == "queued"
    assert rows[0]["reviewable"] is False
    overview = await webui.webui_overview()
    assert len(overview["operations"]) == 1
    assert overview["pending"] == 0


@pytest.mark.asyncio
async def test_fermentation_and_repeat_review(webui):
    create_seed()
    webui.config.thought_cabinet.fermentation_enabled = True
    assert "发酵" in (await webui.webui_approve("seed_test"))["message"]
    assert (await webui.webui_seeds())["rows"][0]["status"] == "fermenting"
    with pytest.raises(PermissionError, match="待审核"):
        await webui.webui_approve("seed_test")


@pytest.mark.asyncio
async def test_reject_preserves_record(webui):
    create_seed()
    await webui.webui_reject("seed_test")
    seed = _import_soul_submodule("models.seeds").get_thought_seed_by_id("seed_test")
    assert seed.status == "rejected"
    assert (await webui.webui_seeds())["rows"] == []
    with pytest.raises(PermissionError):
        await webui.webui_approve("seed_test")


@pytest.mark.asyncio
async def test_cabinet_and_missing_seed_guard(webui):
    with pytest.raises(PermissionError, match="不存在"):
        await webui.webui_approve("missing")
    webui.config.thought_cabinet.enabled = False
    with pytest.raises(PermissionError, match="未启用"):
        await webui.webui_approve("missing")


@pytest.mark.asyncio
async def test_untrusted_seed_id_rejected(webui):
    with pytest.raises(ValidationError):
        await webui.webui_approve("' OR 1=1 --")


@pytest.mark.asyncio
async def test_monitor_detects_finished_task(webui):
    class FinishedTask:
        def done(self) -> bool:
            return True

    webui._task_supervisor.note_started("evolution")
    webui._evolution_task = FinishedTask()
    assert (await webui.webui_overview())["tasks"][0]["status"] == "失败"


def test_webui_declaration():
    import json

    declaration = json.loads((Path(__file__).parents[1] / "webui.json").read_text(encoding="utf-8"))
    assert declaration["schema_version"] == 2
    assert all(page["placement"] == "workspace" for page in declaration["pages"])
    assert all(binding["confirmation"] for binding in declaration["pages"][1]["actions"].values())


@pytest.mark.asyncio
async def test_review_locks_state_until_operation_claim(webui, monkeypatch: pytest.MonkeyPatch):
    create_seed()
    api = _import_soul_submodule("components.webui_api")
    conn = _import_soul_submodule("models._conn")._get_conn()
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    claim = api.claim_seed_operation

    def competing_claim(seed_id: str) -> str | None:
        with sqlite3.connect(db_path, timeout=0) as contender, pytest.raises(sqlite3.OperationalError, match="locked"):
            contender.execute("UPDATE soul_thought_seeds SET status='rejected' WHERE seed_id=?", (seed_id,))
        return claim(seed_id)

    monkeypatch.setattr(api, "claim_seed_operation", competing_claim)
    assert (await webui.webui_approve("seed_test"))["operation_id"].startswith("op_")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "fermenting", "queued"])
async def test_status_filter_matches_displayed_state(webui, status):
    create_seed()
    if status != "pending":
        webui.config.thought_cabinet.fermentation_enabled = status == "fermenting"
        await webui.webui_approve("seed_test")
    for candidate in ("all", "pending", "fermenting", "queued"):
        rows = (await webui.webui_seeds(candidate))["rows"]
        assert len(rows) == (1 if candidate in ("all", status) else 0)
        if rows:
            assert rows[0]["status"] == status
            assert rows[0]["position"] == 1
    with pytest.raises(ValueError):
        await webui.webui_seeds("invalid")
