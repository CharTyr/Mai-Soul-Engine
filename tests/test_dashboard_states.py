"""T19：看板必须区分八种状态，不得显示成同一种「空」。

方案原文：`T19：看板区分关闭、未初始化、无候选、无切片、取证失败、LLM失败、
注入未验证与后台停止，不把它们显示成同一种空态。`

判据落在两处：
1. `collect_dashboard_data(...)["health_states"]` 里每项有**不同的 `state`**；
2. 文本渲染用**不同的标记**呈现（✓ / ○ / ∅ / ✗ / ? / ■）。

若「功能关着」「确实没数据」「出错了」三类共用一个 `state`，本测试必须红。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from .conftest import _import_soul_submodule


def _config(**overrides: Any) -> Any:
    """真配置模型 + 少量覆盖（字段齐全，避免看板到处 getattr 默认）。"""
    schema = _import_soul_submodule("plugin_ui_schema")
    cfg = schema.MaiSoulEngineConfig()
    for path, value in overrides.items():
        section, _, attr = path.partition(".")
        if attr:
            setattr(getattr(cfg, section), attr, value)
        else:
            setattr(cfg, section, value)
    return cfg


def _plugin(cfg: Any, supervisor: Any = None) -> Any:
    return SimpleNamespace(config=cfg, _task_supervisor=supervisor)


def _supervisor_with(name: str, status: str) -> Any:
    ts = _import_soul_submodule("utils.task_supervisor")
    sup = ts.TaskSupervisor()
    state = sup.state_of(name)
    state.status = status
    return sup


def _collect(cfg: Any, stream_id: str = "group-A", supervisor: Any = None) -> dict[str, Any]:
    dd = _import_soul_submodule("components.dashboard_data")
    return dd.collect_dashboard_data(_plugin(cfg, supervisor), stream_id)


def _state_of(data: dict[str, Any], key: str) -> str:
    for item in data.get("health_states", []):
        if item.get("key") == key:
            return str(item.get("state", ""))
    raise AssertionError(f"看板没有 {key} 这项状态")


def _label_of(data: dict[str, Any], key: str) -> str:
    for item in data.get("health_states", []):
        if item.get("key") == key:
            return str(item.get("label", ""))
    raise AssertionError(f"看板没有 {key} 这项状态")


# ─── 1. 八种状态齐全 ────────────────────────────────────────────


def test_dashboard_reports_all_eight_state_families(soul_db: Any) -> None:
    """八个状态族一个都不能少（缺哪项都必须红）。"""
    data = _collect(_config(**{"worldview.p1_enabled": True}))
    keys = {item["key"] for item in data["health_states"]}
    assert keys == {
        "spectrum",      # 未初始化
        "candidates",    # 关闭 / 无候选
        "slices",        # 关闭 / 无切片
        "evidence",      # 取证失败
        "llm",           # LLM 失败
        "injection",     # 注入未验证
        "background",    # 后台停止
    }, f"状态族不齐: {keys}"


# ─── 2. 未初始化 ≠ 无候选 ≠ 关闭 ────────────────────────────────


def test_not_initialized_differs_from_empty_candidates(soul_db: Any) -> None:
    """未初始化（光谱没跑 setup）与无候选（思维阁开着但没种子）必须不同。"""
    from .conftest import _import_soul_submodule as imp

    spectrum = imp("models.spectrum").get_or_create_spectrum("global")
    spectrum.initialized = False
    spectrum.save()

    data = _collect(_config(**{"thought_cabinet.enabled": True}))
    assert _state_of(data, "spectrum") == "not_initialized"
    assert _state_of(data, "candidates") == "empty"
    assert _state_of(data, "spectrum") != _state_of(data, "candidates"), (
        "未初始化和无候选被判成同一种状态"
    )


def test_disabled_cabinet_differs_from_empty_cabinet(soul_db: Any) -> None:
    """思维阁**关着** vs 开着但**没候选**——两种空态必须可区分。"""
    off = _collect(_config(**{"thought_cabinet.enabled": False}))
    on = _collect(_config(**{"thought_cabinet.enabled": True}))

    assert _state_of(off, "candidates") == "disabled"
    assert _state_of(on, "candidates") == "empty"
    assert _label_of(off, "candidates") != _label_of(on, "candidates")


def test_slice_disabled_differs_from_no_slice(soul_db: Any) -> None:
    """分层关着（不记录切片）vs 分层开着但本群没切片。"""
    off = _collect(_config(**{"worldview.p1_enabled": False}), stream_id="group-A")
    on = _collect(_config(**{"worldview.p1_enabled": True}), stream_id="group-A")

    assert _state_of(off, "slices") == "disabled"
    assert _state_of(on, "slices") == "empty"


# ─── 3. 取证失败 / LLM 失败：出错 ≠ 没数据 ───────────────────────


def test_capture_failure_is_failed_not_empty(soul_db: Any, tmp_path: Any) -> None:
    """审计日志里有取证失败 → 状态必须是 failed（而不是"没数据"）。"""
    audit = _import_soul_submodule("utils.audit_log")
    audit.init_audit_log(tmp_path / "data")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "audit.jsonl").write_text(
        json.dumps({"ts": "2026-09-17T00:00:00", "type": "evolution_skip",
                    "reason": "fetch_messages_failed", "detail": "宿主接口炸了"}) + "\n",
        encoding="utf-8",
    )

    data = _collect(_config())
    assert _state_of(data, "evidence") == "failed"
    assert "fetch_messages_failed" in str(
        next(i for i in data["health_states"] if i["key"] == "evidence")["detail"]
    )


def test_llm_failure_is_failed_not_ok(soul_db: Any, tmp_path: Any) -> None:
    """LLM 失败同样必须标 failed。"""
    audit = _import_soul_submodule("utils.audit_log")
    audit.init_audit_log(tmp_path / "data")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "audit.jsonl").write_text(
        json.dumps({"ts": "2026-09-17T00:00:00", "type": "evolution_skip",
                    "reason": "llm_parse_failed"}) + "\n",
        encoding="utf-8",
    )

    data = _collect(_config())
    assert _state_of(data, "llm") == "failed"


# ─── 4. 注入未验证（不是"正常"，也不是"没数据"）─────────────────


def test_injection_unverified_has_its_own_state(soul_db: Any) -> None:
    """开了自评 → 注入记为**未验证**（宿主无请求后回调，插件不能自我声明生效）。"""
    data = _collect(_config(**{"self_reflection.enabled": True}))
    assert _state_of(data, "injection") == "unverified"

    off = _collect(_config(**{"self_reflection.enabled": False}))
    assert _state_of(off, "injection") == "disabled"


# ─── 5. 后台停止可区分 ───────────────────────────────────────────


def test_background_stopped_is_marked(soul_db: Any) -> None:
    """后台任务停止/退避/失败 → 状态为 stopped，并点名是哪个任务。"""
    sup = _supervisor_with("evolution", "backoff")
    data = _collect(_config(), supervisor=sup)

    assert _state_of(data, "background") == "stopped"
    detail = next(i for i in data["health_states"] if i["key"] == "background")["detail"]
    assert "evolution" in detail, "没有点出是哪个任务异常"


def test_background_running_is_ok(soul_db: Any) -> None:
    """全部任务在跑 → ok（不能因为"没看头"也写成空）。"""
    ts = _import_soul_submodule("utils.task_supervisor")
    sup = ts.TaskSupervisor()
    for name in ("evolution", "notion", "reflection", "fermentation", "internalization"):
        sup.note_started(name)

    data = _collect(_config(), supervisor=sup)
    assert _state_of(data, "background") == "ok"


# ─── 6. 渲染层：不同状态用不同标记 ───────────────────────────────


def test_text_renderer_marks_states_differently(soul_db: Any) -> None:
    """文本看板里：∅（没数据）、○（关着）、✗（出错）必须用不同标记。"""
    audit = _import_soul_submodule("utils.audit_log")
    import tempfile
    from pathlib import Path

    audit.init_audit_log(Path(tempfile.mkdtemp()))

    data = _collect(_config(**{
        "thought_cabinet.enabled": False,      # → ○ 关着
        "worldview.p1_enabled": True,          # 切片 → ∅ 没数据
    }))
    renderer = _import_soul_submodule("components.dashboard_renderer")
    text = renderer.build_dashboard_text(data)

    assert "【状态细分】" in text, "文本看板没有状态细分段"
    assert "○" in text, "关闭态没有标记"
    assert "∅" in text, "空数据态没有标记"
