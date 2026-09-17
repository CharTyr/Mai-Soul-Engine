"""data_dir 迁移：附属文件补齐（PR #3 修复的核心行为，此前无测试）。

背景：旧实现在「目标已有 soul.db」或「legacy 无 soul.db」时直接 return，
附属文件（audit.jsonl / injections.jsonl 等）永远不迁 —— 形成「库在新、
日志在旧」的混合状态，旧目录一旦被覆盖，审计随之丢失。

行为契约：
- 目标已有 soul.db → 不覆盖数据库，但**补齐缺失的附属文件**；
- legacy 无 soul.db → 不迁移数据库，但**同样补齐附属文件**；
- 补齐时 `skip_existing=True`：目标已有的同名文件**不得**被旧副本覆盖；
- `migration_marker.json` 是迁移内部件，不参与补齐。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .conftest import _import_soul_submodule


def _plugin(plugin_dir: Path, target: Path | None) -> Any:
    ctx = SimpleNamespace(paths={"data_dir": str(target)}) if target is not None else None
    return SimpleNamespace(_plugin_dir=str(plugin_dir), ctx=ctx)


def test_aux_files_copied_when_target_db_exists(tmp_path: Path) -> None:
    """目标已有 soul.db：缺失的附属文件必须补齐，已有的不许覆盖。"""
    legacy = tmp_path / "legacy"
    (legacy / "data").mkdir(parents=True)
    (legacy / "data" / "audit.jsonl").write_text("legacy-audit", encoding="utf-8")
    (legacy / "data" / "injections.jsonl").write_text("legacy-inj", encoding="utf-8")
    (legacy / "data" / "migration_marker.json").write_text("{}", encoding="utf-8")

    host = tmp_path / "target"
    target = host / "mai_soul_engine"              # 宿主目录下的插件子目录
    target.mkdir(parents=True)
    (target / "soul.db").write_bytes(b"")          # 存在即可，此路径不看内容
    (target / "audit.jsonl").write_text("newer-audit", encoding="utf-8")  # 目标侧更新 → 不许覆盖

    dd = _import_soul_submodule("utils.data_dir")
    res = dd.resolve_and_prepare_data_dir(_plugin(legacy, host))

    assert res["data_dir"] == target.resolve()
    assert (target / "injections.jsonl").read_text(encoding="utf-8") == "legacy-inj", \
        "缺失的附属文件没有补齐（旧实现直接 return）"
    assert (target / "audit.jsonl").read_text(encoding="utf-8") == "newer-audit", \
        "目标已有的文件被旧副本覆盖了"
    assert not (target / "migration_marker.json").exists(), \
        "迁移内部件不该被当作附属文件补齐"


def test_aux_files_copied_when_legacy_has_no_db(tmp_path: Path) -> None:
    """legacy 无 soul.db：不迁数据库，但附属文件仍要补齐。"""
    legacy = tmp_path / "legacy"
    (legacy / "data").mkdir(parents=True)
    (legacy / "data" / "audit.jsonl").write_text("legacy-audit", encoding="utf-8")

    host = tmp_path / "target"
    target = host / "mai_soul_engine"

    dd = _import_soul_submodule("utils.data_dir")
    res = dd.resolve_and_prepare_data_dir(_plugin(legacy, host))

    assert res["data_dir"] == target.resolve()
    assert (target / "audit.jsonl").read_text(encoding="utf-8") == "legacy-audit", \
        "legacy 无库时附属文件没补齐（旧实现直接 return）"


def test_same_dir_is_noop(tmp_path: Path) -> None:
    """目标与 legacy 同一目录（无宿主 ctx）→ 什么都不做。"""
    plugin_dir = tmp_path / "plugin"
    (plugin_dir / "data").mkdir(parents=True)

    dd = _import_soul_submodule("utils.data_dir")
    res = dd.resolve_and_prepare_data_dir(_plugin(plugin_dir, None))

    assert res["data_dir"] == (plugin_dir / "data").resolve()
    assert res["source"] == "plugin_dir"
