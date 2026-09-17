"""数据目录盘点与迁移预演工具（T16）。

要求（来自方案）：
- **只读**：不建表、不写文件、不改 mtime
- **不自动选源**：选哪份决定历史去留，必须操作者决定
- 给事实：schema 版本、迁移记录、各表行数、是否已初始化、是否含实质数据
- 有风险要说出来：两份都有数据 / 版本不一致 / 有空壳库

历史背景：`plugins/<插件>/data/` 与 `data/plugins/<plugin-id>/` 可能同时存在
且内容不同，"按大小或修改时间自动选"会丢历史。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .conftest import _import_soul_submodule


def _inv() -> Any:
    return _import_soul_submodule("migration.inventory")


def _make_db(path: Path, *, with_data: bool = False, schema_version: int | None = None) -> Path:
    """造一个真实的 soul.db（用插件自己的迁移建表）。"""
    conn_mod = _import_soul_submodule("models._conn")
    conn_mod.close_db()
    conn_mod.init_db(path)
    if schema_version is not None:
        conn_mod._set_schema_version(schema_version)
    if with_data:
        conn = conn_mod._get_conn()
        conn.execute(
            "INSERT INTO soul_thought_seeds (seed_id, stream_id, seed_type, event, "
            "intensity, confidence, status, created_at) VALUES ('seed1','qq-1','t','e',0.5,0.5,'pending','2026-01-01T00:00:00')"
        )
        conn.execute(
            "INSERT INTO soul_crystallized_traits (trait_id, stream_id, name, question, "
            "thought, confidence, enabled, deleted) VALUES ('t1','global','n','q','th',80,1,0)"
        )
        conn.commit()
    conn_mod.close_db()
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ─── 只读性 ─────────────────────────────────────────────────────────


def test_inventory_does_not_modify_the_file(tmp_path: Path) -> None:
    """盘点前后文件哈希与 mtime 不变（必须严格只读）。"""
    db = _make_db(tmp_path / "soul.db", with_data=True)
    before_hash, before_mtime = _digest(db), db.stat().st_mtime_ns

    _inv().inventory_database(db)

    assert _digest(db) == before_hash, "盘点不得改写数据库"
    assert db.stat().st_mtime_ns == before_mtime, "盘点不得触碰 mtime"


def test_inventory_of_missing_file_is_structured(tmp_path: Path) -> None:
    """文件不存在 → 结构化结果，不抛异常、不创建文件。"""
    missing = tmp_path / "nope.db"
    inv = _inv().inventory_database(missing)

    assert inv.exists is False
    assert inv.readable is False
    assert inv.error
    assert not missing.exists(), "盘点不得创建文件"


def test_inventory_of_non_db_file_does_not_crash(tmp_path: Path) -> None:
    """不是 sqlite 的文件 → 报不可读，不抛异常。"""
    junk = tmp_path / "junk.db"
    junk.write_text("这不是数据库")

    inv = _inv().inventory_database(junk)

    assert inv.exists is True
    assert inv.readable is False
    assert inv.error


# ─── 盘点内容 ───────────────────────────────────────────────────────


def test_inventory_reports_schema_version_and_counts(tmp_path: Path) -> None:
    """能读到 schema 版本、迁移记录、表计数与实质数据标记。"""
    db = _make_db(tmp_path / "soul.db", with_data=True)
    inv = _inv().inventory_database(db)

    assert inv.readable is True
    assert inv.schema_version == _import_soul_submodule("models._conn").CURRENT_SCHEMA_VERSION
    assert inv.migrations, "应读到迁移记录"
    assert inv.table_counts["soul_thought_seeds"] == 1
    assert inv.table_counts["soul_crystallized_traits"] == 1
    assert inv.seed_counts_by_status == {"pending": 1}
    assert inv.has_substantive_data is True


def test_empty_db_is_not_substantive(tmp_path: Path) -> None:
    """刚初始化的空库：可读但没有实质数据（不能被误当成有历史）。"""
    db = _make_db(tmp_path / "soul.db")
    inv = _inv().inventory_database(db)

    assert inv.readable is True
    assert inv.has_substantive_data is False


# ─── 预演：不自动选源 + 风险告警 ─────────────────────────────────────


def test_preview_never_selects_a_source(tmp_path: Path) -> None:
    """预演结果永远标记需要操作者决定，不给"推荐哪份"。"""
    a = _make_db(tmp_path / "a.db", with_data=True)
    b = _make_db(tmp_path / "b.db")

    preview = _inv().build_migration_preview([a, b])

    assert preview.decision_required is True
    assert "操作者" in preview.note
    payload = preview.to_dict()
    assert "recommended" not in payload and "selected" not in payload


def test_preview_warns_when_two_candidates_both_have_data(tmp_path: Path) -> None:
    """两份都有实质数据 → 必须警告（选错丢历史）。"""
    a = _make_db(tmp_path / "a.db", with_data=True)
    b = _make_db(tmp_path / "b.db", with_data=True)

    preview = _inv().build_migration_preview([a, b])

    assert any("都含实质数据" in w for w in preview.warnings)


def test_preview_warns_on_empty_shell(tmp_path: Path) -> None:
    """一空一满 → 警告空壳库不能因为"更新"被选中。"""
    full = _make_db(tmp_path / "full.db", with_data=True)
    empty = _make_db(tmp_path / "empty.db")

    preview = _inv().build_migration_preview([full, empty])

    assert any("空壳" in w for w in preview.warnings)


def test_preview_warns_on_version_mismatch(tmp_path: Path) -> None:
    """schema 版本不一致 → 警告（两份数据处于不同阶段）。"""
    a = _make_db(tmp_path / "a.db", with_data=True)
    b = _make_db(tmp_path / "b.db", with_data=True, schema_version=2)

    preview = _inv().build_migration_preview([a, b])

    assert any("版本不一致" in w for w in preview.warnings)


def test_preview_handles_no_readable_candidate(tmp_path: Path) -> None:
    """全不可读 → 明确说无法预演，而不是假装成功。"""
    preview = _inv().build_migration_preview([tmp_path / "x.db", tmp_path / "y.db"])

    assert any("没有任何可读候选" in w for w in preview.warnings)


# ─── CLI ────────────────────────────────────────────────────────────


def test_cli_prints_report(tmp_path: Path, capsys: Any) -> None:
    """CLI 打印人读报告并返回 0。"""
    db = _make_db(tmp_path / "soul.db", with_data=True)
    code = _inv().main([str(db)])

    out = capsys.readouterr().out
    assert code == 0
    assert "迁移预演" in out
    assert "soul_thought_seeds" not in out  # 用的是去掉前缀的表名
    assert "含实质数据: 是" in out
    assert "操作者" in out


def test_cli_json_mode_is_parseable(tmp_path: Path, capsys: Any) -> None:
    """--json 输出可被解析（供脚本消费）。"""
    db = _make_db(tmp_path / "soul.db", with_data=True)
    _inv().main([str(db), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["decision_required"] is True
    assert len(payload["candidates"]) == 1


def test_cli_without_args_returns_usage(tmp_path: Path, capsys: Any) -> None:
    """无参数 → 用法提示 + 非 0 退出码。"""
    assert _inv().main([]) == 2
    assert "用法" in capsys.readouterr().out


def test_cli_does_not_modify_candidates(tmp_path: Path, capsys: Any) -> None:
    """CLI 全程只读。"""
    db = _make_db(tmp_path / "soul.db", with_data=True)
    before = _digest(db)

    _inv().main([str(db), "--json"])

    assert _digest(db) == before
