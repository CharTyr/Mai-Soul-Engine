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


def test_tracked_table_names_match_real_schema(tmp_path: Path) -> None:
    """盘点表名单必须与真实 schema 一致。

    写错表名不会报错，只会安静地显示「—」，看起来像"这张表没数据"——
    实际是工具根本没找到它。这条测试就是防这个。
    """
    inv_mod = _inv()
    db = _make_db(tmp_path / "soul.db")

    inv = inv_mod.inventory_database(db)
    missing = [
        table for table, count in inv.table_counts.items()
        if count is None and table not in ("soul_seed_operations", "soul_notifications")
    ]

    assert not missing, f"盘点表名与实际 schema 不符（显示为「—」）：{missing}"


def test_spectrum_state_is_read_from_real_table(tmp_path: Path) -> None:
    """能读到真实光谱表的状态（表名/列名写错会静默变 None）。"""
    conn_mod = _import_soul_submodule("models._conn")
    db = _make_db(tmp_path / "soul.db")
    conn_mod.init_db(db)
    conn = conn_mod._get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO soul_ideology_spectrum (scope_id, initialized) "
        "VALUES ('global', 1)"
    )
    conn.commit()
    conn_mod.close_db()

    inv = _inv().inventory_database(db)

    assert inv.spectrum_initialized is True
    assert (inv.table_counts.get("soul_ideology_spectrum") or 0) == 1


# ─── 谱系观察（只描述，不选源） ──────────────────────────────────────


def _add_seed(path: Path, seed_id: str, status: str) -> None:
    conn_mod = _import_soul_submodule("models._conn")
    conn_mod.init_db(path)
    conn = conn_mod._get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO soul_thought_seeds (seed_id, stream_id, seed_type, event, "
        "intensity, confidence, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (seed_id, "qq-1", "t", "e", 0.5, 0.5, status, "2026-01-01T00:00:00"),
    )
    conn.commit()
    conn_mod.close_db()


def test_lineage_detects_later_snapshot(tmp_path: Path) -> None:
    """A 的种子是 B 的子集 → 报告 B 是后续状态（只描述事实）。"""
    a = _make_db(tmp_path / "a.db")
    b = _make_db(tmp_path / "b.db")
    _add_seed(a, "s1", "pending")
    _add_seed(b, "s1", "rejected")
    _add_seed(b, "s2", "pending")

    notes = _inv().build_migration_preview([a, b]).lineage_notes()

    assert any("后续状态" in n for n in notes)
    assert any("b.db" in n and "a.db" in n for n in notes)


def test_lineage_notes_do_not_choose_a_source(tmp_path: Path) -> None:
    """谱系观察里不得出现"选/推荐"这类选源措辞。"""
    a = _make_db(tmp_path / "a.db")
    b = _make_db(tmp_path / "b.db")
    _add_seed(a, "s1", "pending")
    _add_seed(b, "s1", "rejected")
    _add_seed(b, "s2", "pending")

    preview = _inv().build_migration_preview([a, b])

    for n in preview.lineage_notes():
        assert "推荐" not in n and "应该选" not in n and "建议选" not in n
    assert preview.decision_required is True


def test_lineage_reports_same_set_different_status(tmp_path: Path) -> None:
    """种子集合相同但状态不同 → 说明是同一批数据的两次快照。"""
    a = _make_db(tmp_path / "a.db")
    b = _make_db(tmp_path / "b.db")
    _add_seed(a, "s1", "pending")
    _add_seed(b, "s1", "rejected")

    notes = _inv().build_migration_preview([a, b]).lineage_notes()

    assert any("两次快照" in n for n in notes)


def test_lineage_silent_when_unrelated(tmp_path: Path) -> None:
    """两份互不相干的种子集合 → 不做谱系断言（避免误导）。"""
    a = _make_db(tmp_path / "a.db")
    b = _make_db(tmp_path / "b.db")
    _add_seed(a, "s1", "pending")
    _add_seed(b, "s9", "pending")

    notes = _inv().build_migration_preview([a, b]).lineage_notes()

    assert notes == []


def test_lineage_present_in_json_and_text(tmp_path: Path, capsys: Any) -> None:
    """谱系观察在 JSON 与文本报告里都出现。"""
    a = _make_db(tmp_path / "a.db")
    b = _make_db(tmp_path / "b.db")
    _add_seed(a, "s1", "pending")
    _add_seed(b, "s1", "rejected")
    _add_seed(b, "s2", "pending")

    _inv().main([str(a), str(b), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["lineage_notes"]

    _inv().main([str(a), str(b)])
    assert "谱系观察" in capsys.readouterr().out
