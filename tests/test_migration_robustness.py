"""T16 迁移鲁棒性：WAL 库 / 损坏库 / 迁移中断 / 重复迁移。

这四项此前只有「迁移盘点」（`test_migration_inventory.py`）覆盖——那测的是
「只读预演、不自动选源」。**实际迁移链在异常输入下的行为**没有直接测试，
本文件补上。

统一口径（来自方案 T16）：每种输入都要有**确定结果**，
且**不自动覆盖、不丢行**。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from .conftest import _import_soul_submodule


@pytest.fixture
def conn_mod() -> Any:
    return _import_soul_submodule("models._conn")


@pytest.fixture
def im(conn_mod: Any) -> Any:
    """独立管理 DB 生命周期（本文件的测试要反复 init/close）。"""
    model = _import_soul_submodule("models.ideology_model")
    yield model
    model.close_db()


def _latest() -> int:
    """从模块取最新版本号——**不要硬编码**，否则每次加迁移测试都要改。"""
    return int(_import_soul_submodule("models._conn").CURRENT_SCHEMA_VERSION)


def _version(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


def _has_column(db: Path, table: str, column: str) -> bool:
    conn = sqlite3.connect(str(db))
    try:
        return column in {
            row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
    finally:
        conn.close()


def _ledger(db: Path, version: int) -> list[tuple[str, str]]:
    conn = sqlite3.connect(str(db))
    try:
        return [
            (row[0], row[1])
            for row in conn.execute(
                "SELECT status, error FROM soul_schema_migrations WHERE version = ?",
                (version,),
            ).fetchall()
        ]
    finally:
        conn.close()


def _seed_rows(im: Any) -> None:
    """塞一份可校验的数据（光谱 + trait），后面用它证明「迁移不丢行」。"""
    s = im.get_or_create_spectrum("global")
    s.sincerity = 61
    s.save()
    im.create_crystallized_trait(
        trait_id="trait-keep", stream_id="group-A", seed_id="seed-keep",
        name="保留", question="Q", thought="迁移不得弄丢这行",
        tags_json="[]", confidence=70, evidence_json="[]", spectrum_impact_json="{}",
    )


def _assert_rows_intact(im: Any) -> None:
    assert im.get_or_create_spectrum("global").sincerity == 61, "迁移丢了光谱数值"
    trait = im.get_crystallized_trait_by_id("trait-keep")
    assert trait is not None and trait.enabled, "迁移丢了 trait 行"


# ─── 1. WAL 库 ──────────────────────────────────────────────────────


def test_wal_db_migrates_forward_without_losing_rows(im: Any, tmp_path: Path) -> None:
    """WAL 模式的库 + 存量数据 → 补做缺失迁移、数值不减、仍是 WAL。"""
    db = tmp_path / "soul.db"
    im.init_db(db)
    _seed_rows(im)
    im.close_db()

    # 造「落后于代码」的状态：版本退回 3，并拆掉 v5/v6 的迁移产物
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("DROP TABLE IF EXISTS soul_notifications")
    conn.execute("ALTER TABLE soul_injection_snapshots DROP COLUMN pairing_ambiguous")
    conn.execute("ALTER TABLE soul_injection_snapshots DROP COLUMN bot_identity")
    conn.execute(f"PRAGMA user_version = {_latest() - 2}")
    conn.commit()
    conn.close()

    stale = _latest() - 2
    assert _version(db) == stale
    im.init_db(db)

    assert _version(db) == _latest(), "迁移没有推进到最新版本"
    assert _has_column(db, "soul_injection_snapshots", "pairing_ambiguous")
    assert _has_column(db, "soul_injection_snapshots", "bot_identity")
    _assert_rows_intact(im)

    conn = sqlite3.connect(str(db))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert str(mode).lower() == "wal", "迁移把 journal_mode 改掉了"


# ─── 2. 损坏库 ──────────────────────────────────────────────────────


def test_corrupt_db_fails_loudly_and_is_not_overwritten(im: Any, tmp_path: Path) -> None:
    """损坏文件不是合法 SQLite → 必须**显式报错**，且原文件逐字节不变。"""
    db = tmp_path / "corrupt.db"
    payload = b"this is definitely not a sqlite database\n" * 64
    db.write_bytes(payload)

    with pytest.raises(sqlite3.DatabaseError):
        im.init_db(db)

    assert db.read_bytes() == payload, "init_db 覆盖/截断了损坏库（数据销毁）"
    im.close_db()


# ─── 3. 迁移中断 ────────────────────────────────────────────────────


def test_interrupted_migration_does_not_advance_version(
    im: Any, conn_mod: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    """迁移中途失败 → 不推进 user_version、账本记 failed；放开故障后可重试成功。"""
    db = tmp_path / "soul.db"
    im.init_db(db)
    _seed_rows(im)
    im.close_db()

    # 退回一个版本并拆掉最后一个迁移的产物，制造「待执行」的状态
    latest = _latest()
    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE soul_injection_snapshots DROP COLUMN bot_identity")
    conn.execute(f"PRAGMA user_version = {latest - 1}")
    conn.commit()
    conn.close()

    def _boom() -> None:
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(conn_mod, f"_run_v{latest}_migration", _boom)
    with pytest.raises(RuntimeError):
        im.init_db(db)
    im.close_db()

    assert _version(db) == latest - 1, "失败的迁移推进了 user_version"
    assert any(status == "failed" for status, _ in _ledger(db, latest)), "账本没记失败"

    # 放开故障 → 重试仍从失败版本开始并完成
    monkeypatch.undo()
    im.init_db(db)
    assert _version(db) == latest
    assert _has_column(db, "soul_injection_snapshots", "bot_identity")
    _assert_rows_intact(im)


# ─── 4. 重复迁移 ────────────────────────────────────────────────────


def test_repeated_init_is_idempotent_and_keeps_rows(im: Any, tmp_path: Path) -> None:
    """反复 init（模拟重启多次）→ 版本稳定、无遗留 running、行不减。"""
    db = tmp_path / "soul.db"
    im.init_db(db)
    _seed_rows(im)
    im.close_db()
    # 之后反复迁移：数据必须原样保留（重复迁移不得重建/清表）
    for _ in range(3):
        im.init_db(db)
        _assert_rows_intact(im)
        im.close_db()

    assert _version(db) == _latest()
    assert not any(status == "running" for status, _ in _ledger(db, _latest())), (
        "账本遗留 running 记录（上次迁移没有收尾）"
    )

    im.init_db(db)
    _assert_rows_intact(im)
