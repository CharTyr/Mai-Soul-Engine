"""T17：未审核的 legacy 数据不得进入正式人格。

`legacy_import._import_table` 是裸的 `SELECT * → INSERT OR REPLACE`。要验证的
不变量是：**导入后「处于活跃人格状态的集合」不得大于 legacy 库里本来就是
活跃状态的那部分**——导入不得把候选/已禁用/未审核的行变成生效人格。

（正式人格 = 光谱数值 + 生效中的 trait。种子是候选，不是人格。）
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from .conftest import _import_soul_submodule


@pytest.fixture
def im() -> Any:
    model = _import_soul_submodule("models.ideology_model")
    yield model
    model.close_db()


def _legacy_db(project_root: Path) -> Path:
    """造一个 legacy 宿主库：只含少量 soul_* 表与列（旧结构）。"""
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db = data_dir / "MaiBot.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE soul_ideology_spectrum (
            scope_id TEXT PRIMARY KEY, sincerity INTEGER, engagement INTEGER,
            closeness INTEGER, directness INTEGER, updated_at TEXT
        );
        CREATE TABLE soul_group_evolution (group_id TEXT PRIMARY KEY, last_analyzed TEXT);
        CREATE TABLE soul_evolution_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, group_id TEXT,
            sincerity_delta INTEGER, engagement_delta INTEGER, closeness_delta INTEGER,
            directness_delta INTEGER, reason TEXT
        );
        CREATE TABLE soul_thought_seeds (
            seed_id TEXT PRIMARY KEY, stream_id TEXT, status TEXT, seed_type TEXT,
            event TEXT, reasoning TEXT, intensity REAL, confidence REAL,
            evidence_json TEXT, potential_impact_json TEXT, created_at TEXT
        );
        CREATE TABLE soul_crystallized_traits (
            trait_id TEXT PRIMARY KEY, stream_id TEXT, seed_id TEXT, name TEXT,
            question TEXT, thought TEXT, tags_json TEXT, confidence INTEGER,
            evidence_json TEXT, spectrum_impact_json TEXT, created_at TEXT,
            enabled INTEGER, deleted INTEGER
        );
        """
    )
    conn.execute(
        "INSERT INTO soul_ideology_spectrum VALUES ('global', 99, 88, 77, 66, '2026-01-01')"
    )
    # 未审核：永远停在 pending
    conn.execute(
        "INSERT INTO soul_thought_seeds VALUES "
        "('seed-pending', 'group-A', 'pending', '价值观冲突', '未审核的议题', 'r', 0.9, 0.9,"
        " '[]', '{\"sincerity\": 9}', '2026-01-01')"
    )
    # 被拒绝过的：不得复活成人格
    conn.execute(
        "INSERT INTO soul_thought_seeds VALUES "
        "('seed-rejected', 'group-A', 'rejected', '价值观冲突', '被拒的议题', 'r', 0.9, 0.9,"
        " '[]', '{\"sincerity\": 9}', '2026-01-01')"
    )
    # 已禁用的旧 trait：导入后必须仍然 disabled
    conn.execute(
        "INSERT INTO soul_crystallized_traits VALUES "
        "('trait-disabled', 'group-A', 'seed-old', '已禁用观点', 'q', 't', '[]', 50,"
        " '[]', '{}', '2026-01-01', 0, 0)"
    )
    # 生效中的旧 trait：本来就是活跃人格，导入后可以保持
    conn.execute(
        "INSERT INTO soul_crystallized_traits VALUES "
        "('trait-active', 'group-A', 'seed-old2', '生效观点', 'q', 't', '[]', 50,"
        " '[]', '{}', '2026-01-01', 1, 0)"
    )
    conn.commit()
    conn.close()
    return db


def _run_import(im: Any, project_root: Path, plugin_data_dir: Path) -> dict[str, Any]:
    li = _import_soul_submodule("migration.legacy_import")
    return li.run_legacy_import(plugin_data_dir, project_root)


def test_unapproved_legacy_seed_does_not_become_personality(im: Any, tmp_path: Path) -> None:
    """pending / rejected 的 legacy 种子不得变成 trait，也不得改动光谱。"""
    project_root = tmp_path / "host"
    plugin_data = tmp_path / "plugin-data"
    plugin_data.mkdir(parents=True, exist_ok=True)
    _legacy_db(project_root)

    im.init_db(tmp_path / "soul.db")
    _run_import(im, project_root, plugin_data)

    # 未审核种子只应作为「候选」存在，绝不生成 trait
    from_unapproved = [
        t for t in im.query_crystallized_traits(limit=50)
        if t.seed_id in ("seed-pending", "seed-rejected")
    ]
    assert from_unapproved == [], (
        f"未审核的 legacy 种子进入了正式人格: {[t.trait_id for t in from_unapproved]}"
    )

    # 光谱：legacy 谱行**本身就是同一个插件的历史人格**，迁移它是有意设计。
    # 要守的是「未审核种子的 potential_impact 不得被施加」——种子里写的是
    # sincerity=9，若被判成生效，数值会变成 99-9 或 50+9 之类的推导值。
    after = im.get_or_create_spectrum("global")
    assert (after.sincerity, after.engagement, after.closeness, after.directness) == (
        99, 88, 77, 66,
    ), "光谱不是从 legacy 谱行迁移的（可能被未审核种子的影响改过）"


def test_disabled_legacy_trait_stays_disabled_after_import(im: Any, tmp_path: Path) -> None:
    """导入不得「复活」已禁用的旧 trait（enable 状态必须原样保留）。"""
    project_root = tmp_path / "host"
    plugin_data = tmp_path / "plugin-data"
    plugin_data.mkdir(parents=True, exist_ok=True)
    _legacy_db(project_root)

    im.init_db(tmp_path / "soul.db")
    _run_import(im, project_root, plugin_data)

    disabled = im.get_crystallized_trait_by_id("trait-disabled")
    assert disabled is not None, "已禁用的旧 trait 应导入（保留可审计），但必须仍是禁用"
    assert not disabled.enabled, "导入把已禁用的旧 trait 复活了"

    active = im.get_crystallized_trait_by_id("trait-active")
    assert active is not None and active.enabled, "本来就生效的旧 trait 不应被导入禁用"


def test_import_is_idempotent_and_does_not_create_duplicates(im: Any, tmp_path: Path) -> None:
    """重复导入（重启/重跑）不得产生重复人格行。"""
    project_root = tmp_path / "host"
    plugin_data = tmp_path / "plugin-data"
    plugin_data.mkdir(parents=True, exist_ok=True)
    _legacy_db(project_root)

    im.init_db(tmp_path / "soul.db")
    _run_import(im, project_root, plugin_data)
    first = sorted(t.trait_id for t in im.query_crystallized_traits(limit=50))

    # 删掉状态文件 → 强制重跑一次导入
    state = plugin_data / "migration_state.json"
    assert state.exists(), "导入未写状态文件（无法判断是否已迁移）"
    state.unlink()
    _run_import(im, project_root, plugin_data)

    second = sorted(t.trait_id for t in im.query_crystallized_traits(limit=50))
    assert first == second, f"重复导入产生了重复行: {first} → {second}"
