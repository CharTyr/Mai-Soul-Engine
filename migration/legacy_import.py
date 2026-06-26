"""旧版数据迁移 — 从旧 MaiBot 共享数据库导入 soul_* 表到插件自有 SQLite。

迁移触发时机：插件 on_load() 首次运行。
迁移来源：
  1. 宿主 MaiBot.db 中的 soul_* 五张表（旧版插件直接写入宿主 DB）
  2. 旧插件目录下的 data/ 文件（audit.jsonl、injections.jsonl、notion_frontend_state.json）
迁移目标：插件自有 data/soul.db + data/ 下的文件
幂等保证：data/migration_state.json 记录迁移状态，已完成则跳过。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MIGRATION_STATE_FILE = "migration_state.json"
_LEGACY_TABLES = [
    "soul_ideology_spectrum",
    "soul_group_evolution",
    "soul_evolution_history",
    "soul_thought_seeds",
    "soul_crystallized_traits",
]


def run_legacy_import(plugin_data_dir: Path, project_root: Path) -> dict[str, Any]:
    """执行旧版数据迁移。

    Args:
        plugin_data_dir: 插件 data 目录路径。
        project_root: maibot 项目根目录（用于定位宿主 MaiBot.db）。

    Returns:
        dict: 迁移结果摘要。
    """
    state_file = plugin_data_dir / _MIGRATION_STATE_FILE
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if state.get("legacy_imported", False):
                logger.info("[Mai-Soul-Engine] 旧版数据已迁移过，跳过")
                return state
        except Exception:
            logger.warning("[Mai-Soul-Engine] 迁移状态文件读取失败，重新执行迁移")

    summary: dict[str, Any] = {
        "legacy_imported": False,
        "imported_at": datetime.now().isoformat(),
        "host_db_path": "",
        "tables_imported": {},
        "files_copied": [],
        "errors": [],
    }

    # 定位宿主数据库
    host_db_path = project_root / "data" / "MaiBot.db"
    summary["host_db_path"] = str(host_db_path)

    if not host_db_path.exists():
        logger.info("[Mai-Soul-Engine] 未找到宿主数据库 %s，跳过数据迁移", host_db_path)
        summary["errors"].append(f"宿主数据库不存在: {host_db_path}")
        _save_state(state_file, summary)
        return summary

    # 打开宿主数据库（只读）
    try:
        host_conn = sqlite3.connect(f"file:{host_db_path}?mode=ro", uri=True)
        host_conn.row_factory = sqlite3.Row
    except Exception as e:
        logger.error("[Mai-Soul-Engine] 打开宿主数据库失败: %s", e)
        summary["errors"].append(f"打开宿主数据库失败: {e}")
        _save_state(state_file, summary)
        return summary

    # 检查旧表是否存在
    existing_tables = _check_legacy_tables(host_conn)
    if not existing_tables:
        logger.info("[Mai-Soul-Engine] 宿主数据库中无 soul_* 表，跳过数据迁移")
        summary["errors"].append("宿主数据库中无 soul_* 表")
        host_conn.close()
        _save_state(state_file, summary)
        return summary

    logger.info("[Mai-Soul-Engine] 发现旧版数据表: %s", existing_tables)

    # 打开插件数据库（先通过 init_db 建表，确保目标表存在）
    plugin_db_path = plugin_data_dir / "soul.db"
    try:
        from ..models.ideology_model import init_db, close_db, _get_conn

        init_db(plugin_db_path)
        plugin_conn = _get_conn()
    except Exception as e:
        logger.error("[Mai-Soul-Engine] 打开插件数据库失败: %s", e)
        summary["errors"].append(f"打开插件数据库失败: {e}")
        host_conn.close()
        _save_state(state_file, summary)
        return summary

    # 逐表导入
    for table in existing_tables:
        try:
            count = _import_table(host_conn, plugin_conn, table)
            summary["tables_imported"][table] = count
            logger.info("[Mai-Soul-Engine] 导入 %s: %d 条记录", table, count)
        except Exception as e:
            logger.error("[Mai-Soul-Engine] 导入 %s 失败: %s", table, e)
            summary["errors"].append(f"导入 {table} 失败: {e}")

    plugin_conn.commit()
    # 不关闭全局连接 — on_load 中的 init_db 会复用或重新初始化
    host_conn.close()

    # 拷贝旧插件 data 文件
    copied = _copy_legacy_data_files(plugin_data_dir, project_root)
    summary["files_copied"] = copied

    summary["legacy_imported"] = True
    _save_state(state_file, summary)

    logger.info(
        "[Mai-Soul-Engine] 旧版数据迁移完成: 表 %s, 文件 %s",
        summary["tables_imported"],
        copied,
    )
    return summary


def _check_legacy_tables(host_conn: sqlite3.Connection) -> list[str]:
    """检查宿主数据库中存在哪些 soul_* 表。"""
    found = []
    for table in _LEGACY_TABLES:
        try:
            row = host_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if row:
                found.append(table)
        except Exception:
            continue
    return found


def _import_table(host_conn: sqlite3.Connection, plugin_conn: sqlite3.Connection, table: str) -> int:
    """从宿主数据库导入单张表的数据到插件数据库。"""
    rows = host_conn.execute(f"SELECT * FROM {table}").fetchall()
    if not rows:
        return 0

    count = 0
    for row in rows:
        columns = row.keys()
        placeholders = ", ".join("?" for _ in columns)
        col_names = ", ".join(columns)
        values = tuple(row[col] for col in columns)
        try:
            # INSERT OR REPLACE 避免主键冲突
            plugin_conn.execute(
                f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})",
                values,
            )
            count += 1
        except Exception as e:
            logger.warning("[Mai-Soul-Engine] 导入 %s 单行失败: %s", table, e)
            continue

    return count


def _copy_legacy_data_files(plugin_data_dir: Path, project_root: Path) -> list[str]:
    """拷贝旧插件的 data 文件到新位置。"""
    copied = []
    # 旧插件可能在 plugins/Mai-Soul-Engine/ 或 plugins/CharTyr_Mai-Soul-Engine/
    candidates = [
        project_root / "plugins" / "Mai-Soul-Engine" / "data",
        project_root / "plugins" / "CharTyr_Mai-Soul-Engine" / "data",
    ]

    legacy_files = ["audit.jsonl", "injections.jsonl", "notion_frontend_state.json"]

    for legacy_data_dir in candidates:
        if not legacy_data_dir.exists():
            continue
        for filename in legacy_files:
            src = legacy_data_dir / filename
            if not src.exists():
                continue
            dst = plugin_data_dir / filename
            if dst.exists():
                # 已有文件不覆盖
                continue
            try:
                dst.write_bytes(src.read_bytes())
                copied.append(filename)
                logger.info("[Mai-Soul-Engine] 拷贝旧数据文件: %s", filename)
            except Exception as e:
                logger.warning("[Mai-Soul-Engine] 拷贝 %s 失败: %s", filename, e)

    return copied


def _save_state(state_file: Path, summary: dict[str, Any]) -> None:
    """保存迁移状态文件。"""
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error("[Mai-Soul-Engine] 保存迁移状态失败: %s", e)
