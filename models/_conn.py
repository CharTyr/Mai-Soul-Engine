"""数据库连接管理、建表与迁移、时间转换工具。

为兼容历史导入保留的连接管理模块，所有子模块通过 ``from ._conn import _get_conn`` 引用。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import logging
import sqlite3

from ..worldview.constants import GLOBAL_STREAM

logger = logging.getLogger(__name__)

__all__ = [
    "_get_conn",
    "close_db",
    "datetime",
    "init_db",
]

# ─── 全局连接管理 ───────────────────────────────────────────────────

_db_path: Path | None = None
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """获取当前数据库连接。"""
    if _conn is None:
        raise RuntimeError("数据库尚未初始化，请先调用 init_db()")
    return _conn


def init_db(db_path: Path) -> None:
    """初始化插件自有 SQLite 数据库，幂等建表 + 就地迁移。"""
    global _db_path, _conn
    _db_path = db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(str(db_path), check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL")
    _create_tables()
    _run_migrations()
    logger.debug("Soul 数据库已初始化: %s", db_path)


def close_db() -> None:
    """关闭数据库连接。"""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


# ─── 建表 SQL ─────────────────────────────────────────────────────

_CREATE_SQL = [
    """
    CREATE TABLE IF NOT EXISTS soul_ideology_spectrum (
        scope_id TEXT PRIMARY KEY DEFAULT 'global',
        sincerity INTEGER DEFAULT 50,
        engagement INTEGER DEFAULT 50,
        closeness INTEGER DEFAULT 50,
        directness INTEGER DEFAULT 50,
        last_sincerity_dir INTEGER DEFAULT 0,
        last_engagement_dir INTEGER DEFAULT 0,
        last_closeness_dir INTEGER DEFAULT 0,
        last_directness_dir INTEGER DEFAULT 0,
        initialized INTEGER DEFAULT 0,
        last_evolution TEXT DEFAULT '',
        updated_at TEXT DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_group_evolution (
        group_id TEXT PRIMARY KEY,
        last_analyzed TEXT DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_evolution_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT '',
        group_id TEXT,
        sincerity_delta INTEGER DEFAULT 0,
        engagement_delta INTEGER DEFAULT 0,
        closeness_delta INTEGER DEFAULT 0,
        directness_delta INTEGER DEFAULT 0,
        reason TEXT DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_thought_seeds (
        seed_id TEXT PRIMARY KEY,
        stream_id TEXT DEFAULT '',
        seed_type TEXT,
        event TEXT,
        intensity INTEGER,
        confidence INTEGER DEFAULT 0,
        evidence_json TEXT DEFAULT '[]',
        reasoning TEXT,
        potential_impact_json TEXT,
        created_at TEXT DEFAULT '',
        status TEXT DEFAULT 'pending'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_crystallized_traits (
        trait_id TEXT PRIMARY KEY,
        stream_id TEXT DEFAULT '',
        seed_id TEXT DEFAULT '',
        name TEXT,
        question TEXT DEFAULT '',
        thought TEXT,
        tags_json TEXT DEFAULT '[]',
        confidence INTEGER DEFAULT 0,
        evidence_json TEXT DEFAULT '[]',
        spectrum_impact_json TEXT DEFAULT '{}',
        created_at TEXT DEFAULT '',
        enabled INTEGER DEFAULT 1,
        deleted INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_context_slices (
        scope_type TEXT NOT NULL DEFAULT 'group',
        scope_key TEXT NOT NULL,
        sincerity_offset INTEGER DEFAULT 0,
        engagement_offset INTEGER DEFAULT 0,
        closeness_offset INTEGER DEFAULT 0,
        directness_offset INTEGER DEFAULT 0,
        sample_count INTEGER DEFAULT 0,
        updated_at TEXT DEFAULT '',
        PRIMARY KEY (scope_type, scope_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_mood_state (
        scope_id TEXT PRIMARY KEY DEFAULT 'global',
        valence INTEGER DEFAULT 0,
        arousal INTEGER DEFAULT 0,
        energy INTEGER DEFAULT 0,
        updated_at TEXT DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_thought_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        from_trait_id TEXT NOT NULL,
        to_trait_id TEXT DEFAULT '',
        relation_type TEXT NOT NULL,
        source_ref TEXT DEFAULT '',
        created_at TEXT DEFAULT ''
    )
    """,
]


def _create_tables() -> None:
    """幂等建表。"""
    conn = _get_conn()
    for sql in _CREATE_SQL:
        conn.execute(sql)
    conn.commit()


# ─── 迁移工具 ─────────────────────────────────────────────────────


def _has_column(table_name: str, column_name: str) -> bool:
    """检查表中是否已存在某列。"""
    conn = _get_conn()
    rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return any(row[1] == column_name for row in rows)


def _add_column(table_name: str, column_name: str, ddl: str) -> None:
    """安全添加列。"""
    conn = _get_conn()
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")
    conn.commit()


def _rename_column(table_name: str, old_name: str, new_name: str) -> None:
    """安全重命名列（SQLite >= 3.25）。"""
    conn = _get_conn()
    conn.execute(f"ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name}")
    conn.commit()


def _run_migrations() -> None:
    """就地迁移：补齐旧版可能缺失的列 + v2.1.0 政治轴→社交轴重命名。"""
    if not _has_column("soul_thought_seeds", "stream_id"):
        _add_column("soul_thought_seeds", "stream_id", "TEXT DEFAULT ''")
    if not _has_column("soul_thought_seeds", "confidence"):
        _add_column("soul_thought_seeds", "confidence", "INTEGER DEFAULT 0")
    if not _has_column("soul_thought_seeds", "evidence_json"):
        _add_column("soul_thought_seeds", "evidence_json", "TEXT DEFAULT '[]'")
    if not _has_column("soul_thought_seeds", "context_json"):
        _add_column("soul_thought_seeds", "context_json", "TEXT DEFAULT '[]'")
    if not _has_column("soul_crystallized_traits", "question"):
        _add_column("soul_crystallized_traits", "question", "TEXT DEFAULT ''")
    if not _has_column("soul_crystallized_traits", "tags_json"):
        _add_column("soul_crystallized_traits", "tags_json", "TEXT DEFAULT '[]'")
    if not _has_column("soul_crystallized_traits", "confidence"):
        _add_column("soul_crystallized_traits", "confidence", "INTEGER DEFAULT 0")
    if not _has_column("soul_crystallized_traits", "evidence_json"):
        _add_column("soul_crystallized_traits", "evidence_json", "TEXT DEFAULT '[]'")
    if not _has_column("soul_crystallized_traits", "ideology_layer"):
        _add_column("soul_crystallized_traits", "ideology_layer", "TEXT DEFAULT 'conduct'")
    if not _has_column("soul_crystallized_traits", "lifecycle_state"):
        _add_column("soul_crystallized_traits", "lifecycle_state", "TEXT DEFAULT 'active'")

    # v2.1.0：政治光谱轴 → 群聊社交轴（就地重命名列）
    _rename_spectrum_axes()
    _rename_history_axes()
    _rename_slice_axes()

    # 历史上 trait 用空串表全局作用域，与"未设置"无法区分，统一迁移到显式 "global"
    # 幂等：再次运行时无 '' 行就不影响
    conn = _get_conn()
    conn.execute(
        "UPDATE soul_crystallized_traits SET stream_id = ? WHERE stream_id = ''",
        (GLOBAL_STREAM,),
    )
    conn.commit()


def _rename_spectrum_axes() -> None:
    """soul_ideology_spectrum: economic→sincerity 等。"""
    renames = [
        ("economic", "sincerity"),
        ("social", "engagement"),
        ("diplomatic", "closeness"),
        ("progressive", "directness"),
        ("last_economic_dir", "last_sincerity_dir"),
        ("last_social_dir", "last_engagement_dir"),
        ("last_diplomatic_dir", "last_closeness_dir"),
        ("last_progressive_dir", "last_directness_dir"),
    ]
    for old, new in renames:
        if _has_column("soul_ideology_spectrum", old) and not _has_column("soul_ideology_spectrum", new):
            _rename_column("soul_ideology_spectrum", old, new)


def _rename_history_axes() -> None:
    """soul_evolution_history: *_delta 列重命名。"""
    renames = [
        ("economic_delta", "sincerity_delta"),
        ("social_delta", "engagement_delta"),
        ("diplomatic_delta", "closeness_delta"),
        ("progressive_delta", "directness_delta"),
    ]
    for old, new in renames:
        if _has_column("soul_evolution_history", old) and not _has_column("soul_evolution_history", new):
            _rename_column("soul_evolution_history", old, new)


def _rename_slice_axes() -> None:
    """soul_context_slices: *_offset 列重命名。"""
    renames = [
        ("economic_offset", "sincerity_offset"),
        ("social_offset", "engagement_offset"),
        ("diplomatic_offset", "closeness_offset"),
        ("progressive_offset", "directness_offset"),
    ]
    for old, new in renames:
        if _has_column("soul_context_slices", old) and not _has_column("soul_context_slices", new):
            _rename_column("soul_context_slices", old, new)


# ─── 时间转换工具 ───────────────────────────────────────────────────


def _dt_to_str(dt: datetime) -> str:
    """datetime → ISO 字符串。"""
    return dt.isoformat() if dt else ""


def _str_to_dt(s: str) -> datetime:
    """ISO 字符串 → datetime，失败时返回当前时间。"""
    if not s:
        return datetime.now()
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now()
