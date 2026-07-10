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
    _conn.execute("PRAGMA busy_timeout=5000")      # 5s 等锁而非立即抛
    _conn.execute("PRAGMA synchronous=NORMAL")     # WAL 下 NORMAL 足够安全且更快
    _conn.execute("PRAGMA cache_size=-8000")       # 8MB 页缓存
    _conn.execute("PRAGMA temp_store=MEMORY")      # 临时表/排序在内存
    _conn.execute("PRAGMA foreign_keys=ON")        # 为未来建外键做准备
    _create_tables()
    _run_migrations()
    _create_indexes()
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
    """
    CREATE TABLE IF NOT EXISTS soul_injection_snapshots (
        snapshot_id TEXT PRIMARY KEY,
        stream_id TEXT DEFAULT '',
        session_id TEXT DEFAULT '',
        created_at TEXT DEFAULT '',
        trait_ids_json TEXT DEFAULT '[]',
        spectrum_json TEXT DEFAULT '{}',
        mood_json TEXT DEFAULT '{}',
        selection_mode TEXT DEFAULT '',
        context_fingerprint TEXT DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_pending_reflections (
        pending_id INTEGER PRIMARY KEY AUTOINCREMENT,
        stream_id TEXT DEFAULT '',
        session_id TEXT DEFAULT '',
        reply_message_id TEXT DEFAULT '',
        created_at TEXT DEFAULT '',
        snapshot_id TEXT DEFAULT '',
        source TEXT DEFAULT '',
        response_text TEXT DEFAULT '',
        context_json TEXT DEFAULT '[]',
        status TEXT DEFAULT 'pending'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_self_reflections (
        reflection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        stream_id TEXT DEFAULT '',
        created_at TEXT DEFAULT '',
        pending_id INTEGER DEFAULT 0,
        snapshot_id TEXT DEFAULT '',
        reply_type TEXT DEFAULT '',
        evaluated INTEGER DEFAULT 0,
        consistency_score INTEGER DEFAULT 0,
        deviating_axis TEXT DEFAULT '',
        deviating_direction TEXT DEFAULT '',
        reason TEXT DEFAULT '',
        user_reaction_signal TEXT DEFAULT '',
        seed_id TEXT DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pending_created ON soul_pending_reflections (created_at)",
]


def _create_tables() -> None:
    """幂等建表。"""
    conn = _get_conn()
    for sql in _CREATE_SQL:
        conn.execute(sql)
    conn.commit()


def _create_indexes() -> None:
    """幂等建索引（在迁移之后执行，确保所有列已存在）。"""
    conn = _get_conn()
    index_sqls = [
        "CREATE INDEX IF NOT EXISTS idx_traits_stream_enabled ON soul_crystallized_traits(stream_id, enabled, deleted)",
        "CREATE INDEX IF NOT EXISTS idx_traits_lifecycle ON soul_crystallized_traits(lifecycle_state, enabled, deleted)",
        "CREATE INDEX IF NOT EXISTS idx_seeds_status_stream ON soul_thought_seeds(status, stream_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_edges_from ON soul_thought_edges(from_trait_id)",
        "CREATE INDEX IF NOT EXISTS idx_edges_to ON soul_thought_edges(to_trait_id)",
        "CREATE INDEX IF NOT EXISTS idx_pending_status ON soul_pending_reflections(status, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_snapshot_session ON soul_injection_snapshots(session_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_pending_reply_msg ON soul_pending_reflections(reply_message_id, source)",
        "CREATE INDEX IF NOT EXISTS idx_history_group ON soul_evolution_history(group_id, id)",
    ]
    for sql in index_sqls:
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


def _str_to_dt(s: str) -> datetime | None:
    """ISO 字符串 → datetime，失败或空串时返回 None。

    调用方必须处理 None：通常跳过 TTL 判定或使用默认时间。
    不要用 datetime.now() 兜底——会导致坏数据漂移到当前时间，
    TTL 判定误把刚导入的旧记录判为过期。
    """
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None
