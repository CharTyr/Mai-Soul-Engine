"""数据库连接管理、建表与迁移、时间转换工具。

为兼容历史导入保留的连接管理模块，所有子模块通过 ``from ._conn import _get_conn`` 引用。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import hashlib
import logging
import sqlite3

# 全局作用域标记：trait/光谱以此值表示"不绑定特定群、对所有聊天流生效"的全局作用域。
# 历史上曾用空串 "" 表全局（与"未设置/异常"无法区分），现统一用显式 "global"。
# 定义在 models 层（低层），worldview 层从此处引用，避免 models→worldview 反向依赖。
GLOBAL_STREAM = "global"

logger = logging.getLogger(__name__)

__all__ = [
    "_get_conn",
    "close_db",
    "datetime",
    "init_db",
]

CURRENT_SCHEMA_VERSION = 6

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
        deleted INTEGER DEFAULT 0,
        origin_stream_id TEXT DEFAULT ''
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
        context_fingerprint TEXT DEFAULT '',
        context_json TEXT DEFAULT '[]',
        consumed_at TEXT DEFAULT '',
        consumed_by_reply TEXT DEFAULT '',
        delivery_state TEXT DEFAULT 'selected'
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
        raw_consistency_score INTEGER DEFAULT NULL,
        normalized_consistency_score INTEGER DEFAULT NULL,
        correction_consumed_at TEXT DEFAULT '',
        deviating_axis TEXT DEFAULT '',
        deviating_direction TEXT DEFAULT '',
        reason TEXT DEFAULT '',
        user_reaction_signal TEXT DEFAULT '',
        seed_id TEXT DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pending_created ON soul_pending_reflections (created_at)",
    """
    CREATE TABLE IF NOT EXISTS soul_fermentation_inputs (
        input_id TEXT PRIMARY KEY,
        seed_id TEXT NOT NULL,
        stream_id TEXT DEFAULT '',
        message_text TEXT,
        relevance_score REAL DEFAULT 0,
        added_at TEXT DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_notifications (
        notification_id TEXT PRIMARY KEY,
        dedupe_key TEXT NOT NULL,
        stream_id TEXT DEFAULT '',
        text TEXT DEFAULT '',
        status TEXT DEFAULT 'pending',
        attempts INTEGER DEFAULT 0,
        max_attempts INTEGER DEFAULT 3,
        last_error TEXT DEFAULT '',
        created_at TEXT DEFAULT '',
        updated_at TEXT DEFAULT '',
        sent_at TEXT DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_seed_operations (
        operation_id TEXT PRIMARY KEY,
        seed_id TEXT NOT NULL,
        operation_type TEXT DEFAULT 'internalize',
        status TEXT DEFAULT 'running',
        attempt INTEGER DEFAULT 1,
        lease_expires_at TEXT DEFAULT '',
        result_json TEXT DEFAULT '',
        error TEXT DEFAULT '',
        created_at TEXT DEFAULT '',
        updated_at TEXT DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS soul_schema_migrations (
        version INTEGER NOT NULL,
        name TEXT NOT NULL,
        checksum TEXT NOT NULL DEFAULT '',
        started_at TEXT NOT NULL DEFAULT '',
        finished_at TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT '',
        error TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (version, name)
    )
    """,
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
        "CREATE INDEX IF NOT EXISTS idx_fermentation_seed ON soul_fermentation_inputs(seed_id)",
    ]
    # cabinet_slot_no 列可能由 v2 迁移添加，在 _create_indexes 中追加以保证索引存在
    if _has_column("soul_crystallized_traits", "cabinet_slot_no"):
        index_sqls.append(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_slot_active "
            "ON soul_crystallized_traits(cabinet_slot_no) "
            "WHERE cabinet_slot_no IS NOT NULL AND enabled = 1 AND deleted = 0"
        )
    for sql in index_sqls:
        conn.execute(sql)
    conn.commit()


# ─── 迁移框架 ─────────────────────────────────────────────────────


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


# ─── Schema 版本管理 ────────────────────────────────────────────────


def _get_schema_version() -> int:
    """通过 PRAGMA user_version 获取当前 schema 版本。"""
    conn = _get_conn()
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def _set_schema_version(v: int) -> None:
    """设置 PRAGMA user_version。"""
    conn = _get_conn()
    conn.execute(f"PRAGMA user_version = {v}")


def _migration_checksum(name: str) -> str:
    """迁移名称的 sha256 前 16 字符作为 checksum。"""
    return hashlib.sha256(name.encode()).hexdigest()[:16]


def _record_migration_start(version: int, name: str) -> None:
    """记录迁移开始。"""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO soul_schema_migrations "
        "(version, name, checksum, started_at, status, error) "
        "VALUES (?, ?, ?, ?, 'running', '')",
        (version, name, _migration_checksum(name), now),
    )
    conn.commit()


def _record_migration_success(version: int, name: str) -> None:
    """记录迁移成功。"""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE soul_schema_migrations SET finished_at = ?, status = 'success', error = '' "
        "WHERE version = ? AND name = ?",
        (now, version, name),
    )
    conn.commit()


def _record_migration_failed(version: int, name: str, error: str) -> None:
    """记录迁移失败——不推进 user_version。"""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE soul_schema_migrations SET finished_at = ?, status = 'failed', error = ? "
        "WHERE version = ? AND name = ?",
        (now, error[:500], version, name),
    )
    conn.commit()


def _run_migrations() -> None:
    """版本驱动的就地迁移。

    流程：
        1. 确保 soul_schema_migrations 表已存在
        2. 读取 PRAGMA user_version 得到当前版本
        3. 逐版本检查并执行缺失的迁移
        4. 每个迁移成功后才推进 user_version
        5. 迁移失败不推进版本，之后重试仍从失败版本开始
    """
    current = _get_schema_version()

    if current < 1:
        _record_migration_start(1, "v1_legacy_bootstrap")
        try:
            _run_v1_migration()
            _record_migration_success(1, "v1_legacy_bootstrap")
            _set_schema_version(1)
            current = 1
        except Exception as e:
            _record_migration_failed(1, "v1_legacy_bootstrap", str(e))
            raise

    if current < 2:
        _record_migration_start(2, "v2_cabinet_slot_no")
        try:
            _run_v2_migration()
            _record_migration_success(2, "v2_cabinet_slot_no")
            _set_schema_version(2)
            current = 2
        except Exception as e:
            _record_migration_failed(2, "v2_cabinet_slot_no", str(e))
            raise

    if current < 3:
        _record_migration_start(3, "v3_snapshot_pairing_and_delivery")
        try:
            _run_v3_migration()
            _record_migration_success(3, "v3_snapshot_pairing_and_delivery")
            _set_schema_version(3)
            current = 3
        except Exception as e:
            _record_migration_failed(3, "v3_snapshot_pairing_and_delivery", str(e))
            raise

    if current < 4:
        _record_migration_start(4, "v4_seed_operation_lease")
        try:
            _run_v4_migration()
            _record_migration_success(4, "v4_seed_operation_lease")
            _set_schema_version(4)
            current = 4
        except Exception as e:
            _record_migration_failed(4, "v4_seed_operation_lease", str(e))
            raise

    if current < 5:
        _record_migration_start(5, "v5_notification_outbox")
        try:
            _run_v5_migration()
            _record_migration_success(5, "v5_notification_outbox")
            _set_schema_version(5)
            current = 5
        except Exception as e:
            _record_migration_failed(5, "v5_notification_outbox", str(e))
            raise

    if current < 6:
        _record_migration_start(6, "v6_snapshot_pairing_ambiguous")
        try:
            _run_v6_migration()
            _record_migration_success(6, "v6_snapshot_pairing_ambiguous")
            _set_schema_version(6)
            current = 6
        except Exception as e:
            _record_migration_failed(6, "v6_snapshot_pairing_ambiguous", str(e))
            raise


def _run_v6_migration() -> None:
    """Version 6：快照配对歧义标记。

    宿主 planner / after_response 两个 payload **没有共同请求 id**（已核对宿主源码），
    所以配对只能是启发式。当同一会话存在多条未认领快照时，我们**无法确定**
    这条回复对应哪一次注入——必须标记出来，让下游跳过会改写人格的自评反馈，
    而不是拿一条猜出来的关联去改人格。

    旧行默认 0（视为「不歧义」）：它们是历史数据，当时的判定条件无从追溯。
    """
    conn = _get_conn()
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(soul_injection_snapshots)")}
    if "pairing_ambiguous" not in cols:
        conn.execute(
            "ALTER TABLE soul_injection_snapshots "
            "ADD COLUMN pairing_ambiguous INTEGER NOT NULL DEFAULT 0"
        )
    conn.commit()


def _run_v5_migration() -> None:
    """Version 5：通知 outbox。

    通知发送失败必须可重试、可查询，不能静默丢——否则「管理员没收到通知」
    会被误当成「没有需要通知的事」。``dedupe_key`` 唯一，重排队不会刷屏。
    """
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS soul_notifications (
            notification_id TEXT PRIMARY KEY,
            dedupe_key TEXT NOT NULL,
            stream_id TEXT DEFAULT '',
            text TEXT DEFAULT '',
            status TEXT DEFAULT 'pending',
            attempts INTEGER DEFAULT 0,
            max_attempts INTEGER DEFAULT 3,
            last_error TEXT DEFAULT '',
            created_at TEXT DEFAULT '',
            updated_at TEXT DEFAULT '',
            sent_at TEXT DEFAULT ''
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_notifications_dedupe "
        "ON soul_notifications(dedupe_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_notifications_status "
        "ON soul_notifications(status, created_at)"
    )
    conn.commit()


def _run_v4_migration() -> None:
    """Version 4：种子操作租约（幂等内化的单赢家认领）。

    并发批准或崩后重试会让同一颗种子被内化两次（光谱影响施加两遍）。
    该表以「同一 (seed_id, operation_type) 只允许一条 running」的部分唯一索引
    做单赢家认领；租约过期后可被抢占重试。
    """
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS soul_seed_operations (
            operation_id TEXT PRIMARY KEY,
            seed_id TEXT NOT NULL,
            operation_type TEXT DEFAULT 'internalize',
            status TEXT DEFAULT 'running',
            attempt INTEGER DEFAULT 1,
            lease_expires_at TEXT DEFAULT '',
            result_json TEXT DEFAULT '',
            error TEXT DEFAULT '',
            created_at TEXT DEFAULT '',
            updated_at TEXT DEFAULT ''
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_seed_op_running "
        "ON soul_seed_operations(seed_id, operation_type) WHERE status = 'running'"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_seed_op_status "
        "ON soul_seed_operations(status, lease_expires_at)"
    )
    conn.commit()


def _run_v3_migration() -> None:
    """Version 3：注入快照配对与投递态。

    为 ``soul_injection_snapshots`` 增补：
    - ``context_json``：触发上文随快照落库（此前放在 session 键的内存缓存里，
      同会话并发两轮会互相顶掉）
    - ``consumed_at`` / ``consumed_by_reply``：回复认领标记，保证一条快照
      只被一条回复消费（FIFO 认领，同一 reply 重试可复用）
    - ``delivery_state``：selected → hook_applied → final_request_verified /
      unverified，区分「已选中」「已交回宿主」「已确认进入最终请求」
    """
    for column, ddl in (
        ("context_json", "TEXT DEFAULT '[]'"),
        ("consumed_at", "TEXT DEFAULT ''"),
        ("consumed_by_reply", "TEXT DEFAULT ''"),
        ("delivery_state", "TEXT DEFAULT 'selected'"),
    ):
        if not _has_column("soul_injection_snapshots", column):
            _add_column("soul_injection_snapshots", column, ddl)
    conn = _get_conn()
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_snapshot_claim "
        "ON soul_injection_snapshots(session_id, consumed_at, created_at)"
    )
    conn.commit()


def _run_v1_migration() -> None:
    """Version 1：现有全部 has_column / rename / global 归一 / origin / raw 列（保持幂等）。"""
    if not _has_column("soul_thought_seeds", "stream_id"):
        _add_column("soul_thought_seeds", "stream_id", "TEXT DEFAULT ''")
    if not _has_column("soul_thought_seeds", "confidence"):
        _add_column("soul_thought_seeds", "confidence", "INTEGER DEFAULT 0")
    if not _has_column("soul_thought_seeds", "evidence_json"):
        _add_column("soul_thought_seeds", "evidence_json", "TEXT DEFAULT '[]'")
    if not _has_column("soul_thought_seeds", "context_json"):
        _add_column("soul_thought_seeds", "context_json", "TEXT DEFAULT '[]'")
    if not _has_column("soul_thought_seeds", "fermentation_started_at"):
        _add_column("soul_thought_seeds", "fermentation_started_at", "TEXT DEFAULT ''")
    if not _has_column("soul_thought_seeds", "fermentation_checked_at"):
        _add_column("soul_thought_seeds", "fermentation_checked_at", "TEXT DEFAULT ''")
    if not _has_column("soul_thought_seeds", "fermentation_extension_count"):
        _add_column("soul_thought_seeds", "fermentation_extension_count", "INTEGER DEFAULT 0")
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

    # 0B.1：soul_crystallized_traits 加 origin_stream_id 溯源列
    if not _has_column("soul_crystallized_traits", "origin_stream_id"):
        _add_column("soul_crystallized_traits", "origin_stream_id", "TEXT DEFAULT ''")

    # Phase 0A R1：soul_self_reflections 加列
    if not _has_column("soul_self_reflections", "raw_consistency_score"):
        _add_column("soul_self_reflections", "raw_consistency_score", "INTEGER DEFAULT NULL")
    if not _has_column("soul_self_reflections", "normalized_consistency_score"):
        _add_column("soul_self_reflections", "normalized_consistency_score", "INTEGER DEFAULT NULL")
    if not _has_column("soul_self_reflections", "correction_consumed_at"):
        _add_column("soul_self_reflections", "correction_consumed_at", "TEXT DEFAULT ''")


def _run_v2_migration() -> None:
    """Version 2：cabinet_slot_no 列 + 唯一部分索引。"""
    conn = _get_conn()
    if not _has_column("soul_crystallized_traits", "cabinet_slot_no"):
        _add_column("soul_crystallized_traits", "cabinet_slot_no", "INTEGER DEFAULT NULL")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_slot_active "
        "ON soul_crystallized_traits(cabinet_slot_no) "
        "WHERE cabinet_slot_no IS NOT NULL AND enabled = 1 AND deleted = 0"
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
