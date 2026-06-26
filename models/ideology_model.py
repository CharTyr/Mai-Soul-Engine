"""意识形态光谱数据模型层 — sqlite3 实现。

替代旧版 peewee 模型，使用标准库 sqlite3 + dataclass，
提供与旧 peewee 接口类似的属性访问和 save() 语义，
使上层组件代码改动最小化。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

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


# ─── 数据类定义 ─────────────────────────────────────────────────────


@dataclass
class IdeologySpectrum:
    """意识形态光谱 — 四维数值 (0-100)。"""

    scope_id: str = "global"
    economic: int = 50
    social: int = 50
    diplomatic: int = 50
    progressive: int = 50
    last_economic_dir: int = 0
    last_social_dir: int = 0
    last_diplomatic_dir: int = 0
    last_progressive_dir: int = 0
    initialized: bool = False
    last_evolution: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def save(self) -> None:
        """持久化当前光谱状态。"""
        save_spectrum(self)


@dataclass
class GroupEvolutionRecord:
    """群组演化记录 — 记录每个群上次分析的时间。"""

    group_id: str = ""
    last_analyzed: datetime = field(default_factory=datetime.now)

    def save(self) -> None:
        """持久化当前记录。"""
        save_group_evolution_record(self)


@dataclass
class EvolutionHistory:
    """演化历史 — 每次光谱变化的 delta 记录。"""

    id: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    group_id: str = ""
    economic_delta: int = 0
    social_delta: int = 0
    diplomatic_delta: int = 0
    progressive_delta: int = 0
    reason: str = ""


@dataclass
class ThoughtSeed:
    """思维种子 — 待审核的潜在观点。"""

    seed_id: str = ""
    stream_id: str = ""
    seed_type: str = ""
    event: str = ""
    intensity: int = 0
    confidence: int = 0
    evidence_json: str = "[]"
    reasoning: str = ""
    potential_impact_json: str = "{}"
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"

    def delete_instance(self) -> None:
        """删除当前种子记录。"""
        delete_thought_seed(self.seed_id)


@dataclass
class CrystallizedTrait:
    """固化 trait — 已内化的观点，可禁用/软删除。"""

    trait_id: str = ""
    stream_id: str = ""
    seed_id: str = ""
    name: str = ""
    question: str = ""
    thought: str = ""
    tags_json: str = "[]"
    confidence: int = 0
    evidence_json: str = "[]"
    spectrum_impact_json: str = "{}"
    created_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True
    deleted: bool = False
    ideology_layer: str = "conduct"
    lifecycle_state: str = "active"

    def save(self) -> None:
        """持久化当前 trait。"""
        save_crystallized_trait(self)


@dataclass
class ContextSlice:
    """群/会话上下文切片 — 仅记录 Mai 相对全局的局部偏移（P1-d）。"""

    scope_type: str = "group"
    scope_key: str = ""
    economic_offset: int = 0
    social_offset: int = 0
    diplomatic_offset: int = 0
    progressive_offset: int = 0
    sample_count: int = 0
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class MoodState:
    """短期情绪辅助层（P1-e），不写入长期三观。"""

    scope_id: str = "global"
    valence: int = 0
    arousal: int = 0
    energy: int = 0
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ThoughtEdge:
    """轻量思想图谱边（P1-b）。"""

    id: int = 0
    from_trait_id: str = ""
    to_trait_id: str = ""
    relation_type: str = ""
    source_ref: str = ""
    created_at: datetime = field(default_factory=datetime.now)


# ─── 建表与迁移 ─────────────────────────────────────────────────────

_CREATE_SQL = [
    """
    CREATE TABLE IF NOT EXISTS soul_ideology_spectrum (
        scope_id TEXT PRIMARY KEY DEFAULT 'global',
        economic INTEGER DEFAULT 50,
        social INTEGER DEFAULT 50,
        diplomatic INTEGER DEFAULT 50,
        progressive INTEGER DEFAULT 50,
        last_economic_dir INTEGER DEFAULT 0,
        last_social_dir INTEGER DEFAULT 0,
        last_diplomatic_dir INTEGER DEFAULT 0,
        last_progressive_dir INTEGER DEFAULT 0,
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
        economic_delta INTEGER DEFAULT 0,
        social_delta INTEGER DEFAULT 0,
        diplomatic_delta INTEGER DEFAULT 0,
        progressive_delta INTEGER DEFAULT 0,
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
        economic_offset INTEGER DEFAULT 0,
        social_offset INTEGER DEFAULT 0,
        diplomatic_offset INTEGER DEFAULT 0,
        progressive_offset INTEGER DEFAULT 0,
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


def _run_migrations() -> None:
    """就地迁移：补齐旧版可能缺失的列。"""
    if not _has_column("soul_thought_seeds", "stream_id"):
        _add_column("soul_thought_seeds", "stream_id", "TEXT DEFAULT ''")
    if not _has_column("soul_thought_seeds", "confidence"):
        _add_column("soul_thought_seeds", "confidence", "INTEGER DEFAULT 0")
    if not _has_column("soul_thought_seeds", "evidence_json"):
        _add_column("soul_thought_seeds", "evidence_json", "TEXT DEFAULT '[]'")
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


# ─── IdeologySpectrum CRUD ──────────────────────────────────────────


def get_or_create_spectrum(scope_id: str = "global") -> IdeologySpectrum:
    """获取或创建光谱记录。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_ideology_spectrum WHERE scope_id = ?", (scope_id,)
    ).fetchone()
    if row:
        return _row_to_spectrum(row)
    # 不存在则创建
    now = datetime.now()
    conn.execute(
        """INSERT INTO soul_ideology_spectrum
           (scope_id, economic, social, diplomatic, progressive,
            last_economic_dir, last_social_dir, last_diplomatic_dir, last_progressive_dir,
            initialized, last_evolution, updated_at)
           VALUES (?, 50, 50, 50, 50, 0, 0, 0, 0, 0, ?, ?)""",
        (scope_id, _dt_to_str(now), _dt_to_str(now)),
    )
    conn.commit()
    return IdeologySpectrum(scope_id=scope_id, last_evolution=now, updated_at=now)


def save_spectrum(s: IdeologySpectrum) -> None:
    """更新光谱记录。"""
    conn = _get_conn()
    conn.execute(
        """UPDATE soul_ideology_spectrum SET
           economic = ?, social = ?, diplomatic = ?, progressive = ?,
           last_economic_dir = ?, last_social_dir = ?, last_diplomatic_dir = ?, last_progressive_dir = ?,
           initialized = ?, last_evolution = ?, updated_at = ?
           WHERE scope_id = ?""",
        (
            s.economic, s.social, s.diplomatic, s.progressive,
            s.last_economic_dir, s.last_social_dir, s.last_diplomatic_dir, s.last_progressive_dir,
            int(s.initialized), _dt_to_str(s.last_evolution), _dt_to_str(s.updated_at),
            s.scope_id,
        ),
    )
    conn.commit()


def _row_to_spectrum(row: sqlite3.Row) -> IdeologySpectrum:
    """sqlite3.Row → IdeologySpectrum。"""
    return IdeologySpectrum(
        scope_id=row["scope_id"],
        economic=row["economic"],
        social=row["social"],
        diplomatic=row["diplomatic"],
        progressive=row["progressive"],
        last_economic_dir=row["last_economic_dir"],
        last_social_dir=row["last_social_dir"],
        last_diplomatic_dir=row["last_diplomatic_dir"],
        last_progressive_dir=row["last_progressive_dir"],
        initialized=bool(row["initialized"]),
        last_evolution=_str_to_dt(row["last_evolution"]),
        updated_at=_str_to_dt(row["updated_at"]),
    )


# ─── GroupEvolutionRecord CRUD ──────────────────────────────────────


def get_or_create_group_evolution(group_id: str) -> GroupEvolutionRecord:
    """获取或创建群组演化记录。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_group_evolution WHERE group_id = ?", (group_id,)
    ).fetchone()
    if row:
        return GroupEvolutionRecord(group_id=row["group_id"], last_analyzed=_str_to_dt(row["last_analyzed"]))
    now = datetime.now()
    conn.execute(
        "INSERT INTO soul_group_evolution (group_id, last_analyzed) VALUES (?, ?)",
        (group_id, _dt_to_str(now)),
    )
    conn.commit()
    return GroupEvolutionRecord(group_id=group_id, last_analyzed=now)


def save_group_evolution_record(r: GroupEvolutionRecord) -> None:
    """更新群组演化记录。"""
    conn = _get_conn()
    conn.execute(
        "UPDATE soul_group_evolution SET last_analyzed = ? WHERE group_id = ?",
        (_dt_to_str(r.last_analyzed), r.group_id),
    )
    conn.commit()


# ─── EvolutionHistory CRUD ──────────────────────────────────────────


def create_evolution_history(
    timestamp: datetime,
    group_id: str,
    economic_delta: int,
    social_delta: int,
    diplomatic_delta: int,
    progressive_delta: int,
    reason: str,
) -> None:
    """创建演化历史记录。"""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO soul_evolution_history
           (timestamp, group_id, economic_delta, social_delta, diplomatic_delta, progressive_delta, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            _dt_to_str(timestamp), group_id,
            economic_delta, social_delta, diplomatic_delta, progressive_delta,
            reason,
        ),
    )
    conn.commit()


def get_evolution_history(limit: int = 50) -> list[EvolutionHistory]:
    """获取最近的演化历史。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM soul_evolution_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [
        EvolutionHistory(
            id=row["id"],
            timestamp=_str_to_dt(row["timestamp"]),
            group_id=row["group_id"],
            economic_delta=row["economic_delta"],
            social_delta=row["social_delta"],
            diplomatic_delta=row["diplomatic_delta"],
            progressive_delta=row["progressive_delta"],
            reason=row["reason"],
        )
        for row in rows
    ]


# ─── ThoughtSeed CRUD ───────────────────────────────────────────────


def create_thought_seed(
    seed_id: str,
    stream_id: str,
    seed_type: str,
    event: str,
    intensity: int,
    confidence: int,
    evidence_json: str,
    reasoning: str,
    potential_impact_json: str,
    status: str = "pending",
) -> None:
    """创建思维种子。"""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO soul_thought_seeds
           (seed_id, stream_id, seed_type, event, intensity, confidence,
            evidence_json, reasoning, potential_impact_json, created_at, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            seed_id, stream_id, seed_type, event, intensity, confidence,
            evidence_json, reasoning, potential_impact_json,
            _dt_to_str(datetime.now()), status,
        ),
    )
    conn.commit()


def get_pending_thought_seeds(stream_id: str | None = None) -> list[ThoughtSeed]:
    """获取待审核种子列表。"""
    conn = _get_conn()
    if stream_id and stream_id != "global":
        rows = conn.execute(
            "SELECT * FROM soul_thought_seeds WHERE status = 'pending' AND stream_id = ? ORDER BY created_at DESC",
            (stream_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM soul_thought_seeds WHERE status = 'pending' ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_seed(row) for row in rows]


def get_thought_seed_by_id(seed_id: str) -> ThoughtSeed | None:
    """根据 ID 获取种子。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_thought_seeds WHERE seed_id = ?", (seed_id,)
    ).fetchone()
    return _row_to_seed(row) if row else None


def delete_thought_seed(seed_id: str) -> bool:
    """删除种子。"""
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM soul_thought_seeds WHERE seed_id = ?", (seed_id,))
    conn.commit()
    return cursor.rowcount > 0


def count_pending_thought_seeds() -> int:
    """统计待审核种子数量。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM soul_thought_seeds WHERE status = 'pending'"
    ).fetchone()
    return int(row["cnt"]) if row else 0


def _row_to_seed(row: sqlite3.Row) -> ThoughtSeed:
    """sqlite3.Row → ThoughtSeed。"""
    return ThoughtSeed(
        seed_id=row["seed_id"],
        stream_id=row["stream_id"],
        seed_type=row["seed_type"],
        event=row["event"],
        intensity=row["intensity"],
        confidence=row["confidence"],
        evidence_json=row["evidence_json"],
        reasoning=row["reasoning"],
        potential_impact_json=row["potential_impact_json"],
        created_at=_str_to_dt(row["created_at"]),
        status=row["status"],
    )


# ─── CrystallizedTrait CRUD ─────────────────────────────────────────


def create_crystallized_trait(
    trait_id: str,
    stream_id: str,
    seed_id: str,
    name: str,
    question: str,
    thought: str,
    tags_json: str,
    confidence: int,
    evidence_json: str,
    spectrum_impact_json: str,
    created_at: datetime | None = None,
    enabled: bool = True,
    deleted: bool = False,
    ideology_layer: str = "conduct",
    lifecycle_state: str = "active",
) -> None:
    """创建固化 trait。"""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO soul_crystallized_traits
           (trait_id, stream_id, seed_id, name, question, thought, tags_json,
            confidence, evidence_json, spectrum_impact_json, created_at, enabled, deleted,
            ideology_layer, lifecycle_state)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trait_id, stream_id, seed_id, name, question, thought, tags_json,
            confidence, evidence_json, spectrum_impact_json,
            _dt_to_str(created_at or datetime.now()),
            int(enabled), int(deleted),
            ideology_layer or "conduct",
            lifecycle_state or "active",
        ),
    )
    conn.commit()


def get_crystallized_trait_by_id(trait_id: str) -> CrystallizedTrait | None:
    """根据 ID 获取 trait（不含已软删除的）。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_crystallized_traits WHERE trait_id = ? AND deleted = 0", (trait_id,)
    ).fetchone()
    return _row_to_trait(row) if row else None


def save_crystallized_trait(t: CrystallizedTrait) -> None:
    """更新 trait。"""
    conn = _get_conn()
    conn.execute(
        """UPDATE soul_crystallized_traits SET
           stream_id = ?, seed_id = ?, name = ?, question = ?, thought = ?,
           tags_json = ?, confidence = ?, evidence_json = ?, spectrum_impact_json = ?,
           created_at = ?, enabled = ?, deleted = ?,
           ideology_layer = ?, lifecycle_state = ?
           WHERE trait_id = ?""",
        (
            t.stream_id, t.seed_id, t.name, t.question, t.thought,
            t.tags_json, t.confidence, t.evidence_json, t.spectrum_impact_json,
            _dt_to_str(t.created_at), int(t.enabled), int(t.deleted),
            t.ideology_layer or "conduct",
            t.lifecycle_state or "active",
            t.trait_id,
        ),
    )
    conn.commit()


def query_crystallized_traits(
    *,
    deleted: bool = False,
    enabled: bool | None = None,
    stream_id: str | None = None,
    order_by_created_desc: bool = True,
    limit: int = 50,
) -> list[CrystallizedTrait]:
    """查询固化 trait 列表。"""
    conn = _get_conn()
    conditions = ["deleted = ?"]
    params: list[Any] = [int(deleted)]
    if enabled is not None:
        conditions.append("enabled = ?")
        params.append(int(enabled))
    if stream_id is not None:
        conditions.append("stream_id = ?")
        params.append(stream_id)
    where = " AND ".join(conditions)
    order = "created_at DESC" if order_by_created_desc else "created_at ASC"
    sql = f"SELECT * FROM soul_crystallized_traits WHERE {where} ORDER BY {order} LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_trait(row) for row in rows]


def query_active_traits_for_injection(
    stream_id: str = "",
    limit: int = 80,
) -> list[CrystallizedTrait]:
    """查询用于注入的活跃 trait（按 stream_id 匹配或全局）。"""
    conn = _get_conn()
    if stream_id:
        rows = conn.execute(
            """SELECT * FROM soul_crystallized_traits
               WHERE deleted = 0 AND enabled = 1
               AND (stream_id = ? OR stream_id = '')
               ORDER BY created_at DESC LIMIT ?""",
            (stream_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM soul_crystallized_traits
               WHERE deleted = 0 AND enabled = 1 AND stream_id = ''
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [_row_to_trait(row) for row in rows]


def _row_to_trait(row: sqlite3.Row) -> CrystallizedTrait:
    """sqlite3.Row → CrystallizedTrait。"""
    return CrystallizedTrait(
        trait_id=row["trait_id"],
        stream_id=row["stream_id"],
        seed_id=row["seed_id"],
        name=row["name"],
        question=row["question"],
        thought=row["thought"],
        tags_json=row["tags_json"],
        confidence=row["confidence"],
        evidence_json=row["evidence_json"],
        spectrum_impact_json=row["spectrum_impact_json"],
        created_at=_str_to_dt(row["created_at"]),
        enabled=bool(row["enabled"]),
        deleted=bool(row["deleted"]),
        ideology_layer=row["ideology_layer"] if "ideology_layer" in row.keys() else "conduct",
        lifecycle_state=row["lifecycle_state"] if "lifecycle_state" in row.keys() else "active",
    )


# ─── P1：切片 / 情绪 / 图谱 ───────────────────────────────────────────


def upsert_context_slice(
    scope_type: str,
    scope_key: str,
    economic_offset: int,
    social_offset: int,
    diplomatic_offset: int,
    progressive_offset: int,
    sample_count: int,
) -> None:
    conn = _get_conn()
    now = _dt_to_str(datetime.now())
    conn.execute(
        """INSERT INTO soul_context_slices
           (scope_type, scope_key, economic_offset, social_offset, diplomatic_offset,
            progressive_offset, sample_count, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(scope_type, scope_key) DO UPDATE SET
           economic_offset = economic_offset + excluded.economic_offset,
           social_offset = social_offset + excluded.social_offset,
           diplomatic_offset = diplomatic_offset + excluded.diplomatic_offset,
           progressive_offset = progressive_offset + excluded.progressive_offset,
           sample_count = sample_count + excluded.sample_count,
           updated_at = excluded.updated_at""",
        (
            scope_type,
            scope_key,
            economic_offset,
            social_offset,
            diplomatic_offset,
            progressive_offset,
            sample_count,
            now,
        ),
    )
    conn.commit()


def get_context_slice(scope_type: str, scope_key: str) -> ContextSlice | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_context_slices WHERE scope_type = ? AND scope_key = ?",
        (scope_type, scope_key),
    ).fetchone()
    if not row:
        return None
    return ContextSlice(
        scope_type=row["scope_type"],
        scope_key=row["scope_key"],
        economic_offset=int(row["economic_offset"]),
        social_offset=int(row["social_offset"]),
        diplomatic_offset=int(row["diplomatic_offset"]),
        progressive_offset=int(row["progressive_offset"]),
        sample_count=int(row["sample_count"]),
        updated_at=_str_to_dt(row["updated_at"]),
    )


def get_or_create_mood(scope_id: str = "global") -> MoodState:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM soul_mood_state WHERE scope_id = ?", (scope_id,)).fetchone()
    if row:
        return MoodState(
            scope_id=row["scope_id"],
            valence=int(row["valence"]),
            arousal=int(row["arousal"]),
            energy=int(row["energy"]),
            updated_at=_str_to_dt(row["updated_at"]),
        )
    now = datetime.now()
    conn.execute(
        "INSERT INTO soul_mood_state (scope_id, valence, arousal, energy, updated_at) VALUES (?, 0, 0, 0, ?)",
        (scope_id, _dt_to_str(now)),
    )
    conn.commit()
    return MoodState(scope_id=scope_id, updated_at=now)


def save_mood(m: MoodState) -> None:
    conn = _get_conn()
    conn.execute(
        """UPDATE soul_mood_state SET valence = ?, arousal = ?, energy = ?, updated_at = ?
           WHERE scope_id = ?""",
        (m.valence, m.arousal, m.energy, _dt_to_str(m.updated_at), m.scope_id),
    )
    conn.commit()


def create_thought_edge(
    from_trait_id: str,
    to_trait_id: str,
    relation_type: str,
    source_ref: str = "",
) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO soul_thought_edges
           (from_trait_id, to_trait_id, relation_type, source_ref, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (from_trait_id, to_trait_id or "", relation_type, source_ref, _dt_to_str(datetime.now())),
    )
    conn.commit()


def list_thought_edges_for_trait(trait_id: str, limit: int = 20) -> list[ThoughtEdge]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM soul_thought_edges
           WHERE from_trait_id = ? OR to_trait_id = ?
           ORDER BY id DESC LIMIT ?""",
        (trait_id, trait_id, limit),
    ).fetchall()
    return [
        ThoughtEdge(
            id=row["id"],
            from_trait_id=row["from_trait_id"],
            to_trait_id=row["to_trait_id"],
            relation_type=row["relation_type"],
            source_ref=row["source_ref"],
            created_at=_str_to_dt(row["created_at"]),
        )
        for row in rows
    ]


def count_traits_by_layer() -> dict[str, int]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT ideology_layer, COUNT(*) AS cnt FROM soul_crystallized_traits
           WHERE deleted = 0 AND enabled = 1 GROUP BY ideology_layer"""
    ).fetchall()
    out: dict[str, int] = {"values": 0, "worldview": 0, "conduct": 0}
    for row in rows:
        layer = row["ideology_layer"] or "conduct"
        out[layer] = int(row["cnt"])
    return out
