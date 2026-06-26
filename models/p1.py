"""ContextSlice / MoodState / ThoughtEdge 数据类与 CRUD。

P1 三观生长相关：群切片、短期情绪、思想图谱边。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ._conn import _dt_to_str, _get_conn, _str_to_dt

__all__ = [
    "ContextSlice",
    "MoodState",
    "ThoughtEdge",
    "create_thought_edge",
    "get_context_slice",
    "get_or_create_mood",
    "list_thought_edges_for_trait",
    "list_thought_edges_for_traits",
    "save_mood",
    "upsert_context_slice",
]


@dataclass
class ContextSlice:
    """群/会话上下文切片 — 仅记录 Mai 相对全局的局部偏移（P1-d）。"""

    scope_type: str = "group"
    scope_key: str = ""
    sincerity_offset: int = 0
    engagement_offset: int = 0
    closeness_offset: int = 0
    directness_offset: int = 0
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


# ─── 上下文切片 CRUD ─────────────────────────────────────────────


def upsert_context_slice(
    scope_type: str,
    scope_key: str,
    sincerity_offset: int,
    engagement_offset: int,
    closeness_offset: int,
    directness_offset: int,
    sample_count: int,
) -> None:
    conn = _get_conn()
    now = _dt_to_str(datetime.now())
    conn.execute(
        """INSERT INTO soul_context_slices
           (scope_type, scope_key, sincerity_offset, engagement_offset, closeness_offset,
            directness_offset, sample_count, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(scope_type, scope_key) DO UPDATE SET
           sincerity_offset = sincerity_offset + excluded.sincerity_offset,
           engagement_offset = engagement_offset + excluded.engagement_offset,
           closeness_offset = closeness_offset + excluded.closeness_offset,
           directness_offset = directness_offset + excluded.directness_offset,
           sample_count = sample_count + excluded.sample_count,
           updated_at = excluded.updated_at""",
        (
            scope_type,
            scope_key,
            sincerity_offset,
            engagement_offset,
            closeness_offset,
            directness_offset,
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
        sincerity_offset=int(row["sincerity_offset"]),
        engagement_offset=int(row["engagement_offset"]),
        closeness_offset=int(row["closeness_offset"]),
        directness_offset=int(row["directness_offset"]),
        sample_count=int(row["sample_count"]),
        updated_at=_str_to_dt(row["updated_at"]),
    )


# ─── 情绪 CRUD ──────────────────────────────────────────────────────


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


# ─── 思想图谱边 CRUD ────────────────────────────────────────────────


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


def list_thought_edges_for_traits(trait_ids: list[str]) -> dict[str, list[ThoughtEdge]]:
    """批量查询多个 trait 的思想图谱边，按 trait_id 分组返回。

    Args:
        trait_ids: 要查询的 trait ID 列表。

    Returns:
        dict: {trait_id: [edges where 该 trait 是 from 或 to 端]}。
        trait_ids 为空时返回空 dict。
    """
    if not trait_ids:
        return {}
    conn = _get_conn()
    placeholders = ",".join("?" * len(trait_ids))
    trait_set = set(trait_ids)
    rows = conn.execute(
        f"""SELECT * FROM soul_thought_edges
            WHERE from_trait_id IN ({placeholders}) OR to_trait_id IN ({placeholders})
            ORDER BY id DESC""",
        (*trait_ids, *trait_ids),
    ).fetchall()
    result: dict[str, list[ThoughtEdge]] = {tid: [] for tid in trait_ids}
    for row in rows:
        edge = ThoughtEdge(
            id=row["id"],
            from_trait_id=row["from_trait_id"],
            to_trait_id=row["to_trait_id"],
            relation_type=row["relation_type"],
            source_ref=row["source_ref"],
            created_at=_str_to_dt(row["created_at"]),
        )
        if edge.from_trait_id in trait_set:
            result[edge.from_trait_id].append(edge)
        if edge.to_trait_id in trait_set:
            result[edge.to_trait_id].append(edge)
    return result
