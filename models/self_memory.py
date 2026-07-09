"""Bot 自我事件记忆 CRUD + 检索。

与 A_Memorix 人物画像无关：主体永远是 bot 自己。
检索以关键词 LIKE 为主（中文可靠），不做 embedding。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

from ._conn import _dt_to_str, _get_conn

__all__ = [
    "SelfMemory",
    "create_self_memory",
    "get_self_memory",
    "list_self_memories",
    "search_self_memories",
    "soft_delete_self_memory",
    "count_self_memories",
    "set_self_memory_tags",
]


@dataclass
class SelfMemory:
    """一条 bot 自我事件/约定记忆。"""

    memory_id: str
    content: str
    tags_json: str = "[]"
    source: str = "manual"
    stream_id: str = ""
    event_time: str = ""
    importance: int = 3
    enabled: int = 1
    deleted: int = 0
    created_at: str = ""
    updated_at: str = ""

    @property
    def tags(self) -> list[str]:
        try:
            raw = json.loads(self.tags_json or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(raw, list):
            return []
        return [str(x).strip() for x in raw if str(x).strip()]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _normalize_tags(tags: Iterable[object] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in tags or []:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= 12:
            break
    return out


def _tags_json(tags: Iterable[object] | None) -> str:
    return json.dumps(_normalize_tags(tags), ensure_ascii=False)


def _clamp_importance(value: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 3
    return max(1, min(5, n))


def create_self_memory(
    content: str,
    *,
    tags: Iterable[object] | None = None,
    source: str = "manual",
    stream_id: str = "",
    event_time: str = "",
    importance: int = 3,
) -> SelfMemory | None:
    """写入一条自我记忆。内容为空则返回 None。"""
    text = str(content or "").strip()
    if not text:
        return None
    if len(text) > 2000:
        text = text[:2000]

    now = _dt_to_str(datetime.now())
    memory_id = _new_id()
    mem = SelfMemory(
        memory_id=memory_id,
        content=text,
        tags_json=_tags_json(tags),
        source=str(source or "manual").strip() or "manual",
        stream_id=str(stream_id or "").strip(),
        event_time=str(event_time or "").strip() or now,
        importance=_clamp_importance(importance),
        enabled=1,
        deleted=0,
        created_at=now,
        updated_at=now,
    )
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO soul_self_memories (
            memory_id, content, tags_json, source, stream_id, event_time,
            importance, enabled, deleted, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            mem.memory_id,
            mem.content,
            mem.tags_json,
            mem.source,
            mem.stream_id,
            mem.event_time,
            mem.importance,
            mem.enabled,
            mem.deleted,
            mem.created_at,
            mem.updated_at,
        ),
    )
    conn.commit()
    return mem


def get_self_memory(memory_id: str) -> SelfMemory | None:
    mid = str(memory_id or "").strip()
    if not mid:
        return None
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_self_memories WHERE memory_id = ? AND deleted = 0 LIMIT 1",
        (mid,),
    ).fetchone()
    return _row_to_memory(row) if row else None


def list_self_memories(limit: int = 20) -> list[SelfMemory]:
    lim = max(1, min(50, int(limit or 20)))
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT * FROM soul_self_memories
        WHERE deleted = 0 AND enabled = 1
        ORDER BY importance DESC, created_at DESC
        LIMIT ?
        """,
        (lim,),
    ).fetchall()
    return [_row_to_memory(r) for r in rows]


def search_self_memories(query: str, limit: int = 8) -> list[SelfMemory]:
    """关键词检索自我记忆。

    - 空 query：返回最近/高重要度列表
    - 多词：AND 语义（每条须同时包含所有 token）
    """
    lim = max(1, min(30, int(limit or 8)))
    q = str(query or "").strip()
    if not q:
        return list_self_memories(limit=lim)

    tokens = _tokenize_query(q)
    if not tokens:
        return list_self_memories(limit=lim)

    # 先拉一批再过滤，避免复杂动态 SQL 注入；表量预期不大
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT * FROM soul_self_memories
        WHERE deleted = 0 AND enabled = 1
        ORDER BY importance DESC, created_at DESC
        LIMIT 300
        """
    ).fetchall()

    hits: list[tuple[int, SelfMemory]] = []
    for row in rows:
        mem = _row_to_memory(row)
        hay = f"{mem.content}\n{' '.join(mem.tags)}".lower()
        score = 0
        ok = True
        for tok in tokens:
            t = tok.lower()
            if t not in hay:
                ok = False
                break
            score += hay.count(t)
        if not ok:
            continue
        score += mem.importance
        hits.append((score, mem))

    hits.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in hits[:lim]]


def soft_delete_self_memory(memory_id: str) -> bool:
    mid = str(memory_id or "").strip()
    if not mid:
        return False
    now = _dt_to_str(datetime.now())
    conn = _get_conn()
    cur = conn.execute(
        """
        UPDATE soul_self_memories
        SET deleted = 1, updated_at = ?
        WHERE memory_id = ? AND deleted = 0
        """,
        (now, mid),
    )
    conn.commit()
    return cur.rowcount > 0


def set_self_memory_tags(memory_id: str, tags: Iterable[object] | None) -> SelfMemory | None:
    mid = str(memory_id or "").strip()
    if not mid:
        return None
    now = _dt_to_str(datetime.now())
    tags_json = _tags_json(tags)
    conn = _get_conn()
    cur = conn.execute(
        """
        UPDATE soul_self_memories
        SET tags_json = ?, updated_at = ?
        WHERE memory_id = ? AND deleted = 0
        """,
        (tags_json, now, mid),
    )
    conn.commit()
    if cur.rowcount <= 0:
        return None
    return get_self_memory(mid)


def count_self_memories() -> dict[str, Any]:
    conn = _get_conn()
    total = conn.execute(
        "SELECT COUNT(*) FROM soul_self_memories WHERE deleted = 0"
    ).fetchone()[0]
    enabled = conn.execute(
        "SELECT COUNT(*) FROM soul_self_memories WHERE deleted = 0 AND enabled = 1"
    ).fetchone()[0]
    return {"total": int(total or 0), "enabled": int(enabled or 0)}


def format_memories_for_tool(memories: list[SelfMemory], *, max_chars: int = 1200) -> str:
    """把检索结果压成 planner 可读短文。"""
    if not memories:
        return "（无匹配的自我记忆）"
    lines: list[str] = []
    used = 0
    for idx, mem in enumerate(memories, start=1):
        tag_s = ",".join(mem.tags) if mem.tags else "-"
        line = (
            f"{idx}. [{mem.memory_id}] {mem.content}"
            f"\n   tags={tag_s} · source={mem.source} · t={mem.event_time or mem.created_at}"
        )
        if used + len(line) > max_chars and lines:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def _tokenize_query(query: str) -> list[str]:
    # 中英文词/数字；过滤过短噪声
    raw = re.findall(r"[\w\u4e00-\u9fff]+", query, flags=re.UNICODE)
    out: list[str] = []
    seen: set[str] = set()
    for tok in raw:
        t = tok.strip()
        if len(t) < 1:
            continue
        # 单字符英文跳过；中文单字保留（事件关键词常是单字+语境）
        if len(t) == 1 and t.isascii():
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= 8:
            break
    return out


def _row_to_memory(row: Any) -> SelfMemory:
    return SelfMemory(
        memory_id=str(row["memory_id"] or ""),
        content=str(row["content"] or ""),
        tags_json=str(row["tags_json"] or "[]"),
        source=str(row["source"] or "manual"),
        stream_id=str(row["stream_id"] or ""),
        event_time=str(row["event_time"] or ""),
        importance=int(row["importance"] or 3),
        enabled=int(row["enabled"] or 0),
        deleted=int(row["deleted"] or 0),
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
    )
