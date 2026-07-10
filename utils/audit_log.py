"""Soul 审计 / 运行日志（data/audit.jsonl）。

所有关键闭环事件统一写这里，便于开发观察：
- init / reset / api_set_spectrum
- evolution / evolution_skip / evolution_cycle
- reflection_cycle
- seed_created（可选）
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

_audit_lock = asyncio.Lock()
_audit_file: Path | None = None
_audit_enabled: bool = True
# 审计日志轮转阈值（MB）
_AUDIT_MAX_SIZE_MB: int = 8
AUDIT_MAX_BYTES: int = _AUDIT_MAX_SIZE_MB * 1024 * 1024


def set_audit_enabled(enabled: bool) -> None:
    """设置审计日志开关（由 plugin.on_config_update 调用）。"""
    global _audit_enabled
    _audit_enabled = enabled


def init_audit_log(plugin_dir: Path) -> None:
    global _audit_file
    data_dir = plugin_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _audit_file = data_dir / "audit.jsonl"


def _rotate_if_needed(path: Path) -> None:
    if path.exists() and path.stat().st_size > AUDIT_MAX_BYTES:
        rotated = path.with_suffix(".1.jsonl")
        if rotated.exists():
            rotated.unlink()
        path.rename(rotated)


async def log_audit_event(event_type: str, **fields: Any) -> None:
    """写入一条结构化审计事件。"""
    if not _audit_enabled:
        return
    if not _audit_file:
        return
    entry: dict[str, Any] = {
        "ts": datetime.now().isoformat(),
        "type": str(event_type or "event"),
    }
    for k, v in fields.items():
        if v is not None:
            # 截断 detail 字段到 200 字符，防 audit.jsonl 膨胀
            if k == "detail" and len(str(v)) > 200:
                v = str(v)[:200] + "..."
            entry[k] = v

    path = _audit_file

    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    async with _audit_lock:
        await asyncio.to_thread(_write)


async def log_evolution(
    group_id: str,
    before: dict,
    after: dict,
    deltas: dict,
    reason: str,
    message_count: int,
) -> None:
    """记录成功演化。"""
    await log_audit_event(
        "evolution",
        group_id=group_id,
        before=before,
        after=after,
        deltas=deltas,
        reason=reason,
        message_count=message_count,
    )


async def log_evolution_skip(
    group_id: str,
    reason: str,
    *,
    message_count: int = 0,
    detail: str = "",
) -> None:
    """记录跳过的演化（消息不足 / LLM 失败 / 解析失败等）。"""
    await log_audit_event(
        "evolution_skip",
        group_id=group_id,
        reason=reason,
        message_count=message_count,
        detail=detail or None,
    )


async def log_evolution_cycle(
    *,
    groups_planned: int,
    groups_analyzed: int,
    groups_skipped: int,
    seeds_created: int = 0,
    interval_hours: float = 0,
) -> None:
    """一轮演化循环摘要（所有监控群跑完后）。"""
    await log_audit_event(
        "evolution_cycle",
        groups_planned=groups_planned,
        groups_analyzed=groups_analyzed,
        groups_skipped=groups_skipped,
        seeds_created=seeds_created,
        interval_hours=interval_hours,
    )


async def log_reflection_cycle(
    *,
    pending: int,
    evaluated: int,
    skipped: int,
    seeds: int,
    parse_failed: bool = False,
    llm_failed: bool = False,
    detail: str = "",
) -> None:
    """一轮自我评价摘要。"""
    await log_audit_event(
        "reflection_cycle",
        pending=pending,
        evaluated=evaluated,
        skipped=skipped,
        seeds=seeds,
        parse_failed=parse_failed,
        llm_failed=llm_failed,
        detail=detail or None,
    )


async def log_init(admin_id: str, spectrum: dict) -> None:
    await log_audit_event("init", admin_id=admin_id, spectrum=spectrum)


async def log_reset(admin_id: str) -> None:
    await log_audit_event("reset", admin_id=admin_id)


async def log_api_set_spectrum(before: dict, after: dict) -> None:
    await log_audit_event("api_set_spectrum", before=before, after=after)


def read_recent_audit(limit: int = 20) -> list[dict[str, Any]]:
    """同步读取最近 N 条 audit（供 /soul_observe）。"""
    if not _audit_file or not _audit_file.exists():
        return []
    try:
        lines = _audit_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-max(1, min(200, limit)) :]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out
