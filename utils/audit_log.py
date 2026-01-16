import json
import asyncio
from datetime import datetime
from pathlib import Path

_audit_lock = asyncio.Lock()
_audit_file: Path | None = None


def init_audit_log(plugin_dir: Path):
    global _audit_file
    data_dir = plugin_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _audit_file = data_dir / "audit.jsonl"


async def log_evolution(
    group_id: str,
    before: dict,
    after: dict,
    deltas: dict,
    reason: str,
    message_count: int,
):
    """记录演化日志到audit.jsonl"""
    if not _audit_file:
        return
    
    entry = {
        "ts": datetime.now().isoformat(),
        "type": "evolution",
        "group_id": group_id,
        "before": before,
        "after": after,
        "deltas": deltas,
        "reason": reason,
        "message_count": message_count,
    }
    
    async with _audit_lock:
        with open(_audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def log_init(admin_id: str, spectrum: dict):
    """记录初始化日志"""
    if not _audit_file:
        return
    
    entry = {
        "ts": datetime.now().isoformat(),
        "type": "init",
        "admin_id": admin_id,
        "spectrum": spectrum,
    }
    
    async with _audit_lock:
        with open(_audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def log_reset(admin_id: str):
    """记录重置日志"""
    if not _audit_file:
        return
    
    entry = {
        "ts": datetime.now().isoformat(),
        "type": "reset",
        "admin_id": admin_id,
    }
    
    async with _audit_lock:
        with open(_audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
