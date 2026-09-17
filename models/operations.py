"""种子操作租约与幂等终结（单赢家认领）。

**为什么需要**：内化要调 LLM（几十秒），不能在数据库事务里做。原实现是
「先内化 → 成功后才标种子终态」，于是并发批准或崩后重试会让同一颗种子
被内化两次，光谱影响施加两遍。

**做法**：
1. LLM 调用**之前**先认领租约（``claim_seed_operation``）。同一
   ``(seed_id, operation_type)`` 只允许一条 ``running``（部分唯一索引兜底），
   所以并发时只有一条能进入内化。
2. 成功后在**同一事务**里提交「操作结果 + 种子终态」（``finish_seed_operation``）。
   重复终结幂等返回 False。
3. 失败释放租约（``release_seed_operation``），种子保持非终态，可重试。
4. 进程崩溃留下的 ``running`` 记录在租约过期后可被抢占重试，不会永久卡死。

租约 + 幂等终结替代不了跨 LLM 调用的长事务——那是做不到的；它保证的是
「同一颗种子同时只有一个内化在跑，且结果只落一次」。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ._conn import _dt_to_str, _get_conn, _str_to_dt
from .seeds import TERMINAL_SEED_STATUSES

__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "SeedOperation",
    "claim_seed_operation",
    "get_seed_operation",
    "finish_seed_operation",
    "release_seed_operation",
]

# 默认租约：必须大于一次内化的最坏耗时（含 2 次 LLM 调用），
# 但要短到崩溃后能较快恢复。取 10 分钟。
DEFAULT_LEASE_SECONDS = 600

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


@dataclass
class SeedOperation:
    """一条种子操作记账。"""

    operation_id: str = ""
    seed_id: str = ""
    operation_type: str = "internalize"
    status: str = STATUS_RUNNING
    attempt: int = 1
    lease_expires_at: str = ""
    result_json: str = ""
    error: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None


def _row_to_operation(row: Any) -> SeedOperation:
    return SeedOperation(
        operation_id=row["operation_id"],
        seed_id=row["seed_id"],
        operation_type=row["operation_type"],
        status=row["status"],
        attempt=int(row["attempt"] or 1),
        lease_expires_at=row["lease_expires_at"] or "",
        result_json=row["result_json"] or "",
        error=row["error"] or "",
        created_at=_str_to_dt(row["created_at"]),
        updated_at=_str_to_dt(row["updated_at"]),
    )


def get_seed_operation(operation_id: str) -> SeedOperation | None:
    """按 operation_id 取操作记录。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM soul_seed_operations WHERE operation_id = ?",
        (operation_id,),
    ).fetchone()
    return _row_to_operation(row) if row else None


def claim_seed_operation(
    seed_id: str,
    *,
    operation_type: str = "internalize",
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> str | None:
    """认领一颗种子的操作租约。

    成功返回新的 ``operation_id``；以下情况返回 None（调用方应放弃本次内化）：
    - 已有未过期的 ``running`` 租约（并发批准 / 正在跑）
    - 该种子已存在 ``done`` 记录（已经内化过，防重复施加影响）
    - 已终态的种子（approved/rejected/expired/internalized）

    租约过期的 ``running`` 记录会被抢占（新 operation_id，attempt+1）。
    """
    conn = _get_conn()
    now = datetime.now()
    now_str = _dt_to_str(now)
    expires_str = _dt_to_str(now + timedelta(seconds=max(0, lease_seconds)))

    # 已内化过 → 不再认领（幂等）
    settled = conn.execute(
        "SELECT 1 FROM soul_seed_operations "
        "WHERE seed_id = ? AND operation_type = ? AND status = ? LIMIT 1",
        (seed_id, operation_type, STATUS_DONE),
    ).fetchone()
    if settled is not None:
        return None

    # 种子已是终态 → 不再认领
    seed_row = conn.execute(
        "SELECT status FROM soul_thought_seeds WHERE seed_id = ?", (seed_id,),
    ).fetchone()
    if seed_row is not None and seed_row["status"] in TERMINAL_SEED_STATUSES:
        return None

    operation_id = f"op_{uuid.uuid4().hex[:16]}"
    try:
        conn.execute(
            "INSERT INTO soul_seed_operations "
            "(operation_id, seed_id, operation_type, status, attempt, "
            " lease_expires_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
            (operation_id, seed_id, operation_type, STATUS_RUNNING, expires_str, now_str, now_str),
        )
        conn.commit()
        return operation_id
    except Exception:
        # 撞唯一索引 = 已有 running 租约；下面尝试抢占过期租约
        conn.rollback()

    try:
        cursor = conn.execute(
            "UPDATE soul_seed_operations "
            "SET operation_id = ?, attempt = attempt + 1, lease_expires_at = ?, "
            "    updated_at = ?, error = '' "
            "WHERE seed_id = ? AND operation_type = ? AND status = ? "
            "  AND (lease_expires_at = '' OR lease_expires_at < ?)",
            (operation_id, expires_str, now_str, seed_id, operation_type, STATUS_RUNNING, now_str),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        return None
    return operation_id if cursor.rowcount == 1 else None


def finish_seed_operation(
    operation_id: str,
    *,
    seed_status: str,
    result_json: str = "",
) -> bool:
    """幂等终结：同一事务提交操作结果 + 种子终态。

    仅当操作仍为 ``running`` 时生效；重复调用返回 False（不会二次施加影响）。
    ``seed_status`` 必须是合法终态，否则拒绝写入。
    """
    if seed_status not in TERMINAL_SEED_STATUSES:
        return False

    conn = _get_conn()
    now_str = _dt_to_str(datetime.now())
    try:
        conn.execute("BEGIN")
        row = conn.execute(
            "SELECT seed_id, status FROM soul_seed_operations WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if row is None or row["status"] != STATUS_RUNNING:
            conn.execute("ROLLBACK")
            return False

        cursor = conn.execute(
            "UPDATE soul_seed_operations "
            "SET status = ?, result_json = ?, lease_expires_at = '', updated_at = ? "
            "WHERE operation_id = ? AND status = ?",
            (STATUS_DONE, result_json, now_str, operation_id, STATUS_RUNNING),
        )
        if cursor.rowcount != 1:
            conn.execute("ROLLBACK")
            return False

        conn.execute(
            "UPDATE soul_thought_seeds SET status = ? WHERE seed_id = ?",
            (seed_status, row["seed_id"]),
        )
        conn.execute("COMMIT")
        return True
    except Exception:
        conn.rollback()
        return False


def release_seed_operation(operation_id: str, *, error: str = "") -> bool:
    """释放租约（内化失败）：标记 failed，种子保持非终态以便重试。"""
    conn = _get_conn()
    now_str = _dt_to_str(datetime.now())
    cursor = conn.execute(
        "UPDATE soul_seed_operations "
        "SET status = ?, error = ?, lease_expires_at = '', updated_at = ? "
        "WHERE operation_id = ? AND status = ?",
        (STATUS_FAILED, error[:500], now_str, operation_id, STATUS_RUNNING),
    )
    conn.commit()
    return cursor.rowcount == 1
