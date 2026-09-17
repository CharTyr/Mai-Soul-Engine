"""内化操作队列：命令快速返回，后台按预算执行。

**要解决的问题**：`/soul_approve`（即时内化模式）在命令里直接调 LLM 内化。
插件给 LLM 的超时是 120s，而宿主给命令的 RPC 超时是 60s（`component_timeout.py`），
所以慢一点就会出现：

- 管理员看到命令超时/失败
- 但插件侧内化其实跑完了、光谱影响已经写入

也就是"报错说没成功，实际上成功了"。配合操作租约（`models/operations.py`），
重复批准不会重复施加影响，但管理员的认知是错的。

**做法**：命令只做认领并把操作落进队列，立刻回 `operation_id`；
后台任务按预算执行真正的内化。管理员随时可以查状态。

设计边界：
- 队列就是 `soul_seed_operations` 里 `running` 的行（不另建表）
- 只处理 `pending` 种子（`fermenting` 归发酵循环所有，别抢）
- 单条失败逐条隔离；失败释放租约，下轮可重试
- 完成时「操作结果 + 种子终态」在同一事务提交（见 `finish_seed_operation`）
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["enqueue_internalization", "run_queue_once"]


async def enqueue_internalization(plugin: Any, seed_id: str) -> str | None:
    """把一颗待审种子排进内化队列。

    Returns:
        ``operation_id``；返回 None 表示不能排队（已有进行中的操作 /
        已经内化过 / 种子已是终态）。
    """
    from ..models.operations import claim_seed_operation

    operation_id = claim_seed_operation(seed_id, operation_type="internalize")
    if operation_id is None:
        logger.info("[Soul] 种子 %s 无法排队（已有进行中的操作或已终态）", seed_id)
        return None
    logger.info("[Soul] 种子 %s 已排队内化 operation=%s", seed_id, operation_id)
    return operation_id


async def run_queue_once(plugin: Any, limit: int = 3) -> dict[str, int]:
    """执行一轮队列：处理已认领但未完成的内化操作。

    Returns:
        ``{"done": n, "retry": n, "skipped": n}``
    """
    from ..models.operations import (
        finish_seed_operation,
        list_running_operations,
        release_seed_operation,
    )
    from .internalization_engine import InternalizationEngine
    from .seed_manager import ThoughtSeedManager

    stats = {"done": 0, "retry": 0, "skipped": 0}
    operations = list_running_operations(limit=limit, operation_type="internalize")

    manager = ThoughtSeedManager.from_plugin_config(plugin)

    for op in operations:
        seed = await manager.get_seed_by_id(op.seed_id)
        if seed is None:
            release_seed_operation(op.operation_id, error="种子不存在")
            stats["skipped"] += 1
            continue

        status = (seed.get("status") or "") if isinstance(seed, dict) else ""
        if status != "pending":
            # 终态（已内化/拒绝）或发酵中（归发酵循环）——不该由本队列处理
            release_seed_operation(
                op.operation_id, error=f"种子状态 {status or '未知'} 不归队列处理"
            )
            stats["skipped"] += 1
            continue

        dedup_cfg = {
            "enabled": bool(plugin.config.thought_cabinet.auto_dedup_enabled),
            "threshold": float(plugin.config.thought_cabinet.auto_dedup_threshold),
        }

        try:
            engine = InternalizationEngine(plugin)
            result = await engine.internalize_seed(seed, dedup=dedup_cfg)
        except Exception as e:  # noqa: BLE001 — 单条失败不影响队列其余条目
            logger.exception("[Soul] 内化操作 %s 执行异常", op.operation_id)
            release_seed_operation(op.operation_id, error=f"{type(e).__name__}: {e}")
            stats["retry"] += 1
            continue

        if not result.get("success"):
            reason = str(result.get("error") or result.get("rejection_reason") or "未知原因")
            release_seed_operation(op.operation_id, error=reason)
            stats["retry"] += 1
            logger.warning("[Soul] 内化操作 %s 未成功（种子 %s）: %s", op.operation_id, op.seed_id, reason)
            continue

        # 操作结果与种子终态在同一事务提交
        import json as _json

        finish_seed_operation(
            op.operation_id,
            seed_status="approved",
            result_json=_json.dumps(
                {
                    "trait_id": result.get("trait_id", ""),
                    "merged_into": result.get("merged_into", ""),
                    "relation": result.get("relation", ""),
                },
                ensure_ascii=False,
            ),
        )
        stats["done"] += 1
        logger.info(
            "[Soul] 内化操作 %s 完成：种子 %s → trait %s",
            op.operation_id, op.seed_id, result.get("trait_id", ""),
        )

        # 完成后通知管理员（走 outbox：失败可重放）
        try:
            from ..utils.notify import send_or_queue

            admin_id = plugin.config.admin.admin_user_id
            if admin_id:
                from ..utils.spectrum_utils import parse_user_id

                platform, user_id = parse_user_id(admin_id)
                if platform and user_id:
                    stream = await plugin.ctx.chat.get_stream_by_user_id(
                        platform=platform, user_id=user_id
                    )
                    if stream:
                        trait_id = result.get("trait_id", "")
                        await send_or_queue(
                            plugin,
                            f"🧠 思维种子 {op.seed_id} 内化完成\\ntrait: {trait_id}\\n"
                            f"可用 /soul_slot {trait_id} <槽位> 让它优先被关注",
                            stream,
                            dedupe_key=f"internalize_done:{op.operation_id}",
                        )
        except Exception as e:  # noqa: BLE001 — 通知失败不影响内化结果
            logger.warning("[Soul] 内化完成通知失败（种子 %s）: %s: %s", op.seed_id, type(e).__name__, e)

        if stats["done"] >= limit:
            break

    return stats
