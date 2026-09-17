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

# 单颗种子的最大失败重试次数（超过即放弃，等管理员介入）
MAX_ATTEMPTS = 3


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
        ``blocked`` 非零表示本轮因运行模式未消费（操作仍在队列里，不是失败）。
    """
    from ..models.operations import (
        claim_seed_operation,
        finish_seed_operation_in_tx,
        get_seed_operation,
        list_retryable_operations,
        list_running_operations,
        release_seed_operation,
    )
    from ..utils.runtime_mode import resolve_runtime_mode
    from .internalization_engine import InternalizationEngine
    from .seed_manager import ThoughtSeedManager

    stats = {"done": 0, "retry": 0, "skipped": 0, "blocked": 0}

    # 运行模式闸门：非 apply 时**不消费**（留着，等切回 apply 继续）。
    # 不能「消费了再丢弃」——那会把操作判失败、丢掉管理员的批准意图。
    if not resolve_runtime_mode(plugin.config).mutation_allowed:
        stats["blocked"] = 1
        logger.info(
            "[Soul] 运行模式 %s 禁止改写人格，内化队列本轮不消费",
            resolve_runtime_mode(plugin.config).mode,
        )
        return stats

    operations = list_running_operations(limit=limit, operation_type="internalize")

    # 有界重试：把「失败但没超上限」的种子重新认领。
    # 失败原因可能是瞬时 DB 故障，而管理员的批准意图不该因此丢失。
    retry_seeds = list_retryable_operations(operation_type="internalize", max_attempts=MAX_ATTEMPTS)
    for seed_id in retry_seeds:
        if len(operations) >= limit:
            break
        new_id = claim_seed_operation(seed_id, operation_type="internalize")
        if not new_id:
            continue
        op = get_seed_operation(new_id)
        if op is not None:
            operations.append(op)
            logger.info("[Soul] 重新认领种子 %s 重试（operation=%s）", seed_id, new_id)

    manager = ThoughtSeedManager.from_plugin_config(plugin)

    for op in operations:
        try:
            seed = await manager.get_seed_by_id(op.seed_id)
            if seed is None:
                release_seed_operation(op.operation_id, error="种子不存在")
                stats["skipped"] += 1
                continue

            status = (seed.get("status") or "") if isinstance(seed, dict) else ""
            if status == "fermenting":
                # 发酵中的种子归**发酵循环**所有。这里既不能内化，也绝不能
                # 释放它的租约——释放会让发酵完成后无法终结操作，下一轮重复内化。
                stats["skipped"] += 1
                continue
            if status != "pending":
                # 终态（已内化/拒绝/过期）——但若是本操作自己刚写出的终态，
                # 说明已完成过，交给幂等路径；这里只做释放，不写人格。
                release_seed_operation(
                    op.operation_id, error=f"种子状态 {status or '未知'} 不归队列处理"
                )
                stats["skipped"] += 1
                continue

            dedup_cfg = {
                "enabled": bool(plugin.config.thought_cabinet.auto_dedup_enabled),
                "threshold": float(plugin.config.thought_cabinet.auto_dedup_threshold),
            }

            import json as _json

            def _finalize(conn, _op=op):
                """在同一事务内校验所有权 + 种子 CAS，然后写终态。

                任一条不满足 → 整笔回滚，人格不写。这挡住了：
                - 租约被接管后旧执行者提交（操作已不是我们的）
                - 覆盖管理员在途拒绝（种子状态已变）
                """
                return finish_seed_operation_in_tx(
                    conn,
                    _op.operation_id,
                    seed_status="approved",
                    result_json=_json.dumps({}, ensure_ascii=False),
                    expected_seed_status="pending",
                )

            engine = InternalizationEngine(plugin)
            result = await engine.internalize_seed(seed, dedup=dedup_cfg, finalize=_finalize)
        except Exception as e:  # noqa: BLE001 — 单条失败不影响队列其余条目
            logger.exception("[Soul] 内化操作 %s 执行异常", op.operation_id)
            try:
                release_seed_operation(op.operation_id, error=f"{type(e).__name__}: {e}")
            except Exception:  # noqa: BLE001 — 释放失败也不能中断队列
                logger.exception("[Soul] 释放操作 %s 失败", op.operation_id)
            stats["retry"] += 1
            continue

        if not result.get("success"):
            if result.get("blocked_by_mode"):
                # 模式切走：保留队列，不判失败（下一轮 apply 再跑）
                stats["blocked"] += 1
                continue
            if result.get("finalize_failed"):
                # 所有权/状态校验失败：人格已回滚，操作留给接管者或管理员
                stats["skipped"] += 1
                logger.warning(
                    "[Soul] 内化操作 %s 终态校验失败（人格未写）: %s",
                    op.operation_id, result.get("error"),
                )
                continue
            reason = str(result.get("error") or result.get("rejection_reason") or "未知原因")
            release_seed_operation(op.operation_id, error=reason)
            stats["retry"] += 1
            logger.warning("[Soul] 内化操作 %s 未成功（种子 %s）: %s", op.operation_id, op.seed_id, reason)
            continue

        # 人格 + 种子终态 + 操作终态已在 engine 的同一事务内提交
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
