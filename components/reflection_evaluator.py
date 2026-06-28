"""自我评价评价层：异步消费协程 + 批量 LLM 评价。

独立于演化循环（oracle 7c），通过 ``plugin._self_reflection_task`` 管理，
``on_unload`` cancel，与 ``_evolution_task`` 一致。

每周期：
1. 清理过期/超量 pending（防堆积）
2. 取待评队列（先入先评，上限 max_replies_per_cycle）
3. 批量送 LLM 评价（SELF_REFLECTION_PROMPT：抽象倾向 + 对立视角 + 相关性门槛判例）
4. 落 soul_self_reflections + 更新 pending 状态
5. 显著偏离的 substantive 回复生成 self_observation 种子（走 /soul_approve 人工审批）
6. 可选批次归一化（normalize_across_batch，对冲系统性高估）
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 自我观察种子生成门槛：一致性分低于此值且 LLM 给了 self_observation_trait 才生成种子
_SEED_SCORE_THRESHOLD: int = 70
# 单条回复文本截断（防 prompt 爆炸）
_RESPONSE_MAX_CHARS: int = 500


async def run_reflection_loop(plugin) -> None:
    """周期性自我评价循环。

    由 plugin._self_reflection_loop() 以 asyncio.create_task 启动。
    """
    cfg = plugin.config.self_reflection
    interval_hours = max(0.5, float(cfg.evaluation_interval_hours))
    logger.info("[SelfReflection] 评价循环已启动，间隔 %.1f 小时", interval_hours)
    while True:
        try:
            await _evaluate_cycle(plugin)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[SelfReflection] 评价周期异常")
        await asyncio.sleep(interval_hours * 3600)


async def _evaluate_cycle(plugin) -> None:
    """单次评价周期。"""
    cfg = plugin.config.self_reflection
    # 懒导入避免循环导入（见 reflection_capture 注释）
    from ..models.self_reflection import (
        cleanup_expired_pending,
        create_self_reflection,
        get_injection_snapshot,
        list_pending_reflections,
        update_pending_status,
    )
    from ..models.ideology_model import get_crystallized_trait_by_id, get_or_create_spectrum
    from ..prompts.self_reflection_prompts import (
        build_abstract_tendency,
        build_reply_block,
        build_self_reflection_prompt,
    )

    # 1. 清理过期/超量 pending + 孤儿 snapshot
    cleaned_pending = cleanup_expired_pending(
        max_age_hours=int(cfg.pending_max_age_hours),
        max_rows=int(cfg.pending_max_rows),
    )
    from ..models.self_reflection import cleanup_orphan_snapshots

    cleaned_snaps = cleanup_orphan_snapshots(max_age_hours=int(cfg.pending_max_age_hours))
    if cleaned_pending or cleaned_snaps:
        logger.debug(
            "[SelfReflection] 清理过期 pending %s 条 + 孤儿 snapshot %s 条",
            cleaned_pending,
            cleaned_snaps,
        )

    # 2. 取待评队列
    pendings = list_pending_reflections(
        limit=int(cfg.max_replies_per_cycle),
        max_age_hours=int(cfg.pending_max_age_hours),
    )
    if not pendings:
        return
    logger.info("[SelfReflection] 本轮评价 %s 条待评回复", len(pendings))

    # 3. 构建批量 prompt
    spectrum = get_or_create_spectrum("global")
    tendency = build_abstract_tendency(
        {
            "sincerity": spectrum.sincerity,
            "engagement": spectrum.engagement,
            "closeness": spectrum.closeness,
            "directness": spectrum.directness,
        }
    )

    reply_blocks: list[str] = []
    meta: list[dict[str, Any]] = []  # 每条 pending 的配对元数据
    for idx, p in enumerate(pendings, start=1):
        trait_lines: list[str] = []
        if p.snapshot_id:
            snap = get_injection_snapshot(p.snapshot_id)
            if snap:
                try:
                    trait_ids = _json.loads(snap.trait_ids_json or "[]")
                except (_json.JSONDecodeError, TypeError):
                    trait_ids = []
                for tid in trait_ids:
                    trait = get_crystallized_trait_by_id(tid)
                    if trait and trait.enabled:
                        trait_lines.append(f"{trait.name}: {trait.thought}")
        try:
            context_lines = _json.loads(p.context_json or "[]")
        except (_json.JSONDecodeError, TypeError):
            context_lines = []
        response_text = (p.response_text or "")[:_RESPONSE_MAX_CHARS]
        reply_blocks.append(
            build_reply_block(idx, response_text, context_lines, trait_lines)
        )
        meta.append({"pending": p, "trait_lines": trait_lines, "context_lines": context_lines})

    prompt = build_self_reflection_prompt(
        tendency=tendency,
        replies_block="\n\n".join(reply_blocks),
    )

    # 4. 调 LLM
    try:
        llm_result = await plugin.ctx.llm.generate(prompt)
    except (RuntimeError, ValueError, OSError):
        logger.exception("[SelfReflection] 评价 LLM 请求失败")
        return
    response = ""
    if isinstance(llm_result, dict):
        response = str(llm_result.get("response", "") or "")
    elif isinstance(llm_result, str):
        response = llm_result
    if not response:
        logger.warning("[SelfReflection] 评价 LLM 返回空")
        return

    # 5. 解析 JSON 数组
    results = _parse_evaluation_json(response)
    if not results:
        logger.warning("[SelfReflection] 无法解析评价结果: %s", response[:200])
        return

    # 可选批次归一化（对冲系统性高估）
    if cfg.normalize_across_batch and len(results) > 1:
        results = _normalize_batch_scores(results)

    # 6. 落库 + 更新 pending + 生成种子
    seed_count = 0
    for item in results:
        idx = int(item.get("index", 0) or 0)
        if idx < 1 or idx > len(meta):
            continue
        m = meta[idx - 1]
        p = m["pending"]
        reply_type = str(item.get("reply_type", "") or "")
        evaluated = int(item.get("evaluated", 0) or 0)
        score = int(item.get("consistency_score", 0) or 0)
        axis = str(item.get("deviating_axis", "") or "")
        direction = str(item.get("deviating_direction", "") or "")
        reason = str(item.get("reason", "") or "")
        sot = item.get("self_observation_trait")

        try:
            reflection_id = create_self_reflection(
                stream_id=p.stream_id or "global",
                pending_id=p.pending_id,
                snapshot_id=p.snapshot_id,
                reply_type=reply_type,
                evaluated=evaluated,
                consistency_score=score,
                deviating_axis=axis,
                deviating_direction=direction,
                reason=reason,
                seed_id="",
            )
        except Exception:
            logger.exception("[SelfReflection] 写 self_reflection 失败 (pending=%s)", p.pending_id)
            reflection_id = 0

        # 更新 pending 状态
        status = "done" if evaluated else "skipped"
        try:
            update_pending_status(p.pending_id, status)
        except Exception:
            logger.exception("[SelfReflection] 更新 pending 状态失败 (pending=%s)", p.pending_id)

        # 生成 self_observation 种子（仅 substantive + 显著偏离 + LLM 给了 trait）
        if (
            reflection_id
            and reply_type == "substantive"
            and evaluated == 1
            and score < _SEED_SCORE_THRESHOLD
            and isinstance(sot, dict)
            and sot.get("name")
        ):
            seed_id = await _maybe_create_self_observation_seed(
                plugin, sot, p, m["context_lines"], m["trait_lines"], reason
            )
            if seed_id:
                seed_count += 1

    logger.info(
        "[SelfReflection] 评价完成：%s 条已评，%s 条跳过，%s 条生成自我观察种子",
        sum(1 for r in results if int(r.get("evaluated", 0) or 0)),
        sum(1 for r in results if not int(r.get("evaluated", 0) or 0)),
        seed_count,
    )


def _parse_evaluation_json(response: str) -> list[dict]:
    """解析 LLM 评价 JSON 数组（兼容 ```json 代码围栏）。"""
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        data = _json.loads(text)
    except (_json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


def _normalize_batch_scores(results: list[dict]) -> list[dict]:
    """批次归一化：自评分减本批均值（仅对 evaluated=1 的项），对冲系统性高估。

    归一化后分数仍 clamp 到 0-100。偏离轴/方向不变（相对排序的信息量保留）。
    """
    scored = [r for r in results if int(r.get("evaluated", 0) or 0) == 1]
    if not scored:
        return results
    mean = sum(int(r.get("consistency_score", 0) or 0) for r in scored) / len(scored)
    for r in scored:
        raw = int(r.get("consistency_score", 0) or 0)
        normalized = int(round(raw - mean + 50))  # 重新中心化到 50
        r["consistency_score"] = max(0, min(100, normalized))
    return results


async def _maybe_create_self_observation_seed(
    plugin,
    sot: dict,
    pending,
    context_lines: list[str],
    trait_lines: list[str],
    reason: str,
) -> str | None:
    """为显著偏离的 substantive 回复创建自我观察种子（走 /soul_approve 人工审批）。

    默认 auto_internalize_threshold=0 → 全部人工审批（防自指跑偏）。
    """
    try:
        from ..thought.seed_manager import ThoughtSeedManager

        manager = ThoughtSeedManager.from_plugin_config(plugin)
        confidence = max(0, min(100, int(sot.get("confidence", 0) or 0)))
        seed_data = {
            "type": "self_observation",
            "event": str(sot.get("name", "") or ""),
            "intensity": confidence / 100.0,  # create_seed 内部 ×100
            "confidence": confidence / 100.0,
            "evidence": [reason] if reason else [str(sot.get("thought", "") or "")],
            "reasoning": str(sot.get("thought", "") or ""),
            "potential_impact": sot.get("spectrum_impact", {}) or {},
        }
        # 上下文：回复本身 + 触发上文
        ctx = [f"bot回复: {(pending.response_text or '')[:200]}"] + list(context_lines)
        seed_id = await manager.create_seed(seed_data, stream_id=pending.stream_id or "", context_messages=ctx)
        if seed_id:
            logger.info("[SelfReflection] 生成自我观察种子 %s: %s", seed_id, sot.get("name", ""))
        return seed_id
    except Exception:
        logger.exception("[SelfReflection] 创建自我观察种子失败")
        return None
