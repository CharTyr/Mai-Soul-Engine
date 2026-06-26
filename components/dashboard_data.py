"""Dashboard 数据聚合：将 Soul 引擎全部当前状态聚合成结构化 dict。

纯读聚合，不调 LLM，不写 DB。供 dashboard_renderer 消费。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..models.ideology_model import (
    count_pending_thought_seeds,
    count_traits_by_layer,
    get_context_slice,
    get_evolution_history,
    get_or_create_mood,
    get_or_create_spectrum,
    list_thought_edges_for_traits,
    query_crystallized_traits,
)


def _fmt_dt(dt: datetime | None) -> str | None:
    """格式化 datetime 为 "YYYY-MM-DD HH:MM:SS"，None 保持 None。"""
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def collect_dashboard_data(plugin: Any, stream_id: str = "") -> dict[str, Any]:
    """聚合 Soul 引擎全部当前状态为结构化 dict。

    Args:
        plugin: 插件实例（含 config 属性）。
        stream_id: 群聊流 ID；为空时表示全局视角，group_slice 返回 None。

    Returns:
        按契约定义的完整状态 dict。
    """
    # ── 社交光谱 ──────────────────────────────────────────────────────
    spectrum = get_or_create_spectrum("global")

    # ── 各层 trait 数量 ──────────────────────────────────────────────
    raw_layer_counts = count_traits_by_layer()
    trait_counts_by_layer: dict[str, int] = {
        "values": raw_layer_counts.get("values", 0),
        "worldview": raw_layer_counts.get("worldview", 0),
        "conduct": raw_layer_counts.get("conduct", 0),
    }

    # ── 生命周期分布 & trait 总数（仅统计未删除） ──────────────────
    all_traits = query_crystallized_traits(deleted=False, limit=10000)
    lifecycle_distribution: dict[str, int] = {
        "active": 0,
        "strengthened": 0,
        "expired": 0,
        "contradicted": 0,
        "weakened": 0,
        "revised": 0,
    }
    for t in all_traits:
        state = t.lifecycle_state
        if state in lifecycle_distribution:
            lifecycle_distribution[state] += 1
    trait_total = len(all_traits)

    # ── 短期情绪（P1） ──────────────────────────────────────────────
    mood = get_or_create_mood("global")

    # ── 群切片（stream_id 为空时直接 None） ────────────────────────
    group_slice: dict[str, int] | None = None
    if stream_id:
        cs = get_context_slice("group", stream_id)
        if cs is not None:
            group_slice = {
                "sincerity_offset": cs.sincerity_offset,
                "engagement_offset": cs.engagement_offset,
                "closeness_offset": cs.closeness_offset,
                "directness_offset": cs.directness_offset,
                "sample_count": cs.sample_count,
            }

    # ── 待审种子数 ──────────────────────────────────────────────────
    pending_seeds = count_pending_thought_seeds()

    # ── 最近演化记录（最多 5 条，新→旧） ───────────────────────────
    history_records = get_evolution_history(limit=5)
    recent_evolutions: list[dict[str, Any]] = []
    for rec in history_records:
        reason = (rec.reason or "")[:60]
        recent_evolutions.append({
            "group_id": rec.group_id,
            "timestamp": _fmt_dt(rec.timestamp),
            "deltas": {
                "sincerity": rec.sincerity_delta,
                "engagement": rec.engagement_delta,
                "closeness": rec.closeness_delta,
                "directness": rec.directness_delta,
            },
            "reason": reason,
        })

    # ── 思想图谱边总数（去重：同一条边会同时出现在 from/to 两端） ──
    trait_ids = [t.trait_id for t in all_traits]
    if trait_ids:
        edges_by_trait = list_thought_edges_for_traits(trait_ids)
        unique_edges: set[tuple[str, str, str]] = set()
        for edges in edges_by_trait.values():
            for e in edges:
                key = (e.from_trait_id, e.to_trait_id, e.relation_type)
                unique_edges.add(key)
        graph_edge_total = len(unique_edges)
    else:
        graph_edge_total = 0

    # ── 功能开关一览 ────────────────────────────────────────────────
    cfg = plugin.config
    feature_flags: dict[str, bool] = {
        "p1_enabled": cfg.worldview.p1_enabled,
        "mood_enabled": cfg.worldview.mood_enabled,
        "graph_inject": cfg.worldview.graph_inject,
        "thought_cabinet": cfg.thought_cabinet.enabled,
        "notion": cfg.notion.enabled,
        "api": cfg.api.enabled,
        "card_render": cfg.render.card_enabled,
    }

    return {
        "initialized": spectrum.initialized,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stream_id": stream_id,
        "spectrum": {
            "sincerity": spectrum.sincerity,
            "engagement": spectrum.engagement,
            "closeness": spectrum.closeness,
            "directness": spectrum.directness,
            "updated_at": _fmt_dt(spectrum.updated_at),
            "last_evolution": _fmt_dt(spectrum.last_evolution),
        },
        "p1_enabled": cfg.worldview.p1_enabled,
        "trait_counts_by_layer": trait_counts_by_layer,
        "lifecycle_distribution": lifecycle_distribution,
        "trait_total": trait_total,
        "mood": {
            "enabled": cfg.worldview.mood_enabled,
            "valence": mood.valence,
            "arousal": mood.arousal,
            "energy": mood.energy,
            "updated_at": _fmt_dt(mood.updated_at),
        },
        "group_slice": group_slice,
        "thought_cabinet": {
            "enabled": cfg.thought_cabinet.enabled,
            "pending_seeds": pending_seeds,
        },
        "recent_evolutions": recent_evolutions,
        "graph_edge_total": graph_edge_total,
        "feature_flags": feature_flags,
    }
