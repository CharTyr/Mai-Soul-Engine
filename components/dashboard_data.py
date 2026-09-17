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


# ─── T19：八种空态必须可区分 ─────────────────────────────────────────

# 状态取值（看板/文本渲染据此给出**不同**文案，不得都写成"无数据"）
HEALTH_OK = "ok"
HEALTH_DISABLED = "disabled"                 # 功能关着
HEALTH_NOT_INITIALIZED = "not_initialized"   # 没做 /soul_setup
HEALTH_EMPTY = "empty"                       # 开着但确实没数据
HEALTH_FAILED = "failed"                     # 出错了（取证 / LLM）
HEALTH_UNVERIFIED = "unverified"             # 已交回宿主但无法确认最终生效
HEALTH_STOPPED = "stopped"                   # 后台任务不在跑

# 审计事件里代表「取证失败」与「LLM 失败」的原因码
_EVIDENCE_FAILURE_REASONS = {"fetch_messages_failed", "stream_not_found"}
_LLM_FAILURE_REASONS = {"llm_failed", "llm_empty_response", "llm_parse_failed"}


def _collect_health_states(plugin: Any, stream_id: str, cfg: Any, group_slice: Any,
                           pending_seeds: int) -> list[dict[str, Any]]:
    """把八种状态**分别**算出来（T19）。

    关键：`无候选` 与 `思维阁关着`、`没数据` 与 `出错了` 必须给出不同 state，
    否则看板会把「功能没开」「没东西」「坏了」渲染成同一种空。
    """
    states: list[dict[str, Any]] = []

    # 未初始化 vs 已初始化：光谱
    spectrum = get_or_create_spectrum("global")
    if not spectrum.initialized:
        states.append({"key": "spectrum", "state": HEALTH_NOT_INITIALIZED,
                       "label": "未初始化", "detail": "请执行 /soul_setup"})
    else:
        states.append({"key": "spectrum", "state": HEALTH_OK, "label": "已初始化", "detail": ""})

    # 无候选 vs 思维阁关着
    if not cfg.thought_cabinet.enabled:
        states.append({"key": "candidates", "state": HEALTH_DISABLED,
                       "label": "思维阁已关闭", "detail": "不产生候选"})
    elif pending_seeds <= 0:
        states.append({"key": "candidates", "state": HEALTH_EMPTY,
                       "label": "无候选", "detail": "思维阁开着，暂无待审种子"})
    else:
        states.append({"key": "candidates", "state": HEALTH_OK,
                       "label": f"{pending_seeds} 个待审候选", "detail": ""})

    # 无切片 vs 未开分层 vs 全局视角
    if not getattr(cfg.worldview, "p1_enabled", False):
        states.append({"key": "slices", "state": HEALTH_DISABLED,
                       "label": "分层已关闭", "detail": "不记录群切片"})
    elif not stream_id:
        states.append({"key": "slices", "state": HEALTH_DISABLED,
                       "label": "全局视角", "detail": "切片是分群的，切到群视角才看得到"})
    elif not group_slice:
        states.append({"key": "slices", "state": HEALTH_EMPTY,
                       "label": "无切片", "detail": "本群尚无偏移记录"})
    else:
        states.append({"key": "slices", "state": HEALTH_OK, "label": "有切片", "detail": ""})

    # 取证失败 / LLM 失败：从审计日志尾部找**最近一次**原因
    from ..utils.audit_log import tail_events

    events = tail_events(200)
    last_evidence_fail = ""
    last_llm_fail = ""
    for ev in reversed(events):
        etype = str(ev.get("type", ""))
        reason = str(ev.get("reason", "") or "")
        if not last_evidence_fail and reason in _EVIDENCE_FAILURE_REASONS:
            last_evidence_fail = reason
        if not last_llm_fail and (
            reason in _LLM_FAILURE_REASONS
            or (etype == "reflection_cycle" and ev.get("llm_failed"))
        ):
            last_llm_fail = reason or "reflection_llm_failed"
        if last_evidence_fail and last_llm_fail:
            break

    if last_evidence_fail:
        states.append({"key": "evidence", "state": HEALTH_FAILED,
                       "label": "取证失败", "detail": last_evidence_fail})
    else:
        states.append({"key": "evidence", "state": HEALTH_OK, "label": "取证正常", "detail": ""})

    if last_llm_fail:
        states.append({"key": "llm", "state": HEALTH_FAILED,
                       "label": "LLM 失败", "detail": last_llm_fail})
    else:
        states.append({"key": "llm", "state": HEALTH_OK, "label": "LLM 正常", "detail": ""})

    # 注入未验证：宿主没有请求后回调，插件无法确认最终请求里真的带了注入
    if not getattr(cfg.self_reflection, "enabled", False):
        states.append({"key": "injection", "state": HEALTH_DISABLED,
                       "label": "未开记录", "detail": "自评关着，不落注入快照"})
    else:
        states.append({"key": "injection", "state": HEALTH_UNVERIFIED,
                       "label": "注入未验证",
                       "detail": "已交回宿主；宿主无请求后回调，无法确认最终是否生效"})

    # 后台停止：监督器里有没有没在跑的任务
    supervisor = getattr(plugin, "_task_supervisor", None)
    if supervisor is None:
        states.append({"key": "background", "state": HEALTH_STOPPED,
                       "label": "后台状态不可用", "detail": "未拿到监督器"})
    else:
        _task_names = ("evolution", "notion", "reflection", "fermentation", "internalization")
        try:
            statuses = {name: str(supervisor.state_of(name).status) for name in _task_names}
        except Exception:  # noqa: BLE001 — 看板不能因观测失败而崩
            statuses = {}
        bad = [k for k, v in statuses.items() if v in ("stopped", "failed", "backoff")]
        if bad:
            states.append({"key": "background", "state": HEALTH_STOPPED,
                           "label": "后台异常", "detail": ",".join(sorted(bad))})
        else:
            states.append({"key": "background", "state": HEALTH_OK,
                           "label": "后台运行中", "detail": ""})

    return states


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
        "self_reflection": cfg.self_reflection.enabled,
    }

    # ── 自我评价反馈回路 ────────────────────────────────────────────
    self_reflection_data: dict[str, Any] = {"enabled": cfg.self_reflection.enabled}
    if cfg.self_reflection.enabled:
        from ..models.self_reflection import count_pending_reflections, count_self_reflections
        from .reflection_feedback import build_recent_reflection_summary

        self_reflection_data["pending_counts"] = count_pending_reflections()
        self_reflection_data["reflection_counts"] = count_self_reflections()
        self_reflection_data["recent_summary"] = build_recent_reflection_summary(stream_id, limit=10)

    # ── 思维阁槽位占用 ────────────────────────────────────────────
    cabinet_slots: list[dict[str, Any]] = []
    for t in all_traits:
        if t.cabinet_slot_no is not None:
            cabinet_slots.append({
                "slot_no": t.cabinet_slot_no,
                "trait_id": t.trait_id,
                "name": t.name or "",
            })
    cabinet_slots.sort(key=lambda x: x["slot_no"])
    cabinet_grid: list[dict[str, Any]] = []
    slot_map = {s["slot_no"]: s for s in cabinet_slots}
    for i in range(1, 13):
        if i in slot_map:
            s = slot_map[i]
            cabinet_grid.append({"slot_no": i, "empty": False, "trait_id": s["trait_id"], "name": s["name"]})
        else:
            cabinet_grid.append({"slot_no": i, "empty": True})
    cabinet_data: dict[str, Any] = {
        "slots_used": len(cabinet_slots),
        "slots_total": 12,
        "occupancy": cabinet_slots,
        "grid": cabinet_grid,
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
        "self_reflection": self_reflection_data,
        "cabinet": cabinet_data,
        "health_states": _collect_health_states(plugin, stream_id, cfg, group_slice, pending_seeds),
    }
