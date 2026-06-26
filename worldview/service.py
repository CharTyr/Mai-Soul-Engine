"""P1 三观服务：演化限速、群切片、情绪衰减与注入摘要。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..models import ideology_model as im
from .constants import (
    IDEOLOGY_LAYERS,
    LAYER_LABEL_ZH,
    LIFECYCLE_LABEL_ZH,
    SPECTRUM_DIM_TO_LAYER,
    normalize_ideology_layer,
    normalize_lifecycle_state,
)

logger = logging.getLogger(__name__)

MOOD_AXES: tuple[str, ...] = ("valence", "arousal", "energy")


@dataclass
class WorldviewConfigView:
    """从插件配置提取的 P1 视图（避免强类型耦合）。"""

    p1_enabled: bool = True
    values_max_delta: int = 2
    worldview_max_delta: int = 4
    conduct_max_delta: int = 6
    local_influence_ratio: float = 0.35
    mood_enabled: bool = True
    mood_decay_hours: float = 8.0
    mood_inject: bool = True
    graph_inject: bool = True


def config_from_plugin(plugin: Any) -> WorldviewConfigView:
    wv = plugin.config.worldview
    return WorldviewConfigView(
        p1_enabled=wv.p1_enabled,
        values_max_delta=wv.values_max_delta,
        worldview_max_delta=wv.worldview_max_delta,
        conduct_max_delta=wv.conduct_max_delta,
        local_influence_ratio=wv.local_influence_ratio,
        mood_enabled=wv.mood_enabled,
        mood_decay_hours=wv.mood_decay_hours,
        mood_inject=wv.mood_inject,
        graph_inject=wv.graph_inject,
    )


class WorldviewService:
    def __init__(self, cfg: WorldviewConfigView) -> None:
        self.cfg = cfg

    def layer_cap(self, layer: str) -> int:
        layer = normalize_ideology_layer(layer)
        if layer == "values":
            return max(0, self.cfg.values_max_delta)
        if layer == "worldview":
            return max(0, self.cfg.worldview_max_delta)
        return max(0, self.cfg.conduct_max_delta)

    def apply_layer_caps_to_deltas(self, raw: dict[str, int], evolution_rate: int) -> dict[str, int]:
        """按三观层对四维光谱 delta 限速（P1-c）。"""
        if not self.cfg.p1_enabled:
            return {k: max(-evolution_rate, min(evolution_rate, int(raw.get(k, 0)))) for k in SPECTRUM_DIM_TO_LAYER}

        out: dict[str, int] = {}
        for dim, layer in SPECTRUM_DIM_TO_LAYER.items():
            cap = min(evolution_rate, self.layer_cap(layer))
            val = int(raw.get(dim, 0) or 0)
            out[dim] = max(-cap, min(cap, val))
        return out

    def record_local_slice(self, stream_id: str, deltas: dict[str, int], message_count: int) -> None:
        """记录群聊对全局光谱的局部偏移（P1-d），不写用户画像。"""
        if not self.cfg.p1_enabled or not stream_id:
            return
        ratio = max(0.0, min(1.0, self.cfg.local_influence_ratio))
        local = {dim: int(round(float(deltas.get(dim, 0)) * ratio)) for dim in SPECTRUM_DIM_TO_LAYER}
        im.upsert_context_slice(
            scope_type="group",
            scope_key=stream_id,
            sincerity_offset=local["sincerity"],
            engagement_offset=local["engagement"],
            closeness_offset=local["closeness"],
            directness_offset=local["directness"],
            sample_count=message_count,
        )

    def get_slice_offsets(self, stream_id: str) -> dict[str, int] | None:
        if not self.cfg.p1_enabled or not stream_id:
            return None
        row = im.get_context_slice("group", stream_id)
        if not row:
            return None
        return {
            "sincerity": row.sincerity_offset,
            "engagement": row.engagement_offset,
            "closeness": row.closeness_offset,
            "directness": row.directness_offset,
        }

    def decay_mood_if_needed(self) -> im.MoodState:
        mood = im.get_or_create_mood("global")
        if not self.cfg.p1_enabled or not self.cfg.mood_enabled:
            return mood
        hours = max(0.1, float(self.cfg.mood_decay_hours))
        if mood.updated_at and datetime.now() - mood.updated_at > timedelta(hours=hours):
            mood.valence = 0
            mood.arousal = 0
            mood.energy = 0
            mood.updated_at = datetime.now()
            im.save_mood(mood)
        return mood

    def nudge_mood_from_deltas(self, deltas: dict[str, int]) -> None:
        """由演化结果轻微调整短期情绪，不写入长期三观表。"""
        if not self.cfg.p1_enabled or not self.cfg.mood_enabled:
            return
        mood = im.get_or_create_mood("global")
        total = sum(int(deltas.get(d, 0) or 0) for d in SPECTRUM_DIM_TO_LAYER)
        if total > 0:
            mood.valence = min(2, mood.valence + 1)
            mood.arousal = min(2, mood.arousal + 1)
        elif total < 0:
            mood.valence = max(-2, mood.valence - 1)
            mood.energy = max(-2, mood.energy - 1)
        mood.updated_at = datetime.now()
        im.save_mood(mood)

    def mood_prompt_lines(self) -> list[str]:
        if not self.cfg.p1_enabled or not self.cfg.mood_enabled or not self.cfg.mood_inject:
            return []
        mood = self.decay_mood_if_needed()
        if mood.valence == 0 and mood.arousal == 0 and mood.energy == 0:
            return []
        parts: list[str] = []
        if mood.valence > 0:
            parts.append("语气可稍偏积极")
        elif mood.valence < 0:
            parts.append("语气可稍偏克制、少展开")
        if mood.arousal > 0:
            parts.append("可稍带兴奋感")
        if mood.energy < 0:
            parts.append("可更简短、少铺陈")
        if not parts:
            return []
        return ["【短期情绪辅助】" + "；".join(parts) + "（仅影响语气，不代表三观变化）"]

    def build_layer_trait_summary(
        self,
        stream_id: str,
        limit_per_layer: int = 2,
        exclude_trait_ids: set[str] | None = None,
        traits: list[Any] | None = None,
    ) -> str:
        """按三观层汇总可注入的固化观点（P1-f）。

        Args:
            exclude_trait_ids: 已在详细 trait 注入块中出现的 trait_id，层摘要中跳过避免重复。
            traits: 可选，预查询的 traits 列表；为 None 时内部查询。
        """
        if not self.cfg.p1_enabled:
            return ""
        exclude = exclude_trait_ids or set()
        if traits is None:
            traits = im.query_active_traits_for_injection(stream_id=stream_id, limit=80)
        by_layer: dict[str, list[str]] = {layer: [] for layer in IDEOLOGY_LAYERS}
        for t in traits:
            if t.trait_id in exclude:
                continue
            layer = normalize_ideology_layer(t.ideology_layer)
            if len(by_layer[layer]) >= limit_per_layer:
                continue
            thought = (t.thought or "").replace("\n", " ").strip()[:100]
            if thought:
                by_layer[layer].append(thought)

        lines: list[str] = []
        for layer in IDEOLOGY_LAYERS:
            items = by_layer[layer]
            if items:
                label = LAYER_LABEL_ZH.get(layer, layer)
                lines.append(f"- {label}：" + "；".join(items))
        if not lines:
            return ""
        return "【三观分层摘要】\n" + "\n".join(lines)

    def build_graph_hint(self, stream_id: str, trait_ids: list[str], limit: int = 3) -> str:
        if not self.cfg.p1_enabled or not self.cfg.graph_inject:
            return ""
        hints: list[str] = []
        for tid in trait_ids[:limit]:
            edges = im.list_thought_edges_for_trait(tid, limit=4)
            for e in edges:
                rel = e.relation_type
                if rel == "derived_from" and e.source_ref:
                    hints.append(f"观点 {tid[:8]}… 源自内省/种子 {e.source_ref[:12]}")
                elif rel == "supports" and e.to_trait_id:
                    hints.append(f"观点 {tid[:8]}… 支撑 {e.to_trait_id[:8]}…")
        if not hints:
            return ""
        return "【思想关联】" + "；".join(hints[:limit])

    def format_status_extras(self, stream_id: str) -> str:
        if not self.cfg.p1_enabled:
            return ""
        blocks: list[str] = []
        offsets = self.get_slice_offsets(stream_id)
        if offsets and any(offsets.values()):
            blocks.append(
                "本群局部偏移（相对全局）: "
                f"真诚{offsets['sincerity']:+d} 投入{offsets['engagement']:+d} "
                f"亲近{offsets['closeness']:+d} 直率{offsets['directness']:+d}"
            )
        mood = self.decay_mood_if_needed()
        if self.cfg.mood_enabled and (mood.valence or mood.arousal or mood.energy):
            blocks.append(
                f"短期情绪: 愉悦{mood.valence:+d} 兴奋{mood.arousal:+d} 精力{mood.energy:+d}（自动衰减）"
            )
        counts = im.count_traits_by_layer()
        if counts:
            parts = [f"{LAYER_LABEL_ZH.get(k, k)}{v}条" for k, v in counts.items() if v]
            if parts:
                blocks.append("固化观点分层: " + " ".join(parts))
        return "\n".join(blocks)

    def api_worldview_payload(self, stream_id: str = "") -> dict[str, Any]:
        mood = self.decay_mood_if_needed()
        slice_row = im.get_context_slice("group", stream_id) if stream_id else None
        return {
            "p1_enabled": self.cfg.p1_enabled,
            "mood": {
                "valence": mood.valence,
                "arousal": mood.arousal,
                "energy": mood.energy,
                "updated_at": mood.updated_at.isoformat() if mood.updated_at else None,
            },
            "trait_counts_by_layer": im.count_traits_by_layer(),
            "group_slice": (
                {
                    "sincerity_offset": slice_row.sincerity_offset,
                    "engagement_offset": slice_row.engagement_offset,
                    "closeness_offset": slice_row.closeness_offset,
                    "directness_offset": slice_row.directness_offset,
                    "sample_count": slice_row.sample_count,
                    "updated_at": slice_row.updated_at.isoformat() if slice_row.updated_at else None,
                }
                if slice_row
                else None
            ),
            "layer_labels": LAYER_LABEL_ZH,
            "lifecycle_labels": LIFECYCLE_LABEL_ZH,
        }

    @staticmethod
    def infer_layer_from_tags(tags: list[str]) -> str:
        joined = " ".join(tags).casefold()
        if any(x in joined for x in ("价值观", "value", "原则", "底线")):
            return "values"
        if any(x in joined for x in ("世界观", "world", "理解", "认知")):
            return "worldview"
        return "conduct"

    def register_trait_graph(
        self,
        trait_id: str,
        seed_id: str = "",
        merged_into: str | None = None,
    ) -> None:
        if seed_id:
            im.create_thought_edge(
                from_trait_id=trait_id,
                to_trait_id="",
                relation_type="derived_from",
                source_ref=seed_id,
            )
        if merged_into:
            im.create_thought_edge(
                from_trait_id=trait_id,
                to_trait_id=merged_into,
                relation_type="supports",
                source_ref="dedup_merge",
            )