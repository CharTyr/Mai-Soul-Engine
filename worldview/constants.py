"""P1 三观分层与生命周期常量。"""

from __future__ import annotations

IDEOLOGY_LAYERS: tuple[str, ...] = ("values", "worldview", "conduct")
"""价值观（最慢）→ 世界观（中）→ 处事观（较快）。"""

LIFECYCLE_STATES: tuple[str, ...] = (
    "active",
    "strengthened",
    "weakened",
    "revised",
    "expired",
    "contradicted",
)

SPECTRUM_DIM_TO_LAYER: dict[str, str] = {
    "economic": "values",
    "social": "worldview",
    "diplomatic": "worldview",
    "progressive": "conduct",
}

LAYER_LABEL_ZH: dict[str, str] = {
    "values": "价值观",
    "worldview": "世界观",
    "conduct": "处事观",
}

LIFECYCLE_LABEL_ZH: dict[str, str] = {
    "active": "有效",
    "strengthened": "已强化",
    "weakened": "已弱化",
    "revised": "已修正",
    "expired": "已过期",
    "contradicted": "存在矛盾",
}


def normalize_ideology_layer(raw: str | None, default: str = "conduct") -> str:
    layer = (raw or "").strip().lower()
    if layer in IDEOLOGY_LAYERS:
        return layer
    aliases = {
        "value": "values",
        "values": "values",
        "价值观": "values",
        "world": "worldview",
        "worldview": "worldview",
        "世界观": "worldview",
        "conduct": "conduct",
        "处事": "conduct",
        "处事观": "conduct",
    }
    return aliases.get(layer, default if default in IDEOLOGY_LAYERS else "conduct")


def normalize_lifecycle_state(raw: str | None, default: str = "active") -> str:
    state = (raw or "").strip().lower()
    if state in LIFECYCLE_STATES:
        return state
    return default if default in LIFECYCLE_STATES else "active"