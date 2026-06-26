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
"""trait 生命周期状态枚举。

已有写入路径（实际使用）：
- ``active``：新建 trait 的初始状态。
- ``strengthened``：merge 时被重复证据强化（TTL 过期豁免）。
- ``expired``：长期未强化、超 ``trait_ttl_days`` 自动过期（同时 enabled=0）。

预留枚举（**尚无写入路径**，等待基于 LLM 的内省矛盾检测落地，详见 README/AGENTS）：
- ``weakened`` / ``revised`` / ``contradicted``。
注入侧 ``ideology_injector._trait_quality_score`` 已对 ``weakened`` / ``revised`` 预置降权分，
故保留枚举而非删除；删除会使该降权分支变成死代码且阻断后续迭代。
"""

SPECTRUM_DIM_TO_LAYER: dict[str, str] = {
    "sincerity": "values",
    "engagement": "worldview",
    "closeness": "conduct",
    "directness": "conduct",
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