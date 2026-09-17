"""LLM 输出的内化候选：结构化校验后再落库（Phase D「候选优先」）。

**为什么需要**：LLM 产出的是**候选**，不是人格。旧代码在写人格之前做的是
「静默兜底」——`confidence` 解析失败就归 0、tags 类型不对就清空、
`int(impact.get(...))` 遇到非数字直接抛 ValueError。结果是两种坏局面：

- 模型吐了垃圾 → 人格悄悄变了（静默 clamp / 归零），事后无法归因
- 模型吐了非数字 → 内化崩溃

本模块把「候选是否可接纳」变成一次显式判定：无效候选带**机器可读的原因**返回，
调用方据此跳过写入并记录，而不是猜。

区分两类问题：
- **拒绝**（``valid=False``）：会污染人格或会崩溃的——空观点、数值无法解析、
  越界、类型错误
- **告警**（``warnings``）：可容忍的噪声——未知轴名、tags 类型不对、层名无法识别
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "DELTA_AXES",
    "MAX_TAGS",
    "TraitCandidate",
    "build_trait_candidate",
]

# 只有这四个轴可以改光谱（与 worldview/constants.SPECTRUM_DIM_TO_LAYER 对齐）
DELTA_AXES: tuple[str, ...] = ("sincerity", "engagement", "closeness", "directness")
MAX_TAGS = 8


@dataclass(frozen=True)
class TraitCandidate:
    """校验后的候选。``valid=False`` 时不得写入任何人格状态。"""

    valid: bool
    rejection_reason: str = ""
    thought: str = ""
    layer: str = ""
    tags: tuple[str, ...] = ()
    confidence: float = 0.0
    spectrum_deltas: Mapping[str, int] = field(default_factory=dict)
    reasoning: str = ""
    warnings: tuple[str, ...] = ()


def _reject(reason: str) -> TraitCandidate:
    return TraitCandidate(valid=False, rejection_reason=reason)


def _as_number(value: Any) -> float | None:
    """把 LLM 可能给出的数字表示解析成 float；bool 不算数字。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def build_trait_candidate(
    raw: Any,
    *,
    max_delta: int,
    default_layer: str = "conduct",
) -> TraitCandidate:
    """把内化 LLM 的原始输出校验成候选。

    Args:
        raw: 解析后的 LLM 输出（通常来自 ``json.loads``）。
        max_delta: 单轴允许的最大绝对值（越界即拒绝，不 clamp）。
        default_layer: 层名无法识别时的回落层。

    Returns:
        TraitCandidate：``valid=True`` 才可进入写入流程。
    """
    if not isinstance(raw, Mapping):
        return _reject("not_a_mapping")

    warnings: list[str] = []

    # ── 观点正文 ───────────────────────────────────────────────────
    thought = raw.get("thought")
    thought_text = thought.strip() if isinstance(thought, str) else ""
    if not thought_text:
        return _reject("empty_thought")

    # ── 置信度 ─────────────────────────────────────────────────────
    raw_confidence = raw.get("confidence", 0.0)
    if raw_confidence is None:
        confidence = 0.0
    else:
        parsed_confidence = _as_number(raw_confidence)
        if parsed_confidence is None:
            return _reject("confidence_not_numeric")
        if parsed_confidence < 0.0 or parsed_confidence > 1.0:
            return _reject(
                f"confidence_out_of_range:{parsed_confidence}"
            )
        confidence = parsed_confidence

    # ── 光谱 delta ─────────────────────────────────────────────────
    # 主键是 spectrum_deltas；spectrum_impact 是历史别名（旧 prompt / 旧模型输出），
    # 仍须支持，否则会静默把「有光谱影响」当成「零影响」。
    raw_deltas = raw.get("spectrum_deltas")
    if raw_deltas is None:
        raw_deltas = raw.get("spectrum_impact")
    if raw_deltas is None:
        raw_deltas = {}
    if not isinstance(raw_deltas, Mapping):
        return _reject("spectrum_deltas_not_a_mapping")

    deltas: dict[str, int] = {axis: 0 for axis in DELTA_AXES}
    for key, value in raw_deltas.items():
        axis = str(key)
        if axis not in DELTA_AXES:
            # 未知轴名：丢弃并告警，不让它静默消失也不因它拒绝整条候选
            warnings.append(f"unknown_axis:{axis}")
            continue
        parsed_delta = _as_number(value)
        if parsed_delta is None:
            return _reject(f"delta_not_numeric:{axis}")
        if abs(parsed_delta) > abs(int(max_delta)):
            return _reject(f"delta_exceeds_max:{axis}:{parsed_delta}")
        deltas[axis] = int(parsed_delta)

    # ── 层 ─────────────────────────────────────────────────────────
    from ..worldview.constants import normalize_ideology_layer

    raw_layer = raw.get("ideology_layer")
    layer = normalize_ideology_layer(
        str(raw_layer or "").strip(), default=default_layer,
    )
    if layer == default_layer and str(raw_layer or "").strip() not in ("", default_layer):
        warnings.append(f"layer_unrecognized:{raw_layer}")

    # ── tags ───────────────────────────────────────────────────────
    raw_tags = raw.get("tags")
    tags: tuple[str, ...] = ()
    if raw_tags is None:
        tags = ()
    elif not isinstance(raw_tags, (list, tuple)):
        warnings.append("tags_not_a_list")
    else:
        cleaned = [t.strip() for t in raw_tags if isinstance(t, str) and t.strip()]
        if len(cleaned) != len([t for t in raw_tags]):
            warnings.append("tags_filtered")
        tags = tuple(cleaned[:MAX_TAGS])

    reasoning = raw.get("reasoning")
    return TraitCandidate(
        valid=True,
        thought=thought_text,
        layer=layer,
        tags=tags,
        confidence=confidence,
        spectrum_deltas=deltas,
        reasoning=reasoning.strip() if isinstance(reasoning, str) else "",
        warnings=tuple(warnings),
    )
