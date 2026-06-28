"""P1.5: 双路反馈测试。

验证：
- build_recent_reflection_summary：无偏离返回空、显著偏离出摘要、小偏离跳过
- apply_self_reflection_spectrum_correction：dead zone、方向、weight=0 跳过、历史记录
- _build_injection_block：自评摘要按有无 trait 分场景插入（oracle 修订点 5）

从宿主仓根运行：``uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/test_reflection_feedback.py -q``
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .conftest import _import_soul_submodule


def _cfg(weight: float = 0.5, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        self_reflection_weight=weight,
        evaluation_interval_hours=6.0,
        max_replies_per_cycle=20,
        relevance_gate_enabled=True,
        normalize_across_batch=False,
        pending_max_age_hours=48,
        pending_max_rows=5000,
    )


def _plugin(weight: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(self_reflection=_cfg(weight=weight)))


# ─── build_recent_reflection_summary ──────────────────────────────


def test_summary_empty_when_no_reflections(soul_db: Any) -> None:
    fb = _import_soul_submodule("components.reflection_feedback")
    assert fb.build_recent_reflection_summary("g", limit=10) == ""


def test_summary_with_significant_deviations(soul_db: Any) -> None:
    fb = _import_soul_submodule("components.reflection_feedback")
    # directness 偏低 3 次（净 -3，≥2 阈值）
    for _ in range(3):
        soul_db.create_self_reflection("g", 1, "", "substantive", 1, 50, "directness", "low", "x")
    summary = fb.build_recent_reflection_summary("g", limit=10)
    assert "直率度偏低" in summary
    assert "3/3次" in summary


def test_summary_skips_minor_deviations(soul_db: Any) -> None:
    fb = _import_soul_submodule("components.reflection_feedback")
    # 仅 1 次偏离（净 -1，<2 阈值）
    soul_db.create_self_reflection("g", 1, "", "substantive", 1, 50, "directness", "low", "x")
    assert fb.build_recent_reflection_summary("g", limit=10) == ""


def test_summary_balanced_deviations_skip(soul_db: Any) -> None:
    """高低抵消时净偏离为 0，跳过。"""
    fb = _import_soul_submodule("components.reflection_feedback")
    soul_db.create_self_reflection("g", 1, "", "substantive", 1, 50, "sincerity", "high", "x")
    soul_db.create_self_reflection("g", 2, "", "substantive", 1, 50, "sincerity", "low", "x")
    assert fb.build_recent_reflection_summary("g", limit=10) == ""


# ─── apply_self_reflection_spectrum_correction ────────────────────


def test_correction_dead_zone_skips_small_net(soul_db: Any) -> None:
    """净偏离 <3（dead zone）不修正。"""
    fb = _import_soul_submodule("components.reflection_feedback")
    soul_db.get_or_create_spectrum("global")
    # 仅 2 次偏低（净 -2 < 3）
    for _ in range(2):
        soul_db.create_self_reflection("global", 1, "", "substantive", 1, 50, "directness", "low", "x")
    before = soul_db.get_or_create_spectrum("global").directness
    fb.apply_self_reflection_spectrum_correction(_plugin(weight=0.5), evolution_rate=6)
    after = soul_db.get_or_create_spectrum("global").directness
    assert before == after  # dead zone 拦截


def test_correction_applies_on_significant_net(soul_db: Any) -> None:
    """净偏离 ≥3 时修正：directness 偏低 4 次 → 人设 directness 向下校准。"""
    fb = _import_soul_submodule("components.reflection_feedback")
    soul_db.get_or_create_spectrum("global")
    for _ in range(4):
        soul_db.create_self_reflection("global", 1, "", "substantive", 1, 50, "directness", "low", "x")
    before = soul_db.get_or_create_spectrum("global").directness
    fb.apply_self_reflection_spectrum_correction(_plugin(weight=0.5), evolution_rate=6)
    after = soul_db.get_or_create_spectrum("global").directness
    assert after < before  # 向下校准


def test_correction_weight_zero_skips(soul_db: Any) -> None:
    """weight=0 时不修正。"""
    fb = _import_soul_submodule("components.reflection_feedback")
    soul_db.get_or_create_spectrum("global")
    for _ in range(5):
        soul_db.create_self_reflection("global", 1, "", "substantive", 1, 50, "directness", "low", "x")
    before = soul_db.get_or_create_spectrum("global").directness
    fb.apply_self_reflection_spectrum_correction(_plugin(weight=0.0), evolution_rate=6)
    after = soul_db.get_or_create_spectrum("global").directness
    assert before == after


def test_correction_records_history(soul_db: Any) -> None:
    """修正时写演化历史，reason 含自评成分。"""
    fb = _import_soul_submodule("components.reflection_feedback")
    soul_db.get_or_create_spectrum("global")
    for _ in range(4):
        soul_db.create_self_reflection("global", 1, "", "substantive", 1, 50, "directness", "low", "x")
    fb.apply_self_reflection_spectrum_correction(_plugin(weight=0.5), evolution_rate=6)
    history = soul_db.get_evolution_history(limit=5)
    assert len(history) >= 1
    assert "自评修正" in (history[0].reason or "")


# ─── _build_injection_block 分场景合并（oracle 修订点 5）───────────


def test_injection_block_with_traits_summary_below_trait(soul_db: Any) -> None:
    """有 trait 时自评摘要在 trait 块下方，语态"低优先级自查"。"""
    inj = _import_soul_submodule("components.ideology_injector")
    block = inj._build_injection_block(
        "【光谱提示】",
        [],
        ["- t1: 观点A"],
        "近期自我评价：直率度偏低",
    )
    # trait 块在前，自评摘要在后
    assert block.index("观点A") < block.index("近期自我反思提示")
    assert "低优先级" in block


def test_injection_block_without_traits_summary_after_spectrum(soul_db: Any) -> None:
    """无 trait 时自评摘要在光谱提示后，语态"补充参考"。"""
    inj = _import_soul_submodule("components.ideology_injector")
    block = inj._build_injection_block(
        "【光谱提示】",
        [],
        [],
        "近期自我评价：直率度偏低",
    )
    assert "光谱提示" in block
    assert "最近自我评价洞察" in block
    assert "补充参考" in block
    # 无 trait 块
    assert "已固化的观点" not in block


def test_injection_block_no_summary_unchanged(soul_db: Any) -> None:
    """无自评摘要时注入块与原行为一致。"""
    inj = _import_soul_submodule("components.ideology_injector")
    block = inj._build_injection_block("【光谱提示】", [], ["- t1: A"], "")
    assert "自我反思" not in block
    assert "自我评价洞察" not in block
