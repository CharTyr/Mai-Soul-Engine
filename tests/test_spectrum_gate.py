"""统一光谱写入闸门测试（v2.3.0 三回路收口）。

验证 apply_spectrum_deltas 的 clamp/resistance/smooth/history+source 标记，
以及三条回路经此闸门后的行为一致性。
"""

from __future__ import annotations

from typing import Any

from .conftest import _import_soul_submodule


def _spec(soul_db: Any):
    return soul_db.get_or_create_spectrum("global")


# ─── 基础：写入 + history + source 标记 ─────────────────────────────


def test_gate_writes_spectrum_and_history(soul_db: Any) -> None:
    soul_db.get_or_create_spectrum("global")
    applied = soul_db.apply_spectrum_deltas(
        "evolution",
        {"sincerity": 5, "engagement": -3, "closeness": 0, "directness": 2},
        reason="测试",
        group_id="g1",
    )
    assert applied["sincerity"] == 5
    assert applied["engagement"] == -3
    spec = _spec(soul_db)
    assert spec.sincerity == 55
    assert spec.engagement == 47
    # history 写入且带 source 标记
    hist = soul_db.get_evolution_history(limit=1)
    assert len(hist) == 1
    assert "[evolution]" in (hist[0].reason or "")
    assert "测试" in (hist[0].reason or "")


def test_gate_write_history_false_skips(soul_db: Any) -> None:
    soul_db.get_or_create_spectrum("global")
    soul_db.apply_spectrum_deltas(
        "internalize",
        {"sincerity": 5, "engagement": 0, "closeness": 0, "directness": 0},
        write_history=False,
    )
    assert _spec(soul_db).sincerity == 55
    assert len(soul_db.get_evolution_history(limit=5)) == 0


# ─── clamp（max_per_axis）──────────────────────────────────────────


def test_gate_clamps_to_max_per_axis(soul_db: Any) -> None:
    soul_db.get_or_create_spectrum("global")
    applied = soul_db.apply_spectrum_deltas(
        "internalize",
        {"sincerity": 50, "engagement": 0, "closeness": 0, "directness": 0},
        max_per_axis=10,
    )
    # 50 被 clamp 到 10
    assert applied["sincerity"] == 10
    assert _spec(soul_db).sincerity == 60


# ─── resistance（反向变动阻力）─────────────────────────────────────


def test_gate_resistance_reduces_reversal(soul_db: Any) -> None:
    """先正向 +5 设 last_dir=1，再反向 -5，阻力 0.5 应把反向 delta 砍半到 -2。"""
    soul_db.get_or_create_spectrum("global")
    soul_db.apply_spectrum_deltas(
        "evolution",
        {"sincerity": 5, "engagement": 0, "closeness": 0, "directness": 0},
        resistance=0.5,
    )
    assert _spec(soul_db).sincerity == 55  # 50 + 5（首次无阻力）
    assert _spec(soul_db).last_sincerity_dir == 1
    soul_db.apply_spectrum_deltas(
        "evolution",
        {"sincerity": -5, "engagement": 0, "closeness": 0, "directness": 0},
        resistance=0.5,
    )
    # 反向 -5 经阻力 0.5 → int(-5*0.5)=int(-2.5)=-2；55-2=53
    assert _spec(soul_db).sincerity == 53


# ─── smooth（EMA）──────────────────────────────────────────────────


def test_gate_smooth_reduces_delta(soul_db: Any) -> None:
    """EMA alpha=0.3：50→target 55，smoothed=50*0.7+55*0.3=51.5→round 52，delta=2。"""
    soul_db.get_or_create_spectrum("global")
    applied = soul_db.apply_spectrum_deltas(
        "evolution",
        {"sincerity": 5, "engagement": 0, "closeness": 0, "directness": 0},
        smooth_alpha=0.3,
    )
    assert applied["sincerity"] == 2
    assert _spec(soul_db).sincerity == 52


# ─── 三回路 source 标记区分 ────────────────────────────────────────


def test_three_sources_all_log_history(soul_db: Any) -> None:
    soul_db.get_or_create_spectrum("global")
    soul_db.apply_spectrum_deltas(
        "evolution", {"sincerity": 1, "engagement": 0, "closeness": 0, "directness": 0},
        group_id="g1", reason="群聊",
    )
    soul_db.apply_spectrum_deltas(
        "internalize", {"sincerity": 0, "engagement": 2, "closeness": 0, "directness": 0},
        max_per_axis=10, reason="内化",
    )
    soul_db.apply_spectrum_deltas(
        "self_reflection", {"sincerity": 0, "engagement": 0, "closeness": -1, "directness": 0},
        group_id="global", reason="自评",
    )
    hist = soul_db.get_evolution_history(limit=5)
    assert len(hist) == 3
    reasons = [h.reason or "" for h in hist]
    assert any("[evolution]" in r for r in reasons)
    assert any("[internalize]" in r for r in reasons)
    assert any("[self_reflection]" in r for r in reasons)


# ─── 返回值供调用方做副作用 ──────────────────────────────────────


def test_returns_applied_deltas_for_caller(soul_db: Any) -> None:
    """返回经 resistance/smooth 后的实际 delta，供 evolution_task 做 slice/mood。"""
    soul_db.get_or_create_spectrum("global")
    applied = soul_db.apply_spectrum_deltas(
        "evolution",
        {"sincerity": 10, "engagement": -10, "closeness": 5, "directness": -5},
        max_per_axis=6,
    )
    # 全部被 clamp 到 ±6
    assert applied == {"sincerity": 6, "engagement": -6, "closeness": 5, "directness": -5}


# ─── 边界（ora-2 复审建议）─────────────────────────────────────────


def test_gate_max_per_axis_zero_blocks_all(soul_db: Any) -> None:
    """max_per_axis=0 应把所有 delta 压成 0（光谱不变）。"""
    soul_db.get_or_create_spectrum("global")
    before = _spec(soul_db).sincerity
    applied = soul_db.apply_spectrum_deltas(
        "evolution",
        {"sincerity": 5, "engagement": 3, "closeness": -2, "directness": 8},
        max_per_axis=0,
    )
    assert applied == {"sincerity": 0, "engagement": 0, "closeness": 0, "directness": 0}
    assert _spec(soul_db).sincerity == before  # 不变


def test_gate_resistance_zero_delta_zero_keeps_last_dir(soul_db: Any) -> None:
    """resistance=0 + delta=0 时保留 last_dir（与 apply_resistance 对齐，ora-2 复审#1）。"""
    soul_db.get_or_create_spectrum("global")
    # 先正向 +5 设 last_sincerity_dir=1
    soul_db.apply_spectrum_deltas(
        "evolution",
        {"sincerity": 5, "engagement": 0, "closeness": 0, "directness": 0},
    )
    assert _spec(soul_db).last_sincerity_dir == 1
    # 再传 sincerity delta=0（resistance=0 默认），不应清零 last_dir
    soul_db.apply_spectrum_deltas(
        "evolution",
        {"sincerity": 0, "engagement": 0, "closeness": 0, "directness": 0},
    )
    assert _spec(soul_db).last_sincerity_dir == 1  # 保留，未清零


def test_gate_resistance_and_smooth_combined(soul_db: Any) -> None:
    """resistance + smooth 同轮启用（回路1 真实组合）：clamp→resistance→smooth 链。"""
    soul_db.get_or_create_spectrum("global")
    # 首轮 last_dir=0，resistance 不生效（无反向）；smooth 把 +3 压成 +1
    # smooth_delta(50,3,0.3): target=53, ema=50*0.7+53*0.3=50.9→51, delta=1
    applied = soul_db.apply_spectrum_deltas(
        "evolution",
        {"sincerity": 3, "engagement": 0, "closeness": 0, "directness": 0},
        resistance=0.5,
        smooth_alpha=0.3,
    )
    assert applied["sincerity"] == 1
    assert _spec(soul_db).sincerity == 51
    assert _spec(soul_db).last_sincerity_dir == 1

