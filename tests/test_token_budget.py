"""提示词预算：保守估算 + 稳定裁剪（方案 §4.1「token 预算显式配置」）。"""

from __future__ import annotations

from typing import Any

from .conftest import _import_soul_submodule


def _tb() -> Any:
    return _import_soul_submodule("utils.token_budget")


# ─── 估算：保守（宁可高估） ─────────────────────────────────────────


def test_empty_text_is_zero_tokens() -> None:
    assert _tb().estimate_tokens("") == 0
    assert _tb().estimate_tokens(None) == 0  # type: ignore[arg-type]


def test_cjk_counts_one_token_per_char() -> None:
    """中文按 1 token/字 —— 真实多在 1~1.5 字/token，高估即保守。"""
    tb = _tb()
    assert tb.estimate_tokens("真诚") == 2
    assert tb.estimate_tokens("我觉得应该直接说出来") == 10


def test_ascii_is_cheaper_than_cjk() -> None:
    """同样字符数，ASCII 必须估得比 CJK 少（否则英文场景会过度裁剪）。"""
    tb = _tb()
    ascii_tokens = tb.estimate_tokens("abcdefghij")
    cjk_tokens = tb.estimate_tokens("一二三四五六七八九十")
    assert ascii_tokens < cjk_tokens


def test_estimate_is_annotated_as_estimate() -> None:
    """估算必须能标明出处（不得被当成精确计数）。"""
    assert "估算" in _tb().ESTIMATE_LABEL


# ─── 裁剪：顺序即优先级、稳定、不静默 ──────────────────────────────


def test_fit_keeps_prefix_order() -> None:
    """顺序即优先级：保留的是前缀（高优先级在前）。"""
    kept, dropped = _tb().fit_to_budget(["一二三", "四五六", "七八九"], budget_tokens=6)
    assert kept == ["一二三", "四五六"]
    assert dropped == 1


def test_fit_is_deterministic() -> None:
    """同样的输入永远给出同样的结果（可复现，不依赖 dict 顺序/时间）。"""
    tb = _tb()
    lines = [f"观点{i}" for i in range(20)]
    first = tb.fit_to_budget(lines, 12)
    second = tb.fit_to_budget(lines, 12)
    assert first == second


def test_fit_reports_dropped_count() -> None:
    """丢弃数量必须报出来（不许静默截断）。"""
    # 每条 5 token；预算 10 → 恰好留 2 条，其余 8 条被丢弃且必须报出来
    kept, dropped = _tb().fit_to_budget(["一二三四五"] * 10, budget_tokens=10)
    assert len(kept) == 2
    assert dropped == 8


def test_fit_empty_input_is_empty_result() -> None:
    kept, dropped = _tb().fit_to_budget([], budget_tokens=100)
    assert kept == [] and dropped == 0


def test_fit_keeps_first_line_even_over_budget() -> None:
    """第一条超预算也保留：把「有内容」裁成「没内容」更难排查。"""
    kept, dropped = _tb().fit_to_budget(["很长的第一条内容" * 5, "第二条"], budget_tokens=3)
    assert len(kept) == 1
    assert dropped == 1


def test_fit_never_exceeds_budget_when_it_can_avoid_it() -> None:
    """只要累加会超预算就停 —— 不许「先塞进去再截字符串」。"""
    tb = _tb()
    lines = ["一二三", "四五六", "七八九"]
    kept, _ = tb.fit_to_budget(lines, budget_tokens=6)
    assert tb.estimate_tokens("".join(kept)) <= 6
