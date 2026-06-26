"""矛盾 trait 注入排除 + 弱化 trait 仍可见测试。

验证：
- 矛盾 trait（contradicted + enabled=False）不被注入查询返回
- 弱化 trait（weakened，enabled=1）仍被注入查询返回（降权但可见）
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import _import_soul_submodule


def _create_trait_for_injection(
    im: Any,
    trait_id: str,
    stream_id: str = "global",
    *,
    enabled: bool = True,
    lifecycle: str = "active",
) -> None:
    im.create_crystallized_trait(
        trait_id=trait_id,
        stream_id=stream_id,
        seed_id="",
        name=f"测试_{trait_id}",
        question="测试问题",
        thought="测试观点",
        tags_json="[]",
        confidence=80,
        evidence_json="[]",
        spectrum_impact_json="{}",
        ideology_layer="conduct",
        lifecycle_state=lifecycle,
    )


# ─── 矛盾 trait 被排除 ──────────────────────────────────────────────


def test_contradicted_trait_excluded_from_injection(soul_db: Any) -> None:
    """矛盾 trait（lifecycle=contradicted + enabled=False）不被 query_active_traits_for_injection 返回。"""
    _create_trait_for_injection(soul_db, "ok_trait", stream_id="global")
    _create_trait_for_injection(soul_db, "bad_trait", stream_id="global")

    # 把 bad_trait 标为矛盾并禁用
    soul_db.set_trait_lifecycle_state("bad_trait", "contradicted", enabled=False)

    results = soul_db.query_active_traits_for_injection(stream_id="global")
    ids = {t.trait_id for t in results}

    assert "ok_trait" in ids
    assert "bad_trait" not in ids, "矛盾 trait 不应出现在注入结果中"


# ─── 弱化 trait 仍可见 ──────────────────────────────────────────────


def test_weakened_trait_still_visible_in_injection(soul_db: Any) -> None:
    """弱化 trait（lifecycle=weakened，enabled=1）仍被注入查询返回。"""
    _create_trait_for_injection(soul_db, "weakened_one", stream_id="global")

    # 只改 lifecycle，不动 enabled
    soul_db.set_trait_lifecycle_state("weakened_one", "weakened")

    results = soul_db.query_active_traits_for_injection(stream_id="global")
    ids = {t.trait_id for t in results}

    assert "weakened_one" in ids, "weakened trait 仍应出现在注入结果中"

    # 确认 enabled 仍为 True
    trait = soul_db.get_crystallized_trait_by_id("weakened_one")
    assert trait is not None
    assert trait.enabled is True
    assert trait.lifecycle_state == "weakened"


# ─── 混合场景 ───────────────────────────────────────────────────────


def test_mixed_contradicted_and_weakened(soul_db: Any) -> None:
    """混合场景：多个 trait 中矛盾的被排除，弱化的仍在。"""
    for tid in ("t_active", "t_weak", "t_bad"):
        _create_trait_for_injection(soul_db, tid, stream_id="global")

    soul_db.set_trait_lifecycle_state("t_weak", "weakened")        # 只改 lifecycle
    soul_db.set_trait_lifecycle_state("t_bad", "contradicted", enabled=False)  # 禁用

    results = soul_db.query_active_traits_for_injection(stream_id="global")
    ids = {t.trait_id for t in results}

    assert "t_active" in ids
    assert "t_weak" in ids
    assert "t_bad" not in ids
