"""批量图谱边查询测试。

验证：
- `list_thought_edges_for_traits([id1,id2,id3])` 返回按 trait_id 分组的边
- 每条边归入 from 端和 to 端分组
- 空列表返回空 dict
- 与单数版 `list_thought_edges_for_trait` 对比，同一 trait 的边集一致
"""

from __future__ import annotations

from typing import Any

import pytest


def test_batch_edges_basic_grouping(soul_db: Any) -> None:
    """批量查询：边按 from 和 to 端正确归入各 trait 分组。"""
    # 建 3 个 trait
    for tid in ("ta", "tb", "tc"):
        soul_db.create_crystallized_trait(
            trait_id=tid, stream_id="global", seed_id="",
            name=tid, question="问", thought="答",
            tags_json="[]", confidence=80,
            evidence_json="[]", spectrum_impact_json="{}",
        )

    # 建边
    soul_db.create_thought_edge(from_trait_id="ta", to_trait_id="tb", relation_type="supports")
    soul_db.create_thought_edge(from_trait_id="tb", to_trait_id="tc", relation_type="conflicts")
    soul_db.create_thought_edge(from_trait_id="ta", to_trait_id="tc", relation_type="extends")

    result = soul_db.list_thought_edges_for_traits(["ta", "tb", "tc"])

    # 返回值类型
    assert isinstance(result, dict)
    assert set(result.keys()) == {"ta", "tb", "tc"}

    # ta: 2 条出边（→tb, →tc）
    assert len(result["ta"]) == 2
    ta_rel = {e.relation_type for e in result["ta"]}
    assert ta_rel == {"supports", "extends"}

    # tb: 入边(←ta) + 出边(→tc)
    assert len(result["tb"]) == 2
    tb_as_from = {e.relation_type for e in result["tb"] if e.from_trait_id == "tb"}
    tb_as_to = {e.relation_type for e in result["tb"] if e.to_trait_id == "tb"}
    assert tb_as_from == {"conflicts"}
    assert tb_as_to == {"supports"}

    # tc: 2 条入边（←tb, ←ta）
    assert len(result["tc"]) == 2
    tc_rels = {e.from_trait_id for e in result["tc"]}
    assert tc_rels == {"ta", "tb"}


def test_batch_edges_empty_list(soul_db: Any) -> None:
    """空列表传参返回空 dict。"""
    assert soul_db.list_thought_edges_for_traits([]) == {}


def test_batch_edges_consistency_with_single(soul_db: Any) -> None:
    """批量查询与单数版 list_thought_edges_for_trait 对同一 trait 结果一致。"""
    # 建 2 个 trait
    for tid in ("tx", "ty"):
        soul_db.create_crystallized_trait(
            trait_id=tid, stream_id="global", seed_id="",
            name=tid, question="问", thought="答",
            tags_json="[]", confidence=80,
            evidence_json="[]", spectrum_impact_json="{}",
        )

    soul_db.create_thought_edge(from_trait_id="tx", to_trait_id="ty", relation_type="supports")

    # 批量
    batch = soul_db.list_thought_edges_for_traits(["tx", "ty"])
    # 单数
    single_tx = soul_db.list_thought_edges_for_trait("tx")

    # 比较 tx 的边集（批量用 trait_id 分组，单数版返回全部相关边）
    batch_tx_ids = {e.id for e in batch["tx"]}
    single_tx_ids = {e.id for e in single_tx}
    assert batch_tx_ids == single_tx_ids

    # 验证关系类型一致
    for e in batch["tx"]:
        if e.from_trait_id == "tx":
            assert e.relation_type == "supports"
