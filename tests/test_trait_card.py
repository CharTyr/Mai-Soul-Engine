"""trait 详情卡片：纯文本降级与 HTML 构建测试。

可离线测的部分：
1. `build_trait_text` 满状态(含 tags/光谱影响/证据/边)
2. `build_trait_text` 全部 5 种边类型(回归保护:此前文本版只展示 derived_from/supports)
3. `build_trait_text` 稀疏数据(停用、空 tags/证据/边)
4. `_build_trait_html` 返回含 id="soul-trait" 的非空 HTML
5. `DashboardRenderer.render_trait` ctx=None → 降级返回空串
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import _import_soul_submodule


# ── 样例数据 ──────────────────────────────────────────────────────────

TRAIT_FULL: dict[str, Any] = {
    "trait_id": "trait-abc-123",
    "name": "保持真诚交流",
    "enabled": True,
    "ideology_layer": "values",
    "layer_label": "价值观",
    "lifecycle_state": "active",
    "lifecycle_label": "有效",
    "confidence": 0.85,
    "stream_id": "群A",
    "created_at": "2025-06-20 10:30:00",
    "tags": ["真诚", "沟通", "开放"],
    "question": "如何在群聊中保持真诚？",
    "thought": "真诚能建立信任，促进深度交流",
    "spectrum_impact": {"sincerity": 8, "engagement": 3, "closeness": 0, "directness": 5},
    "evidence": [
        "用户 A 在群聊中主动承认错误",
        "用户 B 表示感谢大家的帮助",
    ],
    "edges": [
        {"relation_type": "derived_from", "label": "源自种子", "target": "seed-001"},
        {"relation_type": "supports", "label": "支撑", "target": "trait-xyz-789"},
    ],
}

TRAIT_SPARSE: dict[str, Any] = {
    "trait_id": "trait-sparse-001",
    "name": "旧观点",
    "enabled": False,
    "ideology_layer": "conduct",
    "layer_label": "处事观",
    "lifecycle_state": "expired",
    "lifecycle_label": "已过期",
    "confidence": 0.3,
    "stream_id": "global",
    "created_at": None,
    "tags": [],
    "question": "",
    "thought": "",
    "spectrum_impact": {},
    "evidence": [],
    "edges": [],
}

TRAIT_ALL_EDGE_TYPES: dict[str, Any] = {
    "trait_id": "trait-edge-all",
    "name": "多边测试",
    "enabled": True,
    "ideology_layer": "worldview",
    "layer_label": "世界观",
    "lifecycle_state": "strengthened",
    "lifecycle_label": "已强化",
    "confidence": 0.9,
    "stream_id": "群B",
    "created_at": "2025-06-21 12:00:00",
    "tags": ["测试"],
    "question": "五种边类型？",
    "thought": "验证全部边类型展示",
    "spectrum_impact": {"sincerity": 2},
    "evidence": ["一条证据"],
    "edges": [
        {"relation_type": "derived_from", "label": "源自种子", "target": "seed-999"},
        {"relation_type": "supports", "label": "支撑", "target": "trait-aaa"},
        {"relation_type": "contradicted_by", "label": "矛盾于", "target": "trait-bbb"},
        {"relation_type": "weakened_by", "label": "弱化于", "target": "trait-ccc"},
        {"relation_type": "revised_by", "label": "修正自", "target": "trait-ddd"},
    ],
}


# ─── 1. build_trait_text 满状态 ──────────────────────────────────────


def test_build_trait_text_full() -> None:
    """build_trait_text 满状态：输出应含 name、layer_label、lifecycle_label、
    置信度、问题、观点、证据内容、边的 label+target。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    text = mod.build_trait_text(TRAIT_FULL)

    # 标题与名称
    assert "◇ Trait" in text
    assert "保持真诚交流" in text

    # 分层与生命周期
    assert "价值观" in text
    assert "有效" in text

    # 置信度
    assert "85%" in text

    # 标签
    assert "真诚" in text
    assert "沟通" in text
    assert "开放" in text

    # 问题与观点
    assert "如何在群聊中保持真诚" in text
    assert "真诚能建立信任" in text

    # 证据内容
    assert "用户 A 在群聊中主动承认错误" in text
    assert "用户 B 表示感谢大家的帮助" in text

    # 边：label → target
    assert "源自种子" in text
    assert "seed-001" in text
    assert "支撑" in text
    assert "trait-xyz-789" in text


# ─── 2. 全部 5 种边类型回归保护 ──────────────────────────────────────


def test_trait_text_all_edge_types() -> None:
    """build_trait_text 包含全部 5 种边类型 label（回归保护：此前文本版
    只展示 derived_from/supports，漏了 contradicted_by/weakened_by/revised_by）。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    text = mod.build_trait_text(TRAIT_ALL_EDGE_TYPES)

    # 全部 5 种边 label
    assert "源自种子" in text
    assert "支撑" in text
    assert "矛盾于" in text
    assert "弱化于" in text
    assert "修正自" in text

    # 对应 target
    assert "seed-999" in text
    assert "trait-aaa" in text
    assert "trait-bbb" in text
    assert "trait-ccc" in text
    assert "trait-ddd" in text

    # 边在【思想关联】下
    assert "思想关联" in text


# ─── 3. 稀疏数据 ──────────────────────────────────────────────────────


def test_trait_text_sparse() -> None:
    """build_trait_text 稀疏数据（停用、空 tags/证据/边、空 spectrum_impact）：
    不报错、不塌陷、含"停用"或类似提示。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    text = mod.build_trait_text(TRAIT_SPARSE)

    # 停用标记
    assert "停用" in text

    # 空标签
    assert "暂无" in text

    # 空问题/观点
    assert "暂无" in text

    # 空光谱影响
    assert "无" in text

    # 空证据
    assert "暂无" in text

    # 空边
    assert "暂无" in text

    # 无错误标记
    assert "Traceback" not in text
    assert "Error" not in text


# ─── 4. _build_trait_html 容器 ──────────────────────────────────────


def test_build_trait_html_container() -> None:
    """DashboardRenderer._build_trait_html(data) 返回含 id="soul-trait" 的非空 HTML。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    renderer = mod.DashboardRenderer(None, 1100, 2.0, 60000)
    html = renderer._build_trait_html(TRAIT_FULL)
    assert isinstance(html, str)
    assert len(html) > 500
    assert 'id="soul-trait"' in html


# ─── 5. render_trait ctx=None 降级 ──────────────────────────────────


@pytest.mark.asyncio
async def test_render_trait_ctx_none() -> None:
    """DashboardRenderer(ctx=None).render_trait(full_data) 应返回空串（降级）。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    renderer = mod.DashboardRenderer(None, 1100, 2.0, 60000)
    result = await renderer.render_trait(TRAIT_FULL)
    assert result == ""
