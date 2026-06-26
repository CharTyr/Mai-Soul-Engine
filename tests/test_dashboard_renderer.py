"""dashboard_renderer 渲染降级与文本构建测试。

可离线测的部分：
1. `build_dashboard_text` 满状态样例 → 输出含关键标签与数值
2. `build_dashboard_text` 未初始化空数据 → 提示未初始化、不报错
3. `DashboardRenderer.render`  ctx=None → 降级返回空串
4. `_build_html` 不报错 → 返回含 id="soul-dashboard" 的 HTML
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import _import_soul_submodule


# ── 满状态样例数据（参考 collect_dashboard_data 返回结构） ──────────

SAMPLE_FULL: dict[str, Any] = {
    "initialized": True,
    "generated_at": "2025-06-20 10:30:00",
    "stream_id": "群A",
    "spectrum": {
        "sincerity": 72,
        "engagement": 45,
        "closeness": 62,
        "directness": 14,
        "updated_at": "2025-06-20 10:00:00",
        "last_evolution": "2025-06-20 09:00:00",
    },
    "p1_enabled": True,
    "trait_counts_by_layer": {"values": 2, "worldview": 4, "conduct": 6},
    "lifecycle_distribution": {
        "active": 5,
        "strengthened": 2,
        "expired": 1,
        "contradicted": 1,
        "weakened": 2,
        "revised": 1,
    },
    "trait_total": 12,
    "mood": {
        "enabled": True,
        "valence": 30,
        "arousal": -10,
        "energy": 15,
        "updated_at": "2025-06-20 10:15:00",
    },
    "group_slice": {
        "sincerity_offset": 5,
        "engagement_offset": -3,
        "closeness_offset": 2,
        "directness_offset": -1,
        "sample_count": 42,
    },
    "thought_cabinet": {
        "enabled": True,
        "pending_seeds": 3,
    },
    "recent_evolutions": [
        {
            "group_id": "群A",
            "timestamp": "2025-06-20 10:00:00",
            "deltas": {"sincerity": 2, "engagement": -1, "closeness": 0, "directness": 1},
            "reason": "群聊氛围积极向上",
        },
        {
            "group_id": "群B",
            "timestamp": "2025-06-19 22:00:00",
            "deltas": {"sincerity": 0, "engagement": 3, "closeness": -2, "directness": 0},
            "reason": "深度讨论话题",
        },
    ],
    "graph_edge_total": 14,
    "feature_flags": {
        "p1_enabled": True,
        "mood_enabled": True,
        "graph_inject": True,
        "thought_cabinet": True,
        "notion": False,
        "api": True,
        "card_render": True,
    },
}

SAMPLE_EMPTY: dict[str, Any] = {
    "initialized": False,
    "generated_at": "2025-06-26 00:00:00",
    "stream_id": "",
    "spectrum": {
        "sincerity": 50,
        "engagement": 50,
        "closeness": 50,
        "directness": 50,
        "updated_at": None,
        "last_evolution": None,
    },
    "p1_enabled": True,
    "trait_counts_by_layer": {"values": 0, "worldview": 0, "conduct": 0},
    "lifecycle_distribution": {
        "active": 0,
        "strengthened": 0,
        "expired": 0,
        "contradicted": 0,
        "weakened": 0,
        "revised": 0,
    },
    "trait_total": 0,
    "mood": {
        "enabled": False,
        "valence": 0,
        "arousal": 0,
        "energy": 0,
        "updated_at": None,
    },
    "group_slice": None,
    "thought_cabinet": {
        "enabled": False,
        "pending_seeds": 0,
    },
    "recent_evolutions": [],
    "graph_edge_total": 0,
    "feature_flags": {
        "p1_enabled": True,
        "mood_enabled": False,
        "graph_inject": True,
        "thought_cabinet": False,
        "notion": False,
        "api": True,
        "card_render": True,
    },
}


# ─── 1. build_dashboard_text 满状态 ──────────────────────────────────


def test_build_dashboard_text_full_state() -> None:
    """build_dashboard_text 满状态：输出应含关键标签与数值。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    text = mod.build_dashboard_text(SAMPLE_FULL)

    # 关键节标题
    assert "Soul 引擎状态" in text
    assert "社交光谱" in text
    assert "三观分层" in text
    assert "生命周期" in text
    assert "思维阁" in text
    assert "功能开关" in text
    assert "本群切片" in text
    assert "思想图谱" in text
    assert "最近演化" in text

    # 轴名称
    assert "真诚" in text
    assert "投入" in text
    assert "亲近" in text
    assert "直率" in text

    # 数值出现（62=closeness, 14=directness, 12=trait_total）
    assert "62" in text
    assert "14" in text
    assert "12" in text

    # 思维阁种子数
    assert "3" in text

        # 功能开关（_FEATURE_LABELS 中为 "P1 三观"）
    assert "P1 三观" in text
    assert "API" in text
    assert "Notion" in text


# ─── 2. build_dashboard_text 未初始化空数据 ──────────────────────────


def test_build_dashboard_text_uninitialized() -> None:
    """build_dashboard_text 空库数据：应提示未初始化，不报错不塌陷。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    text = mod.build_dashboard_text(SAMPLE_EMPTY)

    # 包含未初始化提示
    assert "未初始化" in text
    assert "/soul_setup" in text or "soul_setup" in text

    # 不应有错误标记
    assert "Traceback" not in text
    assert "Error" not in text
    assert "None" not in text

    # 关键节依然出现
    assert "Soul 引擎状态" in text
    assert "社交光谱" in text
    assert "三观分层" in text
    assert "生命周期" in text
    assert "功能开关" in text

    # 全零值
    assert "0" in text


# ─── 3. DashboardRenderer.render ctx=None 降级 ──────────────────────


@pytest.mark.asyncio
async def test_render_ctx_none_returns_empty() -> None:
    """DashboardRenderer(ctx=None).render(full_data) 应返回空串（降级）。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    renderer = mod.DashboardRenderer(None, 1100, 2.0, 60000)
    result = await renderer.render(SAMPLE_FULL)
    assert result == ""


# ─── 4. _build_html 不报错 ──────────────────────────────────────────


def test_build_html_returns_dashboard_container() -> None:
    """DashboardRenderer._build_html(data) 返回含 id="soul-dashboard" 的非空 HTML。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    renderer = mod.DashboardRenderer(None, 1100, 2.0, 60000)
    html = renderer._build_html(SAMPLE_FULL)
    assert isinstance(html, str)
    assert len(html) > 500
    assert 'id="soul-dashboard"' in html
