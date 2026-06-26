"""inspect 命中预览卡片：纯文本降级与 HTML 构建测试。

可离线测的部分：
1. `build_inspect_text` selected 非空 + skipped 非空
2. `build_inspect_text` selected=[] + skipped 若干
3. `build_inspect_text` total_active=0、selected=[]、skipped=[]
4. `_build_inspect_html` 返回含 id="soul-inspect" 的非空 HTML
5. `DashboardRenderer.render_inspect` ctx=None → 降级返回空串
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import _import_soul_submodule


# ── 样例数据 ──────────────────────────────────────────────────────────

INSPECT_WITH_HITS: dict[str, Any] = {
    "query_text": "今天天气真好，大家觉得呢？",
    "stream_id": "群A",
    "selection_mode": "tag_hit",
    "max_traits": 3,
    "total_active": 10,
    "selected": [
        {
            "trait_id": "trait-hit-001",
            "name": "积极社交",
            "layer_label": "价值观",
            "lifecycle_label": "有效",
            "confidence": 0.88,
            "quality_score": 0.75,
            "matched_tags": ["积极", "社交"],
            "thought": "鼓励积极群聊氛围",
        },
        {
            "trait_id": "trait-hit-002",
            "name": "阳光心态",
            "layer_label": "世界观",
            "lifecycle_label": "已强化",
            "confidence": 0.72,
            "quality_score": 0.91,
            "matched_tags": ["心态"],
            "thought": "保持阳光心态有助于群聊",
        },
    ],
    "skipped": [
        {
            "trait_id": "trait-skip-001",
            "name": "旧观点-过期",
            "reason": "已过期",
        },
    ],
}

INSPECT_NO_HITS: dict[str, Any] = {
    "query_text": "有人一起打游戏吗？",
    "stream_id": "global",
    "selection_mode": "tagless_fill",
    "max_traits": 5,
    "total_active": 8,
    "selected": [],
    "skipped": [
        {
            "trait_id": "trait-skip-002",
            "name": "游戏观点",
            "reason": "置信度低于阈值",
        },
        {
            "trait_id": "trait-skip-003",
            "name": "竞技精神",
            "reason": "未命中任何标签",
        },
    ],
}

INSPECT_EMPTY_ACTIVE: dict[str, Any] = {
    "query_text": "",
    "stream_id": "",
    "selection_mode": "spectrum_only",
    "max_traits": 2,
    "total_active": 0,
    "selected": [],
    "skipped": [],
}


# ─── 1. build_inspect_text 有命中 ────────────────────────────────────


def test_build_inspect_text_with_hits() -> None:
    """build_inspect_text selected 非空 + skipped 一条：断言含 query_text、
    selection_mode、命中 trait 的 name、matched_tags、跳过原因。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    text = mod.build_inspect_text(INSPECT_WITH_HITS)

    # 标题
    assert "注入命中预览" in text

    # 视角与模式
    assert "群A" in text
    assert "标签命中" in text

    # 上限与活跃数
    assert "3" in text
    assert "10" in text

    # 待测文本
    assert "今天天气真好" in text

    # 命中的 trait
    assert "#1" in text
    assert "积极社交" in text
    assert "#2" in text
    assert "阳光心态" in text

    # matched_tags
    assert "积极" in text
    assert "社交" in text
    assert "心态" in text

    # 观点
    assert "鼓励积极群聊氛围" in text
    assert "保持阳光心态" in text

    # 跳过原因
    assert "旧观点-过期" in text
    assert "已过期" in text


# ─── 2. build_inspect_text 无命中 ────────────────────────────────────


def test_build_inspect_text_no_hits() -> None:
    """build_inspect_text selected=[]、skipped 若干：断言不报错且含"无命中"提示。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    text = mod.build_inspect_text(INSPECT_NO_HITS)

    # 无命中提示
    assert "无命中" in text

    # 跳过原因仍展示
    assert "游戏观点" in text
    assert "置信度低于阈值" in text
    assert "竞技精神" in text
    assert "未命中任何标签" in text

    # 模式标签
    assert "无标签补位" in text

    # 无错误标记
    assert "Traceback" not in text
    assert "Error" not in text

    # 标题仍出现
    assert "注入命中预览" in text


# ─── 3. build_inspect_text 空活跃（可选边界用例） ────────────────────


def test_inspect_text_empty_active() -> None:
    """build_inspect_text total_active=0、selected=[]、skipped=[]：不塌陷。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    text = mod.build_inspect_text(INSPECT_EMPTY_ACTIVE)

    # 活跃数为 0
    assert "0" in text

    # 无命中提示
    assert "无命中" in text or "无" in text

    # 跳过：无
    assert "无" in text or "无跳过项" in text or "无" in text

    # 无错误标记
    assert "Traceback" not in text
    assert "Error" not in text
    assert "None" not in text


# ─── 4. _build_inspect_html 容器 ─────────────────────────────────────


def test_build_inspect_html_container() -> None:
    """DashboardRenderer._build_inspect_html(data) 返回含 id="soul-inspect" 的非空 HTML。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    renderer = mod.DashboardRenderer(None, 1100, 2.0, 60000)
    html = renderer._build_inspect_html(INSPECT_WITH_HITS)
    assert isinstance(html, str)
    assert len(html) > 500
    assert 'id="soul-inspect"' in html


# ─── 5. render_inspect ctx=None 降级 ──────────────────────────────────


@pytest.mark.asyncio
async def test_render_inspect_ctx_none() -> None:
    """DashboardRenderer(ctx=None).render_inspect(full_data) 应返回空串（降级）。"""
    mod = _import_soul_submodule("components.dashboard_renderer")
    renderer = mod.DashboardRenderer(None, 1100, 2.0, 60000)
    result = await renderer.render_inspect(INSPECT_WITH_HITS)
    assert result == ""
