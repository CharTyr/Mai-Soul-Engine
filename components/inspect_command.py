"""注入命中预览命令 — 模拟文本会命中哪些 trait（管理员，不实际注入）。"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from ..models.ideology_model import query_active_traits_for_injection
from ..utils.spectrum_utils import match_user
from ..utils.trait_tags import parse_tags_json
from ..worldview.constants import LAYER_LABEL_ZH, LIFECYCLE_LABEL_ZH
from .dashboard_renderer import DashboardRenderer, build_inspect_text
from .ideology_injector import _in_cooldown, _select_traits, _trait_quality_score

logger = logging.getLogger(__name__)


async def handle_inspect(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """模拟注入命中预览：输入文本 → 显示会命中哪些 trait 及优先级（不实际注入）。

    Args:
        plugin: MaiSoulEnginePlugin 实例。
        stream_id: 消息流 ID（命令所在视角）。
        **kwargs: 额外参数，包含 message/text 等。

    Returns:
        (success, message, should_send_response) 元组。
    """
    # ── 1. 解析输入文本 ──────────────────────────────────────────────
    message = kwargs.get("message") or {}
    raw_text = kwargs.get("text", "") or message.get("processed_plain_text", "")
    raw_text = str(raw_text) if raw_text else ""

    match = re.match(r"^/soul_inspect\s+(.+)\s*$", raw_text)
    if not match:
        msg = "用法: /soul_inspect <待测文本>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    query_text = match.group(1).strip()
    if not query_text:
        msg = "用法: /soul_inspect <待测文本>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # ── 2. 管理员鉴权 ────────────────────────────────────────────────
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))
    admin_user_id = plugin.config.admin.admin_user_id

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以执行此命令"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # ── 3. 取活跃 trait 池 ──────────────────────────────────────────
    traits = query_active_traits_for_injection(stream_id=stream_id, limit=80)
    total_active = len(traits)

    # 配置值：直接属性访问（无 or 兜底）
    max_traits = plugin.config.injection.max_traits
    cooldown_seconds = plugin.config.injection.trait_cooldown_seconds
    fallback_recent_impact = plugin.config.injection.fallback_recent_impact
    now_ts = time.time()

    # ── 4. 冷却筛选 ──────────────────────────────────────────────────
    cooled_traits: list[Any] = []
    cooldown_trait_refs: list[Any] = []
    for t in traits:
        if await _in_cooldown(stream_id, t.trait_id, now_ts, cooldown_seconds):
            cooldown_trait_refs.append(t)
        else:
            cooled_traits.append(t)

    # ── 5. 干跑选择 ──────────────────────────────────────────────────
    selected, selection_mode, picked = _select_traits(
        cooled_traits, query_text, stream_id,
        max_traits, fallback_recent_impact, now_ts,
    )
    selected_ids = {t.trait_id for t in selected}

    # ── 6. 组装 selected 列表 ────────────────────────────────────────
    text_norm = query_text.casefold()
    selected_list: list[dict[str, Any]] = []
    for t in selected:
        # Tag 命中计算（复用 _select_traits 内部逻辑的等价实现）
        tags = parse_tags_json(t.tags_json or "[]")
        hit_tags: list[str] = []
        for tag in tags:
            if tag and tag.casefold() in text_norm:
                hit_tags.append(tag)

        # 观点摘要截断 ~80 字
        thought = (t.thought or "").replace("\n", " ").replace("\r", " ").strip()
        while "  " in thought:
            thought = thought.replace("  ", " ")
        if len(thought) > 80:
            thought = thought[:80] + "..."

        selected_list.append({
            "trait_id": t.trait_id,
            "name": t.name,
            "layer_label": LAYER_LABEL_ZH.get(t.ideology_layer, t.ideology_layer),
            "lifecycle_label": LIFECYCLE_LABEL_ZH.get(t.lifecycle_state, t.lifecycle_state),
            "confidence": float(t.confidence) / 100.0,
            "quality_score": _trait_quality_score(t),
            "matched_tags": hit_tags,
            "thought": thought,
        })

    # ── 7. 组装 skipped 列表（最多 10 条）─────────────────────────────
    skipped_list: list[dict[str, str]] = []

    # 冷却中的 trait
    for t in cooldown_trait_refs[:10]:
        skipped_list.append({
            "trait_id": t.trait_id,
            "name": t.name,
            "reason": "冷却中",
        })

    # 未选中的非冷却 trait（补满到 10 条）
    remaining_slots = 10 - len(skipped_list)
    if remaining_slots > 0:
        for t in cooled_traits:
            if t.trait_id in selected_ids:
                continue
            if len(skipped_list) >= 10:
                break
            skipped_list.append({
                "trait_id": t.trait_id,
                "name": t.name,
                "reason": "未命中/超出上限",
            })

    # ── 8. 构建数据契约 → 渲染 → 发送 ──────────────────────────────
    data: dict[str, Any] = {
        "query_text": query_text,
        "stream_id": stream_id,
        "selection_mode": selection_mode,
        "max_traits": max_traits,
        "selected": selected_list,
        "skipped": skipped_list,
        "total_active": total_active,
    }

    card_enabled = plugin.config.render.card_enabled

    if card_enabled:
        renderer = DashboardRenderer(
            plugin.ctx,
            plugin.config.render.viewport_width,
            plugin.config.render.device_scale_factor,
            plugin.config.render.render_timeout_ms,
        )
        image_base64 = await renderer.render_inspect(data)

        if image_base64:
            try:
                await plugin.ctx.send.image(image_base64, stream_id)
            except (OSError, RuntimeError) as exc:
                logger.exception("发送注入命中预览卡片图片失败: %s", exc)
                fallback_text = f"卡片渲染失败，以下为文本状态：\n{build_inspect_text(data)}"
                await plugin.ctx.send.text(fallback_text, stream_id)
                return True, "注入命中预览(文本降级)", True
            return True, "已生成注入命中预览卡片", True

        # 渲染返回空串——渲染失败，降级文本
        fallback_text = f"卡片渲染失败，以下为文本状态：\n{build_inspect_text(data)}"
        await plugin.ctx.send.text(fallback_text, stream_id)
        return True, "注入命中预览(文本降级)", True

    # ── 文本降级路径（card_enabled=False） ───────────────────────────
    text = build_inspect_text(data)
    await plugin.ctx.send.text(text, stream_id)
    return True, "注入命中预览(文本)", True
