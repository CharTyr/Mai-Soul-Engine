"""思维种子 / trait 管理命令模块 — maibot_sdk 2.x 版本。"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ===== 种子列表 =====


async def handle_seeds_list(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """查看待审核的思维种子（管理员）。"""
    from ..thought.seed_manager import ThoughtSeedManager
    from ..utils.spectrum_utils import match_user

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以查看思维种子"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    manager = ThoughtSeedManager.from_plugin_config(plugin)
    seeds = await manager.get_pending_seeds()

    msg = manager.format_seeds_list(seeds)
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


# ===== 种子详情 =====


async def handle_seed_detail(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """查看单个思维种子完整详情（管理员）。"""
    from ..thought.seed_manager import ThoughtSeedManager
    from ..utils.spectrum_utils import match_user

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以查看思维种子"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    text = kwargs.get("text", "") or message.get("processed_plain_text", "")
    match = re.match(r"^/soul_seed\s+(\w+)\s*$", str(text))
    if not match:
        msg = "用法: /soul_seed <种子ID>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    seed_id = match.group(1)
    manager = ThoughtSeedManager.from_plugin_config(plugin)
    seed = await manager.get_seed_by_id(seed_id)

    if not seed:
        msg = f"未找到种子 {seed_id}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    lines = [f"🧠 种子详情 {seed_id}", ""]
    lines.append(f"状态: {seed.get('status', '')}")
    if seed.get("stream_id"):
        lines.append(f"来源: {seed['stream_id']}")
    lines.append(f"类型: {seed.get('type', '')}")
    lines.append(f"强度: {seed.get('intensity', 0):.2f}  置信度: {seed.get('confidence', 0):.2f}")
    lines.append(f"创建于: {seed.get('created_at', '')}")
    lines.append("")
    lines.append(f"事件: {seed.get('event', '')}")
    reasoning = seed.get("reasoning", "")
    if reasoning:
        lines.append(f"检测原因: {reasoning}")
    impact = seed.get("potential_impact", {}) or {}
    impact_str = ", ".join([f"{k}:{v:+d}" for k, v in impact.items() if v != 0])
    if impact_str:
        lines.append(f"预期影响: {impact_str}")

    evidence = seed.get("evidence", []) or []
    if evidence:
        lines.append("")
        lines.append("证据片段:")
        lines.extend([f"- {x}" for x in evidence])

    context = seed.get("context", []) or []
    if context:
        lines.append("")
        lines.append("原始对话上下文:")
        lines.extend([f"│ {x}" for x in context])

    if seed.get("status") == "pending":
        lines.append("")
        lines.append(f"/soul_approve {seed_id} - 批准  |  /soul_reject {seed_id} - 拒绝")

    msg = "\n".join(lines)
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


# ===== 种子批准 =====


async def handle_seed_approve(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """批准思维种子内化（管理员）。"""
    from ..thought.seed_manager import ThoughtSeedManager
    from ..thought.internalization_engine import InternalizationEngine
    from ..utils.spectrum_utils import match_user

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以审核思维种子"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # 从文本中提取种子 ID
    text = kwargs.get("text", "") or message.get("processed_plain_text", "")
    match = re.match(r"^/soul_approve\s+(\w+)\s*$", str(text))
    if not match:
        msg = "用法: /soul_approve <种子ID>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    seed_id = match.group(1)

    manager = ThoughtSeedManager.from_plugin_config(plugin)
    seed = await manager.get_seed_by_id(seed_id)

    if not seed:
        msg = f"未找到种子 {seed_id}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if seed.get("status") != "pending":
        msg = f"种子 {seed_id} 不在待审核状态"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    engine = InternalizationEngine(plugin)
    dedup_cfg = {
        "enabled": bool(plugin.config.thought_cabinet.auto_dedup_enabled),
        "threshold": float(plugin.config.thought_cabinet.auto_dedup_threshold),
    }
    result = await engine.internalize_seed(seed, dedup=dedup_cfg)

    if result["success"]:
        manager.mark_seed_status(seed_id, "approved")
        impact = result["spectrum_impact"]
        impact_str = ", ".join([f"{k}:{v:+d}" for k, v in impact.items() if v != 0])
        trait_id = result.get("trait_id", "")
        merged = bool(result.get("merged", False))
        similarity = result.get("dedup_similarity", None)
        trait_line = f"\ntrait_id: {trait_id}" if trait_id else ""
        if merged:
            sim_text = ""
            try:
                if similarity is not None:
                    sim_text = f" (similarity={float(similarity):.2f})"
            except (TypeError, ValueError):
                sim_text = ""
            trait_line = f"\ntrait_id: {trait_id}（已合并）{sim_text}"
        msg = (
            f"✅ 种子 {seed_id} 已批准内化{trait_line}\n\n"
            f"固化观点: {result['thought'][:100]}...\n\n"
            f"光谱影响: {impact_str or '无'}"
        )
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True
    else:
        msg = f"❌ 种子 {seed_id} 内化失败: {result['error']}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True


# ===== 种子拒绝 =====


async def handle_seed_reject(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """拒绝并删除思维种子（管理员）。"""
    from ..thought.seed_manager import ThoughtSeedManager
    from ..utils.spectrum_utils import match_user

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以审核思维种子"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # 从文本中提取种子 ID
    text = kwargs.get("text", "") or message.get("processed_plain_text", "")
    match = re.match(r"^/soul_reject\s+(\w+)\s*$", str(text))
    if not match:
        msg = "用法: /soul_reject <种子ID>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    seed_id = match.group(1)

    manager = ThoughtSeedManager.from_plugin_config(plugin)
    seed = await manager.get_seed_by_id(seed_id)

    if not seed:
        msg = f"未找到种子 {seed_id}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if seed.get("status") != "pending":
        msg = f"种子 {seed_id} 不在待审核状态"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    manager.mark_seed_status(seed_id, "rejected")
    logger.info(f"管理员拒绝思维种子: {seed_id}")
    msg = f"✅ 种子 {seed_id} 已拒绝"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


# ===== Trait 列表 =====


async def handle_traits_list(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """查看已固化的 traits（管理员，可按群过滤）。"""
    from ..utils.spectrum_utils import match_user
    from ..models.ideology_model import query_crystallized_traits

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以查看 traits"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    # 从文本中提取可选的 stream_id 过滤条件
    text = kwargs.get("text", "") or message.get("processed_plain_text", "")
    match = re.match(r"^/soul_traits(?:\s+(\S+))?\s*$", str(text))
    filter_stream_id = match.group(1).strip() if match and match.group(1) else None

    traits = query_crystallized_traits(
        deleted=False,
        stream_id=filter_stream_id if filter_stream_id and filter_stream_id != "global" else None,
        limit=50,
    )
    if not traits:
        msg = "当前没有已固化的 traits"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    lines = ["🧠 已固化 traits：", ""]
    if filter_stream_id:
        lines.append(f"过滤 stream_id: {filter_stream_id}")
        lines.append("")

    for t in traits:
        status = "enabled" if t.enabled else "disabled"
        lines.append(f"- {t.trait_id} [{status}] stream={t.stream_id or '-'} name={t.name}")
        try:
            from ..utils.trait_tags import parse_tags_json

            tags = parse_tags_json(t.tags_json)
        except (ValueError, TypeError):
            tags = []
        try:
            from ..utils.trait_evidence import parse_evidence_json

            evidence_count = len(parse_evidence_json(t.evidence_json))
        except (ValueError, TypeError):
            evidence_count = 0
        try:
            confidence = float(t.confidence or 0) / 100.0
        except (TypeError, ValueError):
            confidence = 0.0
        if tags:
            lines.append(f"  tags: {', '.join(tags)}")
        if confidence > 0.0 or evidence_count > 0:
            lines.append(f"  confidence: {confidence:.2f} evidence: {evidence_count}")
        q = (t.question or "").replace("\n", " ").strip()
        if q:
            if len(q) > 80:
                q = f"{q[:80]}..."
            lines.append(f"  问: {q}")
        snippet = (t.thought or "").replace("\n", " ").strip()
        if len(snippet) > 80:
            snippet = f"{snippet[:80]}..."
        if snippet:
            lines.append(f"  {snippet}")

    msg = "\n".join(lines)
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


# ===== Trait 详情 =====


async def handle_trait_detail(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """查看单个 trait 完整详情（管理员）。"""
    from ..utils.spectrum_utils import match_user
    from ..models.ideology_model import get_crystallized_trait_by_id, list_thought_edges_for_trait
    from ..utils.trait_tags import parse_tags_json
    from ..utils.trait_evidence import parse_evidence_json
    from ..worldview.constants import LAYER_LABEL_ZH, LIFECYCLE_LABEL_ZH

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以查看 trait"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    text = kwargs.get("text", "") or message.get("processed_plain_text", "")
    match = re.match(r"^/soul_trait\s+(\w+)\s*$", str(text))
    if not match:
        msg = "用法: /soul_trait <trait_id>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    trait_id = match.group(1)
    trait = get_crystallized_trait_by_id(trait_id)
    if not trait:
        msg = f"未找到 trait {trait_id}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    import json as _json

    # ── 组装 trait 详情数据契约（供卡片渲染 / 文本降级共用） ──
    layer_label = LAYER_LABEL_ZH.get(trait.ideology_layer, trait.ideology_layer)
    lifecycle_label = LIFECYCLE_LABEL_ZH.get(trait.lifecycle_state, trait.lifecycle_state)

    # 光谱影响：仅保留非 0 项
    try:
        impact_raw = _json.loads(trait.spectrum_impact_json or "{}")
    except (ValueError, TypeError):
        impact_raw = {}
    spectrum_impact = {k: int(v) for k, v in impact_raw.items() if isinstance(v, (int, float)) and v != 0} if isinstance(impact_raw, dict) else {}

    # 证据：每条取 event/reasoning，去换行，截断 100 字
    evidence_list: list[str] = []
    for ev in parse_evidence_json(trait.evidence_json or "[]")[:5]:
        if isinstance(ev, dict):
            ev_text = ev.get("event", "") or ev.get("reasoning", "") or str(ev)
        else:
            ev_text = str(ev)
        ev_text = ev_text.replace("\n", " ").strip()
        if len(ev_text) > 100:
            ev_text = ev_text[:100] + "..."
        if ev_text:
            evidence_list.append(ev_text)

    # 思想关联边：映射全部 5 种关系类型（含矛盾/弱化/修正，此前文本版遗漏）
    edge_label_map = {
        "derived_from": "源自种子",
        "supports": "支撑",
        "contradicted_by": "矛盾于",
        "weakened_by": "弱化于",
        "revised_by": "修正自",
    }
    edges_list: list[dict] = []
    for e in list_thought_edges_for_trait(trait_id, limit=6):
        label = edge_label_map.get(e.relation_type)
        if not label:
            continue  # 未知关系类型跳过，不崩
        if e.relation_type == "derived_from":
            target = (e.source_ref or "")[:12]
        else:
            target = (e.to_trait_id or "")[:12]
        edges_list.append({"relation_type": e.relation_type, "label": label, "target": target})

    data = {
        "trait_id": trait.trait_id,
        "name": trait.name,
        "enabled": bool(trait.enabled),
        "ideology_layer": trait.ideology_layer,
        "layer_label": layer_label,
        "lifecycle_state": trait.lifecycle_state,
        "lifecycle_label": lifecycle_label,
        "confidence": float(trait.confidence or 0) / 100.0,
        "stream_id": trait.stream_id,
        "created_at": trait.created_at.strftime("%Y-%m-%d %H:%M:%S") if trait.created_at else None,
        "tags": parse_tags_json(trait.tags_json or "[]"),
        "question": trait.question or "",
        "thought": trait.thought or "",
        "spectrum_impact": spectrum_impact,
        "evidence": evidence_list,
        "edges": edges_list,
    }

    # ── 出图：card_enabled 且渲染成功则发图，否则降级文本 ──
    from .dashboard_renderer import DashboardRenderer, build_trait_text

    if plugin.config.render.card_enabled:
        renderer = DashboardRenderer(
            plugin.ctx,
            plugin.config.render.viewport_width,
            plugin.config.render.device_scale_factor,
            plugin.config.render.render_timeout_ms,
        )
        image_base64 = await renderer.render_trait(data)
        if image_base64:
            try:
                await plugin.ctx.send.image(image_base64, stream_id)
                return True, "已生成 trait 详情卡片", True
            except (OSError, RuntimeError) as exc:
                logger.exception("发送 trait 详情卡片图片失败: %s", exc)
                fallback_text = f"卡片渲染失败，以下为文本状态：\n{build_trait_text(data)}"
                await plugin.ctx.send.text(fallback_text, stream_id)
                return True, "trait 详情(文本)", True
        # 渲染返回空串——降级文本
        fallback_text = f"卡片渲染失败，以下为文本状态：\n{build_trait_text(data)}"
        await plugin.ctx.send.text(fallback_text, stream_id)
        return True, "trait 详情(文本)", True

    # 配置关闭卡片——纯文本
    text_out = build_trait_text(data)
    await plugin.ctx.send.text(text_out, stream_id)
    return True, "trait 详情(文本)", True


# ===== Trait 设置 Tags =====


async def handle_trait_set_tags(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """设置指定 trait 的 tags（管理员，逗号或空格分隔）。"""
    from ..utils.spectrum_utils import match_user
    from ..models.ideology_model import get_crystallized_trait_by_id
    from ..utils.trait_tags import dumps_tags_json, parse_tags_json

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以设置 trait tags"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    text = kwargs.get("text", "") or message.get("processed_plain_text", "")
    match = re.match(r"^/soul_trait_set_tags\s+(\w+)\s+(.+?)\s*$", str(text))
    if not match:
        msg = "用法: /soul_trait_set_tags <trait_id> <tag1 tag2 / tag1,tag2>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    trait_id = match.group(1)
    tags_text = (match.group(2) or "").strip().replace("，", ",")
    raw_tags = [t for t in re.split(r"[,\\s]+", tags_text) if t]

    trait = get_crystallized_trait_by_id(trait_id)
    if not trait:
        msg = f"未找到 trait {trait_id}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    trait.tags_json = dumps_tags_json(raw_tags)
    trait.save()

    tags = parse_tags_json(trait.tags_json)
    msg = f"✅ trait {trait_id} tags 已更新: {', '.join(tags) if tags else '(empty)'}"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


# ===== Trait 合并 =====


async def handle_trait_merge(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """合并两个 trait（把 source 合并进 target，并软删除 source）。"""
    from ..utils.spectrum_utils import match_user
    from ..models.ideology_model import get_crystallized_trait_by_id
    from ..utils.trait_tags import dumps_tags_json, parse_tags_json
    from ..utils.trait_evidence import dumps_evidence_json, parse_evidence_json

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以合并 trait"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    text = kwargs.get("text", "") or message.get("processed_plain_text", "")
    match = re.match(r"^/soul_trait_merge\s+(\w+)\s+(\w+)\s*$", str(text))
    if not match:
        msg = "用法: /soul_trait_merge <source_trait_id> <target_trait_id>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    source_id = match.group(1)
    target_id = match.group(2)
    if source_id == target_id:
        msg = "source 和 target 不能相同"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    source = get_crystallized_trait_by_id(source_id)
    target = get_crystallized_trait_by_id(target_id)
    if not source:
        msg = f"未找到 source trait {source_id}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True
    if not target:
        msg = f"未找到 target trait {target_id}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    source_tags = parse_tags_json(source.tags_json)
    target_tags = parse_tags_json(target.tags_json)
    merged_tags = list(dict.fromkeys([*target_tags, *source_tags]))
    target.tags_json = dumps_tags_json(merged_tags)

    target_evidence = parse_evidence_json(target.evidence_json)
    source_evidence = parse_evidence_json(source.evidence_json)
    target.evidence_json = dumps_evidence_json([*target_evidence, *source_evidence])

    target.confidence = max(int(target.confidence or 0), int(source.confidence or 0))
    target.save()

    source.enabled = False
    source.deleted = True
    source.save()

    msg = f"✅ 已合并 {source_id} -> {target_id}"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


# ===== Trait 禁用 =====


async def handle_trait_disable(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """禁用指定 trait（管理员）。"""
    from ..utils.spectrum_utils import match_user
    from ..models.ideology_model import get_crystallized_trait_by_id

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以禁用 trait"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    text = kwargs.get("text", "") or message.get("processed_plain_text", "")
    match = re.match(r"^/soul_trait_disable\s+(\w+)\s*$", str(text))
    if not match:
        msg = "用法: /soul_trait_disable <trait_id>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    trait_id = match.group(1)
    trait = get_crystallized_trait_by_id(trait_id)
    if not trait:
        msg = f"未找到 trait {trait_id}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    trait.enabled = False
    trait.save()

    msg = f"✅ trait {trait_id} 已禁用"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


# ===== Trait 启用 =====


async def handle_trait_enable(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """启用指定 trait（管理员）。"""
    from ..utils.spectrum_utils import match_user
    from ..models.ideology_model import get_crystallized_trait_by_id

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以启用 trait"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    text = kwargs.get("text", "") or message.get("processed_plain_text", "")
    match = re.match(r"^/soul_trait_enable\s+(\w+)\s*$", str(text))
    if not match:
        msg = "用法: /soul_trait_enable <trait_id>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    trait_id = match.group(1)
    trait = get_crystallized_trait_by_id(trait_id)
    if not trait:
        msg = f"未找到 trait {trait_id}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    trait.enabled = True
    trait.save()

    msg = f"✅ trait {trait_id} 已启用"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


# ===== Trait 删除 =====


async def handle_trait_delete(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """删除指定 trait（管理员，软删除）。"""
    from ..utils.spectrum_utils import match_user
    from ..models.ideology_model import get_crystallized_trait_by_id

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以删除 trait"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    text = kwargs.get("text", "") or message.get("processed_plain_text", "")
    match = re.match(r"^/soul_trait_delete\s+(\w+)\s*$", str(text))
    if not match:
        msg = "用法: /soul_trait_delete <trait_id>"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    trait_id = match.group(1)
    trait = get_crystallized_trait_by_id(trait_id)
    if not trait:
        msg = f"未找到 trait {trait_id}"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    trait.enabled = False
    trait.deleted = True
    trait.save()

    msg = f"✅ trait {trait_id} 已删除"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True


# ===== 批量拒绝种子 =====


async def handle_seed_reject_all(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """批量拒绝所有待审核种子（管理员）。

    只提供批量拒绝，不提供批量批准——批准会触发大量 LLM 内化调用，
    应逐个人工判断。
    """
    from ..thought.seed_manager import ThoughtSeedManager
    from ..utils.spectrum_utils import match_user

    admin_user_id = plugin.config.admin.admin_user_id
    message = kwargs.get("message") or {}
    platform = message.get("platform", "")
    user_info = message.get("user_info") or {}
    user_id = str(user_info.get("user_id", ""))

    if not match_user(platform, user_id, admin_user_id):
        msg = "只有管理员可以审核思维种子"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    if not plugin.config.thought_cabinet.enabled:
        msg = "思维阁系统未启用"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    manager = ThoughtSeedManager.from_plugin_config(plugin)
    seeds = await manager.get_pending_seeds()

    if not seeds:
        msg = "当前没有待审核的思维种子"
        await plugin.ctx.send.text(msg, stream_id)
        return True, msg, True

    count = 0
    for seed in seeds:
        if manager.mark_seed_status(seed["seed_id"], "rejected"):
            count += 1

    logger.info(f"管理员批量拒绝 {count} 个思维种子")
    msg = f"✅ 已批量拒绝 {count} 个待审核种子"
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
