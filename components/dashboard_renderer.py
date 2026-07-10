"""Soul 引擎全状态可视化卡片：HTML 内联样式 + html2png 渲染。"""

from __future__ import annotations

import logging
from html import escape
from typing import Any

_logger = logging.getLogger(__name__)

_DEBUG_DUMP_HTML = False  # 生产环境不写调试 HTML；调试时改为 True

_ROOT_ID = "soul-dashboard"

from .dashboard_css import _DASHBOARD_CSS
_TRAIT_ROOT_ID = "soul-trait"
_INSPECT_ROOT_ID = "soul-inspect"

_EDGE_RELATION_CSS: dict[str, str] = {
    "derived_from": "edge-derived",
    "supports": "edge-support",
    "contradicted_by": "edge-bad",
    "weakened_by": "edge-warn",
    "revised_by": "edge-revised",
}

_SELECTION_MODE_LABELS: dict[str, str] = {
    "tag_hit": "标签命中",
    "tag_hit+tagless": "标签命中 + 无标签补位",
    "tagless_fill": "无标签补位",
    "fallback_recent_impact": "近期影响回退",
    "spectrum_only": "仅光谱注入",
}

_SPECTRUM_LABELS: dict[str, str] = {
    "sincerity": "真诚",
    "engagement": "投入",
    "closeness": "亲近",
    "directness": "直率",
}

# 0 = left pole, 100 = right pole. Matches spectrum model + /soul_status text.
_SPECTRUM_POLES: dict[str, tuple[str, str]] = {
    "sincerity": ("场面分寸", "真诚直率"),
    "engagement": ("克制怕耗", "热情投入"),
    "closeness": ("保持距离", "容易亲近"),
    "directness": ("含蓄绕弯", "有话直说"),
}

_LAYER_LABELS: dict[str, str] = {
    "values": "价值观",
    "worldview": "世界观",
    "conduct": "处事观",
}

_LIFECYCLE_LABELS: dict[str, str] = {
    "active": "有效",
    "strengthened": "已强化",
    "expired": "已过期",
    "contradicted": "矛盾",
    "weakened": "弱化",
    "revised": "修正",
}

_LIFECYCLE_CSS: dict[str, str] = {
    "active": "lc-active",
    "strengthened": "lc-strong",
    "expired": "lc-expired",
    "contradicted": "lc-bad",
    "weakened": "lc-warn",
    "revised": "lc-revised",
}

_FEATURE_LABELS: dict[str, str] = {
    "p1_enabled": "P1 三观",
    "mood_enabled": "短期情绪",
    "graph_inject": "图谱注入",
    "thought_cabinet": "思维阁",
    "notion": "Notion",
    "api": "API",
    "card_render": "卡片渲染",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _fmt_delta(value: int) -> str:
    if value > 0:
        return f"+{value}"
    return str(value)


def _dash_or(value: Any, *, empty: str = "—") -> str:
    if value is None:
        return empty
    if isinstance(value, str) and not value.strip():
        return empty
    return str(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _short_id(trait_id: str, *, max_len: int = 10) -> str:
    tid = trait_id.strip()
    if len(tid) <= max_len:
        return tid
    return tid[:max_len] + "…"



def _short_perspective(stream_id: str) -> str:
    """Human-readable view label; truncate long stream ids for header density."""
    sid = str(stream_id or "").strip()
    if not sid:
        return "全局"
    if len(sid) <= 14:
        return sid
    return f"{sid[:8]}…{sid[-4:]}"


class DashboardRenderer:
    """使用 Host render.html2png 生成 Soul 状态总览卡片。"""

    def __init__(
        self,
        ctx: Any,
        viewport_width: int,
        device_scale_factor: float,
        render_timeout_ms: int,
    ) -> None:
        self._ctx = ctx
        self.viewport_width = viewport_width
        self.device_scale_factor = device_scale_factor
        self.render_timeout_ms = render_timeout_ms

    async def render(self, data: dict) -> tuple[str, str | None]:
        """把 dashboard 数据渲染为 PNG base64；失败返回 (空串, 错误原因)。"""
        try:
            html = self._build_html(data)
            return await self._render_html(html, root_id=_ROOT_ID)
        except Exception:
            self._log_exception("Soul dashboard 渲染失败")
            return ("", "卡片渲染失败")

    async def render_trait(self, data: dict) -> tuple[str, str | None]:
        """渲染单个 trait 详情卡片为 PNG base64；失败返回 (空串, 错误原因)。"""
        try:
            html = self._build_trait_html(data)
            return await self._render_html(html, root_id=_TRAIT_ROOT_ID, viewport_height=1800)
        except Exception:
            self._log_exception("Soul trait 详情渲染失败")
            return ("", "卡片渲染失败")

    async def render_inspect(self, data: dict) -> tuple[str, str | None]:
        """渲染 inspect 命中预览卡片为 PNG base64；失败返回 (空串, 错误原因)。"""
        try:
            html = self._build_inspect_html(data)
            return await self._render_html(html, root_id=_INSPECT_ROOT_ID, viewport_height=1800)
        except Exception:
            self._log_exception("Soul inspect 预览渲染失败")
            return ("", "卡片渲染失败")



    async def _render_html(
        self,
        html: str,
        *,
        root_id: str,
        viewport_height: int = 1600,
    ) -> tuple[str, str | None]:
        """返回 (base64, error_reason)。成功时 error_reason=None。"""
        if self._ctx is None:
            return ("", "卡片渲染失败")

        if _DEBUG_DUMP_HTML:
            try:
                from datetime import datetime
                from pathlib import Path as _Path

                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                dump_dir = _Path("/tmp/soul-dash-live")
                dump_dir.mkdir(parents=True, exist_ok=True)
                (dump_dir / f"{stamp}-{root_id}.html").write_text(html, encoding="utf-8")
                body_part = html.split("<body", 1)[-1]
                meta = {
                    "root_id": root_id,
                    "viewport_width": self.viewport_width,
                    "viewport_height": viewport_height,
                    "device_scale_factor": self.device_scale_factor,
                    "html_len": len(html),
                    "style_open": html.count("<style"),
                    "style_close": html.count("</style>"),
                    "css_in_body": ("justify-content" in body_part) or (".skip-row" in body_part),
                }
                (dump_dir / f"{stamp}-{root_id}.meta.txt").write_text(repr(meta), encoding="utf-8")
            except Exception:
                pass

        def _png_size(blob: bytes) -> tuple[int | None, int | None]:
            if blob[:8] == b"\x89PNG\r\n\x1a\n" and len(blob) >= 24:
                return int.from_bytes(blob[16:20], "big"), int.from_bytes(blob[20:24], "big")
            return None, None

        def _crop_content_png(blob: bytes) -> bytes:
            try:
                from io import BytesIO
                from PIL import Image

                with Image.open(BytesIO(blob)) as im:
                    im = im.convert("RGB")
                    pixels = im.load()
                    w, h = im.size
                    left, top, right, bottom = w, h, 0, 0
                    found = False
                    step = 2
                    for y in range(0, h, step):
                        for x in range(0, w, step):
                            r, g, b = pixels[x, y]
                            if r + g + b < 735:
                                found = True
                                left = min(left, x)
                                top = min(top, y)
                                right = max(right, x)
                                bottom = max(bottom, y)
                    if not found:
                        return blob
                    pad = 12
                    box = (
                        max(0, left - pad),
                        max(0, top - pad),
                        min(w, right + 1 + pad),
                        min(h, bottom + 1 + pad),
                    )
                    if box[2] - box[0] < 120 or box[3] - box[1] < 120:
                        return blob
                    cropped = im.crop(box)
                    out = BytesIO()
                    cropped.save(out, format="PNG", optimize=True)
                    return out.getvalue()
            except Exception:
                return blob

        def _top_looks_like_css_leak(blob: bytes) -> bool:
            try:
                from io import BytesIO
                from PIL import Image

                with Image.open(BytesIO(blob)) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    band_h = min(28, h)
                    pixels = im.load()
                    dark = total = 0
                    for y in range(band_h):
                        for x in range(0, w, 3):
                            r, g, b = pixels[x, y]
                            total += 1
                            if r + g + b < 420:
                                dark += 1
                    return total > 0 and (dark / total) > 0.08
            except Exception:
                return False

        def _dump(name: str, blob: bytes | None, meta: dict) -> None:
            if not _DEBUG_DUMP_HTML:
                return
            try:
                from datetime import datetime
                from pathlib import Path as _Path

                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                dump_dir = _Path("/tmp/soul-dash-live")
                dump_dir.mkdir(parents=True, exist_ok=True)
                (dump_dir / f"{stamp}-{root_id}-{name}.result.txt").write_text(
                    repr(meta), encoding="utf-8"
                )
                if blob is not None:
                    (dump_dir / f"{stamp}-{root_id}-{name}.png").write_bytes(blob)
            except Exception:
                pass

        async def _once(*, selector: str, full_page: bool, wait_ms: int) -> tuple[str, bytes | None, int | None, int | None]:
            result = await self._ctx.render.html2png(
                html,
                selector=selector,
                viewport={"width": self.viewport_width, "height": viewport_height},
                device_scale_factor=self.device_scale_factor,
                full_page=full_page,
                omit_background=False,
                wait_until="load",
                wait_for_selector=f"#{root_id}",
                wait_for_timeout_ms=wait_ms,
                render_timeout_ms=self.render_timeout_ms,
                allow_network=False,
            )
            if not isinstance(result, dict):
                return "", None, None, None
            image_base64 = result.get("image_base64")
            if not isinstance(image_base64, str) or not image_base64:
                return "", None, None, None
            try:
                import base64 as _b64

                blob = _b64.b64decode(image_base64)
            except Exception:
                return image_base64, None, None, None
            width, height = _png_size(blob)
            return image_base64, blob, width, height

        import asyncio as _asyncio
        import base64 as _b64

        # Prefer body viewport first: live element screenshots are the unstable path.
        attempts = [
            {"name": "a1-body", "selector": "body", "full_page": False, "wait_ms": 300, "crop": True},
            {"name": "a2-element", "selector": f"#{root_id}", "full_page": False, "wait_ms": 300, "crop": False},
            {"name": "a3-full-crop", "selector": "body", "full_page": True, "wait_ms": 250, "crop": True},
        ]

        for idx, attempt in enumerate(attempts):
            try:
                image_base64, blob, width, height = await _once(
                    selector=attempt["selector"],
                    full_page=attempt["full_page"],
                    wait_ms=attempt["wait_ms"],
                )
            except Exception:
                self._log_exception(f"html2png failed ({attempt['name']})")
                # 指数 backoff：重试间递增等待，避免压垮宿主无头浏览器
                await _asyncio.sleep(0.2 * (2 ** idx))
                continue
            if not image_base64 or blob is None:
                # 指数 backoff：重试间递增等待，避免压垮宿主无头浏览器
                await _asyncio.sleep(0.2 * (2 ** idx))
                continue

            if attempt.get("crop"):
                cropped = _crop_content_png(blob)
                if cropped is not blob:
                    blob = cropped
                    width, height = _png_size(blob)
                    image_base64 = _b64.b64encode(blob).decode("ascii")

            leak = _top_looks_like_css_leak(blob)
            _dump(
                attempt["name"],
                blob,
                {
                    "attempt": attempt,
                    "width": width,
                    "height": height,
                    "bytes": len(blob),
                    "css_leak_heuristic": leak,
                },
            )
            if leak:
                _logger.warning(
                    "Soul dashboard rejected CSS-leak-looking frame %s: %sx%s",
                    attempt["name"],
                    width,
                    height,
                )
                # 指数 backoff：重试间递增等待，避免压垮宿主无头浏览器
                await _asyncio.sleep(0.2 * (2 ** idx))
                continue
            if root_id == _ROOT_ID and (height is None or height < 980 or height > 2300):
                _logger.warning(
                    "Soul dashboard size outside recommended range %s: %sx%s — passing through",
                    attempt["name"],
                    width,
                    height,
                )
                # 高度超区间只 log warning 不 reject，让渲染结果通过（可能需要裁剪）
            return (image_base64, None)

        _logger.error("Soul dashboard render failed after all attempts")
        return ("", "卡片渲染失败")

    def _log_exception(self, message: str) -> None:
        ctx = self._ctx
        if ctx is not None and getattr(ctx, "logger", None) is not None:
            ctx.logger.exception(message)
        else:
            _logger.exception(message)

    def _build_html(self, data: dict) -> str:
        initialized = bool(data.get("initialized"))
        generated_at = _dash_or(data.get("generated_at"), empty="未知时间")
        stream_id = str(data.get("stream_id") or "").strip()
        perspective = _short_perspective(stream_id)
        perspective_title = stream_id if stream_id else "全局"

        spectrum = _as_dict(data.get("spectrum"))
        trait_layers = _as_dict(data.get("trait_counts_by_layer"))
        lifecycle = _as_dict(data.get("lifecycle_distribution"))
        mood = _as_dict(data.get("mood"))
        group_slice = data.get("group_slice")
        thought = _as_dict(data.get("thought_cabinet"))
        evolutions = data.get("recent_evolutions")
        if not isinstance(evolutions, list):
            evolutions = []
        flags = _as_dict(data.get("feature_flags"))

        init_badge = (
            '<span class="badge badge-ok">已初始化</span>'
            if initialized
            else '<span class="badge badge-alert">未初始化 · 请 /soul_setup</span>'
        )

        spectrum_block = self._render_spectrum_bars(spectrum)
        layer_block = self._render_layer_counts(trait_layers)
        lifecycle_block = self._render_lifecycle(lifecycle, _as_int(data.get("trait_total")))
        mood_block = self._render_mood(mood)
        slice_block = self._render_group_slice(group_slice) if stream_id and isinstance(group_slice, dict) else ""
        thought_block = self._render_thought_cabinet(thought)
        evolution_block = self._render_evolutions(evolutions[:5])
        flags_block = self._render_feature_flags(flags)
        graph_total = _as_int(data.get("graph_edge_total"))

        body = f"""
        <article id="{_ROOT_ID}" class="dash">
          <header class="hero">
            <div class="hero-stripe" aria-hidden="true"></div>
            <div class="hero-inner">
              <div class="hero-main">
                <div class="eyebrow">Mai Soul Engine</div>
                <h1>Soul 引擎状态</h1>
                <p class="meta" title="视角 {escape(perspective_title)}">生成于 {escape(generated_at)}<span class="meta-sep"> · </span>视角 <code class="meta-id">{escape(perspective)}</code></p>
                {init_badge}
              </div>
            </div>
          </header>

          <section class="grid two">
            <div class="panel">
              <div class="label">社交光谱</div>
              {spectrum_block}
              <p class="footnote">更新 {_dash_or(spectrum.get("updated_at"))} · 上次演化 {_dash_or(spectrum.get("last_evolution"))}</p>
            </div>
            <div class="panel">
              <div class="label">三观分层</div>
              {layer_block}
              <div class="divider"></div>
              <div class="label sub">生命周期 · 共 {graph_total} 条图谱边</div>
              {lifecycle_block}
            </div>
          </section>

          <section class="grid three">
            {mood_block}
            {slice_block or self._render_group_slice_global_placeholder()}
            {thought_block}
          </section>

          <section class="bottom-stack">
            <div class="panel panel-bottom">
              <div class="label">最近演化</div>
              {evolution_block}
            </div>
            <div class="panel panel-flags panel-bottom">
              <div class="label">功能开关</div>
              {flags_block}
            </div>
          </section>
        </article>
        """
        return self._wrap_html(body)

    def _render_spectrum_bars(self, spectrum: dict[str, Any]) -> str:
        rows: list[str] = []
        for key in ("sincerity", "engagement", "closeness", "directness"):
            val = _clamp(_as_int(spectrum.get(key)), 0, 100)
            left_pole, right_pole = _SPECTRUM_POLES[key]
            rows.append(
                f"""
                <div class="axis-row">
                  <div class="axis-head">
                    <span class="axis-left">{escape(left_pole)}</span>
                    <span class="axis-val">{val}</span>
                    <span class="axis-right">{escape(right_pole)}</span>
                  </div>
                  <div class="axis-track">
                    <div class="axis-mid" aria-hidden="true"></div>
                    <div class="axis-fill" style="width:{val}%"></div>
                    <div class="axis-knob" style="left:{val}%"></div>
                  </div>
                </div>
                """
            )
        return '<div class="axes">' + "".join(rows) + "</div>"

    def _render_layer_counts(self, layers: dict[str, Any]) -> str:
        cards: list[str] = []
        for key in ("values", "worldview", "conduct"):
            count = _as_int(layers.get(key))
            cards.append(
                f"""
                <div class="stat-card">
                  <strong>{count}</strong>
                  <span>{escape(_LAYER_LABELS[key])}</span>
                </div>
                """
            )
        return '<div class="stat-grid">' + "".join(cards) + "</div>"

    def _render_lifecycle(self, lifecycle: dict[str, Any], trait_total: int) -> str:
        chips: list[str] = []
        nonzero = 0
        for key in _LIFECYCLE_LABELS:
            count = _as_int(lifecycle.get(key))
            if count:
                nonzero += count
            css = _LIFECYCLE_CSS[key]
            chips.append(
                f'<span class="chip {css}">{escape(_LIFECYCLE_LABELS[key])} <b>{count}</b></span>'
            )
        total_line = f'<p class="trait-total">Trait 总数 <strong>{trait_total}</strong></p>'
        if trait_total <= 0 and nonzero <= 0:
            return (
                total_line
                + '<div class="empty-state empty-state-compact">'
                + '<div class="empty-state-title">暂无 Trait</div>'
                + '<div class="empty-state-desc">思维阁种子通过后会出现在这里</div>'
                + "</div>"
            )
        return total_line + '<div class="chips">' + "".join(chips) + "</div>"

    def _render_mood(self, mood: dict[str, Any]) -> str:
        if not mood.get("enabled"):
            return """
            <div class="panel panel-dim">
              <div class="label">短期情绪</div>
              <div class="empty">未启用</div>
            </div>
            """
        rows: list[str] = []
        for key, label in (("valence", "愉悦"), ("arousal", "兴奋"), ("energy", "精力")):
            val = _clamp(_as_int(mood.get(key)), -100, 100)
            pct = (val + 100) / 2.0
            rows.append(
                f"""
                <div class="bipolar-row">
                  <span class="bar-name">{escape(label)}</span>
                  <div class="bipolar-track">
                    <div class="bipolar-mid"></div>
                    <div class="bipolar-fill" style="left:{pct}%;margin-left:-7px;transform:none;"></div>
                  </div>
                  <span class="bar-val">{val}</span>
                </div>
                """
            )
        updated = _dash_or(mood.get("updated_at"))
        return f"""
        <div class="panel">
          <div class="label">短期情绪</div>
          <div class="bipolar">{"".join(rows)}</div>
          <p class="footnote">更新 {escape(updated)}</p>
        </div>
        """

    def _render_group_slice(self, group_slice: dict[str, Any]) -> str:
        rows: list[str] = []
        for key in ("sincerity", "engagement", "closeness", "directness"):
            offset = _as_int(group_slice.get(f"{key}_offset"))
            rows.append(
                f'<div class="slice-row"><span>{escape(_SPECTRUM_LABELS[key])}</span>'
                f'<strong>{escape(_fmt_delta(offset))}</strong></div>'
            )
        sample = _as_int(group_slice.get("sample_count"))
        return f"""
        <div class="panel">
          <div class="label">本群切片</div>
          <div class="slice-grid">{"".join(rows)}</div>
          <p class="footnote">样本数 {sample}</p>
        </div>
        """


    def _render_group_slice_global_placeholder(self) -> str:
        """Empty-state card when dashboard is opened without a group stream."""
        return """
        <div class="panel panel-dim panel-placeholder">
          <div class="label">本群切片</div>
          <div class="placeholder-body">
            <div class="placeholder-kicker">当前视角</div>
            <div class="placeholder-title">全局 / 非群会话</div>
            <p class="placeholder-desc">群切片只在群聊视角下提供偏移信息。</p>
            <p class="placeholder-hint">到目标群执行 /soul_dashboard 查看本群切片。</p>
          </div>
        </div>
        """

    def _render_thought_cabinet(self, thought: dict[str, Any]) -> str:
        if not thought.get("enabled"):
            inner = '<div class="empty">未启用</div>'
        else:
            pending = _as_int(thought.get("pending_seeds"))
            inner = f'<div class="big-num">{pending}</div><span class="big-label">待审种子</span>'
        return f"""
        <div class="panel">
          <div class="label">思维阁</div>
          <div class="thought-box">{inner}</div>
        </div>
        """

    def _render_evolutions(self, items: list[Any]) -> str:
        if not items:
            return (
                '<div class="empty-state">'
                '<div class="empty-state-title">暂无演化记录</div>'
                '<div class="empty-state-desc">光谱演化产生后会显示在这里</div>'
                "</div>"
            )
        cards: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            deltas = _as_dict(item.get("deltas"))
            delta_parts = [
                f"{_SPECTRUM_LABELS[k]}{_fmt_delta(_as_int(deltas.get(k)))}"
                for k in ("sincerity", "engagement", "closeness", "directness")
            ]
            delta_text = " · ".join(delta_parts)
            reason = _dash_or(item.get("reason"), empty="（无说明）")
            cards.append(
                f"""
                <div class="palette-row">
                  <div class="evo-head">
                    <span>群 {escape(_dash_or(item.get("group_id")))}</span>
                    <span class="muted">{escape(_dash_or(item.get("timestamp")))}</span>
                  </div>
                  <div class="evo-deltas">{escape(delta_text)}</div>
                  <p class="row-detail">{escape(reason)}</p>
                </div>
                """
            )
        return '<div class="evo-stack">' + "".join(cards) + "</div>"

    def _render_feature_flags(self, flags: dict[str, Any]) -> str:
        badges: list[str] = []
        for key, label in _FEATURE_LABELS.items():
            on = bool(flags.get(key))
            css = "on" if on else "off"
            text = "开" if on else "关"
            badges.append(
                f'<span class="flag {css}"><span class="flag-name">{escape(label)}</span>'
                f'<span class="flag-state">{text}</span></span>'
            )
        return '<div class="flags">' + "".join(badges) + "</div>"

    def _lifecycle_chip(self, lifecycle_state: str, lifecycle_label: str) -> str:
        css = _LIFECYCLE_CSS.get(lifecycle_state, "lc-expired")
        label = lifecycle_label.strip() if lifecycle_label.strip() else _LIFECYCLE_LABELS.get(lifecycle_state, lifecycle_state)
        return f'<span class="chip {css}">{escape(label)}</span>'

    def _build_trait_html(self, data: dict) -> str:
        name = _dash_or(data.get("name"), empty="未命名 Trait")
        trait_id = str(data.get("trait_id") or "")
        short_id = escape(_short_id(trait_id))
        enabled = bool(data.get("enabled"))
        enable_badge = (
            '<span class="badge badge-ok">启用</span>'
            if enabled
            else '<span class="badge badge-muted">停用</span>'
        )
        layer_label = _dash_or(data.get("layer_label"), empty=_LAYER_LABELS.get(str(data.get("ideology_layer") or ""), "—"))
        lifecycle_state = str(data.get("lifecycle_state") or "active")
        lifecycle_label = str(data.get("lifecycle_label") or "")
        lc_chip = self._lifecycle_chip(lifecycle_state, lifecycle_label)

        confidence = _clamp(int(round(_as_float(data.get("confidence")) * 100)), 0, 100)
        stream_raw = str(data.get("stream_id") or "").strip()
        stream_display = "全局" if stream_raw == "global" or not stream_raw else stream_raw

        tags = _as_list(data.get("tags"))
        tag_html = self._render_tag_cloud(tags)
        question = _dash_or(data.get("question"), empty="暂无")
        thought = _dash_or(data.get("thought"), empty="暂无")
        impact_block = self._render_spectrum_impact(_as_dict(data.get("spectrum_impact")))
        evidence_block = self._render_evidence_list(_as_list(data.get("evidence")))
        edges_block = self._render_trait_edges(_as_list(data.get("edges")))
        created = _dash_or(data.get("created_at"))

        body = f"""
        <article id="{_TRAIT_ROOT_ID}" class="dash">
          <header class="hero hero-compact">
            <div class="hero-stripe" aria-hidden="true"></div>
            <div class="hero-inner">
              <div class="hero-main">
                <div class="eyebrow">Trait Detail</div>
                <h1 class="title-trait">{escape(name)}</h1>
                <p class="meta">ID {short_id} · 来源 {escape(stream_display)} · 创建于 {escape(created)}</p>
                <div class="badge-row">
                  {enable_badge}
                  <span class="badge badge-layer">{escape(layer_label)}</span>
                  {lc_chip}
                </div>
              </div>
            </div>
          </header>

          <section class="panel">
            <div class="label">置信度</div>
            <div class="bar-row bar-row-wide">
              <span class="bar-name">置信</span>
              <div class="bar-track" style="background-image:linear-gradient(90deg,#3b82f6,#2563eb);background-size:{confidence}% 100%;background-repeat:no-repeat;background-color:#e8eef6;"></div>
              <span class="bar-val">{confidence}%</span>
            </div>
          </section>

          <section class="grid two">
            <div class="panel">
              <div class="label">标签</div>
              {tag_html}
            </div>
            <div class="panel">
              <div class="label">光谱影响</div>
              {impact_block}
            </div>
          </section>

          <section class="panel">
            <div class="label">关联问题</div>
            <p class="body-text">{escape(question)}</p>
          </section>

          <section class="panel panel-accent">
            <div class="label">观点</div>
            <p class="thought-body">{escape(thought)}</p>
          </section>

          <section class="grid two">
            <div class="panel">
              <div class="label">证据</div>
              {evidence_block}
            </div>
            <div class="panel">
              <div class="label">思想关联</div>
              {edges_block}
            </div>
          </section>
        </article>
        """
        return self._wrap_html(body)

    def _render_tag_cloud(self, tags: list[Any]) -> str:
        normalized = [str(t).strip() for t in tags if str(t).strip()]
        if not normalized:
            return '<div class="empty">暂无</div>'
        items = "".join(f'<span class="keycap">{escape(t)}</span>' for t in normalized)
        return f'<div class="tags">{items}</div>'

    def _render_spectrum_impact(self, impact: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("sincerity", "engagement", "closeness", "directness"):
            val = _as_int(impact.get(key))
            if val == 0:
                continue
            parts.append(
                f'<span class="impact-chip">{escape(_SPECTRUM_LABELS[key])} '
                f'<strong>{escape(_fmt_delta(val))}</strong></span>'
            )
        if not parts:
            return '<div class="empty">无光谱偏移</div>'
        return '<div class="impact-row">' + "".join(parts) + "</div>"

    def _render_evidence_list(self, evidence: list[Any]) -> str:
        items = [str(e).strip() for e in evidence if str(e).strip()]
        if not items:
            return '<div class="empty">暂无</div>'
        lis = "".join(f"<li>{escape(item)}</li>" for item in items)
        return f'<ul class="list-body">{lis}</ul>'

    def _render_trait_edges(self, edges: list[Any]) -> str:
        if not edges:
            return '<div class="empty">暂无关联边</div>'
        rows: list[str] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            rel = str(edge.get("relation_type") or "")
            css = _EDGE_RELATION_CSS.get(rel, "edge-derived")
            label = _dash_or(edge.get("label"), empty=rel or "—")
            target = _dash_or(edge.get("target"), empty="—")
            rows.append(
                f"""
                <div class="palette-row edge-palette-row">
                  <span class="edge-badge {css}">{escape(label)}</span>
                  <span class="edge-target">{escape(target)}</span>
                </div>
                """
            )
        if not rows:
            return '<div class="empty">暂无关联边</div>'
        return '<div class="edge-stack">' + "".join(rows) + "</div>"

    def _build_inspect_html(self, data: dict) -> str:
        query_text = _dash_or(data.get("query_text"), empty="（空）")
        stream_id = str(data.get("stream_id") or "").strip()
        perspective = "全局" if not stream_id else stream_id
        mode_key = str(data.get("selection_mode") or "spectrum_only")
        mode_label = _SELECTION_MODE_LABELS.get(mode_key, mode_key)
        max_traits = _as_int(data.get("max_traits"))
        total_active = _as_int(data.get("total_active"))
        selected = _as_list(data.get("selected"))
        skipped = _as_list(data.get("skipped"))

        selected_block = self._render_inspect_selected(selected)
        skipped_block = self._render_inspect_skipped(skipped)
        footnote = (
            f"从 {total_active} 个活跃 trait 中按规则选出最多 {max_traits} 个 · "
            f"模式 {mode_label}"
        )

        body = f"""
        <article id="{_INSPECT_ROOT_ID}" class="dash">
          <header class="hero hero-compact">
            <div class="hero-stripe" aria-hidden="true"></div>
            <div class="hero-inner">
              <div class="hero-main">
                <div class="eyebrow">Injection Inspect</div>
                <h1>注入命中预览</h1>
                <p class="meta">视角 {escape(perspective)}</p>
                <span class="badge badge-mode">{escape(mode_label)}</span>
              </div>
            </div>
          </header>

          <section class="panel">
            <div class="label">待测文本</div>
            <blockquote class="query-block">{escape(query_text)}</blockquote>
          </section>

          <section class="panel">
            <div class="label">命中列表（按优先级）</div>
            {selected_block}
          </section>

          <section class="panel panel-dim">
            <div class="label">跳过</div>
            {skipped_block}
          </section>

          <p class="footnote footnote-block">{escape(footnote)}</p>
        </article>
        """
        return self._wrap_html(body)

    def _render_inspect_selected(self, items: list[Any]) -> str:
        if not items:
            return '<div class="empty">无命中（将走 fallback 或仅光谱注入）</div>'
        cards: list[str] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            name = _dash_or(item.get("name"), empty="—")
            layer = _dash_or(item.get("layer_label"), empty="—")
            lc = _dash_or(item.get("lifecycle_label"), empty="—")
            conf = _as_float(item.get("confidence"))
            quality = _as_float(item.get("quality_score"))
            conf_pct = int(round(conf * 100))
            thought = _dash_or(item.get("thought"), empty="—")
            tags = _as_list(item.get("matched_tags"))
            tag_part = self._render_matched_tags(tags)
            cards.append(
                f"""
                <div class="palette-row palette-row-active hit-card">
                  <div class="hit-head">
                    <span class="hit-rank">#{idx}</span>
                    <span class="hit-name">{escape(name)}</span>
                    <span class="hit-id muted">{escape(_short_id(str(item.get("trait_id") or "")))}</span>
                  </div>
                  <div class="hit-badges">
                    <span class="badge badge-layer">{escape(layer)}</span>
                    <span class="chip lc-active">{escape(lc)}</span>
                    <span class="hit-metric">置信 {conf_pct}%</span>
                    <span class="hit-metric">质量 {quality:.2f}</span>
                  </div>
                  {tag_part}
                  <p class="hit-thought">{escape(thought)}</p>
                </div>
                """
            )
        return '<div class="hit-stack">' + "".join(cards) + "</div>"

    def _render_matched_tags(self, tags: list[Any]) -> str:
        normalized = [str(t).strip() for t in tags if str(t).strip()]
        if not normalized:
            return '<div class="hit-tags empty-inline">未命中标签</div>'
        items = "".join(f'<span class="keycap keycap-hit">{escape(t)}</span>' for t in normalized)
        return f'<div class="hit-tags">{items}</div>'

    def _render_inspect_skipped(self, items: list[Any]) -> str:
        if not items:
            return '<div class="empty">无跳过项</div>'
        rows: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = _dash_or(item.get("name"), empty="—")
            reason = _dash_or(item.get("reason"), empty="—")
            rows.append(
                f"""
                <div class="skip-row">
                  <span class="skip-name">{escape(name)}</span>
                  <span class="skip-reason">{escape(reason)}</span>
                </div>
                """
            )
        return '<div class="skip-stack">' + "".join(rows) + "</div>"

    @staticmethod
    def _wrap_html(body: str) -> str:
        """Assemble a full HTML document for html2png.

        CSS is stored in ``_DASHBOARD_CSS`` as a plain string so curly braces are
        never interpreted by f-string formatting.
        """
        return (
            "<!doctype html>\n"
            '<html lang="zh-CN">\n'
            "<head>\n"
            '<meta charset="utf-8" />\n'
            "<style>\n"
            + _DASHBOARD_CSS
            + "</style>\n"
            "</head>\n"
            "<body>"
            + body
            + "</body>\n"
            "</html>"
        )


def build_dashboard_text(data: dict) -> str:
    """纯文本降级版：渲染关闭或失败时使用。"""
    initialized = bool(data.get("initialized"))
    generated_at = _dash_or(data.get("generated_at"), empty="未知")
    stream_id = str(data.get("stream_id") or "").strip()
    perspective = "全局" if not stream_id else stream_id

    lines: list[str] = [
        "◈ Soul 引擎状态",
        f"时间 {generated_at} · 视角 {perspective}",
        "已初始化" if initialized else "未初始化 — 请执行 /soul_setup",
        "",
        "【社交光谱 0–100】",
    ]

    spectrum = _as_dict(data.get("spectrum"))
    for key in ("sincerity", "engagement", "closeness", "directness"):
        left_p, right_p = _SPECTRUM_POLES[key]
        val = _as_int(spectrum.get(key))
        lines.append(f"  {left_p} <- {val} -> {right_p}")
    lines.append(
        f"  更新 {_dash_or(spectrum.get('updated_at'))} · 演化 {_dash_or(spectrum.get('last_evolution'))}"
    )

    layers = _as_dict(data.get("trait_counts_by_layer"))
    lines.extend(
        [
            "",
            "【三观分层】",
            f"  价值观 {_as_int(layers.get('values'))} · 世界观 {_as_int(layers.get('worldview'))} · 处事观 {_as_int(layers.get('conduct'))}",
        ]
    )

    lifecycle = _as_dict(data.get("lifecycle_distribution"))
    lc_parts = [f"{_LIFECYCLE_LABELS[k]} {_as_int(lifecycle.get(k))}" for k in _LIFECYCLE_LABELS]
    lines.extend(["", "【生命周期】", "  " + " · ".join(lc_parts), f"  Trait 总数 {_as_int(data.get('trait_total'))}"])

    mood = _as_dict(data.get("mood"))
    lines.append("")
    if mood.get("enabled"):
        lines.append("【短期情绪 -100~100】")
        lines.append(
            f"  愉悦 {_as_int(mood.get('valence'))} · 兴奋 {_as_int(mood.get('arousal'))} · 精力 {_as_int(mood.get('energy'))}"
        )
    else:
        lines.append("【短期情绪】未启用")

    group_slice = data.get("group_slice")
    if stream_id and isinstance(group_slice, dict):
        offsets = " · ".join(
            f"{_SPECTRUM_LABELS[k]}{_fmt_delta(_as_int(group_slice.get(f'{k}_offset')))}"
            for k in ("sincerity", "engagement", "closeness", "directness")
        )
        lines.extend(["", "【本群切片】", f"  {offsets}", f"  样本 {_as_int(group_slice.get('sample_count'))}"])

    thought = _as_dict(data.get("thought_cabinet"))
    lines.append("")
    if thought.get("enabled"):
        lines.append(f"【思维阁】待审种子 {_as_int(thought.get('pending_seeds'))}")
    else:
        lines.append("【思维阁】未启用")

    lines.extend(["", f"【思想图谱】边总数 {_as_int(data.get('graph_edge_total'))}"])

    evolutions = data.get("recent_evolutions")
    if not isinstance(evolutions, list):
        evolutions = []
    lines.append("")
    lines.append("【最近演化】")
    if not evolutions:
        lines.append("  暂无")
    else:
        for item in evolutions[:3]:
            if not isinstance(item, dict):
                continue
            deltas = _as_dict(item.get("deltas"))
            delta_s = " ".join(
                f"{_SPECTRUM_LABELS[k]}{_fmt_delta(_as_int(deltas.get(k)))}"
                for k in ("sincerity", "engagement", "closeness", "directness")
            )
            reason = _dash_or(item.get("reason"), empty="")
            lines.append(
                f"  · 群 {_dash_or(item.get('group_id'))} {_dash_or(item.get('timestamp'))} | {delta_s}"
            )
            if reason and reason != "—":
                lines.append(f"    {reason}")

    flags = _as_dict(data.get("feature_flags"))
    flag_s = " · ".join(
        f"{_FEATURE_LABELS[k]}{'开' if flags.get(k) else '关'}" for k in _FEATURE_LABELS
    )
    lines.extend(["", "【功能开关】", f"  {flag_s}"])

    return "\n".join(lines)


def build_trait_text(data: dict) -> str:
    """trait 详情纯文本降级版。"""
    name = _dash_or(data.get("name"), empty="未命名")
    trait_id = str(data.get("trait_id") or "")
    enabled = "启用" if bool(data.get("enabled")) else "停用"
    layer = _dash_or(data.get("layer_label"), empty="—")
    lifecycle = _dash_or(data.get("lifecycle_label"), empty="—")
    conf = int(round(_as_float(data.get("confidence")) * 100))
    stream_raw = str(data.get("stream_id") or "").strip()
    stream_display = "全局" if stream_raw == "global" or not stream_raw else stream_raw

    lines: list[str] = [
        f"◇ Trait · {name}",
        f"ID {trait_id} · {enabled} · {layer} · {lifecycle}",
        f"来源 {stream_display} · 置信 {conf}% · 创建 {_dash_or(data.get('created_at'))}",
        "",
    ]

    tags = _as_list(data.get("tags"))
    if tags:
        tag_s = " ".join(str(t).strip() for t in tags if str(t).strip())
        lines.extend(["【标签】", f"  {tag_s}", ""])
    else:
        lines.extend(["【标签】", "  暂无", ""])

    lines.extend(["【问题】", f"  {_dash_or(data.get('question'), empty='暂无')}", ""])
    lines.extend(["【观点】", f"  {_dash_or(data.get('thought'), empty='暂无')}", ""])

    impact = _as_dict(data.get("spectrum_impact"))
    impact_parts = [
        f"{_SPECTRUM_LABELS[k]}{_fmt_delta(_as_int(impact.get(k)))}"
        for k in ("sincerity", "engagement", "closeness", "directness")
        if _as_int(impact.get(k)) != 0
    ]
    lines.append("【光谱影响】")
    lines.append(f"  {' · '.join(impact_parts) if impact_parts else '无'}")
    lines.append("")

    evidence = _as_list(data.get("evidence"))
    lines.append("【证据】")
    if not evidence:
        lines.append("  暂无")
    else:
        for ev in evidence:
            text = str(ev).strip()
            if text:
                lines.append(f"  · {text}")

    lines.append("")
    lines.append("【思想关联】")
    edges = _as_list(data.get("edges"))
    if not edges:
        lines.append("  暂无")
    else:
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            label = _dash_or(edge.get("label"), empty="—")
            target = _dash_or(edge.get("target"), empty="—")
            lines.append(f"  · {label} → {target}")

    return "\n".join(lines)


def build_inspect_text(data: dict) -> str:
    """inspect 命中纯文本降级版。"""
    query_text = _dash_or(data.get("query_text"), empty="（空）")
    stream_id = str(data.get("stream_id") or "").strip()
    perspective = "全局" if not stream_id else stream_id
    mode_key = str(data.get("selection_mode") or "spectrum_only")
    mode_label = _SELECTION_MODE_LABELS.get(mode_key, mode_key)
    max_traits = _as_int(data.get("max_traits"))
    total_active = _as_int(data.get("total_active"))

    lines: list[str] = [
        "◎ 注入命中预览",
        f"视角 {perspective} · 模式 {mode_label}",
        f"上限 {max_traits} · 活跃总数 {total_active}",
        "",
        "【待测文本】",
        f"  {query_text}",
        "",
        "【命中】",
    ]

    selected = _as_list(data.get("selected"))
    if not selected:
        lines.append("  无命中（将走 fallback 或仅光谱注入）")
    else:
        for idx, item in enumerate(selected, start=1):
            if not isinstance(item, dict):
                continue
            name = _dash_or(item.get("name"), empty="—")
            layer = _dash_or(item.get("layer_label"), empty="—")
            lc = _dash_or(item.get("lifecycle_label"), empty="—")
            conf = int(round(_as_float(item.get("confidence")) * 100))
            quality = _as_float(item.get("quality_score"))
            tags = _as_list(item.get("matched_tags"))
            tag_s = " ".join(str(t).strip() for t in tags if str(t).strip()) if tags else "无标签"
            thought = _dash_or(item.get("thought"), empty="—")
            lines.append(
                f"  #{idx} {name} [{layer}/{lc}] 置信{conf}% 质量{quality:.2f} 标签:{tag_s}"
            )
            lines.append(f"    {thought}")

    lines.extend(["", "【跳过】"])
    skipped = _as_list(data.get("skipped"))
    if not skipped:
        lines.append("  无")
    else:
        for item in skipped:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"  · {_dash_or(item.get('name'), empty='—')} — {_dash_or(item.get('reason'), empty='—')}"
            )

    return "\n".join(lines)