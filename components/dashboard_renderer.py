"""Soul 引擎全状态可视化卡片：HTML 内联样式 + html2png 渲染。"""

from __future__ import annotations

import logging
from html import escape
from typing import Any

_logger = logging.getLogger(__name__)

_ROOT_ID = "soul-dashboard"

_SPECTRUM_LABELS: dict[str, str] = {
    "sincerity": "真诚",
    "engagement": "投入",
    "closeness": "亲近",
    "directness": "直率",
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

    async def render(self, data: dict) -> str:
        """把 dashboard 数据渲染为 PNG base64；失败返回空串。"""
        try:
            html = self._build_html(data)
            return await self._render_html(html)
        except Exception:
            self._log_exception("Soul dashboard 渲染失败")
            return ""

    async def _render_html(self, html: str) -> str:
        if self._ctx is None:
            return ""
        try:
            result = await self._ctx.render.html2png(
                html,
                selector=f"#{_ROOT_ID}",
                viewport={"width": self.viewport_width, "height": 1600},
                device_scale_factor=self.device_scale_factor,
                full_page=False,
                omit_background=False,
                wait_until="load",
                timeout_ms=self.render_timeout_ms,
                allow_network=False,
            )
        except Exception:
            self._log_exception("html2png 调用失败")
            return ""
        if isinstance(result, dict):
            image_base64 = result.get("image_base64")
            return image_base64 if isinstance(image_base64, str) else ""
        return ""

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
        perspective = "全局" if not stream_id else stream_id

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
            <div class="hero-main">
              <div class="eyebrow">Mai Soul Engine</div>
              <h1>Soul 引擎状态</h1>
              <p class="meta">生成于 {escape(generated_at)} · 视角 {escape(perspective)}</p>
              {init_badge}
            </div>
            <div class="hero-glyph" aria-hidden="true">◈</div>
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
            {slice_block or '<div class="panel panel-dim"><div class="label">本群切片</div><div class="empty">全局视角不显示群切片</div></div>'}
            {thought_block}
          </section>

          <section class="panel">
            <div class="label">最近演化</div>
            {evolution_block}
          </section>

          <section class="panel panel-flags">
            <div class="label">功能开关</div>
            {flags_block}
          </section>
        </article>
        """
        return self._wrap_html(body)

    def _render_spectrum_bars(self, spectrum: dict[str, Any]) -> str:
        rows: list[str] = []
        for key in ("sincerity", "engagement", "closeness", "directness"):
            val = _clamp(_as_int(spectrum.get(key)), 0, 100)
            label = _SPECTRUM_LABELS[key]
            rows.append(
                f"""
                <div class="bar-row">
                  <span class="bar-name">{escape(label)}</span>
                  <div class="bar-track"><div class="bar-fill" style="width:{val}%"></div></div>
                  <span class="bar-val">{val}</span>
                </div>
                """
            )
        return '<div class="bars">' + "".join(rows) + "</div>"

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
        for key in _LIFECYCLE_LABELS:
            count = _as_int(lifecycle.get(key))
            css = _LIFECYCLE_CSS[key]
            chips.append(
                f'<span class="chip {css}">{escape(_LIFECYCLE_LABELS[key])} <b>{count}</b></span>'
            )
        total_line = f'<p class="trait-total">Trait 总数 <strong>{trait_total}</strong></p>'
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
                    <div class="bipolar-fill" style="left:{pct}%"></div>
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
            return '<div class="empty">暂无演化记录</div>'
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
                <div class="evo-card">
                  <div class="evo-head">
                    <span>群 {escape(_dash_or(item.get("group_id")))}</span>
                    <span class="muted">{escape(_dash_or(item.get("timestamp")))}</span>
                  </div>
                  <div class="evo-deltas">{escape(delta_text)}</div>
                  <p>{escape(reason)}</p>
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

    @staticmethod
    def _wrap_html(body: str) -> str:
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<style>
:root {{
  --ink: #e8eaf6;
  --ink-soft: rgba(232, 234, 246, 0.72);
  --bg-deep: #12131f;
  --card: rgba(28, 30, 48, 0.94);
  --line: rgba(140, 150, 220, 0.22);
  --accent: #7c6cf0;
  --accent-2: #3dd6c5;
  --warn: #f0a04b;
  --bad: #e85d6c;
  --good: #5fd38d;
  --shadow: 0 28px 90px rgba(8, 10, 24, 0.55);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 32px;
  color: var(--ink);
  font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
  background:
    radial-gradient(circle at 12% 8%, rgba(124, 108, 240, 0.35), transparent 38%),
    radial-gradient(circle at 88% 12%, rgba(61, 214, 197, 0.22), transparent 32%),
    linear-gradient(145deg, #0d0e18 0%, #171a2b 55%, #10121c 100%);
}}
.dash {{
  width: 100%;
  max-width: 100%;
  margin: 0 auto;
  padding: 28px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 28px;
  background: linear-gradient(180deg, rgba(255,255,255,0.04), transparent), var(--card);
  box-shadow: var(--shadow);
}}
.hero {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 20px;
  padding: 26px 28px;
  border-radius: 22px;
  background: linear-gradient(120deg, rgba(124,108,240,0.28), rgba(61,214,197,0.12));
  border: 1px solid var(--line);
  margin-bottom: 18px;
}}
.eyebrow {{
  font-size: 13px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--accent-2);
}}
h1 {{
  margin: 6px 0 8px;
  font-size: 36px;
  letter-spacing: -0.03em;
}}
.meta {{
  margin: 0 0 12px;
  font-size: 15px;
  color: var(--ink-soft);
}}
.hero-glyph {{
  font-size: 42px;
  color: var(--accent);
  opacity: 0.85;
}}
.badge {{
  display: inline-flex;
  padding: 8px 14px;
  border-radius: 999px;
  font-size: 14px;
  font-weight: 700;
}}
.badge-ok {{
  background: rgba(95, 211, 141, 0.18);
  color: #b8f5d0;
  border: 1px solid rgba(95, 211, 141, 0.35);
}}
.badge-alert {{
  background: rgba(232, 93, 108, 0.2);
  color: #ffc4cb;
  border: 1px solid rgba(232, 93, 108, 0.45);
}}
.grid {{
  display: grid;
  gap: 16px;
  margin-bottom: 16px;
}}
.grid.two {{ grid-template-columns: 1fr 1fr; }}
.grid.three {{ grid-template-columns: 1fr 1fr 1fr; }}
.panel {{
  padding: 20px 22px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--line);
}}
.panel-dim {{ opacity: 0.92; }}
.label {{
  margin-bottom: 12px;
  font-size: 14px;
  font-weight: 800;
  letter-spacing: 0.14em;
  color: var(--accent-2);
}}
.label.sub {{ margin-top: 14px; }}
.bars {{ display: grid; gap: 12px; }}
.bar-row {{
  display: grid;
  grid-template-columns: 52px 1fr 40px;
  gap: 10px;
  align-items: center;
}}
.bar-name {{ font-size: 14px; color: var(--ink-soft); }}
.bar-track {{
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  overflow: hidden;
}}
.bar-fill {{
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
}}
.bar-val {{ font-weight: 700; font-size: 15px; text-align: right; }}
.stat-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
}}
.stat-card {{
  padding: 14px;
  border-radius: 16px;
  background: rgba(124, 108, 240, 0.12);
  border: 1px solid rgba(124, 108, 240, 0.25);
  text-align: center;
}}
.stat-card strong {{
  display: block;
  font-size: 28px;
  color: #c4b8ff;
}}
.stat-card span {{ font-size: 13px; color: var(--ink-soft); }}
.divider {{
  height: 1px;
  margin: 14px 0;
  background: var(--line);
}}
.chips {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}}
.chip {{
  display: inline-flex;
  gap: 6px;
  padding: 7px 11px;
  border-radius: 999px;
  font-size: 13px;
  border: 1px solid var(--line);
}}
.chip b {{ font-weight: 800; }}
.lc-active {{ background: rgba(95,211,141,0.12); color: #b8f5d0; }}
.lc-strong {{ background: rgba(61,214,197,0.14); color: #aaf5ec; }}
.lc-expired {{ background: rgba(160,170,200,0.12); color: #c5cbe0; }}
.lc-bad {{ background: rgba(232,93,108,0.16); color: #ffc4cb; }}
.lc-warn {{ background: rgba(240,160,75,0.16); color: #ffe0b8; }}
.lc-revised {{ background: rgba(124,108,240,0.16); color: #d5ccff; }}
.trait-total {{
  margin: 0 0 10px;
  font-size: 14px;
  color: var(--ink-soft);
}}
.trait-total strong {{ color: var(--ink); font-size: 18px; }}
.bipolar {{ display: grid; gap: 10px; }}
.bipolar-row {{
  display: grid;
  grid-template-columns: 52px 1fr 40px;
  gap: 10px;
  align-items: center;
}}
.bipolar-track {{
  position: relative;
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
}}
.bipolar-mid {{
  position: absolute;
  left: 50%;
  top: -2px;
  width: 2px;
  height: 14px;
  background: rgba(255,255,255,0.35);
  transform: translateX(-50%);
}}
.bipolar-fill {{
  position: absolute;
  top: 1px;
  width: 8px;
  height: 8px;
  margin-left: -4px;
  border-radius: 50%;
  background: var(--accent-2);
  box-shadow: 0 0 10px rgba(61,214,197,0.6);
}}
.slice-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.slice-row {{
  display: flex;
  justify-content: space-between;
  padding: 8px 10px;
  border-radius: 12px;
  background: rgba(255,255,255,0.04);
  font-size: 14px;
}}
.thought-box {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 88px;
}}
.big-num {{
  font-size: 40px;
  font-weight: 800;
  color: #c4b8ff;
  line-height: 1;
}}
.big-label {{ font-size: 14px; color: var(--ink-soft); margin-top: 6px; }}
.evo-stack {{ display: grid; gap: 10px; }}
.evo-card {{
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--line);
}}
.evo-head {{
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: var(--ink-soft);
  margin-bottom: 6px;
}}
.evo-deltas {{
  font-size: 14px;
  font-weight: 700;
  color: #d5ccff;
  margin-bottom: 6px;
}}
.evo-card p {{
  margin: 0;
  font-size: 15px;
  line-height: 1.5;
  color: var(--ink);
}}
.muted {{ opacity: 0.85; }}
.footnote {{
  margin: 12px 0 0;
  font-size: 12px;
  color: var(--ink-soft);
}}
.empty {{
  font-size: 15px;
  color: var(--ink-soft);
  padding: 8px 0;
}}
.flags {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}}
.flag {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 13px;
  border: 1px solid var(--line);
}}
.flag.on {{
  background: rgba(95, 211, 141, 0.12);
  border-color: rgba(95, 211, 141, 0.35);
}}
.flag.off {{
  background: rgba(255,255,255,0.04);
  opacity: 0.85;
}}
.flag-name {{ color: var(--ink-soft); }}
.flag-state {{ font-weight: 800; }}
.panel-flags {{ margin-bottom: 0; }}
</style>
</head>
<body>{body}</body>
</html>"""


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
        lines.append(f"  {_SPECTRUM_LABELS[key]} {_as_int(spectrum.get(key))}")
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