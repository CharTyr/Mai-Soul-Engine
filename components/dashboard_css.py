"""Dashboard card CSS. PLAIN string only — never convert this to an f-string."""

_DASHBOARD_CSS = """
:root {
  --ink: #14161a;
  --body: #3a3f47;
  --mute: #6b7280;
  --ash: #9aa1ab;
  --canvas: #f3f5f8;
  --surface: #ffffff;
  --surface-elevated: #f8fafc;
  --surface-card: #e8eef6;
  --hairline: #e2e8f0;
  --hairline-soft: rgba(15,23,42,0.06);
  --hairline-strong: rgba(15,23,42,0.14);
  --accent-blue: #2563eb;
  --accent-blue-soft: rgba(37,99,235,0.14);
  --accent-red: #dc2626;
  --accent-red-soft: rgba(220,38,38,0.10);
  --accent-green: #059669;
  --accent-green-soft: rgba(5,150,105,0.12);
  --accent-yellow: #d97706;
  --accent-yellow-soft: rgba(217,119,6,0.12);
  --hero-stripe-start: #60a5fa;
  --hero-stripe-end: #2563eb;
  --key-bg-start: #ffffff;
  --key-bg-end: #f1f5f9;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  background: var(--canvas);
}
style { display: none !important; }
body {
  margin: 0;
  padding: 20px;
  color: var(--body);
  font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
  font-size: 15px;
  line-height: 1.5;
  background: var(--canvas);
}
.dash {
  width: 100%;
  max-width: 100%;
  margin: 0 auto;
  padding: 14px;
  border: 1px solid var(--hairline);
  border-radius: 16px;
  background: var(--surface);
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}
.hero {
  position: relative;
  overflow: hidden;
  margin-bottom: 12px;
  border: 1px solid var(--hairline);
  border-radius: 10px;
  background: var(--surface-elevated);
}
.hero-stripe {
  height: 6px;
  background:
    repeating-linear-gradient(
      105deg,
      var(--hero-stripe-start) 0px,
      var(--hero-stripe-end) 28px,
      transparent 28px,
      transparent 36px
    ),
    linear-gradient(90deg, var(--hero-stripe-start), var(--hero-stripe-end));
}
.hero-inner { padding: 12px 16px; }
.hero-compact .hero-inner { padding: 10px 14px; }
.hero-main { min-width: 0; }
.eyebrow {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.4px;
  text-transform: uppercase;
  color: var(--mute);
}
h1 {
  margin: 4px 0 6px;
  font-size: 22px;
  font-weight: 600;
  line-height: 1.2;
  color: var(--ink);
}
.title-trait { font-size: 20px; }
.meta {
  margin: 0 0 8px;
  font-size: 12px;
  color: var(--mute);
  line-height: 1.45;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 2px 0;
}
.badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  border: 1px solid var(--hairline);
}
.badge-ok {
  background: var(--accent-green-soft);
  color: var(--accent-green);
  border-color: transparent;
}
.badge-alert {
  background: var(--accent-red-soft);
  color: var(--accent-red);
  border-color: transparent;
}
.badge-muted {
  background: var(--surface-card);
  color: var(--ash);
}
.badge-layer, .badge-mode {
  background: var(--accent-blue-soft);
  color: var(--accent-blue);
  border-color: transparent;
}
.badge-slot {
  background: var(--accent-yellow-soft);
  color: var(--accent-yellow);
  border-color: transparent;
}
.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.grid {
  display: grid;
  gap: 10px;
  margin-bottom: 12px;
}
.grid.two { grid-template-columns: 1fr 1fr; }
.grid.three { grid-template-columns: 1fr 1fr 1fr; }
.panel {
  padding: 12px;
  border-radius: 10px;
  background: var(--surface);
  border: 1px solid var(--hairline);
}
.panel-dim { opacity: 0.95; }
.panel-accent {
  background: var(--surface-elevated);
  border-color: var(--hairline-strong);
}
.panel-flags { margin-bottom: 0; margin-top: 0; }
.label {
  margin-bottom: 8px;
  font-size: 13px;
  font-weight: 600;
  color: var(--mute);
}
.label.sub { margin-top: 10px; }
.bars, .bipolar { display: grid; gap: 10px; }
.bar-row, .bipolar-row {
  display: grid;
  grid-template-columns: 48px minmax(0, 1fr) 42px;
  gap: 8px;
  align-items: center;
}
.bar-row-wide { grid-template-columns: 48px minmax(0, 1fr) 52px; }
.bar-name { font-size: 13px; color: var(--mute); font-weight: 500; }
.bar-track {
  height: 16px;
  border-radius: 999px;
  background-color: #e8eef6;
  border: 1px solid var(--hairline);
  overflow: hidden;
  position: relative;
}
.bipolar-track {
  height: 16px;
  border-radius: 999px;
  background: #e8eef6;
  border: 1px solid var(--hairline);
  overflow: visible;
  position: relative;
}
.bar-fill {
  display: none;
}
.bar-val {
  font-weight: 700;
  font-size: 15px;
  text-align: right;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}

.axes { display: grid; gap: 12px; }
.axis-row { display: grid; gap: 6px; }
.axis-head {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 8px;
  align-items: center;
}
.axis-left {
  font-size: 12px;
  color: var(--mute);
  text-align: left;
  line-height: 1.3;
}
.axis-right {
  font-size: 12px;
  color: var(--mute);
  text-align: right;
  line-height: 1.3;
}
.axis-val {
  font-size: 14px;
  font-weight: 700;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
  min-width: 2.2em;
  text-align: center;
}
.axis-track {
  position: relative;
  height: 12px;
  border-radius: 999px;
  background: #e8eef6;
  border: 1px solid var(--hairline);
  overflow: visible;
}
.axis-mid {
  position: absolute;
  left: 50%;
  top: -3px;
  width: 2px;
  height: 18px;
  background: var(--hairline-strong);
  transform: translateX(-50%);
  opacity: 0.7;
}
.axis-fill {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  border-radius: 999px;
  background: linear-gradient(90deg, #93c5fd, #2563eb);
  min-width: 0;
}
.axis-knob {
  position: absolute;
  top: 50%;
  width: 14px;
  height: 14px;
  margin-left: -7px;
  margin-top: -7px;
  border-radius: 50%;
  background: #2563eb;
  border: 2px solid #fff;
  box-shadow: 0 0 0 1px rgba(37,99,235,0.35);
}
.bipolar-track { overflow: visible; }
.bipolar-mid {
  position: absolute;
  left: 50%;
  top: -3px;
  width: 2px;
  height: 20px;
  background: var(--hairline-strong);
  transform: translateX(-50%);
}
.bipolar-fill {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 14px;
  height: 14px;
  margin-top: -7px;
  margin-left: -7px;
  border-radius: 50%;
  background: #2563eb;
  border: 2px solid #fff;
  box-shadow: 0 0 0 1px rgba(37,99,235,0.35);
  transform: none;
}
.stat-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}
.stat-card {
  padding: 10px 8px;
  border-radius: 8px;
  background: var(--surface-elevated);
  border: 1px solid var(--hairline);
  text-align: center;
}
.stat-card strong {
  display: block;
  font-size: 24px;
  font-weight: 700;
  color: var(--ink);
  line-height: 1.1;
}
.stat-card span { font-size: 12px; color: var(--mute); }
.divider {
  height: 1px;
  margin: 10px 0;
  background: var(--hairline);
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.chip {
  display: inline-flex;
  gap: 4px;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  border: 1px solid transparent;
}
.chip b { font-weight: 700; }
.lc-active, .lc-strong { background: var(--accent-green-soft); color: var(--accent-green); }
.lc-expired, .lc-warn { background: var(--accent-yellow-soft); color: var(--accent-yellow); }
.lc-bad { background: var(--accent-red-soft); color: var(--accent-red); }
.lc-revised { background: var(--accent-blue-soft); color: var(--accent-blue); }
.trait-total {
  margin: 0 0 8px;
  font-size: 13px;
  color: var(--mute);
}
.trait-total strong { color: var(--ink); font-size: 15px; font-weight: 700; }
.slice-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.slice-row {
  display: flex;
  justify-content: space-between;
  padding: 6px 10px;
  border-radius: 6px;
  background: var(--surface-card);
  font-size: 13px;
  color: var(--body);
}
.thought-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 84px;
}
.big-num {
  font-size: 36px;
  font-weight: 700;
  color: var(--ink);
  line-height: 1;
}
.big-label { font-size: 13px; color: var(--mute); margin-top: 4px; }
.slot-item { font-size: 13px; color: var(--body); padding: 3px 6px; }
.slot-item.muted { color: var(--ash); font-size: 12px; }
.evo-stack { display: grid; gap: 6px; }
.palette-row {
  padding: 8px 10px;
  border-radius: 8px;
  background: transparent;
  border: 1px solid transparent;
}
.palette-row-active {
  background: var(--surface-elevated);
  border-color: var(--hairline);
}
.evo-head {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--mute);
  margin-bottom: 4px;
  gap: 8px;
}
.evo-deltas {
  font-size: 13px;
  font-weight: 600;
  color: var(--accent-blue);
  margin-bottom: 4px;
}
.row-detail {
  margin: 0;
  font-size: 13px;
  line-height: 1.45;
  color: var(--body);
}
.muted { color: var(--mute); }
.footnote {
  margin: 8px 0 0;
  font-size: 12px;
  color: var(--mute);
}
.footnote-block { margin: 0 4px 4px; padding: 0 4px; }
.empty {
  font-size: 13px;
  color: var(--mute);
  padding: 8px 0;
}
.empty-inline { font-size: 13px; color: var(--mute); }
.flags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.flag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  border: 1px solid var(--hairline);
  background: var(--surface-card);
  color: var(--mute);
}
.flag.on {
  background: #d1fae5;
  color: #047857;
  border-color: #a7f3d0;
}
.flag.off {
  background: #eef1f5;
  color: #6b7280;
  border-color: #e5e7eb;
}
.flag-name { opacity: 0.95; }
.flag-state { font-weight: 700; }
.tags { display: flex; flex-wrap: wrap; gap: 6px; }
.keycap, .tag {
  display: inline-flex;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 12px;
  color: var(--body);
  background: linear-gradient(180deg, var(--key-bg-start), var(--key-bg-end));
  border: 1px solid var(--hairline);
}
.keycap-hit, .tag-hit {
  color: var(--accent-green);
  border-color: var(--hairline-strong);
}
.edge-stack, .hit-stack, .skip-stack { display: grid; gap: 6px; }
.edge-palette-row, .hit-card {
  display: grid;
  gap: 4px;
}
.edge-badge {
  display: inline-flex;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
}
.edge-target { color: var(--body); font-size: 13px; }
.hit-head {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.hit-rank { color: var(--accent-blue); font-weight: 700; }
.hit-name { font-weight: 600; color: var(--ink); }
.hit-id, .hit-metric { font-size: 12px; color: var(--mute); }
.hit-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.hit-thought { margin: 0; font-size: 13px; line-height: 1.5; color: var(--mute); }
.skip-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 6px 10px;
  border-radius: 6px;
  background: transparent;
  font-size: 13px;
}
.skip-name { color: var(--ash); }
.skip-reason { color: var(--accent-yellow); font-weight: 600; text-align: right; }
.query-block, .list-body, .body-text, .thought-body {
  margin: 0;
  font-size: 14px;
  line-height: 1.55;
  color: var(--body);
}
.impact-row { display: flex; flex-wrap: wrap; gap: 6px; }
.impact-chip {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--accent-blue-soft);
  color: var(--body);
  font-size: 12px;
}
.impact-chip strong { color: var(--accent-blue); margin-left: 4px; font-weight: 700; }

.empty-state {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 4px;
  min-height: 56px;
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--surface-elevated);
  border: 1px dashed var(--hairline);
}
.empty-state-compact {
  min-height: 48px;
  margin-top: 4px;
}
.empty-state-title {
  font-size: 13px;
  font-weight: 700;
  color: var(--body);
}
.empty-state-desc {
  font-size: 12px;
  color: var(--mute);
  line-height: 1.45;
}
.bottom-stack {
  gap: 14px;
  margin-top: 4px;
}
.panel-bottom .label {
  margin-bottom: 10px;
}
.placeholder-kicker {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--ash);
}
.placeholder-title {
  margin-top: 2px;
}
.dash > .grid,
.dash > .bottom-stack {
  width: 100%;
}

.meta-sep { color: var(--ash); }
.meta-id {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  font-weight: 600;
  color: var(--body);
  background: var(--surface-card);
  border: 1px solid var(--hairline);
  border-radius: 6px;
  padding: 1px 6px;
}
.grid.three {
  align-items: stretch;
}
.grid.three > .panel {
  min-height: 168px;
  display: flex;
  flex-direction: column;
}
.grid.three > .panel > .label {
  flex: 0 0 auto;
}
.grid.three > .panel > .bipolar,
.grid.three > .panel > .thought-box,
.grid.three > .panel > .placeholder-body,
.grid.three > .panel > .slice-grid {
  flex: 1 1 auto;
}
.panel-placeholder .placeholder-body {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 6px;
  min-height: 110px;
  padding: 4px 2px 2px;
}
.placeholder-title {
  font-size: 16px;
  font-weight: 700;
  color: var(--ink);
}
.placeholder-desc {
  margin: 0;
  font-size: 13px;
  line-height: 1.5;
  color: var(--body);
}
.placeholder-hint {
  margin: 0;
  font-size: 12px;
  line-height: 1.45;
  color: var(--mute);
}
.bottom-stack {
  display: grid;
  gap: 14px;
  margin-top: 4px;
}
.bottom-stack > .panel {
  margin: 0;
}
.panel-flags .flags {
  row-gap: 8px;
}
.empty {
  min-height: 28px;
}
"""
