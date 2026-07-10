"""Soul 运行观察命令（管理员）— 开发用状态摘要。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.spectrum_utils import check_admin_permission


async def handle_observe(plugin: Any, stream_id: str, **kwargs: Any) -> tuple[bool, str, bool]:
    """汇总光谱 / 队列 / 最近 audit / 最近注入。"""
    ok, err = check_admin_permission(plugin, kwargs, "查看运行观察")
    if not ok:
        await plugin.ctx.send.text(err, stream_id)
        return True, err, True

    from ..models.ideology_model import (
        count_pending_thought_seeds,
        get_or_create_spectrum,
        query_crystallized_traits,
    )
    from ..models.self_reflection import count_pending_reflections, count_self_reflections
    from ..utils.audit_log import init_audit_log, read_recent_audit

    spectrum = get_or_create_spectrum("global")
    lines: list[str] = []
    lines.append("◈ Soul 运行观察")
    lines.append(
        f"光谱 initialized={bool(spectrum.initialized)} "
        f"S{spectrum.sincerity}/E{spectrum.engagement}/C{spectrum.closeness}/D{spectrum.directness}"
    )
    lines.append(f"上次演化: {spectrum.last_evolution or '-'}")

    try:
        pending_seeds = int(count_pending_thought_seeds() or 0)
    except Exception:
        pending_seeds = -1
    lines.append(f"待审种子: {pending_seeds}")

    try:
        traits = query_crystallized_traits(deleted=False, limit=500)
        active = sum(1 for t in traits if getattr(t, "enabled", True))
        lines.append(f"Trait: total={len(traits)} enabled={active}")
    except Exception as exc:
        lines.append(f"Trait: (读失败 {exc})")

    try:
        pr = count_pending_reflections()
        sr = count_self_reflections()
        lines.append(f"自评 pending={pr} reflections={sr}")
    except Exception as exc:
        lines.append(f"自评: (读失败 {exc})")

    init_audit_log(Path(plugin._plugin_dir))
    events = read_recent_audit(12)
    if events:
        lines.append("—— 最近 audit ——")
        for e in events[-8:]:
            ts = str(e.get("ts", ""))[:19]
            typ = e.get("type", "?")
            extra = {
                k: v
                for k, v in e.items()
                if k not in {"ts", "type", "before", "after"} and v not in (None, "", [], {})
            }
            brief = json.dumps(extra, ensure_ascii=False, default=str)
            if len(brief) > 120:
                brief = brief[:117] + "..."
            lines.append(f"{ts} · {typ} {brief}")
    else:
        lines.append("—— audit 暂无（演化/自评跑过后会出现）——")

    inj_path = Path(plugin._plugin_dir) / "data" / "injections.jsonl"
    if inj_path.exists():
        try:
            inj_lines = inj_path.read_text(encoding="utf-8").splitlines()
            tail = [ln for ln in inj_lines[-5:] if ln.strip()]
            if tail:
                lines.append("—— 最近 inject ——")
                for ln in tail:
                    try:
                        obj = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    ts = str(obj.get("ts", ""))[:19]
                    pol = obj.get("policy", "")
                    mode = obj.get("selection_mode", "")
                    picked = obj.get("picked") or []
                    lines.append(f"{ts} · {pol}/{mode} picked={len(picked)}")
        except OSError:
            pass

    msg = chr(10).join(lines)
    await plugin.ctx.send.text(msg, stream_id)
    return True, msg, True
