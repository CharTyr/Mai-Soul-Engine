import os
import secrets
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException


SCHEMA_VERSION = 142


@dataclass(frozen=True)
class SoulApiConfig:
    enabled: bool
    token: str


_START_TIME = time.time()
_LAST_INJECTION: dict[str, Any] = {}
_LAST_INJECTION_BY_STREAM: dict[str, dict[str, Any]] = {}
_INJECTION_HISTORY: deque[dict[str, Any]] = deque(maxlen=400)
_INJECTION_HISTORY_BY_STREAM: dict[str, deque[dict[str, Any]]] = {}
_INJECTION_LOG_LOCK = Lock()


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with _INJECTION_LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _trim_jsonl(path: Path, *, max_bytes: int, max_lines: int) -> None:
    if max_bytes <= 0 or max_lines <= 0:
        return
    try:
        size = int(path.stat().st_size)
    except FileNotFoundError:
        return
    except Exception:
        return
    if size <= max_bytes:
        return

    tail: deque[str] = deque(maxlen=max_lines)
    with _INJECTION_LOG_LOCK:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tail.append(line)

        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for line in tail:
                f.write(line + "\n")
        tmp.replace(path)


def record_last_injection(payload: dict[str, Any], *, stream_id: str | None = None, plugin_dir: Path | None = None) -> None:
    _LAST_INJECTION.clear()
    _LAST_INJECTION.update(payload)

    sid = stream_id or "global"
    if len(_LAST_INJECTION_BY_STREAM) > 256:
        _LAST_INJECTION_BY_STREAM.clear()
    _LAST_INJECTION_BY_STREAM[sid] = dict(payload)

    entry = dict(payload)
    entry.setdefault("ts", _now_iso())
    entry["stream_id"] = sid
    _INJECTION_HISTORY.append(entry)
    if len(_INJECTION_HISTORY_BY_STREAM) > 512:
        _INJECTION_HISTORY_BY_STREAM.clear()
    bucket = _INJECTION_HISTORY_BY_STREAM.get(sid)
    if bucket is None:
        bucket = deque(maxlen=400)
        _INJECTION_HISTORY_BY_STREAM[sid] = bucket
    bucket.append(entry)

    if plugin_dir:
        try:
            path = plugin_dir / "data" / "injections.jsonl"
            _append_jsonl(path, entry)
            _trim_jsonl(path, max_bytes=2_000_000, max_lines=4_000)
        except Exception:
            pass


def _format_uptime(seconds: float) -> str:
    if seconds < 0:
        return "0h 0m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def _now_iso() -> str:
    return datetime.now().isoformat()


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _value_0_100_to_signed(value: int) -> int:
    return _clamp_int(int(round((value - 50) * 2)), -100, 100)


def _impact_dict_to_effects(impact: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """将 impact dict（economic/social/diplomatic/progressive）转换为前端所需 impacts 列表与 source_dimension。"""
    impacts: list[dict[str, Any]] = []
    axis_map = [
        ("sincerity-absurdism", impact.get("economic", 0), "sincerity", "absurdism"),
        ("normies-otakuism", impact.get("social", 0), "normies", "otakuism"),
        ("traditionalism-radicalism", impact.get("diplomatic", 0), "traditionalism", "radicalism"),
        ("heroism-nihilism", impact.get("progressive", 0), "heroism", "nihilism"),
    ]

    strongest: tuple[str, int, str, str] | None = None
    for axis, raw_delta, left_pole, right_pole in axis_map:
        try:
            delta = int(raw_delta)
        except Exception:
            continue
        if strongest is None or abs(delta) > abs(strongest[1]):
            strongest = (axis, delta, left_pole, right_pole)
        if delta == 0:
            continue
        impacts.append(
            {
                "dimension": axis,
                "direction": "right" if delta > 0 else "left",
                "strength": _clamp_int(delta, -10, 10),
            }
        )

    source_dimension = strongest[3] if strongest and strongest[1] > 0 else strongest[2] if strongest else "sincerity"
    return impacts, source_dimension


def _get_api_config(plugin) -> SoulApiConfig:
    enabled = bool(plugin.get_config("api.enabled", True))

    token = str(plugin.get_config("api.token", "") or "").strip()
    env_token = os.getenv("SOUL_API_TOKEN", "").strip()
    if env_token:
        token = env_token

    return SoulApiConfig(enabled=enabled, token=token)


def _require_token(plugin, token: str | None) -> None:
    cfg = _get_api_config(plugin)
    if not cfg.enabled:
        raise HTTPException(status_code=404, detail="Soul API disabled")
    if cfg.token and cfg.token != (token or ""):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _read_audit_fragments(plugin_dir: Path, limit: int, stream_id: str | None = None) -> list[dict[str, Any]]:
    audit_path = plugin_dir / "data" / "audit.jsonl"
    if not audit_path.exists():
        return []

    lines: deque[str] = deque(maxlen=max(1, limit))
    with open(audit_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    import json

    fragments: list[dict[str, Any]] = []
    for idx, raw in enumerate(reversed(lines)):
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue

        ts = str(entry.get("ts") or _now_iso())
        typ = str(entry.get("type") or "event")
        group_id = str(entry.get("group_id") or "")
        reason = str(entry.get("reason") or "")
        deltas = entry.get("deltas") or {}

        if stream_id and stream_id != "global" and typ == "evolution" and group_id != stream_id:
            continue

        if typ == "evolution":
            delta_str = ", ".join(
                f"{k}:{int(v):+d}" for k, v in deltas.items() if isinstance(v, (int, float)) and int(v) != 0
            )
            content = f"演化记录 {group_id}：{reason or '无原因'}（{delta_str or '无变化'}）"
            tags = ["evolution", group_id] if group_id else ["evolution"]
        elif typ == "init":
            content = "完成意识形态光谱初始化"
            tags = ["init"]
        elif typ == "reset":
            content = "意识形态光谱已重置"
            tags = ["reset"]
        else:
            content = f"{typ}: {reason}" if reason else typ
            tags = [typ]

        fragments.append(
            {
                "id": f"audit-{ts}-{idx}",
                "content": content,
                "source": "introspection",
                "timestamp": ts,
                "tags": tags,
                "redacted": False,
            }
        )

    return fragments


def create_soul_api_router(plugin) -> APIRouter:
    router = APIRouter(prefix="/api/v1/soul", tags=["soul"])
    plugin_dir = Path(plugin.plugin_dir)

    def _public_mode() -> bool:
        try:
            return bool(plugin.get_config("api.public_mode", False))
        except Exception:
            return False

    def _sanitize_injection_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if not payload:
            return {}
        out = dict(payload)
        out["picked"] = []
        out.pop("cooldown_skipped", None)
        return out

    @router.get("/spectrum")
    async def get_spectrum(x_soul_token: str | None = Header(default=None, alias="X-Soul-Token")) -> dict[str, Any]:
        _require_token(plugin, x_soul_token)

        from ..models.ideology_model import get_or_create_spectrum, init_tables, EvolutionHistory

        init_tables()
        spectrum = get_or_create_spectrum("global")

        dims_raw = {
            "economic": _value_0_100_to_signed(spectrum.economic),
            "social": _value_0_100_to_signed(spectrum.social),
            "diplomatic": _value_0_100_to_signed(spectrum.diplomatic),
            "progressive": _value_0_100_to_signed(spectrum.progressive),
        }

        axis_map = [
            ("sincerity-absurdism", "sincerity", "absurdism", "economic"),
            ("normies-otakuism", "normies", "otakuism", "social"),
            ("traditionalism-radicalism", "traditionalism", "radicalism", "diplomatic"),
            ("heroism-nihilism", "heroism", "nihilism", "progressive"),
        ]

        recent = list(EvolutionHistory.select().order_by(EvolutionHistory.timestamp.desc()).limit(50))
        volatility: dict[str, float] = {k: 0.0 for k in dims_raw}
        if recent:
            for dim_key, field_name in [
                ("economic", "economic_delta"),
                ("social", "social_delta"),
                ("diplomatic", "diplomatic_delta"),
                ("progressive", "progressive_delta"),
            ]:
                values = [abs(int(getattr(r, field_name, 0) or 0)) for r in recent]
                volatility[dim_key] = round(sum(values) / max(1, len(values)) / 10.0, 3)

        scored_poles: list[tuple[int, str]] = []
        for axis, left_pole, right_pole, dim_key in axis_map:
            current = dims_raw[dim_key]
            pole = right_pole if current > 0 else left_pole
            scored_poles.append((abs(current), pole))

        scored_poles.sort(key=lambda x: x[0], reverse=True)
        primary = scored_poles[0][1] if scored_poles else "sincerity"
        secondary = scored_poles[1][1] if len(scored_poles) > 1 else None

        dominant_trait = f"主导倾向: {primary}" if primary else ""

        last_shift = spectrum.last_evolution.isoformat() if spectrum.last_evolution else _now_iso()
        major = next((r for r in recent if abs(r.economic_delta) + abs(r.social_delta) + abs(r.diplomatic_delta) + abs(r.progressive_delta) >= 8), None)
        if major and major.timestamp:
            last_shift = major.timestamp.isoformat()

        return {
            "dimensions": [
                {
                    "axis": axis,
                    "values": {"current": dims_raw[dim_key], "ema": dims_raw[dim_key], "baseline": 0},
                    "volatility": volatility.get(dim_key, 0.0),
                }
                for axis, _lp, _rp, dim_key in axis_map
            ],
            "base_tone": {"primary": primary, **({"secondary": secondary} if secondary else {})},
            "dominant_trait": dominant_trait,
            "last_major_shift": last_shift,
            "updated_at": spectrum.updated_at.isoformat() if spectrum.updated_at else _now_iso(),
        }

    @router.get("/cabinet")
    async def get_cabinet(
        stream_id: str | None = None, x_soul_token: str | None = Header(default=None, alias="X-Soul-Token")
    ) -> dict[str, Any]:
        _require_token(plugin, x_soul_token)
        public_mode = _public_mode()
        if public_mode:
            stream_id = None

        from ..models.ideology_model import ThoughtSeed, CrystallizedTrait, init_tables

        init_tables()
        seed_query = ThoughtSeed.select().where(ThoughtSeed.status == "pending")
        if stream_id and stream_id != "global":
            seed_query = seed_query.where(ThoughtSeed.stream_id == stream_id)
        seeds = list(seed_query.order_by(ThoughtSeed.created_at.desc()))

        import json

        slots: list[dict[str, Any]] = []
        for seed in seeds:
            try:
                impact = json.loads(seed.potential_impact_json or "{}")
            except json.JSONDecodeError:
                impact = {}

            impacts, source_dimension = _impact_dict_to_effects(impact)

            slots.append(
                {
                    "id": seed.seed_id,
                    "name": seed.seed_type,
                    "source_dimension": source_dimension,
                    "status": "internalizing",
                    "progress": _clamp_int(seed.intensity, 0, 100),
                    "conflict_reason": None,
                    "crystallized_at": None,
                    "introspection": (seed.reasoning or seed.event or "")[:200],
                    "impacts": impacts,
                    "debug_params": {"created_at": seed.created_at.isoformat() if seed.created_at else None},
                }
            )

        trait_query = CrystallizedTrait.select().where((CrystallizedTrait.deleted == False) & (CrystallizedTrait.enabled == True))  # noqa: E712
        if stream_id and stream_id != "global":
            trait_query = trait_query.where(CrystallizedTrait.stream_id == stream_id)
        traits = list(trait_query.order_by(CrystallizedTrait.created_at.desc()).limit(80))

        trait_cards: list[dict[str, Any]] = []
        for t in traits:
            try:
                impact = json.loads(t.spectrum_impact_json or "{}")
            except json.JSONDecodeError:
                impact = {}
            effects, source_dimension = _impact_dict_to_effects(impact)
            try:
                tags = json.loads(getattr(t, "tags_json", "[]") or "[]")
                if not isinstance(tags, list):
                    tags = []
                tags = [str(x) for x in tags if str(x).strip()]
            except Exception:
                tags = []
            try:
                from ..utils.trait_evidence import parse_evidence_json

                evidence = parse_evidence_json(getattr(t, "evidence_json", "[]") or "[]")
            except Exception:
                evidence = []
            try:
                confidence = float(getattr(t, "confidence", 0) or 0) / 100.0
            except Exception:
                confidence = 0.0
            trait_cards.append(
                {
                    "id": t.trait_id,
                    "name": t.name,
                    "source_dimension": source_dimension,
                    "crystallized_at": t.created_at.isoformat() if t.created_at else _now_iso(),
                    "description": t.thought or "",
                    "permanent_effects": effects,
                    "definition": getattr(t, "question", "") or "",
                    "digest": ", ".join(tags) if tags else "",
                    "tags": tags,
                    "confidence": confidence,
                    "evidence": evidence,
                }
            )

        if public_mode:
            for s in slots:
                s["introspection"] = "（已隐藏）"
                s["debug_params"] = None
            for t in trait_cards:
                t["evidence"] = []

        return {
            "slots": slots,
            "traits": trait_cards,
            "dissonance": {"active": False, "conflicting_thoughts": [], "severity": "low"},
        }

    @router.get("/trait_merge_suggestions")
    async def get_trait_merge_suggestions(
        limit: int = 20,
        threshold: float | None = None,
        stream_id: str | None = None,
        include_disabled: bool = False,
        x_soul_token: str | None = Header(default=None, alias="X-Soul-Token"),
    ) -> dict[str, Any]:
        _require_token(plugin, x_soul_token)
        if _public_mode():
            if threshold is None:
                try:
                    threshold = float(plugin.get_config("thought_cabinet.auto_dedup_threshold", 0.78))
                except Exception:
                    threshold = 0.78
            threshold = float(max(0.0, min(1.0, float(threshold))))
            return {"threshold": threshold, "suggestions": [], "updated_at": _now_iso()}

        from ..models.ideology_model import CrystallizedTrait, init_tables
        from ..utils.trait_tags import parse_tags_json

        init_tables()

        try:
            limit = int(limit)
        except Exception:
            limit = 20
        limit = max(1, min(200, limit))

        if threshold is None:
            try:
                threshold = float(plugin.get_config("thought_cabinet.auto_dedup_threshold", 0.78))
            except Exception:
                threshold = 0.78
        threshold = float(max(0.0, min(1.0, float(threshold))))

        query = CrystallizedTrait.select().where(CrystallizedTrait.deleted == False)  # noqa: E712
        if not include_disabled:
            query = query.where(CrystallizedTrait.enabled == True)  # noqa: E712

        requested_sid = str(stream_id or "").strip()
        if not requested_sid or requested_sid == "global":
            requested_sid = ""
        query = query.where(CrystallizedTrait.stream_id == requested_sid)

        traits = list(query.order_by(CrystallizedTrait.created_at.desc()).limit(140))

        import difflib

        def _sim_text(a: str, b: str, *, max_len: int) -> float:
            a = (a or "").strip()
            b = (b or "").strip()
            if not a or not b:
                return 0.0
            return float(difflib.SequenceMatcher(None, a[:max_len], b[:max_len]).ratio())

        def _jaccard(a: list[str], b: list[str]) -> float:
            sa = {x.casefold() for x in (a or []) if str(x).strip()}
            sb = {x.casefold() for x in (b or []) if str(x).strip()}
            if not sa or not sb:
                return 0.0
            return float(len(sa & sb) / len(sa | sb))

        def _confidence(t: CrystallizedTrait) -> float:
            try:
                return float(getattr(t, "confidence", 0) or 0) / 100.0
            except Exception:
                return 0.0

        def _tags(t: CrystallizedTrait) -> list[str]:
            return parse_tags_json(getattr(t, "tags_json", "[]") or "[]")

        suggestions: list[dict[str, Any]] = []
        scope_stream_id = requested_sid or "global"

        for i in range(len(traits)):
            a = traits[i]
            a_tags = _tags(a)
            for j in range(i + 1, len(traits)):
                b = traits[j]
                b_tags = _tags(b)

                tags_sim = _jaccard(a_tags, b_tags)
                thought_sim = _sim_text(a.thought or "", b.thought or "", max_len=420)
                question_sim = _sim_text(getattr(a, "question", "") or "", getattr(b, "question", "") or "", max_len=220)
                text_sim = max(thought_sim, question_sim)
                sim = (text_sim * 0.65) + (tags_sim * 0.35)

                shared = sorted({x for x in a_tags if x.casefold() in {y.casefold() for y in b_tags}})
                if sim < threshold:
                    continue
                if tags_sim <= 0.0 and text_sim < 0.92:
                    continue

                a_conf = _confidence(a)
                b_conf = _confidence(b)

                target, source = (a, b)
                if b_conf > a_conf:
                    target, source = (b, a)
                elif b_conf == a_conf:
                    a_ts = a.created_at.timestamp() if getattr(a, "created_at", None) else 0.0
                    b_ts = b.created_at.timestamp() if getattr(b, "created_at", None) else 0.0
                    target, source = (a, b) if a_ts <= b_ts else (b, a)

                suggestions.append(
                    {
                        "source_trait_id": source.trait_id,
                        "target_trait_id": target.trait_id,
                        "similarity": round(sim, 4),
                        "metrics": {"text": round(text_sim, 4), "tags": round(tags_sim, 4)},
                        "shared_tags": shared[:16],
                        "scope_stream_id": scope_stream_id,
                        "source": {
                            "id": source.trait_id,
                            "name": source.name,
                            "confidence": round(_confidence(source), 4),
                            "created_at": source.created_at.isoformat() if getattr(source, "created_at", None) else _now_iso(),
                            "tags": _tags(source),
                        },
                        "target": {
                            "id": target.trait_id,
                            "name": target.name,
                            "confidence": round(_confidence(target), 4),
                            "created_at": target.created_at.isoformat() if getattr(target, "created_at", None) else _now_iso(),
                            "tags": _tags(target),
                        },
                        "merge_command": f"/soul_trait_merge {source.trait_id} {target.trait_id}",
                    }
                )

        suggestions.sort(key=lambda x: (x.get("similarity", 0.0), x.get("scope_stream_id", "")), reverse=True)
        return {"threshold": threshold, "suggestions": suggestions[:limit], "updated_at": _now_iso()}

    @router.get("/introspection")
    async def get_introspection(
        limit: int = 60,
        stream_id: str | None = None,
        x_soul_token: str | None = Header(default=None, alias="X-Soul-Token"),
    ) -> dict[str, Any]:
        _require_token(plugin, x_soul_token)
        public_mode = _public_mode()
        if public_mode:
            stream_id = None

        fragments = _read_audit_fragments(plugin_dir, limit=limit, stream_id=stream_id)
        if public_mode:
            for f in fragments:
                f["content"] = "[REDACTED]"
                f["redacted"] = True
                tags = f.get("tags") or []
                if isinstance(tags, list) and tags:
                    f["tags"] = [str(tags[0])]
                else:
                    f["tags"] = []
        return {"fragments": fragments, "next_injection": None, "updated_at": _now_iso()}

    @router.get("/fragments")
    async def get_fragments(
        limit: int = 60, x_soul_token: str | None = Header(default=None, alias="X-Soul-Token")
    ) -> dict[str, Any]:
        return await get_introspection(limit=limit, x_soul_token=x_soul_token)

    @router.get("/pulse")
    async def get_pulse(x_soul_token: str | None = Header(default=None, alias="X-Soul-Token")) -> dict[str, Any]:
        _require_token(plugin, x_soul_token)

        evolution_enabled = bool(plugin.get_config("evolution.evolution_enabled", True))
        interval_hours = float(plugin.get_config("evolution.evolution_interval_hours", 1.0))

        last_run: Optional[str] = None
        try:
            from ..models.ideology_model import EvolutionHistory

            r = EvolutionHistory.select().order_by(EvolutionHistory.timestamp.desc()).first()
            if r and r.timestamp:
                last_run = r.timestamp.isoformat()
        except Exception:
            last_run = None

        return {
            "temperature": 0.72,
            "life_state": "active" if evolution_enabled else "idle",
            "status": "Mai-Soul-Engine 运行中",
            "uptime": _format_uptime(time.time() - _START_TIME),
            "heartbeat": int(time.time()),
            "dissonance": {"active": False},
            "introspection": {
                "enabled": evolution_enabled,
                "last_run": last_run,
                "next_run": None if not evolution_enabled else f"+{interval_hours}h",
            },
            "updated_at": _now_iso(),
        }

    @router.get("/targets")
    async def get_targets(x_soul_token: str | None = Header(default=None, alias="X-Soul-Token")) -> dict[str, Any]:
        _require_token(plugin, x_soul_token)
        if _public_mode():
            return {"targets": [], "updated_at": _now_iso()}

        from ..utils.spectrum_utils import chat_config_to_stream_id, parse_chat_id

        monitored = plugin.get_config("monitor.monitored_groups", []) or []
        targets: list[dict[str, Any]] = []
        for config_id in monitored:
            platform, chat_id, chat_type = parse_chat_id(str(config_id))
            stream_id = chat_config_to_stream_id(str(config_id))
            target_name = str(config_id)
            last_activity = None
            message_count = 0

            try:
                from src.common.database.database_model import ChatStreams, Messages

                stream = ChatStreams.get_or_none(ChatStreams.stream_id == stream_id)
                if stream:
                    target_name = stream.group_name or stream.user_cardname or stream.user_nickname or target_name
                    last_activity = datetime.fromtimestamp(float(stream.last_active_time)).isoformat()
                message_count = int(Messages.select().where(Messages.chat_id == stream_id).count())
            except Exception:
                pass

            targets.append(
                {
                    "stream_id": stream_id,
                    "target": target_name,
                    "target_type": "private" if chat_type == "private" else "group" if chat_type == "group" else "channel",
                    "last_activity": last_activity or _now_iso(),
                    "message_count": message_count,
                }
            )

        return {"targets": targets, "updated_at": _now_iso()}

    @router.get("/injection")
    async def get_injection(
        stream_id: str | None = None, x_soul_token: str | None = Header(default=None, alias="X-Soul-Token")
    ) -> dict[str, Any]:
        _require_token(plugin, x_soul_token)
        public_mode = _public_mode()
        if public_mode:
            stream_id = None
        if stream_id and stream_id != "global":
            payload: dict[str, Any] = _LAST_INJECTION_BY_STREAM.get(stream_id, {})
        else:
            payload = _LAST_INJECTION_BY_STREAM.get("global", {}) or _LAST_INJECTION
        if public_mode:
            payload = _sanitize_injection_payload(payload)
        return {"last_injection": payload, "updated_at": _now_iso()}

    @router.get("/injections")
    async def get_injections(
        limit: int = 30,
        stream_id: str | None = None,
        x_soul_token: str | None = Header(default=None, alias="X-Soul-Token"),
    ) -> dict[str, Any]:
        _require_token(plugin, x_soul_token)
        public_mode = _public_mode()
        if public_mode:
            stream_id = None

        try:
            limit = int(limit)
        except Exception:
            limit = 30
        limit = max(1, min(200, limit))

        if stream_id and stream_id != "global":
            bucket = _INJECTION_HISTORY_BY_STREAM.get(stream_id)
            entries = list(bucket) if bucket else []
        else:
            entries = list(_INJECTION_HISTORY)
        if public_mode:
            redacted: list[dict[str, Any]] = []
            for e in entries:
                out = _sanitize_injection_payload(e)
                out.pop("stream_id", None)
                redacted.append(out)
            entries = redacted
        entries = entries[-limit:][::-1]
        return {"entries": entries, "updated_at": _now_iso()}

    @router.get("/health")
    async def get_health(x_soul_token: str | None = Header(default=None, alias="X-Soul-Token")) -> dict[str, Any]:
        _require_token(plugin, x_soul_token)

        from ..models.ideology_model import init_tables, ThoughtSeed, EvolutionHistory

        init_tables()

        monitored = plugin.get_config("monitor.monitored_groups", []) or []
        seed_queue_total = int(ThoughtSeed.select().where(ThoughtSeed.status == "pending").count())
        history_total = int(EvolutionHistory.select().count())

        audit_path = plugin_dir / "data" / "audit.jsonl"
        audit_bytes = int(audit_path.stat().st_size) if audit_path.exists() else 0

        return {
            "ok": True,
            "enabled": bool(plugin.get_config("admin.enabled", True)),
            "schema_version": SCHEMA_VERSION,
            "uptime": _format_uptime(time.time() - _START_TIME),
            "groups": {
                "count": len(monitored),
                "pending_messages_total": 0,
                "seed_queue_total": seed_queue_total,
                "thoughts_total": 0,
                "active_introspection_tasks": 1 if bool(plugin.get_config("evolution.evolution_enabled", True)) else 0,
                "introspection_runs_total": history_total,
            },
            "files": {
                "state_json": {"exists": False, "bytes": 0},
                "audit_log": {"bytes": audit_bytes},
            },
            "updated_at": _now_iso(),
        }

    @router.get("/export")
    async def export_state(x_soul_token: str | None = Header(default=None, alias="X-Soul-Token")) -> dict[str, Any]:
        _require_token(plugin, x_soul_token)

        from ..models.ideology_model import init_tables, get_or_create_spectrum, EvolutionHistory, ThoughtSeed

        init_tables()

        spectrum = get_or_create_spectrum("global")
        history = list(EvolutionHistory.select().order_by(EvolutionHistory.timestamp.desc()).limit(200))
        seeds = list(ThoughtSeed.select().where(ThoughtSeed.status == "pending").order_by(ThoughtSeed.created_at.desc()))

        exported = {
            "spectrum": {
                "economic": spectrum.economic,
                "social": spectrum.social,
                "diplomatic": spectrum.diplomatic,
                "progressive": spectrum.progressive,
                "initialized": spectrum.initialized,
                "last_evolution": spectrum.last_evolution.isoformat() if spectrum.last_evolution else None,
                "updated_at": spectrum.updated_at.isoformat() if spectrum.updated_at else None,
            },
            "evolution_history": [
                {
                    "ts": r.timestamp.isoformat() if r.timestamp else None,
                    "group_id": r.group_id,
                    "deltas": {
                        "economic": r.economic_delta,
                        "social": r.social_delta,
                        "diplomatic": r.diplomatic_delta,
                        "progressive": r.progressive_delta,
                    },
                    "reason": r.reason,
                }
                for r in history
            ],
            "pending_seeds": [
                {
                    "seed_id": s.seed_id,
                    "type": s.seed_type,
                    "intensity": s.intensity,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                }
                for s in seeds
            ],
        }

        return {
            "schema_version": SCHEMA_VERSION,
            "exported_at": _now_iso(),
            "salt": secrets.token_hex(16),
            "groups": {"global": exported},
        }

    return router
