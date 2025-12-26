import asyncio
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from src.common.logger import get_logger
from src.common.server import get_global_server
from src.config.config import global_config, model_config
from src.manager.async_task_manager import AsyncTask, async_task_manager
from src.plugin_system import (
    BaseCommand,
    BaseEventHandler,
    BasePlugin,
    ConfigField,
    ConfigLayout,
    ConfigTab,
    EventType,
    register_plugin,
)
from src.plugin_system.apis import llm_api
from src.plugin_system.base.config_types import section_meta
from src.plugin_system.base.component_types import CustomEventHandlerResult, MaiMessages

logger = get_logger("mai_soul_engine")

AXES: Tuple[str, ...] = (
    "sincerity_absurdism",
    "normies_otakuism",
    "traditionalism_radicalism",
    "heroism_nihilism",
)

AXES_FRONTEND: Dict[str, str] = {
    "sincerity_absurdism": "sincerity-absurdism",
    "normies_otakuism": "normies-otakuism",
    "traditionalism_radicalism": "traditionalism-radicalism",
    "heroism_nihilism": "heroism-nihilism",
}

POLES: Dict[str, Tuple[str, str]] = {
    "sincerity_absurdism": ("Construction", "Abstractions"),
    "normies_otakuism": ("Normies", "Otakuism"),
    "traditionalism_radicalism": ("Traditionalism", "CyberRadicalism"),
    "heroism_nihilism": ("Heroism", "Nihilism"),
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _now_ts() -> float:
    return time.time()


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _to_iso(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class ThoughtSeed:
    seed_id: str
    topic: str
    tags: List[str] = field(default_factory=list)
    energy: float = 0.0
    fragments: List[str] = field(default_factory=list)
    created_ts: float = 0.0


@dataclass
class Thought:
    thought_id: str
    topic: str
    name: str
    tags: List[str]
    style_hint: str
    definition: str
    digest: str
    impact_points: Dict[str, float]
    created_ts: float


@dataclass
class IntrospectionLogItem:
    ts: float
    kind: str
    content: str
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupState:
    target: str = ""
    message_count: int = 0
    last_activity_ts: float = 0.0
    pending_messages: List[Dict[str, Any]] = field(default_factory=list)

    # Spectrum values are normalized [-1, 1]; frontend scales to [-100, 100]
    base_tone: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    values: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    ema: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    recent_deltas: Dict[str, List[float]] = field(default_factory=lambda: {a: [] for a in AXES})

    last_introspection_ts: float = 0.0
    introspection_logs: List[IntrospectionLogItem] = field(default_factory=list)

    seed_queue: List[ThoughtSeed] = field(default_factory=list)
    thoughts: List[Thought] = field(default_factory=list)


class SoulEngine:
    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self._boot_ts = _now_ts()
        self._config: Dict[str, Any] = {}
        self._groups: Dict[str, GroupState] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._tick_lock = asyncio.Lock()
        self._introspection_tasks: Dict[str, asyncio.Task] = {}
        self._router_registered = False
        self._cors_registered = False
        self._api_router = APIRouter()
        self._state_file = self.plugin_dir / "data" / "state.json"
        self._last_persist_ts = 0.0

    def set_config(self, config: Dict[str, Any]) -> None:
        self._config = config or {}

    def get_config(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        cur: Any = self._config
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def is_enabled(self) -> bool:
        return bool(self.get_config("plugin.enabled", True))

    def _get_lock(self, stream_id: str) -> asyncio.Lock:
        if stream_id not in self._locks:
            self._locks[stream_id] = asyncio.Lock()
        return self._locks[stream_id]

    def _get_group(self, stream_id: str) -> GroupState:
        if stream_id not in self._groups:
            self._groups[stream_id] = GroupState()
        return self._groups[stream_id]

    # -------------------------
    # Privacy / Sanitization
    # -------------------------
    def _sanitize_text(self, text: str, *, max_chars: int) -> str:
        s = (text or "").replace("\n", " ").replace("\r", " ").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"https?://\\S+", "<url>", s, flags=re.IGNORECASE)
        s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", "<email>", s)
        s = re.sub(r"@\\S+", "@某人", s)
        s = re.sub(r"\\b\\d{5,}\\b", "<num>", s)
        max_chars = max(40, min(int(max_chars), 800))
        return s[:max_chars]

    def _pending_cap(self) -> int:
        n = int(self.get_config("introspection.pending_max_messages", 600))
        return max(50, min(n, 2000))

    def _log_cap(self) -> int:
        n = int(self.get_config("introspection.max_log_items", 300))
        return max(50, min(n, 5000))

    def _fluctuation_window(self) -> int:
        n = int(self.get_config("spectrum.fluctuation_window", 30))
        return max(10, min(n, 500))

    def _append_log_locked(self, state: GroupState, *, kind: str, content: str, tags: Optional[List[str]] = None, extra: Optional[dict] = None) -> None:
        item = IntrospectionLogItem(
            ts=_now_ts(),
            kind=str(kind)[:32],
            content=str(content or "").strip()[:2000],
            tags=[str(t)[:32] for t in (tags or []) if str(t).strip()][:12],
            extra=extra if isinstance(extra, dict) else {},
        )
        state.introspection_logs.append(item)
        cap = self._log_cap()
        if len(state.introspection_logs) > cap:
            state.introspection_logs = state.introspection_logs[-cap:]

    # -------------------------
    # Message ingestion
    # -------------------------
    async def on_message(self, *, stream_id: str, platform: str, group_id: str, user_id: str, user_name: str, text: str) -> None:
        if not self.is_enabled():
            return
        max_chars = int(self.get_config("performance.max_message_chars", 800))
        max_chars = max(100, min(max_chars, 5000))
        text2 = self._sanitize_text(text, max_chars=max_chars)
        if not text2:
            return

        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            state.target = f"{platform}:{group_id}:group"
            state.message_count = int(state.message_count or 0) + 1
            state.last_activity_ts = _now_ts()
            state.pending_messages.append(
                {
                    "ts": state.last_activity_ts,
                    "user_id": str(user_id),
                    "user_name": str(user_name or user_id)[:32],
                    "text": text2,
                }
            )
            cap = self._pending_cap()
            if len(state.pending_messages) > cap:
                state.pending_messages = state.pending_messages[-cap:]

    # -------------------------
    # Spectrum
    # -------------------------
    def _tier_by_points_abs(self, abs_points: float) -> int:
        v = float(_clamp(abs_points, 0.0, 100.0))
        # 温和曲线：0~5 基本不影响；5~25 才进入第一档
        if v < 5.0:
            return 0
        if v < 25.0:
            return 1
        if v < 50.0:
            return 2
        if v < 75.0:
            return 3
        return 4

    def _axis_instruction(self, axis: str, *, direction: float, tier: int) -> str:
        tier = int(max(0, min(tier, 4)))
        neg, pos = POLES.get(axis, ("NEG", "POS"))
        pole = pos if direction >= 0 else neg

        strength_words = ["几乎中立", "轻微", "偏向", "明显", "强烈"]
        strength = strength_words[tier]

        axis_zh = {
            "sincerity_absurdism": "建构/解构",
            "normies_otakuism": "现充/二次元",
            "traditionalism_radicalism": "传统/赛博激进",
            "heroism_nihilism": "英雄/虚无",
        }.get(axis, axis)
        pole_zh = {
            "Construction": "建构",
            "Abstractions": "解构",
            "Normies": "现充",
            "Otakuism": "二次元",
            "Traditionalism": "传统",
            "CyberRadicalism": "赛博激进",
            "Heroism": "英雄主义",
            "Nihilism": "虚无主义",
        }.get(pole, pole)

        return f"{axis_zh}：{strength}偏 {pole_zh}"

    def _spectrum_instruction_block(self, values: Dict[str, float]) -> str:
        lines: List[str] = []
        for axis in AXES:
            v = float(values.get(axis, 0.0) or 0.0)
            tier = self._tier_by_points_abs(abs(v) * 100.0)
            lines.append(f"- {self._axis_instruction(axis, direction=v, tier=tier)}")
        return "\n".join(lines)

    def _apply_shift_locked(self, state: GroupState, *, shift_points: Dict[str, float], intensity: float, confidence: float, window_size: int) -> Dict[str, float]:
        clamp_abs = float(self.get_config("spectrum.clamp_abs", 1.0))
        regression = float(self.get_config("spectrum.regression_rate", 0.01))
        ema_alpha = float(self.get_config("spectrum.ema_alpha", 0.15))
        window = self._fluctuation_window()

        strength_scale = float(self.get_config("introspection.window_strength_scale", 12.0))
        strength_scale = max(1.0, min(strength_scale, 200.0))
        size_scale = math.log1p(max(1, int(window_size))) / strength_scale
        size_scale = float(_clamp(size_scale, 0.05, 1.0))

        effective = _clamp(float(intensity), 0.0, 1.0) * _clamp(float(confidence), 0.0, 1.0) * size_scale
        applied_points: Dict[str, float] = {a: 0.0 for a in AXES}
        if effective <= 0:
            return applied_points

        for axis in AXES:
            pts = float(shift_points.get(axis, 0.0) or 0.0)
            pts = float(_clamp(pts, -100.0, 100.0))
            delta = float(_clamp(pts / 100.0, -1.0, 1.0))
            d_eff = delta * effective

            state.values[axis] = _clamp(float(state.values.get(axis, 0.0)) + d_eff, -clamp_abs, clamp_abs)
            base = float(state.base_tone.get(axis, 0.0) or 0.0)
            state.values[axis] = state.values[axis] + (base - state.values[axis]) * regression
            state.ema[axis] = (1 - ema_alpha) * float(state.ema.get(axis, 0.0)) + ema_alpha * float(state.values.get(axis, 0.0))

            state.recent_deltas[axis].append(d_eff)
            if len(state.recent_deltas[axis]) > window:
                state.recent_deltas[axis] = state.recent_deltas[axis][-window:]

            applied_points[axis] = float(_clamp(d_eff * 100.0, -100.0, 100.0))
        return applied_points

    # -------------------------
    # Introspection (core loop)
    # -------------------------
    def _self_ids(self) -> set[str]:
        ids: set[str] = set()
        try:
            bot_cfg = getattr(global_config, "bot", None)
            if not bot_cfg:
                return ids

            qq_account = str(getattr(bot_cfg, "qq_account", "") or "").strip()
            if qq_account:
                ids.add(qq_account)
                ids.add(f"qq:{qq_account}")

            platforms = getattr(bot_cfg, "platforms", None) or []
            if isinstance(platforms, list):
                for item in platforms:
                    s = str(item or "").strip()
                    if not s:
                        continue
                    ids.add(s)
                    if ":" in s:
                        p, uid = s.split(":", 1)
                        p = p.strip()
                        uid = uid.strip()
                        if uid:
                            ids.add(uid)
                            if p:
                                ids.add(f"{p}:{uid}")
        except Exception as e:
            logger.debug(f"[soul] read bot_config self ids failed: {e}")
        return ids

    def _main_personality_hint(self) -> str:
        if not bool(self.get_config("introspection.use_main_personality", True)):
            return ""
        try:
            p = getattr(global_config, "personality", None)
            hint = str(getattr(p, "personality", "") or "").strip()
            return hint[:1200]
        except Exception as e:
            logger.debug(f"[soul] read bot_config personality failed: {e}")
            return ""

    def _extract_window_locked(self, state: GroupState, *, now: float) -> Tuple[float, List[Dict[str, Any]]]:
        window_minutes = float(self.get_config("introspection.window_minutes", 30.0))
        window_minutes = max(1.0, min(window_minutes, 24 * 60.0))
        max_messages = int(self.get_config("introspection.max_messages_per_group", 500))
        max_messages = max(20, min(max_messages, 2000))

        from_ts = max(0.0, now - window_minutes * 60.0)
        if state.last_introspection_ts > 0:
            from_ts = max(from_ts, float(state.last_introspection_ts))

        items = []
        for m in state.pending_messages:
            if not isinstance(m, dict):
                continue
            ts = float(m.get("ts", 0.0) or 0.0)
            if ts <= 0 or ts < from_ts:
                continue
            text = str(m.get("text", "") or "").strip()
            if not text:
                continue
            items.append(m)

        if len(items) > max_messages:
            items = items[-max_messages:]
        return from_ts, items

    async def _llm_json(self, *, prompt: str, temperature: float, max_tokens: int, request_type: str) -> Optional[dict]:
        models = llm_api.get_available_models()
        task_config = models.get("utils") or model_config.model_task_config.utils
        ok, content, _reasoning, model_name = await llm_api.generate_with_model(
            prompt,
            model_config=task_config,
            request_type=request_type,
            temperature=float(_clamp(float(temperature), 0.0, 2.0)),
            max_tokens=max(128, min(int(max_tokens), 200_000)),
        )
        if not ok:
            logger.debug(f"[soul] llm failed model={model_name}: {content[:80]}")
            return None
        return _extract_json_object(content)

    async def _internalize_seed(self, stream_id: str, seed: ThoughtSeed) -> Optional[Thought]:
        rounds = int(self.get_config("cabinet.internalization_rounds", 3))
        rounds = max(2, min(rounds, 6))
        temperature = float(self.get_config("cabinet.temperature", 0.35))
        max_tokens = int(self.get_config("cabinet.max_tokens", 60000))
        persona_hint = self._main_personality_hint()

        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            values_snapshot = dict(state.values)
            thought_names = [t.name for t in state.thoughts[-12:]]

        carry: List[str] = []
        final: Optional[dict] = None

        for i in range(1, rounds + 1):
            is_final = i == rounds
            prev = "\n".join([f"- {x}" for x in carry[-10:] if str(x).strip()]) or "- (none)"
            persona = persona_hint if persona_hint else "（沿用主程序人格设定；不要复述。）"

            if not is_final:
                schema = {"round": i, "monologue": "（60~220字内心独白）", "notes": ["要点（1~6条）"]}
                task = f"第 {i} 轮：品鉴思想种子，逐步形成可固化的思想轮廓。"
            else:
                schema = {
                    "round": i,
                    "monologue": "（60~220字内心独白）",
                    "notes": ["结论要点（1~8条）"],
                    "name": "思想名称（短）",
                    "definition": "思想定义（脱敏，不复刻原句）",
                    "digest": "内化结论（脱敏，不复刻原句）",
                    "style_hint": "未来表达风格偏置（1~3句）",
                    "impact_points": {a: 0.0 for a in AXES},
                    "tags": ["..."],
                    "confidence": 0.0,
                    "intensity": 0.0,
                }
                task = "最终轮：输出思想名称/定义/内化结论/风格偏置/宏偏移（点数，建议单轴>=10才算显著）。"

            prompt = f"""
你是“麦麦的思维内省”。你正在内化一个思想种子，并将其固化为可复用的思想。

人格设定（不要复述）：{persona}
当前思想光谱坐标（-1~1；不要复述）：{json.dumps(values_snapshot, ensure_ascii=False)}
已固化思想名（不要复述）：{json.dumps(thought_names, ensure_ascii=False)}

思想种子：
- topic: {seed.topic}
- tags: {json.dumps(seed.tags, ensure_ascii=False)}
- energy: {seed.energy}
- fragments（抽象转述；禁止复刻原句）：{json.dumps(seed.fragments, ensure_ascii=False)}

上一轮累计要点：{prev}

硬性要求：
- 不要输出任何可识别个人信息、链接/号码/邮箱。
- 所有内容不得复刻聊天原句。
- impact_points 以“点”为单位（-100~100）。

任务：{task}
输出必须严格 JSON。JSON schema 示例：
{json.dumps(schema, ensure_ascii=False)}
""".strip()

            obj = await self._llm_json(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens, request_type=f"mai_soul_engine.cabinet.round.{i}"
            )
            if not isinstance(obj, dict):
                break

            monologue = str(obj.get("monologue", "") or "").strip()
            notes = obj.get("notes", [])
            if not isinstance(notes, list):
                notes = []
            notes2 = [str(x).strip() for x in notes if str(x).strip()][:12]
            carry.extend(notes2)

            async with lock:
                state = self._get_group(stream_id)
                self._append_log_locked(
                    state,
                    kind="cabinet_round",
                    content=f"【思想内化 第{i}/{rounds}轮】\n{monologue}\n要点：\n" + "\n".join([f"- {x}" for x in notes2]),
                    tags=["introspection", "cabinet"],
                    extra={"topic": seed.topic, "round": i, "rounds": rounds},
                )

            if is_final:
                final = obj

        if not isinstance(final, dict):
            return None

        name = str(final.get("name", "") or "").strip()[:40] or seed.topic[:40] or "Thought"
        definition = str(final.get("definition", "") or "").strip()[:800]
        digest = str(final.get("digest", "") or "").strip()[:1600]
        style_hint = str(final.get("style_hint", "") or "").strip()[:500]
        tags = final.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags2 = [str(t)[:32] for t in tags if str(t).strip()][:12]

        impact = final.get("impact_points", {})
        impact_points = {a: 0.0 for a in AXES}
        if isinstance(impact, dict):
            for axis in AXES:
                if axis in impact:
                    try:
                        impact_points[axis] = float(_clamp(float(impact[axis]), -100.0, 100.0))
                    except Exception:
                        continue

        thought_id = f"thought:{int(_now_ts())}:{abs(hash(seed.topic)) % 100000}"
        return Thought(
            thought_id=thought_id,
            topic=seed.topic[:60],
            name=name,
            tags=tags2,
            style_hint=style_hint,
            definition=definition,
            digest=digest,
            impact_points=impact_points,
            created_ts=_now_ts(),
        )

    async def _introspection_run(self, stream_id: str, *, force: bool) -> bool:
        if not self.is_enabled():
            return False

        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            from_ts, window_items = self._extract_window_locked(state, now=now)
            if not window_items and not state.seed_queue:
                state.last_introspection_ts = max(float(state.last_introspection_ts or 0.0), now)
                return False

        # 1) 内化候选思想（若有）
        internalize_n = int(self.get_config("cabinet.internalize_seeds_per_run", 1))
        internalize_n = max(0, min(internalize_n, 5))
        if internalize_n > 0:
            for _ in range(internalize_n):
                async with lock:
                    state = self._get_group(stream_id)
                    seed = state.seed_queue.pop(0) if state.seed_queue else None
                if not seed:
                    break
                thought = await self._internalize_seed(stream_id, seed)
                if not thought:
                    async with lock:
                        state = self._get_group(stream_id)
                        self._append_log_locked(
                            state,
                            kind="cabinet_failed",
                            content=f"【思想内化失败】topic={seed.topic}",
                            tags=["introspection", "cabinet"],
                        )
                    continue

                async with lock:
                    state = self._get_group(stream_id)
                    state.thoughts.append(thought)
                    max_slots = int(self.get_config("cabinet.max_slots", 6))
                    max_slots = max(1, min(max_slots, 20))
                    if len(state.thoughts) > max_slots:
                        state.thoughts = state.thoughts[-max_slots:]

                    # 固化思想产生一次明显偏移（宏偏移）
                    applied = self._apply_shift_locked(
                        state,
                        shift_points=thought.impact_points,
                        intensity=1.0,
                        confidence=1.0,
                        window_size=max(10, len(window_items)),
                    )
                    self._append_log_locked(
                        state,
                        kind="cabinet_crystallized",
                        content=f"【思想固化】{thought.name}\nΔpoints={json.dumps({a: round(float(applied.get(a, 0.0) or 0.0), 2) for a in AXES}, ensure_ascii=False)}",
                        tags=["introspection", "cabinet"],
                        extra={"thought_id": thought.thought_id, "topic": thought.topic},
                    )

        # 2) 对聊天窗口做多轮内省
        rounds = int(self.get_config("introspection.rounds", 4))
        rounds = max(2, min(rounds, 8))
        temperature = float(self.get_config("introspection.temperature", 0.7))
        max_tokens = int(self.get_config("introspection.max_tokens", 700))
        persona_hint = self._main_personality_hint()

        async with lock:
            state = self._get_group(stream_id)
            values_snapshot = dict(state.values)
            thought_names = [t.name for t in state.thoughts[-12:]]
            self_ids = self._self_ids()

        context: List[Dict[str, Any]] = []
        for m in window_items:
            uid = str(m.get("user_id", "") or "")
            role = "self" if (uid and uid in self_ids) else "other"
            context.append({"role": role, "user_id": uid, "name": str(m.get("user_name", "") or "")[:32], "text": str(m.get("text", "") or "")})

        carry: List[str] = []
        final: Optional[dict] = None

        for i in range(1, rounds + 1):
            is_final = i == rounds
            prev = "\n".join([f"- {x}" for x in carry[-10:] if str(x).strip()]) or "- (none)"
            persona = persona_hint if persona_hint else "（沿用主程序人格设定；不要复述。）"

            if not is_final:
                schema = {"round": i, "monologue": "（60~220字内心独白）", "notes": ["阶段要点（1~6条）"]}
                task = f"第 {i} 轮：回想聊天内容，形成阶段结论。"
            else:
                schema = {
                    "round": i,
                    "monologue": "（60~220字内心独白）",
                    "notes": ["最终要点（1~8条）"],
                    "spectrum_shift_points": {a: 0.0 for a in AXES},
                    "confidence": 0.0,
                    "intensity": 0.0,
                    "seed_topics": [{"topic": "短标题", "tags": ["..."], "energy": 0.0, "fragments": ["抽象转述切片"]}],
                }
                task = "最终轮：输出四轴偏移（点数）、并生成思想种子（如有）。"

            prompt = f"""
你是“麦麦的思维内省”。你在回想一段时间内的群聊讨论，并形成自己的立场变化与候选思想。

人格设定（不要复述）：{persona}
当前思想光谱坐标（-1~1；不要复述）：{json.dumps(values_snapshot, ensure_ascii=False)}
已固化思想名（不要复述）：{json.dumps(thought_names, ensure_ascii=False)}

本次回想时间窗：from_ts={from_ts} to_ts={now}
聊天记录（已脱敏/截断；禁止复刻原句）：{json.dumps(context, ensure_ascii=False)}

上一轮累计要点：{prev}

硬性要求：
- 不要输出任何可识别个人信息、链接/号码/邮箱。
- 所有内容不得复刻聊天原句。
- 偏移以“点”为单位（-100~100），日常偏移应偏小（通常每轴<5），除非你确信出现明显变化。
- 种子 fragments 必须抽象转述。

任务：{task}
输出必须严格 JSON。JSON schema 示例：
{json.dumps(schema, ensure_ascii=False)}
""".strip()

            obj = await self._llm_json(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                request_type=f"mai_soul_engine.introspection.round.{i}",
            )
            if not isinstance(obj, dict):
                break

            monologue = str(obj.get("monologue", "") or "").strip()
            notes = obj.get("notes", [])
            if not isinstance(notes, list):
                notes = []
            notes2 = [str(x).strip() for x in notes if str(x).strip()][:12]
            carry.extend(notes2)

            async with lock:
                state = self._get_group(stream_id)
                self._append_log_locked(
                    state,
                    kind="introspection_round",
                    content=f"【思维内省 第{i}/{rounds}轮】\n{monologue}\n要点：\n" + "\n".join([f"- {x}" for x in notes2]),
                    tags=["introspection"],
                    extra={"round": i, "rounds": rounds},
                )

            if is_final:
                final = obj

        if not isinstance(final, dict):
            async with lock:
                state = self._get_group(stream_id)
                state.last_introspection_ts = max(float(state.last_introspection_ts or 0.0), now)
            return False

        # 应用最终偏移 & 生成种子
        shift_obj = final.get("spectrum_shift_points", {})
        shift_points: Dict[str, float] = {a: 0.0 for a in AXES}
        if isinstance(shift_obj, dict):
            for axis in AXES:
                if axis in shift_obj:
                    try:
                        shift_points[axis] = float(_clamp(float(shift_obj[axis]), -100.0, 100.0))
                    except Exception:
                        continue

        intensity = float(final.get("intensity", 0.3) or 0.0)
        confidence = float(final.get("confidence", 0.3) or 0.0)

        seeds = final.get("seed_topics", [])
        if not isinstance(seeds, list):
            seeds = []

        async with lock:
            state = self._get_group(stream_id)
            applied = self._apply_shift_locked(
                state,
                shift_points=shift_points,
                intensity=intensity,
                confidence=confidence,
                window_size=len(window_items) if window_items else 1,
            )

            # 生成候选思想种子：进入队列，等下一次内省内化
            seed_threshold = float(self.get_config("cabinet.seed_energy_threshold", 0.5))
            seed_threshold = float(_clamp(seed_threshold, 0.0, 1.0))
            seed_cap = int(self.get_config("cabinet.max_seed_queue", 30))
            seed_cap = max(0, min(seed_cap, 200))

            for tp in seeds[:10]:
                if not isinstance(tp, dict):
                    continue
                topic = str(tp.get("topic", "") or "").strip()
                if not topic:
                    continue
                energy = float(tp.get("energy", 0.0) or 0.0)
                if energy < seed_threshold:
                    continue
                if any(s.topic == topic for s in state.seed_queue):
                    continue

                ttags = tp.get("tags", [])
                if not isinstance(ttags, list):
                    ttags = []
                ttags2 = [str(t)[:32] for t in ttags if str(t).strip()][:12]

                frags = tp.get("fragments", [])
                if not isinstance(frags, list):
                    frags = []
                frags2 = [self._sanitize_text(str(x), max_chars=140) for x in frags if str(x).strip()][:12]
                frags2 = [f for f in frags2 if f]

                seed_id = f"seed:{int(now)}:{abs(hash(topic)) % 100000}"
                state.seed_queue.append(
                    ThoughtSeed(
                        seed_id=seed_id,
                        topic=topic[:60],
                        tags=ttags2,
                        energy=float(_clamp(energy, 0.0, 1.0)),
                        fragments=frags2,
                        created_ts=now,
                    )
                )
                if seed_cap and len(state.seed_queue) > seed_cap:
                    state.seed_queue = state.seed_queue[-seed_cap:]

            state.last_introspection_ts = max(float(state.last_introspection_ts or 0.0), now)
            self._append_log_locked(
                state,
                kind="introspection_result",
                content="【思维内省裁决】\n"
                + f"Δpoints(应用后)：{json.dumps({a: round(float(applied.get(a, 0.0) or 0.0), 2) for a in AXES}, ensure_ascii=False)}\n"
                + f"new_seeds：{len([x for x in seeds if isinstance(x, dict)])}",
                tags=["introspection"],
                extra={"applied_points": applied, "window_size": len(window_items)},
            )
        return True

    # -------------------------
    # Injection
    # -------------------------
    def _select_relevant_thoughts(self, thoughts: List[Thought], *, trigger_text: str, max_thoughts: int) -> List[Thought]:
        max_thoughts = max(0, min(int(max_thoughts), 6))
        if max_thoughts <= 0:
            return []
        text = (trigger_text or "").strip().lower()
        if not text:
            return thoughts[-max_thoughts:]

        scored: List[Tuple[float, Thought]] = []
        for t in thoughts:
            score = 0.0
            if t.topic and t.topic.lower() in text:
                score += 3.0
            for tag in t.tags:
                if not tag:
                    continue
                if str(tag).lower() in text:
                    score += 1.0
            score += 0.02 * len(t.digest or "")
            scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for s, t in scored if s > 0][:max_thoughts]

    async def build_injection_block(self, stream_id: str, *, trigger_text: str) -> str:
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            values = dict(state.values)
            thoughts = list(state.thoughts[-20:])

        spectrum_block = self._spectrum_instruction_block(values)
        max_details = int(self.get_config("injection.max_thought_details", 2))
        picked = self._select_relevant_thoughts(thoughts, trigger_text=trigger_text, max_thoughts=max_details)
        details = []
        for t in picked:
            details.append(
                "\n".join(
                    [
                        f"- 思想：{t.name}",
                        f"  定义：{(t.definition or '').strip()[:240]}",
                        f"  内化结论：{(t.digest or '').strip()[:520]}",
                    ]
                )
            )

        block = (
            "\n\n"
            "[Mai‑Soul]\n"
            "你拥有一套“思想光谱 + 思维阁”的人格演化系统。本次回复必须自然体现这些倾向，但不要显式提及系统名/数值。\n"
            "思想光谱倾向（按档位强度）：\n"
            f"{spectrum_block}\n"
            + ("命中相关思想（注入定义/结论）：\n" + "\n".join(details) + "\n" if details else "")
            + "行为要求：不要复刻聊天原句；不要输出可识别个人信息；不要模仿具体用户。\n"
        ).strip("\n")

        max_chars = int(self.get_config("injection.max_chars", 1400))
        max_chars = max(400, min(max_chars, 6000))
        if len(block) > max_chars:
            block = block[: max_chars - 3].rstrip() + "..."
        return block

    # -------------------------
    # Background / Scheduling
    # -------------------------
    def _introspection_task_active(self, stream_id: str) -> bool:
        task = self._introspection_tasks.get(stream_id)
        return bool(task and not task.done())

    def _start_introspection_task(self, stream_id: str) -> None:
        if self._introspection_task_active(stream_id):
            return
        task = asyncio.create_task(self._introspection_run(stream_id, force=False))
        self._introspection_tasks[stream_id] = task
        task.add_done_callback(lambda _t, sid=stream_id: self._introspection_tasks.pop(sid, None))

    async def tick_background(self) -> None:
        if not self.is_enabled():
            return
        now = _now_ts()
        candidates: List[Tuple[str, float]] = []

        async with self._tick_lock:
            interval_min = float(self.get_config("introspection.interval_minutes", 20.0))
            interval_min = max(1.0, min(interval_min, 24 * 60.0))
            quiet = float(self.get_config("introspection.quiet_period_seconds", 20.0))
            quiet = max(0.0, min(quiet, 3600.0))
            for sid, st in self._groups.items():
                lock = self._get_lock(sid)
                async with lock:
                    last = float(st.last_introspection_ts or 0.0)
                    due = (now - last) >= interval_min * 60.0
                    quiet_ok = (quiet <= 0.0) or ((now - float(st.last_activity_ts or 0.0)) >= quiet)
                    has_new = bool(st.pending_messages) and (float(st.last_activity_ts or 0.0) > last)
                    has_seeds = bool(st.seed_queue)
                    if (has_new or has_seeds) and due and quiet_ok and not self._introspection_task_active(sid):
                        candidates.append((sid, float(st.last_activity_ts or 0.0)))

            # persistence
            if bool(self.get_config("persistence.enabled", True)):
                save_interval = float(self.get_config("persistence.save_interval_seconds", 15.0))
                save_interval = max(3.0, min(save_interval, 3600.0))
                if now - self._last_persist_ts >= save_interval:
                    await self.persist()
                    self._last_persist_ts = now

        if not candidates:
            return
        candidates.sort(key=lambda x: x[1], reverse=True)
        max_n = int(self.get_config("introspection.max_groups_per_tick", 1))
        max_n = max(1, min(max_n, 50))
        for sid, _ts in candidates[:max_n]:
            self._start_introspection_task(sid)

    # -------------------------
    # Persistence
    # -------------------------
    async def persist(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {"version": 1, "groups": {}}
            for sid, st in self._groups.items():
                lock = self._get_lock(sid)
                async with lock:
                    payload["groups"][sid] = {
                        "target": st.target,
                        "message_count": int(st.message_count or 0),
                        "last_activity_ts": float(st.last_activity_ts or 0.0),
                        "pending_messages": list(st.pending_messages or [])[-self._pending_cap():],
                        "base_tone": st.base_tone,
                        "values": st.values,
                        "ema": st.ema,
                        "recent_deltas": st.recent_deltas,
                        "last_introspection_ts": float(st.last_introspection_ts or 0.0),
                        "introspection_logs": [
                            {"ts": it.ts, "kind": it.kind, "content": it.content, "tags": it.tags, "extra": it.extra}
                            for it in (st.introspection_logs or [])[-self._log_cap():]
                        ],
                        "seed_queue": [s.__dict__ for s in (st.seed_queue or [])[-200:]],
                        "thoughts": [t.__dict__ for t in (st.thoughts or [])[-200:]],
                    }
            tmp = self._state_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._state_file)
        except Exception as e:
            logger.debug(f"[soul] persist failed: {e}")

    def load_persisted(self) -> None:
        if not self._state_file.exists():
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
            groups = raw.get("groups", {})
            if not isinstance(groups, dict):
                return
            for sid, obj in groups.items():
                if not isinstance(obj, dict):
                    continue
                st = GroupState()
                st.target = str(obj.get("target", "") or "")
                st.message_count = int(obj.get("message_count", 0) or 0)
                st.last_activity_ts = float(obj.get("last_activity_ts", 0.0) or 0.0)
                pm = obj.get("pending_messages", [])
                st.pending_messages = pm if isinstance(pm, list) else []

                bt = obj.get("base_tone", {})
                if isinstance(bt, dict):
                    st.base_tone = {a: float(_clamp(float(bt.get(a, 0.0) or 0.0), -1.0, 1.0)) for a in AXES}
                vals = obj.get("values", {})
                if isinstance(vals, dict):
                    st.values = {a: float(_clamp(float(vals.get(a, 0.0) or 0.0), -1.0, 1.0)) for a in AXES}
                ema = obj.get("ema", {})
                if isinstance(ema, dict):
                    st.ema = {a: float(_clamp(float(ema.get(a, 0.0) or 0.0), -1.0, 1.0)) for a in AXES}
                rd = obj.get("recent_deltas", {})
                if isinstance(rd, dict):
                    st.recent_deltas = {a: [float(x) for x in (rd.get(a, []) or []) if isinstance(x, (int, float))][-500:] for a in AXES}

                st.last_introspection_ts = float(obj.get("last_introspection_ts", 0.0) or 0.0)

                logs = obj.get("introspection_logs", [])
                if isinstance(logs, list):
                    out: List[IntrospectionLogItem] = []
                    for it in logs[-self._log_cap():]:
                        if not isinstance(it, dict):
                            continue
                        out.append(
                            IntrospectionLogItem(
                                ts=float(it.get("ts", 0.0) or 0.0),
                                kind=str(it.get("kind", "") or "")[:32],
                                content=str(it.get("content", "") or "")[:2000],
                                tags=[str(t)[:32] for t in (it.get("tags") or []) if isinstance(t, (str, int, float))][:12]
                                if isinstance(it.get("tags"), list)
                                else [],
                                extra=it.get("extra") if isinstance(it.get("extra"), dict) else {},
                            )
                        )
                    st.introspection_logs = out

                seeds = obj.get("seed_queue", [])
                if isinstance(seeds, list):
                    st.seed_queue = [ThoughtSeed(**s) for s in seeds if isinstance(s, dict)]
                thoughts = obj.get("thoughts", [])
                if isinstance(thoughts, list):
                    st.thoughts = [Thought(**t) for t in thoughts if isinstance(t, dict)]

                self._groups[str(sid)] = st
        except Exception as e:
            logger.debug(f"[soul] load_persisted failed: {e}")

    # -------------------------
    # API
    # -------------------------
    def _require_api_token(self, request: Request) -> None:
        token = str(self.get_config("api.token", "") or "").strip()
        if not token:
            return
        auth = request.headers.get("authorization", "") or request.headers.get("Authorization", "")
        x = request.headers.get("x-soul-token", "") or request.headers.get("X-Soul-Token", "")
        if auth.startswith("Bearer ") and auth[7:].strip() == token:
            return
        if x.strip() == token:
            return
        raise HTTPException(status_code=401, detail="unauthorized")

    def _cors_preflight_response(self, request: Request) -> Response:
        origin = request.headers.get("origin", "") or request.headers.get("Origin", "")
        allow = self._cors_allow_origin(origin)
        headers = {
            "Access-Control-Allow-Origin": allow,
            "Access-Control-Allow-Methods": "GET,OPTIONS",
            "Access-Control-Allow-Headers": "Authorization,Content-Type,X-Soul-Token",
            "Access-Control-Max-Age": "600",
        }
        return Response(status_code=204, headers=headers)

    def _cors_allow_origin(self, origin: str) -> str:
        origins = self.get_config("api.cors_allow_origins", ["http://localhost:5173", "http://127.0.0.1:5173"])
        if origins == ["*"] or origins == "*":
            return "*" if origin else "*"
        if isinstance(origins, list) and origin in origins:
            return origin
        return origins[0] if isinstance(origins, list) and origins else "*"

    def _register_cors_middleware(self, app) -> None:
        @app.middleware("http")
        async def soul_cors_middleware(request: Request, call_next):
            if request.url.path.startswith("/api/v1/soul") and request.method.upper() == "OPTIONS":
                return self._cors_preflight_response(request)
            response = await call_next(request)
            if request.url.path.startswith("/api/v1/soul"):
                origin = request.headers.get("origin", "") or request.headers.get("Origin", "")
                response.headers["Access-Control-Allow-Origin"] = self._cors_allow_origin(origin)
                response.headers["Vary"] = "Origin"
            return response

    def _resolve_stream_id(self, *, stream_id: Optional[str], target: Optional[str]) -> str:
        if stream_id:
            return str(stream_id)
        if target:
            needle = str(target).strip()
            for sid, st in self._groups.items():
                if st.target == needle:
                    return sid
        # fallback：取最近活跃的群
        best = ""
        best_ts = -1.0
        for sid, st in self._groups.items():
            if st.last_activity_ts > best_ts:
                best_ts = st.last_activity_ts
                best = sid
        if not best:
            raise HTTPException(status_code=404, detail="no active targets")
        return best

    async def _snapshot_spectrum_frontend(self, stream_id: str) -> dict:
        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            st = self._get_group(stream_id)
            dims = []
            for axis in AXES:
                dims.append(
                    {
                        "axis": AXES_FRONTEND.get(axis, axis),
                        "values": {
                            "current": float(_clamp(float(st.values.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                            "ema": float(_clamp(float(st.ema.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                            "baseline": float(_clamp(float(st.base_tone.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                        },
                        "volatility": 0.0,
                    }
                )
        return {
            "dimensions": dims,
            "base_tone": {"primary": "sincerity", "secondary": "normies"},
            "dominant_trait": "",
            "last_major_shift": "",
            "updated_at": _to_iso(now),
        }

    async def _snapshot_cabinet_frontend(self, stream_id: str) -> dict:
        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            st = self._get_group(stream_id)
            slots = []
            for seed in st.seed_queue[:20]:
                slots.append(
                    {
                        "id": seed.seed_id,
                        "name": seed.topic[:40],
                        "source_dimension": "sincerity",
                        "status": "internalizing",
                        "progress": 0.0,
                        "conflict_reason": None,
                        "crystallized_at": None,
                        "introspection": "等待下一次思维内省进行内化。",
                        "impacts": [],
                        "debug_params": {
                            "energy": float(seed.energy),
                            "tags": list(seed.tags or [])[:12],
                            "fragments": list(seed.fragments or [])[:6],
                        },
                    }
                )
            traits = []
            for t in st.thoughts[-50:]:
                traits.append(
                    {
                        "id": t.thought_id,
                        "name": t.name[:40],
                        "source_dimension": "sincerity",
                        "crystallized_at": _to_iso(float(t.created_ts or now)),
                        "description": (t.definition or t.digest or "")[:400],
                        "permanent_effects": [],
                        "definition": (t.definition or "")[:200],
                        "digest": (t.digest or "")[:900],
                    }
                )
            # 兼容：把固化思想也放到 slots 里，方便前端展示
            for t in st.thoughts[-50:]:
                slots.append(
                    {
                        "id": t.thought_id,
                        "name": t.name[:40],
                        "source_dimension": "sincerity",
                        "status": "crystallized",
                        "progress": 100.0,
                        "conflict_reason": None,
                        "crystallized_at": _to_iso(float(t.created_ts or now)),
                        "introspection": ((t.definition or "") + "\n\n" + (t.digest or "")).strip()[:600],
                        "impacts": [],
                        "debug_params": None,
                    }
                )
        return {"slots": slots, "traits": traits, "dissonance": {"active": False, "conflicting_thoughts": [], "severity": "low"}}

    async def _snapshot_introspection_logs_frontend(self, stream_id: str, *, limit: int) -> dict:
        now = _now_ts()
        limit = max(1, min(int(limit), 200))
        lock = self._get_lock(stream_id)
        async with lock:
            st = self._get_group(stream_id)
            items = list(st.introspection_logs or [])[-limit:]
            last_intro = float(st.last_introspection_ts or 0.0)
            interval_min = float(self.get_config("introspection.interval_minutes", 20.0))
            interval_min = max(1.0, min(interval_min, 24 * 60.0))
            next_intro = _to_iso(last_intro + interval_min * 60.0) if last_intro else None

        out = []
        for it in items:
            out.append(
                {
                    "id": f"log-{int(it.ts)}-{it.kind}",
                    "content": it.content,
                    "source": "introspection",
                    "timestamp": _to_iso(float(it.ts or now)),
                    "tags": list(it.tags or [])[:12],
                    "redacted": False,
                    "check_result": "success",
                }
            )
        return {"fragments": out, "next_injection": next_intro, "updated_at": _to_iso(now)}

    def _uptime_str(self, now: float) -> str:
        seconds = max(0, int(now - float(self._boot_ts or now)))
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)
        if days:
            return f"{days}d {hours}h {minutes}m"
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    async def _snapshot_pulse_frontend(self, stream_id: str) -> dict:
        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            st = self._get_group(stream_id)
            values = dict(st.values)
            avg_abs = sum(abs(float(values.get(a, 0.0) or 0.0)) for a in AXES) / float(len(AXES))
            temperature = float(_clamp(0.35 + 0.65 * avg_abs, 0.0, 1.0))
            heartbeat = int(st.message_count or 0)
            last_intro = float(st.last_introspection_ts or 0.0)
            interval_min = float(self.get_config("introspection.interval_minutes", 20.0))
            interval_min = max(1.0, min(interval_min, 24 * 60.0))
            next_intro = _to_iso(last_intro + interval_min * 60.0) if last_intro else None

        return {
            "temperature": temperature,
            "life_state": "contemplating",
            "status": "思维内省系统运行中",
            "uptime": self._uptime_str(now),
            "heartbeat": heartbeat,
            "dissonance": {"active": False},
            "introspection": {"enabled": True, "last_run": _to_iso(last_intro) if last_intro else None, "next_run": next_intro},
            "updated_at": _to_iso(now),
        }

    async def _snapshot_targets_frontend(self, *, limit: int, offset: int) -> dict:
        now = _now_ts()
        items = []
        for sid, st in self._groups.items():
            items.append({"stream_id": sid, "target": st.target, "last_activity_ts": st.last_activity_ts, "message_count": st.message_count})
        items.sort(key=lambda x: float(x.get("last_activity_ts", 0.0) or 0.0), reverse=True)
        limit = max(1, min(int(limit), 200))
        offset = max(0, min(int(offset), 5000))
        slice_items = items[offset : offset + limit]
        out = []
        for it in slice_items:
            out.append(
                {
                    "stream_id": str(it.get("stream_id", "") or ""),
                    "target": str(it.get("target", "") or ""),
                    "target_type": "group",
                    "last_activity": _to_iso(float(it.get("last_activity_ts", now) or now)),
                    "message_count": int(it.get("message_count", 0) or 0),
                }
            )
        return {"targets": out, "updated_at": _to_iso(now)}

    def setup_router(self) -> APIRouter:
        router = APIRouter(prefix="/api/v1/soul", tags=["soul"])

        @router.options("/{path:path}")
        async def soul_preflight(path: str, request: Request):
            return self._cors_preflight_response(request)

        @router.get("/spectrum")
        async def get_spectrum(request: Request, stream_id: Optional[str] = None, target: Optional[str] = None, format: str = "frontend"):
            self._require_api_token(request)
            sid = self._resolve_stream_id(stream_id=stream_id, target=target)
            return JSONResponse(await self._snapshot_spectrum_frontend(sid))

        @router.get("/cabinet")
        async def get_cabinet(request: Request, stream_id: Optional[str] = None, target: Optional[str] = None, format: str = "frontend"):
            self._require_api_token(request)
            sid = self._resolve_stream_id(stream_id=stream_id, target=target)
            return JSONResponse(await self._snapshot_cabinet_frontend(sid))

        @router.get("/introspection")
        async def get_introspection(request: Request, stream_id: Optional[str] = None, target: Optional[str] = None, limit: int = 60):
            self._require_api_token(request)
            sid = self._resolve_stream_id(stream_id=stream_id, target=target)
            return JSONResponse(await self._snapshot_introspection_logs_frontend(sid, limit=limit))

        # 兼容旧前端：/fragments 指向内省日志
        @router.get("/fragments")
        async def get_fragments(request: Request, stream_id: Optional[str] = None, target: Optional[str] = None, limit: int = 60):
            self._require_api_token(request)
            sid = self._resolve_stream_id(stream_id=stream_id, target=target)
            return JSONResponse(await self._snapshot_introspection_logs_frontend(sid, limit=limit))

        @router.get("/pulse")
        async def get_pulse(request: Request, stream_id: Optional[str] = None, target: Optional[str] = None):
            self._require_api_token(request)
            sid = self._resolve_stream_id(stream_id=stream_id, target=target)
            return JSONResponse(await self._snapshot_pulse_frontend(sid))

        @router.get("/targets")
        async def get_targets(request: Request, limit: int = 50, offset: int = 0):
            self._require_api_token(request)
            return JSONResponse(await self._snapshot_targets_frontend(limit=limit, offset=offset))

        return router

    def register_api_routes(self) -> None:
        if self._router_registered:
            return
        server = get_global_server()
        server.register_router(self.setup_router())
        self._router_registered = True
        if not self._cors_registered:
            self._register_cors_middleware(server.get_app())
            self._cors_registered = True


class SoulBackgroundTask(AsyncTask):
    def __init__(self, engine: SoulEngine):
        super().__init__(task_name="MaiSoulIntrospectionBackground", wait_before_start=1, run_interval=5)
        self.engine = engine

    async def run(self):
        await self.engine.tick_background()


_ENGINE: Optional[SoulEngine] = None


def get_engine(plugin_dir: Path) -> SoulEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SoulEngine(plugin_dir=plugin_dir)
    return _ENGINE


class SoulOnStartEventHandler(BaseEventHandler):
    event_type = EventType.ON_START
    handler_name = "mai_soul_engine_on_start"
    handler_description = "Mai-Soul: 注册 API 与后台任务"
    intercept_message = True
    weight = 100

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]:
        try:
            plugin_dir = Path(self.get_config("runtime.plugin_dir", "") or "") or Path(__file__).parent
            engine = get_engine(plugin_dir=plugin_dir)
            engine.set_config(self.plugin_config or {})
            if engine.get_config("persistence.enabled", True):
                engine.load_persisted()
            engine.register_api_routes()
            await async_task_manager.add_task(SoulBackgroundTask(engine))
            return True, True, None, None, None
        except Exception as e:
            logger.error(f"[soul] on_start failed: {e}", exc_info=True)
            return False, True, None, None, None


class SoulOnMessageEventHandler(BaseEventHandler):
    event_type = EventType.ON_MESSAGE
    handler_name = "mai_soul_engine_on_message"
    handler_description = "Mai-Soul: 记录群聊消息供思维内省回想"
    intercept_message = False
    weight = -10

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]:
        if not message or not message.is_group_message:
            return True, True, None, None, None
        try:
            plugin_dir = Path(self.get_config("runtime.plugin_dir", "") or "") or Path(__file__).parent
            engine = get_engine(plugin_dir=plugin_dir)
            engine.set_config(self.plugin_config or {})
            if not engine.is_enabled():
                return True, True, None, None, None

            text = (message.plain_text or "").strip()
            if not text:
                return True, True, None, None, None

            info = message.message_base_info if isinstance(message.message_base_info, dict) else {}
            platform = str(info.get("platform", "") or "")
            group_id = str(info.get("group_id", "") or "")
            user_id = str(info.get("user_id", "") or "")
            user_name = str(info.get("user_cardname", "") or "").strip() or str(info.get("user_nickname", "") or "").strip()
            stream_id = str(message.stream_id or "")
            if not (platform and group_id and user_id and stream_id):
                return True, True, None, None, None

            await engine.on_message(
                stream_id=stream_id,
                platform=platform,
                group_id=group_id,
                user_id=user_id,
                user_name=user_name,
                text=text,
            )
            return True, True, None, None, None
        except Exception as e:
            logger.debug(f"[soul] on_message failed: {e}")
            return True, True, None, None, None


class SoulPostLlmEventHandler(BaseEventHandler):
    event_type = EventType.POST_LLM
    handler_name = "mai_soul_engine_post_llm"
    handler_description = "Mai-Soul: 在回复前注入思想光谱与思维阁"
    intercept_message = True
    weight = 50

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]:
        if not message or not message.llm_prompt or not message.stream_id:
            return True, True, None, None, None
        try:
            plugin_dir = Path(self.get_config("runtime.plugin_dir", "") or "") or Path(__file__).parent
            engine = get_engine(plugin_dir=plugin_dir)
            engine.set_config(self.plugin_config or {})
            if not engine.is_enabled():
                return True, True, None, None, None
            if not bool(engine.get_config("injection.enabled", True)):
                return True, True, None, None, None

            trigger_text = str(getattr(message, "plain_text", "") or "")
            block = await engine.build_injection_block(message.stream_id, trigger_text=trigger_text)
            prompt = str(message.llm_prompt)

            marker = "现在，你说："
            idx = prompt.rfind(marker)
            if idx != -1:
                new_prompt = prompt[:idx].rstrip() + "\n\n" + block + "\n\n" + prompt[idx:]
            else:
                new_prompt = prompt.rstrip() + "\n\n" + block

            message.modify_llm_prompt(new_prompt, suppress_warning=True)
            return True, True, None, None, message
        except Exception as e:
            logger.debug(f"[soul] post_llm injection failed: {e}")
            return True, True, None, None, None


class SoulDebugCommand(BaseCommand):
    command_name = "soul_debug"
    command_description = "Mai-Soul 调试命令（status / introspect）"
    command_pattern = r"^[/／]soul(?:\\s+(?P<sub>help|status|introspect))?(?:\\s+(?P<arg>.+))?$"

    def _is_allowed(self) -> tuple[bool, str]:
        if not bool(self.get_config("debug.enabled", False)):
            return False, "debug_disabled"
        if not bool(self.get_config("debug.admin_only", True)):
            return True, "ok"
        platform = str(getattr(self.message.message_info, "platform", "") or "")
        user_info = getattr(self.message.message_info, "user_info", None)
        user_id = str(getattr(user_info, "user_id", "") or "")
        needle_full = f"{platform}:{user_id}" if platform and user_id else ""
        allowed = self.get_config("debug.admin_user_ids", [])
        if not isinstance(allowed, list):
            allowed = []
        allowed_set = {str(x).strip() for x in allowed if str(x).strip()}
        if user_id and user_id in allowed_set:
            return True, "ok"
        if needle_full and needle_full in allowed_set:
            return True, "ok"
        return False, "not_admin"

    def _get_engine(self) -> SoulEngine:
        plugin_dir = Path(self.get_config("runtime.plugin_dir", "") or "") or Path(__file__).parent
        engine = get_engine(plugin_dir=plugin_dir)
        engine.set_config(self.plugin_config or {})
        return engine

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        ok, reason = self._is_allowed()
        if not ok:
            return True, None, 0
        sub = str(self.matched_groups.get("sub") or "").strip().lower() or "help"
        engine = self._get_engine()
        if not engine.is_enabled():
            await self.send_text("Mai-Soul 当前未启用（[plugin].enabled=false）。")
            return True, "engine_disabled", 1
        stream_id = getattr(getattr(self.message, "chat_stream", None), "stream_id", None)
        if not stream_id:
            await self.send_text("无法获取 stream_id。")
            return True, "no_stream", 1

        if sub == "help":
            await self.send_text(
                "\n".join(
                    [
                        "Mai-Soul 调试命令：",
                        "- /soul status          查看当前群状态",
                        "- /soul introspect      强制执行一次思维内省",
                    ]
                )
            )
            return True, "help", 1

        if sub == "status":
            pulse = await engine._snapshot_pulse_frontend(str(stream_id))
            await self.send_text(json.dumps(pulse, ensure_ascii=False, indent=2)[:1500])
            return True, "status", 1

        if sub == "introspect":
            ok2 = await engine._introspection_run(str(stream_id), force=True)
            await self.send_text("思维内省执行完成。" if ok2 else "思维内省未执行（窗口为空或 LLM 失败）。")
            return True, "introspect", 1

        return True, "unknown", 1


@register_plugin
class MaiSoulEnginePlugin(BasePlugin):
    plugin_name = "mai_soul_engine"
    enable_plugin = True
    dependencies: List[str] = []
    python_dependencies: List[Any] = []
    config_file_name = "config.toml"

    config_section_descriptions = {
        "plugin": section_meta("插件开关", icon="settings", order=0),
        "api": section_meta("API", icon="cloud", order=1),
        "spectrum": section_meta("思想光谱", icon="activity", order=2),
        "cabinet": section_meta("思维阁", icon="archive", order=3),
        "introspection": section_meta("思维内省", icon="brain", order=4),
        "injection": section_meta("回复注入", icon="message-square", order=5),
        "persistence": section_meta("持久化", icon="save", order=6),
        "performance": section_meta("性能", icon="zap", order=7),
        "debug": section_meta("调试", icon="bug", order=99),
    }

    config_layout = ConfigLayout(
        tabs=[
            ConfigTab(id="general", title="General", icon="settings", order=0, sections=["plugin", "api"]),
            ConfigTab(id="mind", title="Mind", icon="brain", order=1, sections=["introspection", "cabinet", "spectrum", "injection"]),
            ConfigTab(id="ops", title="Ops", icon="zap", order=2, sections=["persistence", "performance", "debug"]),
        ]
    )

    config_schema = {
        "plugin": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用 Mai‑Soul（思维内省系统）", order=0),
        },
        "api": {
            "token": ConfigField(type=str, default="", description="API Token（留空则不鉴权；生产建议设置）", input_type="password", order=0),
            "cors_allow_origins": ConfigField(
                type=list,
                default=["http://localhost:5173", "http://127.0.0.1:5173"],
                description="允许跨域的 Origin 列表（['*'] 表示全部允许）",
                item_type="string",
                order=1,
            ),
        },
        "spectrum": {
            "clamp_abs": ConfigField(type=float, default=1.0, description="光谱数值夹紧（内部 [-1,1]）", min=0.2, max=1.0, order=0),
            "regression_rate": ConfigField(type=float, default=0.01, description="回归阻力：越大越快回到 base_tone", min=0.0, max=0.2, order=1),
            "ema_alpha": ConfigField(type=float, default=0.15, description="EMA 平滑系数", min=0.01, max=0.8, order=2),
            "fluctuation_window": ConfigField(type=int, default=30, description="波动率窗口（近 N 次）", min=10, max=200, order=3),
        },
        "cabinet": {
            "max_slots": ConfigField(type=int, default=6, description="思维阁槽位数（固化思想上限）", min=1, max=20, order=0),
            "max_seed_queue": ConfigField(type=int, default=30, description="思想种子队列上限", min=0, max=200, order=1),
            "seed_energy_threshold": ConfigField(type=float, default=0.5, description="生成思想种子所需最低 energy（0~1）", min=0.0, max=1.0, order=2),
            "internalize_seeds_per_run": ConfigField(type=int, default=1, description="每次内省最多内化几个种子", min=0, max=5, order=3),
            "internalization_rounds": ConfigField(type=int, default=3, description="内化轮数（>=2）", min=2, max=6, order=4),
            "temperature": ConfigField(type=float, default=0.35, description="内化温度", min=0.0, max=2.0, order=5),
            "max_tokens": ConfigField(type=int, default=60000, description="内化最大 tokens", min=128, max=200_000, order=6),
        },
        "introspection": {
            "interval_minutes": ConfigField(type=float, default=20.0, description="内省触发间隔（分钟）", min=1.0, max=24 * 60.0, order=0),
            "window_minutes": ConfigField(type=float, default=30.0, description="回看聊天时间窗（分钟）", min=1.0, max=24 * 60.0, order=1),
            "rounds": ConfigField(type=int, default=4, description="内省轮数（>=2）", min=2, max=8, order=2),
            "max_messages_per_group": ConfigField(type=int, default=500, description="单次内省最多读取消息条数", min=20, max=2000, order=3),
            "quiet_period_seconds": ConfigField(type=float, default=20.0, description="静默窗口（秒）：避免边聊边回想", min=0.0, max=3600.0, order=4),
            "max_groups_per_tick": ConfigField(type=int, default=1, description="每次 tick 最多启动几个群的内省任务", min=1, max=50, order=5),
            "temperature": ConfigField(type=float, default=0.7, description="内省温度", min=0.0, max=2.0, order=6),
            "max_tokens": ConfigField(type=int, default=700, description="内省最大 tokens", min=128, max=4096, order=7),
            "window_strength_scale": ConfigField(type=float, default=12.0, description="窗口规模强度缩放（越大越保守）", min=1.0, max=200.0, order=8),
            "use_main_personality": ConfigField(type=bool, default=True, description="是否读取主程序人格设定参与内省/内化", order=9),
            "pending_max_messages": ConfigField(type=int, default=600, description="每群最多缓存多少条待回想消息", min=50, max=2000, order=10),
            "max_log_items": ConfigField(type=int, default=300, description="最多保留多少条内省日志", min=50, max=5000, order=11),
        },
        "injection": {
            "enabled": ConfigField(type=bool, default=True, description="是否在回复前注入光谱与固化思想（POST_LLM）", order=0),
            "max_thought_details": ConfigField(type=int, default=2, description="最多注入几条相关固化思想详情", min=0, max=6, order=1),
            "max_chars": ConfigField(type=int, default=1400, description="注入块最大字符数", min=400, max=6000, order=2),
        },
        "persistence": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用持久化（state.json）", order=0),
            "save_interval_seconds": ConfigField(type=float, default=15.0, description="保存间隔（秒）", min=3.0, max=3600.0, order=1),
        },
        "performance": {
            "max_message_chars": ConfigField(type=int, default=800, description="单条消息最大字符数（入库前截断）", min=100, max=5000, order=0),
        },
        "debug": {
            "enabled": ConfigField(type=bool, default=False, description="是否启用 /soul 调试命令（生产建议关闭）", order=0),
            "admin_only": ConfigField(type=bool, default=True, description="调试命令是否仅管理员可用", order=1),
            "admin_user_ids": ConfigField(type=list, default=[], description="允许调试命令的用户列表（platform:user_id 或 user_id）", item_type="string", order=2),
        },
        "runtime": {"plugin_dir": ConfigField(type=str, default="", description="运行时注入：插件目录（一般不用填）", order=0)},
    }
