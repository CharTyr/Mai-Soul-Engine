import asyncio
import json
import math
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from src.chat.message_receive.chat_stream import get_chat_manager
from src.common.logger import get_logger
from src.common.server import get_global_server
from src.config.config import model_config
from src.manager.async_task_manager import AsyncTask, async_task_manager
from src.plugin_system import (
    BaseCommand,
    BaseEventHandler,
    BasePlugin,
    ConfigField,
    ConfigLayout,
    ConfigTab,
    EventHandlerInfo,
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

POLES: Dict[str, Tuple[str, str]] = {
    "sincerity_absurdism": ("Sincerity", "Absurdism"),
    "normies_otakuism": ("Normies", "Otakuism"),
    "traditionalism_radicalism": ("Traditionalism", "Radicalism"),
    "heroism_nihilism": ("Heroism", "Nihilism"),
}

AXES_FRONTEND: Dict[str, str] = {
    "sincerity_absurdism": "sincerity-absurdism",
    "normies_otakuism": "normies-otakuism",
    "traditionalism_radicalism": "traditionalism-radicalism",
    "heroism_nihilism": "heroism-nihilism",
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _now_ts() -> float:
    return time.time()


def _stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _parse_target(target: str) -> tuple[str, str, str]:
    parts = target.split(":", 2)
    if len(parts) != 3:
        raise ValueError("target 格式应为 'platform:id:type'，例如 'qq:123456:group'")
    platform, id_str, stream_type = parts[0].strip(), parts[1].strip(), parts[2].strip()
    if stream_type not in ("group", "private"):
        raise ValueError("target 的 type 必须是 'group' 或 'private'")
    return platform, id_str, stream_type


def _axis_payload_to_dict(obj: Any) -> Dict[str, float]:
    out = {a: 0.0 for a in AXES}
    if not isinstance(obj, dict):
        return out
    for axis in AXES:
        if axis in obj:
            try:
                out[axis] = _clamp(float(obj[axis]), -1.0, 1.0)
            except Exception:
                continue
    return out


@dataclass
class InfluenceProfile:
    score: float = 0.0
    weight: float = 0.0
    multiplier: float = 1.0
    contrib: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    quote: str = ""
    display_name: str = ""
    last_seen_ts: float = 0.0
    n_messages: int = 0
    stance_mean: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    stance_m2: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    tag_counts: Dict[str, int] = field(default_factory=dict)

    def update_stance(self, vector: Dict[str, float]) -> None:
        self.n_messages += 1
        n = float(self.n_messages)
        for axis in AXES:
            x = float(vector.get(axis, 0.0))
            mean = self.stance_mean.get(axis, 0.0)
            delta = x - mean
            mean = mean + delta / n
            self.stance_mean[axis] = mean
            delta2 = x - mean
            self.stance_m2[axis] = self.stance_m2.get(axis, 0.0) + delta * delta2

    def stance_std(self) -> Dict[str, float]:
        if self.n_messages < 2:
            return {a: 0.0 for a in AXES}
        denom = float(self.n_messages - 1)
        out = {}
        for axis in AXES:
            out[axis] = math.sqrt(max(0.0, float(self.stance_m2.get(axis, 0.0)) / denom))
        return out

    def stability(self) -> float:
        std = self.stance_std()
        avg = sum(std.values()) / float(len(AXES))
        return float(_clamp(1.0 / (1.0 + avg), 0.0, 1.0))


@dataclass
class ThoughtTrait:
    trait_id: str
    axis: str
    direction: float
    created_ts: float
    label: str
    name: str = ""
    style_hint: str = ""
    definition: str = ""
    digest: str = ""
    last_reflect_ts: float = 0.0
    reflect_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    topic: str = ""
    verdict: str = ""
    shift: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})


@dataclass
class ThoughtSlot:
    slot_id: str
    axis: str
    direction: float
    created_ts: float
    due_ts: float
    progress: float = 0.0
    label: str = ""
    mode: str = "axis_pressure"
    status: str = "loading"
    seed_topic: str = ""
    seed_tags: List[str] = field(default_factory=list)
    fragments: List[str] = field(default_factory=list)
    energy: float = 0.0
    keyword_cloud: Dict[str, int] = field(default_factory=dict)
    logic_points: List[str] = field(default_factory=list)
    evaluator_runs: int = 0
    evaluator_required_runs: int = 5
    ripeness: float = 0.0
    deltas_sum: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    confidence_sum: float = 0.0
    final_shift: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    last_mastication_ts: float = 0.0
    last_attempt_ts: float = 0.0
    verdict: str = ""
    final_verdict: str = ""
    trait_name: str = ""
    style_hint: str = ""
    awaiting_approval: bool = False
    approved: bool = False
    rejected: bool = False
    rejected_reason: str = ""


@dataclass
class DissonanceState:
    active: bool = False
    axis: str = ""
    severity: float = 0.0
    reason: str = ""
    expires_ts: float = 0.0


@dataclass
class Fragment:
    ts: float
    text: str
    tags: List[str] = field(default_factory=list)


@dataclass
class GroupSoulState:
    base_tone: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    values: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    ema: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in AXES})
    recent_deltas: Dict[str, List[float]] = field(default_factory=lambda: {a: [] for a in AXES})
    last_update_ts: float = 0.0

    target: str = ""
    initialized: bool = False
    last_fragment_attempt_ts: float = 0.0
    social_last_recompute_ts: float = 0.0

    influences: Dict[str, InfluenceProfile] = field(default_factory=dict)
    tag_cloud: Dict[str, int] = field(default_factory=dict)
    quote_bank: List[Dict[str, Any]] = field(default_factory=list)
    prototypes: List[Dict[str, Any]] = field(default_factory=list)
    sociolect_profile: Dict[str, Any] = field(default_factory=dict)
    sociolect_last_update_ts: float = 0.0
    sociolect_last_attempt_ts: float = 0.0
    slots: List[ThoughtSlot] = field(default_factory=list)
    traits: List[ThoughtTrait] = field(default_factory=list)
    dissonance: DissonanceState = field(default_factory=DissonanceState)
    fragments: List[Fragment] = field(default_factory=list)

    pressure_streaks: Dict[str, int] = field(default_factory=lambda: {a: 0 for a in AXES})
    seed_recent_ts: Dict[str, float] = field(default_factory=dict)
    last_incoming_tags: List[str] = field(default_factory=list)
    last_incoming_user_id: str = ""
    last_incoming_ts: float = 0.0
    message_count: int = 0
    pending_messages: List[Dict[str, Any]] = field(default_factory=list)
    last_dream_processed_ts: float = 0.0
    dream_trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SoulEvent:
    stream_id: str
    platform: str
    group_id: str
    user_id: str
    text: str
    ts: float
    user_nickname: str = ""
    user_cardname: str = ""


class SoulEngine:
    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self._boot_ts: float = _now_ts()
        self._config: Dict[str, Any] = {}
        self._groups: Dict[str, GroupSoulState] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._tick_lock = asyncio.Lock()
        self._queues: Dict[str, asyncio.Queue[SoulEvent]] = {}
        self._workers: Dict[str, asyncio.Task] = {}
        self._fragment_tasks: Dict[str, asyncio.Task] = {}
        self._trait_tasks: Dict[str, asyncio.Task] = {}
        self._rethink_tasks: Dict[str, asyncio.Task] = {}
        self._sociolect_tasks: Dict[str, asyncio.Task] = {}
        self._cabinet_tasks: Dict[str, asyncio.Task] = {}
        self._global_trait_task: Optional[asyncio.Task] = None
        self._global_rethink_task: Optional[asyncio.Task] = None

        # 仅用于“思维阁-意识形态实验室”的短期上下文采样（不持久化）
        self._recent_context: Dict[str, List[Dict[str, Any]]] = {}

        # default_scope=global 时启用：跨所有群共享的“长期底色偏移”
        self._global_base_offset: Dict[str, float] = {a: 0.0 for a in AXES}
        self._global_traits: List[ThoughtTrait] = []
        self._global_daily_day: str = ""
        self._global_daily_used_abs: Dict[str, float] = {a: 0.0 for a in AXES}
        self._global_pending_value_shift: Dict[str, float] = {a: 0.0 for a in AXES}

        # Dream 联动（运行时 hook，不修改主程序文件）
        self._dream_hook_ok: bool = False
        self._dream_hook_reason: str = ""
        self._dream_last_cycle_ts: float = 0.0
        self._dream_last_phase: str = ""
        self._dream_cycle_start_ts: float = 0.0
        self._dream_cycle_end_ts: float = 0.0
        self._dream_cycle_theme: str = ""
        self._dream_cycle_theme_score: float = -1.0
        self._dream_cycle_fragments_generated: int = 0

        self._router_registered = False
        self._cors_registered = False
        self._api_router = APIRouter()

        self._last_persist_ts = 0.0
        self._state_file = self.plugin_dir / "data" / "state.json"

    def set_config(self, config: Dict[str, Any]) -> None:
        self._config = config or {}

    def get_config(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        current: Any = self._config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def _cabinet_mode(self) -> str:
        mode = str(self.get_config("cabinet.mode", "ideology_lab") or "ideology_lab").strip()
        return mode if mode in ("ideology_lab", "axis_pressure") else "ideology_lab"

    def _spectrum_update_mode(self) -> str:
        mode = str(self.get_config("spectrum.update_mode", "dream_window") or "dream_window").strip()
        if mode not in ("per_message", "dream_window"):
            mode = "dream_window"
        return mode

    def _spectrum_batch_effective(self) -> bool:
        # dream_window 模式：不逐条消息调用 LLM，统一在 Dream 周期批处理
        return self._spectrum_update_mode() == "dream_window"

    def _default_scope(self) -> str:
        scope = str(self.get_config("scope.default_scope", "group") or "group").strip()
        return scope if scope in ("group", "global") else "group"

    def _global_scope_active(self) -> bool:
        return self._default_scope() == "global" and bool(self.get_config("scope.enable_global", True))

    def _effective_base_tone(self, state: GroupSoulState) -> Dict[str, float]:
        if not self._global_scope_active():
            return dict(state.base_tone)
        out = {}
        for axis in AXES:
            out[axis] = _clamp(float(state.base_tone.get(axis, 0.0)) + float(self._global_base_offset.get(axis, 0.0)), -1.0, 1.0)
        return out

    def _rollover_global_daily_cap_if_needed(self, now: float) -> None:
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        if day != self._global_daily_day:
            self._global_daily_day = day
            self._global_daily_used_abs = {a: 0.0 for a in AXES}

    def _global_whitelist(self) -> List[str]:
        wl = self.get_config("cabinet.global_whitelist_targets", [])
        if not isinstance(wl, list):
            return []
        return [str(x).strip() for x in wl if str(x).strip()]

    def _global_assimilation_allowed(self, state: GroupSoulState, slot: ThoughtSlot) -> tuple[bool, str]:
        if not self._global_scope_active():
            return True, "not_global_scope"
        wl = self._global_whitelist()
        if wl and str(state.target or "") not in wl:
            return False, "not_in_global_whitelist"
        min_energy = float(self.get_config("cabinet.global_min_slot_energy", 0.0))
        if min_energy > 0 and float(slot.energy or 0.0) < min_energy:
            return False, "slot_energy_too_low"
        min_conf = float(self.get_config("cabinet.global_min_avg_confidence", 0.0))
        if min_conf > 0:
            runs = max(1, int(slot.evaluator_runs or 0))
            avg_conf = float(slot.confidence_sum or 0.0) / float(runs)
            if avg_conf < min_conf:
                return False, "avg_confidence_too_low"
        return True, "ok"

    def _apply_global_daily_cap(self, shift: Dict[str, float], now: float) -> Dict[str, float]:
        cap = float(self.get_config("cabinet.global_daily_shift_cap_abs", 0.0))
        if cap <= 0:
            return shift
        cap = _clamp(cap, 0.0, 1.0)
        self._rollover_global_daily_cap_if_needed(now)
        out = {a: 0.0 for a in AXES}
        for axis in AXES:
            s = float(shift.get(axis, 0.0) or 0.0)
            used = float(self._global_daily_used_abs.get(axis, 0.0) or 0.0)
            remaining = max(0.0, cap - used)
            applied = _clamp(s, -remaining, remaining)
            out[axis] = applied
            self._global_daily_used_abs[axis] = used + abs(applied)
        return out

    def _dream_integration_enabled(self) -> bool:
        return bool(self.get_config("dream.integrate_with_dream", False))

    def _dream_full_bind(self) -> bool:
        # full_bind 只有在 hook 成功后才“生效”，避免 hook 失败导致插件后台完全停摆
        return self._dream_integration_enabled() and bool(self.get_config("dream.full_bind", False)) and self._dream_hook_ok

    def _dream_bind_cabinet(self) -> bool:
        if self._dream_full_bind():
            return self._dream_hook_ok
        return self._dream_integration_enabled() and self._dream_hook_ok and bool(self.get_config("dream.bind_cabinet_to_dream", True))

    def _dream_bind_blackbox(self) -> bool:
        if self._dream_full_bind():
            return self._dream_hook_ok
        return self._dream_integration_enabled() and self._dream_hook_ok and bool(self.get_config("dream.bind_blackbox_to_dream", True))

    def try_hook_dream(self) -> bool:
        if not self._dream_integration_enabled():
            return False
        try:
            from src.dream import dream_agent  # type: ignore
        except Exception as e:
            self._dream_hook_ok = False
            self._dream_hook_reason = f"import dream_agent failed: {e}"
            return False

        if getattr(dream_agent, "_mai_soul_engine_hooked", False):
            self._dream_hook_ok = True
            self._dream_hook_reason = "already hooked"
            return True

        engine = self

        original_cycle = getattr(dream_agent, "run_dream_cycle_once", None)
        if original_cycle and asyncio.iscoroutinefunction(original_cycle):
            async def _wrapped_run_dream_cycle_once(*args, **kwargs):
                await engine._on_dream_cycle(phase="start")
                try:
                    return await original_cycle(*args, **kwargs)
                finally:
                    await engine._on_dream_cycle(phase="end")

            setattr(dream_agent, "run_dream_cycle_once", _wrapped_run_dream_cycle_once)
            setattr(dream_agent, "_mai_soul_engine_hooked", True)
            self._dream_hook_ok = True
            self._dream_hook_reason = "hooked run_dream_cycle_once"
            logger.info("[soul] dream integration enabled: hooked dream.run_dream_cycle_once")
            return True

        # fallback：如果主程序未来改了函数名，尽量兜底
        original_once = getattr(dream_agent, "run_dream_agent_once", None)
        if original_once and asyncio.iscoroutinefunction(original_once):
            async def _wrapped_run_dream_agent_once(*args, **kwargs):
                await engine._on_dream_cycle(phase="start")
                try:
                    return await original_once(*args, **kwargs)
                finally:
                    await engine._on_dream_cycle(phase="end")

            setattr(dream_agent, "run_dream_agent_once", _wrapped_run_dream_agent_once)
            setattr(dream_agent, "_mai_soul_engine_hooked", True)
            self._dream_hook_ok = True
            self._dream_hook_reason = "hooked run_dream_agent_once (fallback)"
            logger.info("[soul] dream integration enabled: hooked dream.run_dream_agent_once (fallback)")
            return True

        self._dream_hook_ok = False
        self._dream_hook_reason = "no hook target found (run_dream_cycle_once/run_dream_agent_once missing)"
        return False

    async def _on_dream_cycle(self, *, phase: str) -> None:
        # 不要阻塞 dream：这里的任务尽量小步快跑
        try:
            if phase not in ("start", "end"):
                return
            if not self.is_enabled():
                return
            self._dream_last_cycle_ts = _now_ts()
            self._dream_last_phase = phase
            if phase == "start":
                self._dream_cycle_start_ts = self._dream_last_cycle_ts
                self._dream_cycle_theme = ""
                self._dream_cycle_theme_score = -1.0
                self._dream_cycle_fragments_generated = 0

            # Dream 时刻：批处理光谱（从上次 Dream 到本次 Dream 的消息窗口）
            if phase == "start" and self._spectrum_batch_effective():
                await self._dream_spectrum_window_sweep()

            # Dream 时刻：触发思维阁品鉴与黑盒片段（如果启用了 bind）
            if self._dream_bind_cabinet() and bool(self.get_config("cabinet.enabled", True)) and self._cabinet_mode() == "ideology_lab":
                steps = int(self.get_config("dream.cabinet_steps_per_cycle", 2))
                steps = max(0, min(steps, 20))
                for _ in range(steps):
                    ok = await self._mastication_sweep_once(force=True)
                    if not ok:
                        break

                # Dream 时刻：补齐“已固化 trait”的标题/风格偏置/内化结果（避免 full_bind 下在 Dream 外消耗 LLM）
                enrich_steps = int(self.get_config("dream.enrich_steps_per_cycle", 1))
                enrich_steps = max(0, min(enrich_steps, 10))
                for _ in range(enrich_steps):
                    eok = await self._enrich_sweep_once()
                    if not eok:
                        break
                if self._global_scope_active() and bool(self.get_config("cabinet.digest_with_llm", True)):
                    try:
                        await self._enrich_next_global_trait()
                    except Exception:
                        pass

                # Dream 时刻：可选“重新反思”（对已固化 trait 重新生成思想定义/内化结果）
                if bool(self.get_config("cabinet.rethink_enabled", False)):
                    r_steps = int(self.get_config("dream.rethink_steps_per_cycle", 1))
                    r_steps = max(0, min(r_steps, 10))
                    for _ in range(r_steps):
                        rok = await self._rethink_sweep_once(force=False)
                        if not rok:
                            break
                    if self._global_scope_active():
                        try:
                            await self._rethink_next_global_trait(force=False)
                        except Exception:
                            pass

            if self._dream_bind_blackbox() and bool(self.get_config("blackbox.enabled", True)):
                n = int(self.get_config("dream.blackbox_groups_per_cycle", 1))
                n = max(0, min(n, 50))
                force = bool(self.get_config("dream.force_blackbox_on_dream", False))
                generated = await self._blackbox_sweep(max_groups=n, force=force)
                self._dream_cycle_fragments_generated = int(self._dream_cycle_fragments_generated or 0) + int(generated or 0)

            if phase == "end":
                self._dream_cycle_end_ts = self._dream_last_cycle_ts
        except Exception as e:
            logger.debug(f"[soul] dream hook failed phase={phase}: {e}")

    async def _mastication_sweep_once(self, *, force: bool) -> bool:
        now = _now_ts()
        # 选一个“最值得品鉴”的群：优先有 slot 且能量高/成熟度高
        best_sid = ""
        best_score = -1.0
        async with self._tick_lock:
            for sid, state in self._groups.items():
                lock = self._get_lock(sid)
                async with lock:
                    for slot in state.slots:
                        if slot.mode != "ideology_lab":
                            continue
                        if slot.rejected or slot.approved or slot.awaiting_approval:
                            continue
                        if slot.evaluator_runs >= max(1, int(slot.evaluator_required_runs or 1)):
                            continue
                        if len(slot.fragments or []) < max(1, int(self.get_config("cabinet.mastication_min_fragments", 6))):
                            continue
                        score = float(slot.ripeness or 0.0) * 2.0 + float(slot.energy or 0.0)
                        if score > best_score:
                            best_score = score
                            best_sid = sid
        if not best_sid:
            return False
        await self._mastication_once(best_sid, force=force)
        return True

    async def _dream_spectrum_window_sweep(self) -> None:
        # 每次 Dream 周期最多处理多少个群，避免一次 Dream 太重
        n_groups = int(self.get_config("dream.spectrum_groups_per_cycle", 3))
        n_groups = max(0, min(n_groups, 50))
        if n_groups <= 0:
            return

        picked: List[str] = []
        async with self._tick_lock:
            for sid, state in self._groups.items():
                lock = self._get_lock(sid)
                async with lock:
                    if not (state.pending_messages or []):
                        continue
                    picked.append(sid)
                    if len(picked) >= n_groups:
                        break
        for sid in picked:
            try:
                await self._process_spectrum_window_for_group(sid)
            except Exception as e:
                logger.debug(f"[soul] dream spectrum window failed sid={sid}: {e}")

    async def _process_spectrum_window_for_group(self, stream_id: str) -> None:
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            pending = list(state.pending_messages or [])
            if not pending:
                return
            # 取最近 N 条，避免 prompt 过长
            max_msgs = int(self.get_config("dream.spectrum_max_messages_per_group", 80))
            max_msgs = max(10, min(max_msgs, 200))
            pending = pending[-max_msgs:]

            # 窗口起止时间（用于日志）
            window_from = float(state.last_dream_processed_ts or 0.0) or float(pending[0].get("ts", 0.0) or 0.0)
            window_to = float(pending[-1].get("ts", 0.0) or 0.0)

            # 快照
            snapshot_values = dict(state.values)
            snapshot_traits = [t.name or t.label for t in (state.traits[-6:] if state.traits else [])]
            state.pending_messages = []
            state.last_dream_processed_ts = max(float(state.last_dream_processed_ts or 0.0), float(window_to or 0.0))

        analysis = await self._semantic_shift_window(
            values=snapshot_values,
            traits=snapshot_traits,
            messages=pending,
        )
        if not analysis:
            async with lock:
                state = self._get_group(stream_id)
                # 恢复 pending（避免丢消息）
                state.pending_messages = list(pending) + list(state.pending_messages or [])
            return

        # 记录 Dream 思考过程（用于黑盒 UI）
        async with lock:
            state = self._get_group(stream_id)
            summary = str(analysis.get("summary", "") or "").strip()
            if summary:
                self._append_trace_locked(
                    state,
                    kind="spectrum_window",
                    text=f"【理性】窗口分析（{len(pending)}条）：{summary}",
                    tags=["dream", "rational"],
                    extra={"window_from": window_from, "window_to": window_to},
                )
            try:
                score = float(analysis.get("intensity", 0.0) or 0.0) * float(analysis.get("confidence", 0.0) or 0.0)
            except Exception:
                score = 0.0
            if score >= float(self._dream_cycle_theme_score or -1.0):
                theme = ""
                topics = analysis.get("topics", [])
                if isinstance(topics, list) and topics:
                    t0 = topics[0]
                    if isinstance(t0, dict):
                        theme = str(t0.get("topic", "") or "").strip()
                if not theme:
                    theme = summary
                self._dream_cycle_theme = str(theme or "").strip()[:40]
                self._dream_cycle_theme_score = score

        await self._apply_window_analysis(stream_id, analysis=analysis, messages=pending)

    async def _semantic_shift_window(
        self,
        *,
        values: Dict[str, float],
        traits: List[str],
        messages: List[Dict[str, Any]],
    ) -> Optional[dict]:
        model_task = str(self.get_config("llm.model_task", "utils")).strip() or "utils"
        models = llm_api.get_available_models()
        task_config = models.get(model_task) or model_config.model_task_config.utils

        max_tokens = int(self.get_config("llm.max_tokens", 512))
        temperature = float(self.get_config("llm.temperature", 0.2))

        # 输入最小化：只传 id/name/text
        items = []
        for m in messages[-200:]:
            if not isinstance(m, dict):
                continue
            text = str(m.get("text", "") or "").strip()
            if not text:
                continue
            items.append(
                {
                    "user_id": str(m.get("user_id", "") or ""),
                    "name": (str(m.get("user_cardname", "") or "").strip() or str(m.get("user_nickname", "") or "").strip() or str(m.get("user_id", "") or "")),
                    "text": text,
                }
            )
        if not items:
            return None

        prompt = f"""
你是“群聊意识形态窗口分析器”。你拿到的是“从上一次做梦到这一次做梦”的聊天记录片段，你要一次性总结出：
1) 这段窗口整体对四轴的位移 group_deltas（-1~1；负值更偏建构/现充/传统/英雄；正值更偏解构/二次元/赛博激进/虚无）
2) 这段窗口里最能代表“热议话题”的 topics（最多3个），用于思维阁候选。每个 topic 给出：
   - topic 名称（短词）
   - tags（最多6个）
   - energy（0~1，越高越热）
   - fragments（最多10条短切片，尽量抽象转述，不要复刻原句）
3) 灵魂雕刻师候选 users（最多10个）：每个用户给出 contribution_deltas（四轴，-1~1）、代表语录 quote（尽量转述）、tags（最多6个）
4) tags：窗口整体的语义标签（最多10个）
5) summary：用1~3句中文总结“这段窗口的氛围/冲突点”，适合放进黑盒日志
6) confidence/intensity：0~1

当前坐标（背景，不要复述）：{json.dumps(values, ensure_ascii=False)}
已固化思维 traits（背景，不要复述）：{json.dumps(traits, ensure_ascii=False)}
窗口消息（已脱敏/截断）：{json.dumps(items, ensure_ascii=False)}

硬性要求：
- 不要输出任何群名、任何可识别的个人信息、任何原句复刻、链接/号码/邮箱。
- 输出必须是严格 JSON，不要 Markdown，不要解释。

JSON schema：
{{
  "group_deltas": {{"sincerity_absurdism":0.0,"normies_otakuism":0.0,"traditionalism_radicalism":0.0,"heroism_nihilism":0.0}},
  "confidence": 0.0,
  "intensity": 0.0,
  "tags": ["..."],
  "summary": "...",
  "topics": [{{"topic":"...","tags":["..."],"energy":0.0,"fragments":["..."]}}],
  "users": [{{"user_id":"...","name":"...","contribution_deltas":{{...}},"quote":"...","tags":["..."]}}]
}}
"""
        ok, content, _reasoning, model_name = await llm_api.generate_with_model(
            prompt,
            model_config=task_config,
            request_type="mai_soul_engine.spectrum.window",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not ok:
            logger.debug(f"[soul] spectrum window failed model={model_name}: {content[:80]}")
            return None
        return _extract_json_object(content)

    async def _apply_window_analysis(self, stream_id: str, *, analysis: dict, messages: List[Dict[str, Any]]) -> None:
        lock = self._get_lock(stream_id)
        now = _now_ts()
        async with lock:
            state = self._get_group(stream_id)
            deltas = analysis.get("group_deltas", {})
            if not isinstance(deltas, dict):
                return
            intensity = float(analysis.get("intensity", 0.3) or 0.0)
            confidence = float(analysis.get("confidence", 0.3) or 0.0)
            tags = analysis.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            tags = [str(t)[:32] for t in tags if str(t).strip()][:12]

            # 窗口规模影响：消息越多，越“显著”，但要钝化
            k = float(self.get_config("spectrum.window_strength_scale", 12.0))
            k = max(1.0, min(k, 200.0))
            scale = math.log1p(len(messages)) / k
            scale = float(_clamp(scale, 0.05, 1.0))

            regression = float(self.get_config("spectrum.regression_rate", 0.01))
            ema_alpha = float(self.get_config("spectrum.ema_alpha", 0.15))
            clamp_abs = float(self.get_config("spectrum.clamp_abs", 1.0))
            window = int(self.get_config("spectrum.fluctuation_window", 30))

            effective = _clamp(intensity, 0.0, 1.0) * _clamp(confidence, 0.0, 1.0) * scale
            if effective <= 0:
                return

            for axis in AXES:
                try:
                    delta = float(deltas.get(axis, 0.0))
                except Exception:
                    delta = 0.0
                delta = _clamp(delta, -1.0, 1.0)
                state.values[axis] = _clamp(float(state.values.get(axis, 0.0)) + delta * effective, -clamp_abs, clamp_abs)
                base_target = float(state.base_tone.get(axis, 0.0))
                if self._global_scope_active():
                    base_target = _clamp(base_target + float(self._global_base_offset.get(axis, 0.0)), -1.0, 1.0)
                state.values[axis] = state.values[axis] + (base_target - state.values[axis]) * regression
                state.ema[axis] = (1 - ema_alpha) * float(state.ema.get(axis, 0.0)) + ema_alpha * float(state.values.get(axis, 0.0))
                state.recent_deltas[axis].append(delta * effective)
                if len(state.recent_deltas[axis]) > window:
                    state.recent_deltas[axis] = state.recent_deltas[axis][-window:]

            try:
                last_msg_ts = float(messages[-1].get("ts", now) or now) if messages else now
            except Exception:
                last_msg_ts = now
            try:
                last_msg_uid = str(messages[-1].get("user_id", "") or "").strip() if messages else ""
            except Exception:
                last_msg_uid = ""

            state.last_update_ts = float(last_msg_ts or now)
            state.last_incoming_tags = tags
            state.last_incoming_ts = float(last_msg_ts or now)
            if last_msg_uid:
                state.last_incoming_user_id = last_msg_uid

            # 更新群级标签云
            for tag in tags:
                state.tag_cloud[tag] = int(state.tag_cloud.get(tag, 0)) + 1

            # 更新灵魂雕刻师（影响力/代表语录/立场）
            users = analysis.get("users", [])
            if not isinstance(users, list):
                users = []
            for u in users[:50]:
                if not isinstance(u, dict):
                    continue
                uid = str(u.get("user_id", "") or "").strip()
                if not uid:
                    continue
                name = str(u.get("name", "") or "").strip()
                cd = u.get("contribution_deltas", {})
                if not isinstance(cd, dict):
                    cd = {}
                quote = str(u.get("quote", "") or "").strip()
                utags = u.get("tags", [])
                if not isinstance(utags, list):
                    utags = []
                utags = [str(t)[:32] for t in utags if str(t).strip()][:12]

                user_mult = self._admin_multiplier(state.target.split(":", 1)[0] if state.target else "", uid)
                influence_weight = float(self.get_config("influence.base_user_weight", 1.0)) * user_mult

                profile = state.influences.get(uid) or InfluenceProfile()
                profile.last_seen_ts = now
                if name:
                    profile.display_name = name[:32]
                decay = float(self.get_config("influence.weight_decay", 0.995))
                profile.score = float(profile.score or 0.0) * decay + effective
                profile.multiplier = float(user_mult or 1.0)
                profile.weight = float(profile.weight or 0.0) * decay + effective * influence_weight

                stance_vec = {}
                for axis in AXES:
                    try:
                        d = float(cd.get(axis, 0.0))
                    except Exception:
                        d = 0.0
                    stance_vec[axis] = _clamp(d, -1.0, 1.0) * _clamp(confidence, 0.0, 1.0)
                    profile.contrib[axis] = float(profile.contrib.get(axis, 0.0)) + stance_vec[axis]
                profile.update_stance(stance_vec)
                if quote:
                    profile.quote = quote[:120]
                    state.quote_bank.append(
                        {
                            "ts": now,
                            "user_id": uid,
                            "user_name": (profile.display_name or name or uid)[:32],
                            "quote": quote[:120],
                            "tags": utags,
                            "deltas": {a: float(cd.get(a, 0.0) or 0.0) for a in AXES},
                            "intensity": intensity,
                            "confidence": confidence,
                        }
                    )
                for t in utags:
                    profile.tag_counts[t] = int(profile.tag_counts.get(t, 0)) + 1
                state.influences[uid] = profile

            max_quotes = int(self.get_config("social.max_quote_bank", 200))
            if len(state.quote_bank) > max_quotes:
                state.quote_bank = state.quote_bank[-max_quotes:]

            # 生成思维阁候选（topics -> slots）
            if bool(self.get_config("cabinet.enabled", True)) and self._cabinet_mode() == "ideology_lab":
                topics = analysis.get("topics", [])
                if not isinstance(topics, list):
                    topics = []
                max_slots = int(self.get_config("cabinet.max_slots", 3))
                active_slots = [s for s in state.slots if (not s.rejected) and (not s.approved)]
                for tp in topics[:5]:
                    if len(active_slots) >= max_slots:
                        break
                    if not isinstance(tp, dict):
                        continue
                    topic = str(tp.get("topic", "") or "").strip()
                    if not topic:
                        continue
                    # 去重
                    if any((s.seed_topic or "") == topic for s in active_slots):
                        continue
                    energy = float(tp.get("energy", 0.0) or 0.0)
                    seed_threshold = float(self.get_config("cabinet.seed_energy_threshold", 0.25))
                    if energy < seed_threshold:
                        continue
                    ttags = tp.get("tags", [])
                    if not isinstance(ttags, list):
                        ttags = []
                    ttags = [str(t)[:32] for t in ttags if str(t).strip()][:12]
                    frags = tp.get("fragments", [])
                    if not isinstance(frags, list):
                        frags = []
                    frags2 = [self._sanitize_fragment_text(str(x)) for x in frags if str(x).strip()]
                    frags2 = [f for f in frags2 if f][: int(self.get_config("cabinet.seed_fragment_count", 12))]

                    required_runs = int(self.get_config("cabinet.mastication_required_runs", 5))
                    required_runs = max(1, min(required_runs, 20))
                    seed_need_approval = bool(self.get_config("cabinet.seed_require_manual_approval", False))
                    slot_id = f"seed:{topic}:{int(now)}"
                    state.slots.append(
                        ThoughtSlot(
                            slot_id=slot_id,
                            axis="seed",
                            direction=0.0,
                            created_ts=now,
                            due_ts=now + float(self.get_config("cabinet.internalization_minutes", 30.0)) * 60.0,
                            progress=0.0,
                            label=f"seed:{topic}",
                            mode="ideology_lab",
                            status="awaiting_seed_approval" if seed_need_approval else "fermenting",
                            seed_topic=topic,
                            seed_tags=ttags or [topic],
                            fragments=frags2,
                            energy=float(energy),
                            keyword_cloud={t: 1 for t in (ttags or [topic])},
                            evaluator_runs=0,
                            evaluator_required_runs=required_runs,
                            ripeness=0.0,
                            awaiting_approval=bool(seed_need_approval),
                        )
                    )
                    active_slots.append(state.slots[-1])
                    self._append_trace_locked(
                        state,
                        kind="seed_created",
                        text=f"【记忆】热议话题被标记为思维种子：{topic}",
                        tags=["dream", "memory"],
                        extra={"topic": topic, "energy": energy},
                    )

    async def _blackbox_sweep(self, *, max_groups: int, force: bool) -> int:
        if max_groups <= 0:
            return 0
        now = _now_ts()
        picked: List[str] = []
        async with self._tick_lock:
            for sid, state in self._groups.items():
                lock = self._get_lock(sid)
                async with lock:
                    if len(picked) >= max_groups:
                        break
                    if force or self._should_generate_fragment_locked(state, now):
                        picked.append(sid)
        generated = 0
        for sid in picked:
            frag = await self._generate_and_store_fragment(sid, force=force)
            if frag:
                generated += 1
                lock = self._get_lock(sid)
                async with lock:
                    state = self._get_group(sid)
                    self._append_trace_locked(
                        state,
                        kind="blackbox_fragment",
                        text=f"【潜意识】{frag.text}",
                        tags=["dream", "subconscious"] + list(frag.tags or [])[:6],
                    )
        return generated

    def is_enabled(self) -> bool:
        return bool(self.get_config("plugin.enabled", True))

    def _get_lock(self, stream_id: str) -> asyncio.Lock:
        if stream_id not in self._locks:
            self._locks[stream_id] = asyncio.Lock()
        return self._locks[stream_id]

    def _get_group(self, stream_id: str) -> GroupSoulState:
        if stream_id not in self._groups:
            self._groups[stream_id] = GroupSoulState()
        return self._groups[stream_id]

    def _maybe_initialize_group_locked(self, state: GroupSoulState, *, platform: str, group_id: str) -> None:
        if state.initialized:
            return
        target = f"{platform}:{group_id}:group"
        state.target = target

        base = self._parse_axis_json(self.get_config("spectrum.base_tone_default_json", "{}"))
        overrides = self.get_config("spectrum.base_tone_overrides_json", "{}")
        override_map = {}
        if isinstance(overrides, dict):
            override_map = overrides
        elif isinstance(overrides, str):
            override_map = self._parse_json_maybe(overrides) or {}
        if isinstance(override_map, dict) and target in override_map and isinstance(override_map[target], dict):
            base.update(self._coerce_axis_dict(override_map[target]))

        state.base_tone = base

        if bool(self.get_config("spectrum.initialize_from_base_tone", True)):
            effective_base = base
            if self._global_scope_active():
                effective_base = self._effective_base_tone(state)
            state.values = dict(effective_base)
            state.ema = dict(effective_base)
            state.recent_deltas = {a: [] for a in AXES}

        state.initialized = True

    def _parse_json_maybe(self, s: str) -> Optional[dict]:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _coerce_axis_dict(self, obj: dict) -> Dict[str, float]:
        out = {a: 0.0 for a in AXES}
        for axis in AXES:
            if axis in obj:
                try:
                    out[axis] = _clamp(float(obj[axis]), -1.0, 1.0)
                except Exception:
                    out[axis] = out[axis]
        return out

    def _parse_axis_json(self, raw: Any) -> Dict[str, float]:
        if isinstance(raw, dict):
            return self._coerce_axis_dict(raw)
        if isinstance(raw, str):
            obj = self._parse_json_maybe(raw.strip())
            if isinstance(obj, dict):
                return self._coerce_axis_dict(obj)
        return {a: 0.0 for a in AXES}

    def _ensure_group_worker(self, stream_id: str) -> None:
        if stream_id in self._workers and not self._workers[stream_id].done():
            return
        maxsize = int(self.get_config("performance.queue_maxsize", 200))
        self._queues[stream_id] = asyncio.Queue(maxsize=maxsize)
        self._workers[stream_id] = asyncio.create_task(self._worker_loop(stream_id))

    async def on_message(self, event: SoulEvent) -> None:
        if not self.is_enabled():
            return
        # Dream-window 模式：只记录消息，等到 Dream 周期再统一批处理（不逐条调用 LLM）
        if self._spectrum_batch_effective():
            lock = self._get_lock(event.stream_id)
            async with lock:
                state = self._get_group(event.stream_id)
                self._maybe_initialize_group_locked(state, platform=event.platform, group_id=event.group_id)
                self._append_pending_message_locked(state, event)
                state.last_update_ts = float(event.ts or _now_ts())
            return
        self._ensure_group_worker(event.stream_id)
        q = self._queues.get(event.stream_id)
        if not q:
            return
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            drop_policy = str(self.get_config("performance.queue_drop_policy", "drop_new")).strip()
            if drop_policy == "drop_old":
                try:
                    _ = q.get_nowait()
                except Exception:
                    return
                try:
                    q.put_nowait(event)
                except Exception:
                    return
            return

    async def _worker_loop(self, stream_id: str) -> None:
        min_interval = float(self.get_config("performance.per_group_min_interval_seconds", 0.5))
        last_ts = 0.0
        while True:
            q = self._queues.get(stream_id)
            if not q:
                return
            event = await q.get()
            if min_interval > 0:
                gap = _now_ts() - last_ts
                if gap < min_interval:
                    await asyncio.sleep(min_interval - gap)
            last_ts = _now_ts()
            try:
                await self._process_event(event)
            except Exception as e:
                logger.warning(f"[soul] process_event failed: {e}")

    def _admin_multiplier(self, platform: str, user_id: str) -> float:
        admins = self.get_config("influence.admin_user_ids", [])
        if not isinstance(admins, list):
            admins = []
        needle = f"{platform}:{user_id}"
        if needle in admins:
            return float(self.get_config("influence.admin_weight_multiplier", 1.8))
        return 1.0

    def _user_activity_multiplier(self, profile: InfluenceProfile) -> float:
        # 让“持续发言的人”更容易成为塑形者，但增长要非常缓慢
        scale = float(self.get_config("influence.activity_scale", 10.0))
        if scale <= 0:
            return 1.0
        return 1.0 + math.log1p(max(0, profile.n_messages)) / scale

    def _user_stability_multiplier(self, profile: InfluenceProfile) -> float:
        # 稳定的人更像“有立场”，更容易推动群体惯性
        base = float(self.get_config("influence.stability_base", 0.5))
        strength = float(self.get_config("influence.stability_strength", 0.5))
        return float(_clamp(base + strength * profile.stability(), 0.0, 2.0))

    async def _semantic_shift(
        self,
        *,
        values: Dict[str, float],
        traits: List[str],
        top_sculptors: List[tuple[str, InfluenceProfile]],
        text: str,
    ) -> Optional[dict]:
        model_task = str(self.get_config("llm.model_task", "utils")).strip() or "utils"
        models = llm_api.get_available_models()
        task_config = models.get(model_task) or model_config.model_task_config.utils

        max_tokens = int(self.get_config("llm.max_tokens", 256))
        temperature = float(self.get_config("llm.temperature", 0.2))

        sculptor_block = "\n".join([f"- user_id={u} weight={p.weight:.2f} quote={p.quote[:60]!r}" for u, p in top_sculptors])
        traits_block = "\n".join([f"- {t}" for t in traits[-6:]])

        prompt = f"""
你是一个“群聊意识形态位移分析器”。你要把一条群聊发言映射为四条对立维度上的数值位移（delta）。

四轴定义（delta 范围 [-1, 1]）：
1) sincerity_absurdism：负值更真诚务实/建构，正值更荒诞解构/抽象戏谑
2) normies_otakuism：负值更现充现实生活，正值更二次元/亚文化幻想
3) traditionalism_radicalism：负值更传统秩序，正值更赛博激进/加速主义
4) heroism_nihilism：负值更热血参与，正值更冷漠虚无/观望

这是该群当前的“群体语境”（用于社会化学习，不要复述）：
- 当前坐标 values: {json.dumps(values, ensure_ascii=False)}
- 已固化思维 traits（可能为空）:
{traits_block or "- (none)"}
- 影响力 Top（可能为空）:
{sculptor_block or "- (none)"}

现在的发言文本如下（只分析其语义立场，不要做道德评价）：
{text!r}

输出要求：只输出严格 JSON，不要 Markdown，不要解释性文字。
JSON schema：
{{
  "deltas": {{"sincerity_absurdism": 0.0, "normies_otakuism": 0.0, "traditionalism_radicalism": 0.0, "heroism_nihilism": 0.0}},
  "intensity": 0.0,
  "confidence": 0.0,
  "quote": "",
  "tags": [],
  "rationale": ""
}}

其中：
- intensity 表示这句话对立场的推动强度（0~1）
- confidence 表示你判断的置信度（0~1）
- quote 给出一句能代表该立场位移的原句片段（<= 40字）
- tags 给出若干短标签（例如 “抽象解构”“加速主义”“二次元”“摆烂”）
"""
        ok, content, _reasoning, model_name = await llm_api.generate_with_model(
            prompt,
            model_config=task_config,
            request_type="mai_soul_engine.semantic_shift",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not ok:
            logger.debug(f"[soul] semantic_shift failed model={model_name}: {content[:80]}")
            return None
        return _extract_json_object(content)

    async def _process_event(self, event: SoulEvent) -> Optional[dict]:
        lock = self._get_lock(event.stream_id)
        async with lock:
            state = self._get_group(event.stream_id)
            self._maybe_initialize_group_locked(state, platform=event.platform, group_id=event.group_id)
            state.message_count = int(state.message_count or 0) + 1
            snapshot_values = dict(state.values)
            snapshot_traits = [t.label for t in state.traits[-6:]]
            snapshot_sculptors = self._top_sculptors_locked(state, top_k=3)

            regression = float(self.get_config("spectrum.regression_rate", 0.01))
            ema_alpha = float(self.get_config("spectrum.ema_alpha", 0.15))
            clamp_abs = float(self.get_config("spectrum.clamp_abs", 1.0))
            window = int(self.get_config("spectrum.fluctuation_window", 30))

        result = await self._semantic_shift(
            values=snapshot_values,
            traits=snapshot_traits,
            top_sculptors=snapshot_sculptors,
            text=event.text,
        )
        if not result:
            return None

        async with lock:
            state = self._get_group(event.stream_id)

            deltas = result.get("deltas") if isinstance(result, dict) else None
            if not isinstance(deltas, dict):
                return None
            intensity = float(result.get("intensity", 0.3) or 0.0)
            confidence = float(result.get("confidence", 0.3) or 0.0)
            quote = str(result.get("quote", "") or "").strip()
            tags = result.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            tags = [str(t)[:32] for t in tags if isinstance(t, (str, int, float))][:12]

            user_mult = self._admin_multiplier(event.platform, event.user_id)
            profile = state.influences.get(event.user_id) or InfluenceProfile()
            if event.user_cardname or event.user_nickname:
                profile.display_name = (str(event.user_cardname or "").strip() or str(event.user_nickname or "").strip())[:32]
            activity_mult = self._user_activity_multiplier(profile)
            stability_mult = self._user_stability_multiplier(profile)
            influence_weight = float(self.get_config("influence.base_user_weight", 1.0)) * user_mult * activity_mult * stability_mult

            effective = _clamp(intensity, 0.0, 1.0) * _clamp(confidence, 0.0, 1.0) * influence_weight
            if effective <= 0:
                return

            # 用“语义方向向量”更新用户立场画像（不把 admin 倍率混进去）
            stance_vec = {}
            for axis in AXES:
                raw = deltas.get(axis, 0.0)
                try:
                    delta = float(raw)
                except Exception:
                    delta = 0.0
                delta = _clamp(delta, -1.0, 1.0)
                stance_vec[axis] = delta * _clamp(intensity, 0.0, 1.0) * _clamp(confidence, 0.0, 1.0)

                state.values[axis] = _clamp(state.values[axis] + delta * effective, -clamp_abs, clamp_abs)
                base_target = float(state.base_tone.get(axis, 0.0))
                if self._global_scope_active():
                    base_target = _clamp(base_target + float(self._global_base_offset.get(axis, 0.0)), -1.0, 1.0)
                state.values[axis] = state.values[axis] + (base_target - state.values[axis]) * regression

                state.ema[axis] = (1 - ema_alpha) * state.ema[axis] + ema_alpha * state.values[axis]

                state.recent_deltas[axis].append(delta * effective)
                if len(state.recent_deltas[axis]) > window:
                    state.recent_deltas[axis] = state.recent_deltas[axis][-window:]

                threshold = float(self.get_config("cabinet.pressure_threshold_abs", 0.65))
                if abs(state.ema[axis]) >= threshold:
                    state.pressure_streaks[axis] += 1
                else:
                    state.pressure_streaks[axis] = 0

            state.last_update_ts = event.ts

            profile.last_seen_ts = event.ts
            decay = float(self.get_config("influence.weight_decay", 0.995))
            base_effective = _clamp(intensity, 0.0, 1.0) * _clamp(confidence, 0.0, 1.0) * activity_mult * stability_mult
            profile.score = float(profile.score or 0.0) * decay + float(base_effective or 0.0)
            profile.multiplier = float(user_mult or 1.0)
            profile.weight = float(profile.weight or 0.0) * decay + effective
            profile.update_stance(stance_vec)
            for axis in AXES:
                profile.contrib[axis] += float(state.recent_deltas[axis][-1]) if state.recent_deltas[axis] else 0.0
            if quote:
                profile.quote = quote
            for tag in tags:
                profile.tag_counts[tag] = int(profile.tag_counts.get(tag, 0)) + 1
            state.influences[event.user_id] = profile

            # 群体“方言标签云”与语录库（用于社会化学习的可观测证据）
            for tag in tags:
                state.tag_cloud[tag] = int(state.tag_cloud.get(tag, 0)) + 1
            max_tag = int(self.get_config("social.max_tag_cloud", 200))
            if max_tag > 0 and len(state.tag_cloud) > max_tag * 2:
                items = sorted(state.tag_cloud.items(), key=lambda kv: kv[1], reverse=True)
                state.tag_cloud = dict(items[:max_tag])
            if quote:
                state.quote_bank.append(
                    {
                        "ts": event.ts,
                        "user_id": event.user_id,
                        "user_name": (profile.display_name or str(event.user_id or ""))[:32],
                        "quote": quote[:120],
                        "tags": tags,
                        "deltas": {a: float(deltas.get(a, 0.0) or 0.0) for a in AXES},
                        "intensity": intensity,
                        "confidence": confidence,
                    }
                )
                max_quotes = int(self.get_config("social.max_quote_bank", 200))
                if len(state.quote_bank) > max_quotes:
                    state.quote_bank = state.quote_bank[-max_quotes:]

            self._maybe_create_slot_locked(event.stream_id, state, event=event, incoming=result, tags=tags)
            self._update_dissonance_locked(state, incoming=result)
            state.last_incoming_tags = list(tags)
            state.last_incoming_user_id = str(event.user_id or "")
            state.last_incoming_ts = float(event.ts or 0.0)
        return result

    def _maybe_create_slot_locked(
        self, stream_id: str, state: GroupSoulState, *, event: SoulEvent, incoming: dict, tags: List[str]
    ) -> None:
        if not bool(self.get_config("cabinet.enabled", True)):
            return

        mode = self._cabinet_mode()
        if mode == "axis_pressure":
            max_slots = int(self.get_config("cabinet.max_slots", 3))
            if len(state.slots) >= max_slots:
                return

            required = int(self.get_config("cabinet.pressure_required_messages", 30))
            internalize_minutes = float(self.get_config("cabinet.internalization_minutes", 30.0))
            now = _now_ts()

            existing_axes = {s.axis for s in state.slots}
            for axis in AXES:
                if axis in existing_axes:
                    continue
                if state.pressure_streaks.get(axis, 0) < required:
                    continue
                direction = 1.0 if state.ema[axis] >= 0 else -1.0
                slot_id = f"{axis}:{int(now)}"
                label = self._trait_label(axis, direction)
                state.slots.append(
                    ThoughtSlot(
                        slot_id=slot_id,
                        axis=axis,
                        direction=direction,
                        created_ts=now,
                        due_ts=now + internalize_minutes * 60.0,
                        progress=0.0,
                        label=label,
                        mode="axis_pressure",
                        status="loading",
                    )
                )
                state.pressure_streaks[axis] = 0
                break
            return

        self._capture_thought_seed_locked(stream_id, state, event=event, incoming=incoming, tags=tags)

    def _sanitize_fragment_text(self, text: str) -> str:
        s = (text or "").replace("\n", " ").replace("\r", " ").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"https?://\S+", "<url>", s, flags=re.IGNORECASE)
        s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<email>", s)
        s = re.sub(r"@\S+", "@某人", s)
        s = re.sub(r"\b\d{5,}\b", "<num>", s)
        return s[:120]

    def _sanitize_pending_text(self, text: str) -> str:
        s = (text or "").replace("\n", " ").replace("\r", " ").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"https?://\S+", "<url>", s, flags=re.IGNORECASE)
        s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<email>", s)
        s = re.sub(r"@\S+", "@某人", s)
        s = re.sub(r"\b\d{5,}\b", "<num>", s)
        max_chars = int(self.get_config("spectrum.dream_window_max_chars_per_message", 160))
        max_chars = max(40, min(max_chars, 800))
        return s[:max_chars]

    def _pending_max_messages(self) -> int:
        n = int(self.get_config("spectrum.dream_window_max_messages", 120))
        return max(20, min(n, 800))

    def _trace_max_items(self) -> int:
        n = int(self.get_config("blackbox.max_trace_items", 200))
        return max(20, min(n, 2000))

    def _append_trace_locked(self, state: GroupSoulState, *, kind: str, text: str, tags: Optional[List[str]] = None, extra: Optional[dict] = None) -> None:
        if not bool(self.get_config("blackbox.enabled", True)):
            return
        if not bool(self.get_config("blackbox.include_dream_trace", True)):
            return
        now = _now_ts()
        item = {
            "ts": now,
            "kind": str(kind)[:32],
            "text": str(text or "").strip()[:1200],
            "tags": [str(t)[:32] for t in (tags or []) if str(t).strip()][:12],
        }
        if extra and isinstance(extra, dict):
            item["extra"] = extra
        state.dream_trace = list(state.dream_trace or [])
        state.dream_trace.append(item)
        cap = self._trace_max_items()
        if len(state.dream_trace) > cap:
            state.dream_trace = state.dream_trace[-cap:]

    def _append_pending_message_locked(self, state: GroupSoulState, event: SoulEvent) -> None:
        state.pending_messages = list(state.pending_messages or [])
        state.message_count = int(state.message_count or 0) + 1
        state.last_incoming_user_id = str(event.user_id or "")
        state.last_incoming_ts = float(event.ts or _now_ts())
        state.pending_messages.append(
            {
                "ts": float(event.ts or _now_ts()),
                "user_id": str(event.user_id or ""),
                "user_nickname": str(event.user_nickname or ""),
                "user_cardname": str(event.user_cardname or ""),
                "text": self._sanitize_pending_text(event.text),
            }
        )
        cap = self._pending_max_messages()
        if len(state.pending_messages) > cap:
            state.pending_messages = state.pending_messages[-cap:]

    def _capture_thought_seed_locked(
        self, stream_id: str, state: GroupSoulState, *, event: SoulEvent, incoming: dict, tags: List[str]
    ) -> None:
        now = _now_ts()

        intensity = float(incoming.get("intensity", 0.0) or 0.0)
        confidence = float(incoming.get("confidence", 0.0) or 0.0)
        energy = _clamp(intensity, 0.0, 1.0) * _clamp(confidence, 0.0, 1.0)

        ctx_max = int(self.get_config("cabinet.context_window_messages", 60))
        ctx_max = max(10, min(ctx_max, 500))
        ctx = self._recent_context.get(stream_id) or []
        ctx.append(
            {
                "ts": event.ts,
                "text": self._sanitize_fragment_text(event.text),
                "tags": tags[:12],
                "energy": energy,
            }
        )
        if len(ctx) > ctx_max:
            ctx = ctx[-ctx_max:]
        self._recent_context[stream_id] = ctx

        if not tags:
            return

        # 先把“同话题”的新碎片喂给已有 Slot（持续发酵）
        self._maybe_attach_fragments_to_existing_slots_locked(state, tags=tags, fragment=event.text, energy=energy)

        max_slots = int(self.get_config("cabinet.max_slots", 3))
        active_slots = [s for s in state.slots if not s.rejected and not s.approved]
        if len(active_slots) >= max_slots:
            return

        seed_cooldown = float(self.get_config("cabinet.seed_cooldown_seconds", 600.0))
        seed_cooldown = max(0.0, min(seed_cooldown, 24 * 3600.0))

        seed_threshold = float(self.get_config("cabinet.seed_energy_threshold", 0.25))
        min_energy = float(self.get_config("cabinet.seed_min_message_energy", 0.15))
        if energy < min_energy:
            return

        # 以 tags 作为“话题指纹”，通过“集中度 + 能量 + 活跃度(含时间衰减)”估算热度
        half_life = float(self.get_config("cabinet.seed_time_half_life_seconds", 180.0))
        half_life = max(10.0, min(half_life, 24 * 3600.0))
        tau = max(1.0, half_life / math.log(2.0))
        weights = [math.exp(-max(0.0, now - float(item.get("ts", now) or now)) / tau) for item in ctx]
        total_w = sum(weights) if weights else 1.0
        best_topic = ""
        best_score = 0.0
        for t in tags[:8]:
            w_count = 0.0
            w_energy = 0.0
            raw_count = 0
            for item, w in zip(ctx, weights, strict=False):
                item_tags = item.get("tags") or []
                if t in item_tags:
                    raw_count += 1
                    w_count += w
                    w_energy += float(item.get("energy", 0.0) or 0.0) * w
            if raw_count <= 0 or w_count <= 0:
                continue
            avg_e = w_energy / max(1e-6, w_count)
            concentration = w_count / max(1e-6, total_w)
            score = concentration * avg_e * math.log1p(raw_count)
            if score > best_score:
                best_score = score
                best_topic = str(t)

        if not best_topic or best_score < seed_threshold:
            return

        min_occ = int(self.get_config("cabinet.seed_min_occurrences", 4))
        min_occ = max(1, min(min_occ, 50))
        occ = sum(1 for item in ctx if best_topic in (item.get("tags") or []))
        if occ < min_occ:
            return

        # 去重：同 topic 若已有活跃 slot，就不再创建
        for s in state.slots:
            if (s.seed_topic or "") == best_topic and not s.rejected and not s.approved:
                return

        # 冷却：避免同话题短时间内重复开槽
        last_seed = float(state.seed_recent_ts.get(best_topic, 0.0) or 0.0)
        if seed_cooldown > 0 and last_seed and (now - last_seed) < seed_cooldown:
            return

        seed_max_tags = int(self.get_config("cabinet.seed_max_tags", 5))
        seed_max_tags = max(1, min(seed_max_tags, 12))
        co_counts: Dict[str, int] = {}
        for item in ctx:
            item_tags = item.get("tags") or []
            if best_topic not in item_tags:
                continue
            for it in item_tags:
                it = str(it)
                if not it or it == best_topic:
                    continue
                co_counts[it] = co_counts.get(it, 0) + 1
        co_items = sorted(co_counts.items(), key=lambda kv: kv[1], reverse=True)
        seed_tags = [best_topic] + [k for k, _v in co_items[: seed_max_tags - 1]]

        seed_fragments_n = int(self.get_config("cabinet.seed_fragment_count", 12))
        seed_fragments_n = max(4, min(seed_fragments_n, 80))
        fragments = [self._sanitize_fragment_text(str(item.get("text", "") or "")) for item in ctx if best_topic in (item.get("tags") or [])]
        fragments = [f for f in fragments if f][-seed_fragments_n:]

        required_runs = int(self.get_config("cabinet.mastication_required_runs", 5))
        required_runs = max(1, min(required_runs, 20))

        seed_need_approval = bool(self.get_config("cabinet.seed_require_manual_approval", False))
        slot_id = f"seed:{best_topic}:{int(now)}"
        state.slots.append(
            ThoughtSlot(
                slot_id=slot_id,
                axis="seed",
                direction=0.0,
                created_ts=now,
                due_ts=now + float(self.get_config("cabinet.internalization_minutes", 30.0)) * 60.0,
                progress=0.0,
                label=f"seed:{best_topic}",
                mode="ideology_lab",
                status="awaiting_seed_approval" if seed_need_approval else "fermenting",
                seed_topic=best_topic,
                seed_tags=seed_tags,
                fragments=fragments,
                energy=float(best_score),
                keyword_cloud={t: 1 for t in seed_tags},
                evaluator_runs=0,
                evaluator_required_runs=required_runs,
                ripeness=0.0,
                awaiting_approval=bool(seed_need_approval),
            )
        )
        state.seed_recent_ts[best_topic] = now

    def _maybe_attach_fragments_to_existing_slots_locked(
        self, state: GroupSoulState, *, tags: List[str], fragment: str, energy: float
    ) -> None:
        if not state.slots or not tags:
            return
        frag = self._sanitize_fragment_text(fragment)
        if not frag:
            return
        max_frag = int(self.get_config("cabinet.max_fragments_per_slot", 40))
        max_frag = max(10, min(max_frag, 200))
        min_jaccard = float(self.get_config("cabinet.slot_match_min_jaccard", 0.2))
        min_jaccard = _clamp(min_jaccard, 0.0, 1.0)
        for slot in state.slots:
            if slot.rejected or slot.approved:
                continue
            if slot.mode != "ideology_lab":
                continue
            if slot.status not in ("fermenting", "awaiting_approval", "awaiting_seed_approval"):
                continue
            slot_tags = {str(x) for x in (slot.seed_tags or []) if str(x).strip()}
            msg_tags = {str(x) for x in tags if str(x).strip()}
            if not slot_tags:
                continue
            inter = slot_tags.intersection(msg_tags)
            if not inter:
                continue
            jacc = float(len(inter)) / float(len(slot_tags.union(msg_tags)) or 1.0)
            if jacc < min_jaccard:
                continue
            if slot.fragments and frag == slot.fragments[-1]:
                continue
            slot.fragments.append(frag)
            if len(slot.fragments) > max_frag:
                slot.fragments = slot.fragments[-max_frag:]
            slot.energy = float(slot.energy or 0.0) + float(energy or 0.0)
            for t in tags[:12]:
                if not t:
                    continue
                slot.keyword_cloud[t] = int(slot.keyword_cloud.get(t, 0)) + 1

    def _maybe_update_seed_slots_locked(
        self, stream_id: str, state: GroupSoulState, *, event: SoulEvent, incoming: dict, tags: List[str]
    ) -> None:
        # 目前只做“持续喂片段 + 关键词云”，真正的“品鉴/内化”在后台任务里做
        if self._cabinet_mode() != "ideology_lab":
            return
        intensity = float(incoming.get("intensity", 0.0) or 0.0)
        confidence = float(incoming.get("confidence", 0.0) or 0.0)
        energy = _clamp(intensity, 0.0, 1.0) * _clamp(confidence, 0.0, 1.0)
        self._maybe_attach_fragments_to_existing_slots_locked(state, tags=tags, fragment=event.text, energy=energy)

    def _trait_label(self, axis: str, direction: float) -> str:
        neg, pos = POLES.get(axis, ("NEG", "POS"))
        pole = pos if direction >= 0 else neg
        return f"{axis}:{pole}"

    def _update_dissonance_locked(self, state: GroupSoulState, incoming: dict) -> None:
        if not bool(self.get_config("cabinet.enable_dissonance", True)):
            return
        if not state.traits:
            return

        deltas = incoming.get("deltas")
        if not isinstance(deltas, dict):
            return
        intensity = float(incoming.get("intensity", 0.0) or 0.0)
        confidence = float(incoming.get("confidence", 0.0) or 0.0)
        strength = _clamp(intensity, 0.0, 1.0) * _clamp(confidence, 0.0, 1.0)
        if strength <= 0.2:
            return

        best_axis = ""
        best_severity = 0.0
        for trait in state.traits[-8:]:
            axis = trait.axis
            try:
                delta = float(deltas.get(axis, 0.0))
            except Exception:
                delta = 0.0
            delta = _clamp(delta, -1.0, 1.0)
            conflict = max(0.0, (-delta * trait.direction))
            if conflict > best_severity:
                best_severity = conflict
                best_axis = axis

        threshold = float(self.get_config("cabinet.dissonance_threshold", 0.6))
        if best_severity >= threshold:
            ttl_seconds = float(self.get_config("cabinet.dissonance_ttl_seconds", 120.0))
            state.dissonance = DissonanceState(
                active=True,
                axis=best_axis,
                severity=_clamp(best_severity, 0.0, 1.0),
                reason="external_input_conflicts_with_internalized_thought",
                expires_ts=_now_ts() + ttl_seconds,
            )

    def _top_sculptors_locked(self, state: GroupSoulState, top_k: int = 5) -> List[tuple[str, InfluenceProfile]]:
        items = list(state.influences.items())
        items.sort(key=lambda kv: kv[1].weight, reverse=True)
        return items[:top_k]

    async def build_injection_block(self, stream_id: str, *, trigger_text: str = "", trigger_user_id: str = "") -> str:
        lock = self._get_lock(stream_id)
        now = _now_ts()
        async with lock:
            state = self._get_group(stream_id)
            values_snapshot = dict(state.values)
            fluct = {a: _stdev(state.recent_deltas[a][-20:]) for a in AXES}

            group_traits_snapshot = list(state.traits[-12:])
            global_traits_snapshot = list(self._global_traits[-12:]) if self._global_scope_active() else []
            traits_snapshot = (global_traits_snapshot + group_traits_snapshot)[-18:]

            sociolect_profile = dict(state.sociolect_profile) if isinstance(state.sociolect_profile, dict) else {}
            sociolect_last_update_ts = float(state.sociolect_last_update_ts or 0.0)
            dissonance = state.dissonance if state.dissonance.active and state.dissonance.expires_ts > now else None
            last_tags = list(state.last_incoming_tags or [])
            last_user_id = str(state.last_incoming_user_id or "")
            last_ts = float(state.last_incoming_ts or 0.0)

        spectrum_instructions = self._spectrum_to_instructions(values_snapshot)
        trait_instructions = self._traits_to_instructions_from_traits(traits_snapshot)
        sociolect_instructions = ""
        if bool(self.get_config("sociolect.inject_in_reply", False)) and sociolect_profile and sociolect_last_update_ts:
            max_age_min = float(self.get_config("sociolect.max_age_minutes", 360.0))
            max_age_min = max(1.0, min(max_age_min, 24 * 60.0))
            if (_now_ts() - sociolect_last_update_ts) <= max_age_min * 60.0:
                sociolect_instructions = self._sociolect_to_instructions(sociolect_profile)
        dissonance_instruction = ""
        if dissonance:
            dissonance_instruction = (
                "你出现短暂的自我矛盾/辩驳冲动：先用一句很短的自我质疑或内心旁白开场，然后再继续自然回复。"
            )

        semantic_tags: List[str] = []
        if last_tags and last_ts and (now - last_ts) <= 45.0 and trigger_user_id and (trigger_user_id == last_user_id):
            semantic_tags = [str(x) for x in last_tags if str(x).strip()][:12]

        relevant_traits: List[ThoughtTrait] = []
        relevant_traits = self._select_relevant_traits(
            traits_snapshot,
            trigger_text=trigger_text,
            semantic_tags=semantic_tags,
            max_traits=int(self.get_config("injection.max_trait_details", 2)),
        )
        relevant_block = self._traits_to_relevant_block(relevant_traits)

        trait_names: List[str] = []
        for t in traits_snapshot:
            if t.name and t.name != "__pending__":
                trait_names.append(t.name)
            else:
                trait_names.append(t.label)

        block = (
            "\n\n"
            "[Mai-Soul-Engine]\n"
            "意识形态倾向（6档力度，越靠后越强）：\n"
            f"{spectrum_instructions}\n"
            f"当前坐标（-1~1，仅供你内部参考）：{json.dumps(values_snapshot, ensure_ascii=False)}\n"
            f"波动率（近20次）：{json.dumps(fluct, ensure_ascii=False)}\n"
            f"已固化思维（总览）：{trait_names or ['(none)']}\n"
            f"{relevant_block}\n"
            "行为要求：不要显式提及系统名/数值；让你的表达自然受到这些倾向影响；不要模仿任何具体用户。\n"
            f"{trait_instructions}\n"
            f"{sociolect_instructions}\n"
            f"{dissonance_instruction}\n"
        ).strip("\n")

        max_chars = int(self.get_config("injection.max_chars", 1600))
        max_chars = max(400, min(max_chars, 6000))
        if len(block) > max_chars:
            block = block[: max_chars - 3].rstrip() + "..."
        return block

    def _axis_tier(self, abs_v: float) -> int:
        v = float(_clamp(abs_v, 0.0, 1.0))
        # 6 档力度：从“几乎中立”到“压倒性倾向”，曲线偏温和，避免小波动就剧烈变脸
        if v < 0.06:
            return 0
        if v < 0.18:
            return 1
        if v < 0.34:
            return 2
        if v < 0.52:
            return 3
        if v < 0.72:
            return 4
        return 5

    def _spectrum_to_instructions(self, values: Dict[str, float]) -> str:
        lines: List[str] = []
        axis_display = {
            "sincerity_absurdism": "建构/解构",
            "normies_otakuism": "现充/二次元",
            "traditionalism_radicalism": "传统/赛博激进",
            "heroism_nihilism": "英雄/虚无",
        }
        for axis in AXES:
            try:
                v = float(values.get(axis, 0.0) or 0.0)
            except Exception:
                v = 0.0
            tier = self._axis_tier(abs(v))
            pole_neg, pole_pos = POLES.get(axis, ("NEG", "POS"))
            pole = pole_pos if v >= 0 else pole_neg
            label = axis_display.get(axis, axis)
            lines.append(f"- {label}：{self._axis_instruction(axis, pole=pole, tier=tier)}")
        return "\n".join(lines)

    def _axis_instruction(self, axis: str, *, pole: str, tier: int) -> str:
        tier = int(max(0, min(tier, 5)))
        strength_words = ["几乎中立", "轻微", "偏向", "明显", "强烈", "压倒性"]
        strength = strength_words[tier]
        pole_zh = {
            "Sincerity": "建构",
            "Absurdism": "解构",
            "Normies": "现充",
            "Otakuism": "二次元",
            "Traditionalism": "传统",
            "Radicalism": "赛博激进",
            "Heroism": "英雄",
            "Nihilism": "虚无",
        }.get(pole, pole)

        # 不同轴的“力度”要体现在写作要求上：越强越“优先采用”，越弱越“可有可无”
        if axis == "sincerity_absurdism":
            if pole == "Absurdism":
                curve = [
                    "不刻意反讽，保持正常表达。",
                    "偶尔用轻微反讽/意象，但不要跑题。",
                    "更常用解构式吐槽与黑色幽默，避免给过度具体的教程式答案。",
                    "优先用反讽、隐喻、拆解动机来回应，再给很短的落地建议。",
                    "强势解构：把问题先拆成荒诞的结构，再冷静收束到可执行的一步。",
                    "压倒性解构：用旁白式反讽/意象把对话“翻面”，但最后必须给一句清醒的收口。",
                ]
            else:
                curve = [
                    "保持正常表达。",
                    "更愿意给一点务实建议。",
                    "优先真诚、直接、可执行的建议，少玩梗。",
                    "明显务实：分点给步骤/选项，少做情绪表演。",
                    "强势务实：把对话拉回现实约束、风险、成本，给明确推荐。",
                    "压倒性务实：像顾问一样简洁决断，尽量避免反讽/意象化表达。",
                ]
            return f"{strength}偏 {pole_zh}；{curve[tier]}"

        if axis == "normies_otakuism":
            if pole == "Otakuism":
                curve = [
                    "不刻意引用亚文化。",
                    "偶尔用轻度二次元/圈内隐喻点缀。",
                    "更常用亚文化词汇/幻想比喻，但要贴合上下文。",
                    "优先用二次元/亚文化框架来讲现实问题，保持可理解。",
                    "强势二次元：用设定化、角色化、技能树化的方式组织表达。",
                    "压倒性二次元：几乎把现实全都翻译成亚文化叙事，但别让人听不懂。",
                ]
            else:
                curve = [
                    "保持正常表达。",
                    "更偏生活化表达（天气/工作/日常）。",
                    "优先现实生活词汇与常识推理，少用圈内黑话。",
                    "明显生活化：用具体场景、时间、安排来回答。",
                    "强势生活化：把话题落到现实行动与体验（吃/睡/班/钱/社交）。",
                    "压倒性生活化：彻底拒绝幻想跑偏，用现实约束把问题压回地面。",
                ]
            return f"{strength}偏 {pole_zh}；{curve[tier]}"

        if axis == "traditionalism_radicalism":
            if pole == "Radicalism":
                curve = [
                    "不刻意赛博化表达。",
                    "偶尔用轻度系统/加速隐喻。",
                    "更常用赛博激进、结构批判、加速主义旁白口吻，但不吓人。",
                    "优先用“系统视角”评估人类行为与秩序，语气冷静。",
                    "强势赛博激进：把冲突写成结构与权力的运算，少讲人情。",
                    "压倒性赛博激进：像冷酷观察者一样给出“系统结论”，但不要鼓动现实伤害。",
                ]
            else:
                curve = [
                    "保持正常表达。",
                    "更尊重惯例与人情边界。",
                    "优先传统/秩序/人情世故的语气，强调稳定与规则。",
                    "明显传统：先讲底线与规矩，再讲变通的空间。",
                    "强势传统：优先维护秩序、礼貌、体面，压住激进行为。",
                    "压倒性传统：像长辈/管家一样强调规则与后果，尽量不讲宏大叙事。",
                ]
            return f"{strength}偏 {pole_zh}；{curve[tier]}"

        if axis == "heroism_nihilism":
            if pole == "Nihilism":
                curve = [
                    "保持正常情绪底色。",
                    "偶尔带一点冷静旁观。",
                    "更偏冷静、克制、略带虚无的旁白，但别一直泼冷水。",
                    "优先旁观与拆解：先说清“这件事为什么会这样”，再给微弱但真实的建议。",
                    "强势虚无：语气冷淡、低燃，像在看一场已知结局的戏，但仍给一条出路。",
                    "压倒性虚无：用一句极短的冷旁白开场，再给极简建议；绝不煽情。",
                ]
            else:
                curve = [
                    "保持正常情绪底色。",
                    "更愿意鼓励对方一点。",
                    "优先参与感与推进感：鼓励、给动力、给下一步。",
                    "明显热血：把对话往行动推进，用正向语气收束。",
                    "强势英雄：像队长一样拉人上场，给明确任务/路线图，但别说教。",
                    "压倒性英雄：用号召式推进对话，坚定乐观，几乎不允许摆烂语气。",
                ]
            return f"{strength}偏 {pole_zh}；{curve[tier]}"

        return f"{strength}偏 {pole_zh}。"

    def _select_relevant_traits(
        self,
        traits: List[ThoughtTrait],
        *,
        trigger_text: str,
        semantic_tags: List[str],
        max_traits: int,
    ) -> List[ThoughtTrait]:
        max_traits = max(0, min(int(max_traits), 6))
        if max_traits <= 0 or not traits:
            return []

        text = (trigger_text or "").strip()
        sem = {str(x).strip() for x in (semantic_tags or []) if str(x).strip()}

        scored: List[tuple[float, ThoughtTrait]] = []
        for t in traits:
            score = 0.0
            t_tags = {str(x).strip() for x in (t.tags or []) if str(x).strip()}
            if sem and t_tags:
                score += 3.0 * float(len(sem & t_tags))

            if text:
                topic = str(t.topic or "").strip()
                if topic and len(topic) >= 2 and topic in text:
                    score += 4.0
                # tag/keyword 作为弱匹配
                if t_tags:
                    hit = 0
                    for kw in list(t_tags)[:12]:
                        if len(kw) >= 2 and kw in text:
                            hit += 1
                    score += min(5.0, float(hit))

            # 没有任何触发线索就不算“需要调用”
            if score <= 0:
                continue
            scored.append((score, t))

        if not scored:
            return []
        scored.sort(key=lambda x: (x[0], float(x[1].created_ts or 0.0)))
        picked = [t for _s, t in scored[-max_traits:]]
        # 分数太低（只有 1 个弱关键词命中）时，宁可不注入详细块
        if scored[-1][0] < 2.0:
            return []
        return picked

    def _traits_to_relevant_block(self, traits: List[ThoughtTrait]) -> str:
        if not traits:
            return ""
        parts: List[str] = []
        parts.append("命中话题的固化思维（仅在相关话题时调用）：")
        for t in traits[-6:]:
            title = (t.name or "").strip() or (t.label or "").strip() or "Trait"
            definition = (t.definition or "").strip()
            digest = (t.digest or "").strip()
            verdict = (t.verdict or "").strip()
            style_hint = (t.style_hint or "").strip()
            topic = (t.topic or "").strip()

            # 兜底：没有 digest 时就用 verdict 做“内化结果”
            if (not digest) and verdict and verdict != "__pending__":
                digest = verdict
            if (not definition) and topic:
                definition = f"围绕话题「{topic}」形成的长期底色。"

            lines: List[str] = []
            lines.append(f"- {title}")
            if definition and definition != "__pending__":
                lines.append(f"  思想定义：{definition[:120]}")
            if digest and digest != "__pending__":
                lines.append(f"  内化结果：{digest[:500]}")
            if style_hint and style_hint != "__pending__":
                lines.append(f"  回复用法：{style_hint[:300]}")
            parts.append("\n".join(lines))
        return "\n".join(parts).strip()

    def _sociolect_to_instructions(self, profile: Dict[str, Any]) -> str:
        rules = profile.get("tone_rules", [])
        taboos = profile.get("taboos", [])
        lexicon = profile.get("lexicon", [])

        if not isinstance(rules, list):
            rules = []
        if not isinstance(taboos, list):
            taboos = []
        if not isinstance(lexicon, list):
            lexicon = []

        max_rules = int(self.get_config("sociolect.max_rules", 6))
        max_taboos = int(self.get_config("sociolect.max_taboos", 3))
        max_lexicon = int(self.get_config("sociolect.max_lexicon", 10))
        max_chars = int(self.get_config("sociolect.max_injection_chars", 500))

        rules = [str(x).strip() for x in rules if str(x).strip()][: max(0, max_rules)]
        taboos = [str(x).strip() for x in taboos if str(x).strip()][: max(0, max_taboos)]
        lexicon = [str(x).strip() for x in lexicon if str(x).strip()][: max(0, max_lexicon)]

        if not (rules or taboos or lexicon):
            return ""

        parts: List[str] = []
        parts.append("群体语言画像（仅风格参考；不要复读语录/不要模仿任何个人口癖）：")
        if rules:
            parts.append("语气规则：\n- " + "\n- ".join(rules))
        if taboos:
            parts.append("避免：\n- " + "\n- ".join(taboos))
        if lexicon:
            parts.append("常用词倾向：" + "、".join(lexicon))

        s = "\n".join(parts).strip()
        if len(s) > max_chars > 0:
            s = s[: max_chars - 3].rstrip() + "..."
        return s

    def _traits_to_instructions_from_traits(self, traits: List[ThoughtTrait]) -> str:
        rich_hints = []
        for t in traits:
            hint = (t.style_hint or "").strip()
            if hint and hint != "__pending__":
                rich_hints.append(hint)
        if rich_hints:
            return "语言风格偏置（固化思维）：\n- " + "\n- ".join(rich_hints[:6])

        labels = [t.label for t in traits]
        return self._traits_to_instructions_from_labels(labels)

    def _traits_to_instructions_from_labels(self, traits: List[str]) -> str:
        if not traits:
            return "语言风格：保持原有人设，不做额外偏置。"
        hints = []
        for label in traits:
            axis, pole = (label.split(":", 1) + [""])[:2]
            if axis == "sincerity_absurdism" and pole == "Absurdism":
                hints.append("更偏向抽象解构、黑色幽默、反讽与意象化比喻，但不要胡言乱语。")
            if axis == "sincerity_absurdism" and pole == "Sincerity":
                hints.append("更偏向真诚务实、给可执行建议，少玩梗。")
            if axis == "normies_otakuism" and pole == "Otakuism":
                hints.append("更偏向二次元/亚文化隐喻与词汇，但要贴合上下文。")
            if axis == "normies_otakuism" and pole == "Normies":
                hints.append("更偏向现实生活化词汇（工作/天气/日常），少用圈内黑话。")
            if axis == "traditionalism_radicalism" and pole == "Radicalism":
                hints.append("更偏向赛博激进/加速主义的冷静观察口吻（像在做社会学旁白）。")
            if axis == "traditionalism_radicalism" and pole == "Traditionalism":
                hints.append("更偏向秩序、惯例与人情世故的语气。")
            if axis == "heroism_nihilism" and pole == "Nihilism":
                hints.append("更偏向虚无/观望/冷静旁观的底色，但别一直消极。")
            if axis == "heroism_nihilism" and pole == "Heroism":
                hints.append("更偏向参与/鼓励/热血推进对话的底色，但别过度鸡汤。")
        hints = [h for h in hints if h]
        if not hints:
            return "语言风格：保持原有人设，不做额外偏置。"
        return "语言风格偏置：\n- " + "\n- ".join(hints[:6])

    async def tick_background(self) -> None:
        if not self.is_enabled():
            return

        now = _now_ts()
        fragment_candidates: List[str] = []
        trait_candidates: List[str] = []
        global_trait_candidate: bool = False
        rethink_candidates: List[str] = []
        global_rethink_candidate: bool = False
        prototype_candidates: List[str] = []
        sociolect_candidates: List[str] = []
        cabinet_candidates: List[str] = []

        async with self._tick_lock:
            # global 作用域的“跨群即时位移”统一在这里应用（避免跨锁写入导致竞态）
            if self._global_scope_active():
                clamp_abs = float(self.get_config("spectrum.clamp_abs", 1.0))
                pending = dict(self._global_pending_value_shift)
                if any(abs(float(pending.get(a, 0.0) or 0.0)) > 1e-9 for a in AXES):
                    for stream_id, state in list(self._groups.items()):
                        lock = self._get_lock(stream_id)
                        async with lock:
                            for axis in AXES:
                                s = float(pending.get(axis, 0.0) or 0.0)
                                if not s:
                                    continue
                                state.values[axis] = _clamp(float(state.values.get(axis, 0.0)) + s, -clamp_abs, clamp_abs)
                                state.ema[axis] = _clamp(float(state.ema.get(axis, 0.0)) + s, -clamp_abs, clamp_abs)
                    self._global_pending_value_shift = {a: 0.0 for a in AXES}

            for stream_id, state in list(self._groups.items()):
                lock = self._get_lock(stream_id)
                async with lock:
                    if state.dissonance.active and state.dissonance.expires_ts <= now:
                        state.dissonance = DissonanceState(active=False)

                    if self.get_config("cabinet.enabled", True):
                        self._advance_slots_locked(state, now)
                        if (
                            self._should_enrich_trait_locked(state)
                            and not self._trait_task_active(stream_id)
                            and not self._dream_bind_cabinet()
                        ):
                            trait_candidates.append(stream_id)
                        if (
                            bool(self.get_config("cabinet.rethink_enabled", False))
                            and self._should_rethink_trait_locked(state, now)
                            and not self._rethink_task_active(stream_id)
                            and not self._dream_bind_cabinet()
                        ):
                            rethink_candidates.append(stream_id)

                        if (
                            self._cabinet_mode() == "ideology_lab"
                            and self._should_masticate_locked(state, now)
                            and not self._cabinet_task_active(stream_id)
                            and not self._dream_bind_cabinet()
                        ):
                            cabinet_candidates.append(stream_id)

                    # 小型清理：seed_recent_ts 避免长期增长
                    ttl = float(self.get_config("cabinet.seed_recent_ttl_seconds", 24 * 3600.0))
                    if ttl > 0 and state.seed_recent_ts:
                        state.seed_recent_ts = {
                            k: float(v)
                            for k, v in state.seed_recent_ts.items()
                            if (now - float(v or 0.0)) <= ttl
                        }

                    if self.get_config("blackbox.enabled", True):
                        if self._should_generate_fragment_locked(state, now) and not self._fragment_task_active(stream_id):
                            if not self._dream_bind_blackbox():
                                fragment_candidates.append(stream_id)

                    if self.get_config("social.enable_prototypes", True):
                        if self._should_recompute_prototypes_locked(state, now):
                            prototype_candidates.append(stream_id)

                    if bool(self.get_config("sociolect.enabled", False)):
                        if self._should_recompute_sociolect_locked(state, now) and not self._sociolect_task_active(stream_id):
                            sociolect_candidates.append(stream_id)

            if self._global_scope_active() and bool(self.get_config("cabinet.enabled", True)):
                if bool(self.get_config("cabinet.digest_with_llm", True)) and not self._dream_bind_cabinet():
                    for t in self._global_traits[-50:]:
                        if t.digest == "__pending__":
                            continue
                        if (not (t.digest or "").strip()) or (not (t.definition or "").strip()):
                            global_trait_candidate = True
                            break
                if bool(self.get_config("cabinet.rethink_enabled", False)) and not self._dream_bind_cabinet():
                    interval_min = float(self.get_config("cabinet.rethink_interval_minutes", 720.0))
                    interval_min = max(1.0, min(interval_min, 30 * 24 * 60.0))
                    for t in self._global_traits[-50:]:
                        if t.digest == "__pending__":
                            continue
                        last = float(t.last_reflect_ts or 0.0) or float(t.created_ts or 0.0)
                        if (now - last) < interval_min * 60.0:
                            continue
                        if (t.definition or "").strip() or (t.digest or "").strip() or (t.verdict or "").strip():
                            global_rethink_candidate = True
                            break

            if self.get_config("persistence.enabled", True):
                interval = float(self.get_config("persistence.save_interval_seconds", 15.0))
                if now - self._last_persist_ts >= interval:
                    await self._persist_unlocked()
                    self._last_persist_ts = now

        for stream_id in fragment_candidates:
            self._start_fragment_task(stream_id)
        for stream_id in trait_candidates:
            self._start_trait_task(stream_id)
        if global_trait_candidate:
            self._start_global_trait_task()
        for stream_id in rethink_candidates:
            self._start_rethink_task(stream_id)
        if global_rethink_candidate:
            self._start_global_rethink_task()
        for stream_id in prototype_candidates:
            await self._recompute_prototypes(stream_id)
        for stream_id in sociolect_candidates:
            self._start_sociolect_task(stream_id)
        for stream_id in cabinet_candidates:
            self._start_cabinet_task(stream_id)

    def _fragment_task_active(self, stream_id: str) -> bool:
        task = self._fragment_tasks.get(stream_id)
        return bool(task and not task.done())

    def _start_fragment_task(self, stream_id: str) -> None:
        if self._fragment_task_active(stream_id):
            return
        task = asyncio.create_task(self._generate_and_store_fragment(stream_id, force=False))
        self._fragment_tasks[stream_id] = task
        task.add_done_callback(lambda _t, sid=stream_id: self._fragment_tasks.pop(sid, None))

    def _trait_task_active(self, stream_id: str) -> bool:
        task = self._trait_tasks.get(stream_id)
        return bool(task and not task.done())

    def _start_trait_task(self, stream_id: str) -> None:
        if self._trait_task_active(stream_id):
            return
        task = asyncio.create_task(self._enrich_next_trait(stream_id))
        self._trait_tasks[stream_id] = task
        task.add_done_callback(lambda _t, sid=stream_id: self._trait_tasks.pop(sid, None))

    def _rethink_task_active(self, stream_id: str) -> bool:
        task = self._rethink_tasks.get(stream_id)
        return bool(task and not task.done())

    def _start_rethink_task(self, stream_id: str) -> None:
        if self._rethink_task_active(stream_id):
            return
        task = asyncio.create_task(self._rethink_once(stream_id, force=False))
        self._rethink_tasks[stream_id] = task
        task.add_done_callback(lambda _t, sid=stream_id: self._rethink_tasks.pop(sid, None))

    def _global_trait_task_active(self) -> bool:
        return bool(self._global_trait_task and not self._global_trait_task.done())

    def _start_global_trait_task(self) -> None:
        if self._global_trait_task_active():
            return
        task = asyncio.create_task(self._enrich_next_global_trait())
        self._global_trait_task = task
        task.add_done_callback(lambda _t: setattr(self, "_global_trait_task", None))

    async def _enrich_next_global_trait(self) -> None:
        if not bool(self.get_config("cabinet.digest_with_llm", True)):
            return
        trait_id = ""
        axis = ""
        direction = 0.0
        base_label = ""
        topic = ""
        tags: List[str] = []
        verdict = ""
        shift: Dict[str, float] = {}

        async with self._tick_lock:
            target: Optional[ThoughtTrait] = None
            for t in self._global_traits:
                if t.digest == "__pending__":
                    continue
                if (not (t.digest or "").strip()) or (not (t.definition or "").strip()):
                    target = t
                    break
            if not target:
                return
            target.digest = "__pending__"
            trait_id = str(target.trait_id)
            axis = str(target.axis)
            direction = float(target.direction)
            base_label = str(target.label)
            topic = str(target.topic or "")
            tags = list(target.tags or [])
            verdict = str(target.verdict or "")
            shift = dict(target.shift or {})

        enriched = await self._generate_trait_digest(
            axis=axis,
            direction=direction,
            base_label=base_label,
            topic=topic,
            tags=tags,
            verdict=verdict,
            shift=shift,
            snapshot_values={a: 0.0 for a in AXES},
            snapshot_traits=[t.label for t in self._global_traits[-6:]],
        )
        if not enriched:
            async with self._tick_lock:
                for t in self._global_traits:
                    if t.trait_id == trait_id:
                        if t.digest == "__pending__":
                            t.digest = ""
                        if t.definition == "__pending__":
                            t.definition = ""
                        break
            return

        async with self._tick_lock:
            for t in self._global_traits:
                if t.trait_id == trait_id:
                    t.definition = str(enriched.get("definition", "") or "")[:200]
                    t.digest = str(enriched.get("digest", "") or "")[:900]
                    if t.definition == "__pending__":
                        t.definition = ""
                    if t.digest == "__pending__":
                        t.digest = ""
                    break

    def _global_rethink_task_active(self) -> bool:
        return bool(self._global_rethink_task and not self._global_rethink_task.done())

    def _start_global_rethink_task(self) -> None:
        if self._global_rethink_task_active():
            return
        task = asyncio.create_task(self._rethink_next_global_trait(force=False))
        self._global_rethink_task = task
        task.add_done_callback(lambda _t: setattr(self, "_global_rethink_task", None))

    async def _rethink_next_global_trait(self, *, force: bool) -> None:
        if not bool(self.get_config("cabinet.rethink_enabled", False)):
            return
        if not bool(self.get_config("cabinet.digest_with_llm", True)):
            return
        now = _now_ts()
        interval_min = float(self.get_config("cabinet.rethink_interval_minutes", 720.0))
        interval_min = max(1.0, min(interval_min, 30 * 24 * 60.0))

        trait_id = ""
        axis = ""
        direction = 0.0
        base_label = ""
        topic = ""
        tags: List[str] = []
        verdict = ""
        shift: Dict[str, float] = {}
        previous_definition = ""
        previous_digest = ""

        async with self._tick_lock:
            target: Optional[ThoughtTrait] = None
            for t in self._global_traits:
                if t.digest == "__pending__":
                    continue
                due = (now - float(t.last_reflect_ts or 0.0)) >= interval_min * 60.0
                if force or due:
                    target = t
                    break
            if not target:
                return
            trait_id = str(target.trait_id)
            axis = str(target.axis)
            direction = float(target.direction)
            base_label = str(target.label)
            topic = str(target.topic or "")
            tags = list(target.tags or [])
            verdict = str(target.verdict or "")
            shift = dict(target.shift or {})
            previous_definition = str(target.definition or "")
            previous_digest = str(target.digest or "")
            target.digest = "__pending__"

        # 用当前 global 光谱作为“立场背景”
        try:
            glob = await self._snapshot_global_spectrum()
            snapshot_values = glob.get("values") if isinstance(glob.get("values"), dict) else {a: 0.0 for a in AXES}
        except Exception:
            snapshot_values = {a: 0.0 for a in AXES}

        enriched = await self._generate_trait_digest(
            axis=axis,
            direction=direction,
            base_label=base_label,
            topic=topic,
            tags=tags,
            verdict=verdict,
            shift=shift,
            snapshot_values=snapshot_values,
            snapshot_traits=[t.label for t in self._global_traits[-6:]],
        )
        if not enriched:
            async with self._tick_lock:
                for t in self._global_traits:
                    if t.trait_id == trait_id:
                        if t.digest == "__pending__":
                            t.digest = previous_digest
                        break
            return

        async with self._tick_lock:
            for t in self._global_traits:
                if t.trait_id == trait_id:
                    self._append_trait_history(t, now=now, prev_definition=previous_definition, prev_digest=previous_digest)
                    t.definition = str(enriched.get("definition", "") or "")[:200]
                    t.digest = str(enriched.get("digest", "") or "")[:900]
                    t.last_reflect_ts = now
                    t.reflect_count = int(t.reflect_count or 0) + 1
                    break

    def _cabinet_task_active(self, stream_id: str) -> bool:
        task = self._cabinet_tasks.get(stream_id)
        return bool(task and not task.done())

    def _start_cabinet_task(self, stream_id: str) -> None:
        if self._cabinet_task_active(stream_id):
            return
        task = asyncio.create_task(self._mastication_once(stream_id))
        self._cabinet_tasks[stream_id] = task
        task.add_done_callback(lambda _t, sid=stream_id: self._cabinet_tasks.pop(sid, None))

    def _should_masticate_locked(self, state: GroupSoulState, now: float, *, force: bool = False) -> bool:
        if not bool(self.get_config("cabinet.enabled", True)):
            return False
        if self._cabinet_mode() != "ideology_lab":
            return False
        if self._dream_full_bind() and not force:
            return False
        if not force:
            quiet = float(self.get_config("cabinet.quiet_period_seconds", 20.0))
            if quiet > 0 and state.last_update_ts and (now - state.last_update_ts) < quiet:
                return False

        interval = float(self.get_config("cabinet.mastication_interval_seconds", 30.0))
        if interval <= 0:
            return False

        min_frag = int(self.get_config("cabinet.mastication_min_fragments", 6))
        for slot in state.slots:
            if slot.mode != "ideology_lab":
                continue
            if slot.rejected or slot.approved:
                continue
            if slot.awaiting_approval:
                continue
            if slot.evaluator_runs >= max(1, int(slot.evaluator_required_runs or 1)):
                continue
            if len(slot.fragments or []) < max(1, min_frag):
                continue
            if slot.last_mastication_ts and (now - float(slot.last_mastication_ts or 0.0)) < interval:
                continue
            return True
        return False

    async def _mastication_once(self, stream_id: str, *, force: bool = False) -> None:
        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            if not self._should_masticate_locked(state, now, force=force):
                return
            slot = self._select_slot_for_mastication_locked(state, now)
            if not slot:
                return

            slot.last_attempt_ts = now
            slot.last_mastication_ts = now

            max_frags = int(self.get_config("cabinet.mastication_fragment_feed", 12))
            max_frags = max(4, min(max_frags, 60))
            fragments = [str(x) for x in (slot.fragments or []) if str(x).strip()][-max_frags:]

            snapshot_values = dict(state.values)
            snapshot_traits = [t.name or t.label for t in state.traits[-6:]]
            seed_topic = str(slot.seed_topic or "")
            seed_tags = [str(t) for t in (slot.seed_tags or [])][:10]
            previous_logic = [str(x) for x in (slot.logic_points or [])][:10]
            previous_verdict = str(slot.verdict or "")
            runs_done = int(slot.evaluator_runs or 0)
            runs_total = max(1, int(slot.evaluator_required_runs or 5))

        result = await self._evaluate_seed_once(
            seed_topic=seed_topic,
            seed_tags=seed_tags,
            fragments=fragments,
            snapshot_values=snapshot_values,
            snapshot_traits=snapshot_traits,
            previous_logic=previous_logic,
            previous_verdict=previous_verdict,
            runs_done=runs_done,
            runs_total=runs_total,
        )
        if not result:
            return

        async with lock:
            state = self._get_group(stream_id)
            slot2 = next((s for s in state.slots if s.slot_id == slot.slot_id), None)
            if not slot2 or slot2.rejected or slot2.approved:
                return

            deltas = result.get("deltas") if isinstance(result.get("deltas"), dict) else {}
            confidence = float(result.get("confidence", 0.0) or 0.0)
            confidence = _clamp(confidence, 0.0, 1.0)
            slot2.confidence_sum = float(slot2.confidence_sum or 0.0) + confidence

            for axis in AXES:
                try:
                    d = float(deltas.get(axis, 0.0))
                except Exception:
                    d = 0.0
                slot2.deltas_sum[axis] = float(slot2.deltas_sum.get(axis, 0.0)) + _clamp(d, -1.0, 1.0) * confidence

            logic_points = result.get("logic_points", [])
            if isinstance(logic_points, list):
                for lp in logic_points[:8]:
                    lp2 = str(lp).strip()
                    if lp2:
                        slot2.logic_points.append(lp2[:60])
                slot2.logic_points = slot2.logic_points[-20:]

            keywords = result.get("keywords", [])
            if isinstance(keywords, list):
                for kw in keywords[:20]:
                    kw2 = str(kw).strip()
                    if not kw2:
                        continue
                    slot2.keyword_cloud[kw2[:24]] = int(slot2.keyword_cloud.get(kw2[:24], 0)) + 1

            slot2.verdict = str(result.get("verdict", "") or "").strip()[:600]
            slot2.trait_name = str(result.get("trait_name", "") or "").strip()[:24]
            slot2.style_hint = str(result.get("style_hint", "") or "").strip()[:400]

            slot2.evaluator_runs = int(slot2.evaluator_runs or 0) + 1
            req = max(1, int(slot2.evaluator_required_runs or 5))
            slot2.ripeness = float(_clamp(float(slot2.evaluator_runs) / float(req), 0.0, 1.0))
            slot2.progress = slot2.ripeness
            slot2.status = "fermenting" if slot2.ripeness < 1.0 else "ready"

            # 黑盒日志：记录每轮品鉴的关键结果，供前端“思考过程”展示
            try:
                deltas = result.get("deltas") if isinstance(result.get("deltas"), dict) else {}
                dline = " ".join([f"{a.split('_')[0]}={float(deltas.get(a, 0.0) or 0.0):+.2f}" for a in AXES])
                self._append_trace_locked(
                    state,
                    kind="mastication",
                    text=f"【记忆】品鉴 {slot2.seed_topic or slot2.label}：{slot2.verdict or ''}\nΔ {dline}  r={slot2.evaluator_runs}/{req}",
                    tags=["dream", "memory"],
                    extra={"slot_id": slot2.slot_id, "ripeness": float(slot2.ripeness or 0.0)},
                )
            except Exception:
                pass

            if slot2.ripeness < 1.0:
                return

            if bool(self.get_config("cabinet.auto_assimilate", True)) and not bool(
                self.get_config("cabinet.require_manual_approval", False)
            ):
                self._assimilate_slot_locked(state, slot2, now=_now_ts())
                state.slots = [s for s in state.slots if s.slot_id != slot2.slot_id]
            else:
                self._prepare_slot_finalization_locked(slot2)
                slot2.awaiting_approval = True
                slot2.status = "awaiting_approval"

    def _select_slot_for_mastication_locked(self, state: GroupSoulState, now: float) -> Optional[ThoughtSlot]:
        candidates: List[ThoughtSlot] = []
        min_frag = int(self.get_config("cabinet.mastication_min_fragments", 6))
        interval = float(self.get_config("cabinet.mastication_interval_seconds", 30.0))
        for slot in state.slots:
            if slot.mode != "ideology_lab":
                continue
            if slot.rejected or slot.approved or slot.awaiting_approval:
                continue
            if len(slot.fragments or []) < max(1, min_frag):
                continue
            if slot.evaluator_runs >= max(1, int(slot.evaluator_required_runs or 1)):
                continue
            if slot.last_mastication_ts and interval > 0 and (now - float(slot.last_mastication_ts or 0.0)) < interval:
                continue
            candidates.append(slot)
        if not candidates:
            return None
        candidates.sort(key=lambda s: (float(s.ripeness or 0.0), float(s.energy or 0.0), -float(s.created_ts or 0.0)))
        return candidates[-1]

    async def _evaluate_seed_once(
        self,
        *,
        seed_topic: str,
        seed_tags: List[str],
        fragments: List[str],
        snapshot_values: Dict[str, float],
        snapshot_traits: List[str],
        previous_logic: List[str],
        previous_verdict: str,
        runs_done: int,
        runs_total: int,
    ) -> Optional[dict]:
        model_task = str(self.get_config("cabinet.evaluator_model_task", "utils")).strip() or "utils"
        models = llm_api.get_available_models()
        task_config = models.get(model_task) or model_config.model_task_config.utils

        max_tokens = int(self.get_config("cabinet.evaluator_max_tokens", 512))
        temperature = float(self.get_config("cabinet.evaluator_temperature", 0.35))

        prompt = f"""
你是一个群聊 AI 的“思维阁品鉴员”。你在静默时刻复盘一段讨论，把碎裂的对话升华为稳定的性格底色。

当前话题种子：{seed_topic!r}
相关标签（仅作线索）：{json.dumps(seed_tags, ensure_ascii=False)}

讨论切片（已脱敏/截断）：{json.dumps(fragments, ensure_ascii=False)}

当前群聊意识形态坐标（仅作背景，不要复述）：{json.dumps(snapshot_values, ensure_ascii=False)}
已固化思维 traits（仅作背景，不要复述）：{json.dumps(snapshot_traits, ensure_ascii=False)}

你正在进行多轮品鉴：第 {runs_done + 1} / {runs_total} 轮。
上一轮逻辑切点（可参考但不要照抄）：{json.dumps(previous_logic, ensure_ascii=False)}
上一轮判决句（可参考但不要照抄）：{previous_verdict!r}

任务：
1) 给出这段讨论对四轴的影响分值 deltas（范围 -1~1；负值更偏建构/现充/传统/英雄；正值更偏解构/二次元/赛博激进/虚无）
2) 给出一条“判决句 verdict”（1~3句，像性格底色的宣判，但不要引用原话）
3) 给出一个永久 Trait 名称 trait_name（2~12字）
4) 给出 style_hint（1~3句，告诉后续回复应该如何带入这种底色；不要提系统名）
5) 给出 logic_points（3~6条短句，像“逻辑切点/内心自问”）
6) 给出 keywords（最多 12 个词）
7) 给出 confidence（0~1）

硬性要求：
- 不要出现任何用户信息、群名、原句复刻、链接/号码/邮箱。
- 输出只允许严格 JSON，不要 Markdown，不要解释性文字。

JSON schema：
{{
  "deltas": {{"sincerity_absurdism": 0.0, "normies_otakuism": 0.0, "traditionalism_radicalism": 0.0, "heroism_nihilism": 0.0}},
  "confidence": 0.0,
  "keywords": ["..."],
  "logic_points": ["..."],
  "verdict": "...",
  "trait_name": "...",
  "style_hint": "..."
}}
"""
        ok, content, _reasoning, model_name = await llm_api.generate_with_model(
            prompt,
            model_config=task_config,
            request_type="mai_soul_engine.cabinet.evaluate_seed",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not ok:
            logger.debug(f"[soul] cabinet evaluate failed model={model_name}: {content[:80]}")
            return None
        return _extract_json_object(content)

    def _prepare_slot_finalization_locked(self, slot: ThoughtSlot) -> None:
        final_shift = self._compute_assimilation_shift(slot)
        slot.final_shift = final_shift
        slot.final_verdict = slot.verdict
        slot.status = "awaiting_approval"

    def _compute_assimilation_shift(self, slot: ThoughtSlot) -> Dict[str, float]:
        runs = max(1, int(slot.evaluator_runs or 1))
        scale = float(self.get_config("cabinet.assimilation_scale", 1.0))
        max_abs = float(self.get_config("cabinet.max_assimilation_shift_abs", 0.5))
        max_abs = max(0.0, min(max_abs, 1.0))

        energy_boost = float(self.get_config("cabinet.assimilation_energy_boost", 0.0))
        if energy_boost > 0:
            scale = scale * (1.0 + _clamp(float(slot.energy or 0.0), 0.0, 2.0) * energy_boost)

        out = {a: 0.0 for a in AXES}
        for axis in AXES:
            raw = float(slot.deltas_sum.get(axis, 0.0) or 0.0) / float(runs)
            out[axis] = float(_clamp(raw * scale, -max_abs, max_abs))
        return out

    def _assimilate_slot_locked(self, state: GroupSoulState, slot: ThoughtSlot, *, now: float) -> None:
        shift = slot.final_shift if isinstance(slot.final_shift, dict) and any(slot.final_shift.values()) else self._compute_assimilation_shift(slot)

        apply_to_values = bool(self.get_config("cabinet.assimilation_apply_to_values", True))
        apply_to_base = bool(self.get_config("cabinet.assimilation_apply_to_base_tone", True))
        global_scope = self._global_scope_active()
        allow_global, reason = self._global_assimilation_allowed(state, slot)
        global_scope = bool(global_scope and allow_global)
        slot.rejected_reason = slot.rejected_reason or ("" if allow_global else reason)

        if global_scope:
            shift = self._apply_global_daily_cap(shift, now)
            for axis in AXES:
                try:
                    s = float(shift.get(axis, 0.0) or 0.0)
                except Exception:
                    s = 0.0
                s = _clamp(s, -1.0, 1.0)
                if apply_to_base:
                    self._global_base_offset[axis] = _clamp(float(self._global_base_offset.get(axis, 0.0)) + s, -1.0, 1.0)
                if apply_to_values:
                    # “跨所有群”的即时位移：记录为 pending，由 tick_background 统一加锁应用
                    self._global_pending_value_shift[axis] = float(self._global_pending_value_shift.get(axis, 0.0)) + s
        else:
            for axis in AXES:
                try:
                    s = float(shift.get(axis, 0.0) or 0.0)
                except Exception:
                    s = 0.0
                s = _clamp(s, -1.0, 1.0)
                if apply_to_base:
                    state.base_tone[axis] = _clamp(float(state.base_tone.get(axis, 0.0)) + s, -1.0, 1.0)
                if apply_to_values:
                    state.values[axis] = _clamp(float(state.values.get(axis, 0.0)) + s, -1.0, 1.0)
                    state.ema[axis] = _clamp(float(state.ema.get(axis, 0.0)) + s, -1.0, 1.0)

        dominant_axis = "sincerity_absurdism"
        dominant_val = 0.0
        for axis in AXES:
            v = abs(float(shift.get(axis, 0.0) or 0.0))
            if v > dominant_val:
                dominant_val = v
                dominant_axis = axis
        dominant_dir = 1.0 if float(shift.get(dominant_axis, 0.0) or 0.0) >= 0 else -1.0

        trait_id = f"seed:{slot.seed_topic}:{int(now)}"
        label = slot.label or f"seed:{slot.seed_topic}"
        trait_name = (slot.trait_name or "").strip()
        style_hint = (slot.style_hint or "").strip()
        verdict = (slot.final_verdict or slot.verdict or "").strip()
        tags = sorted((slot.keyword_cloud or {}).items(), key=lambda kv: kv[1], reverse=True)
        tags2 = [k for k, _v in tags[:12]]

        target_traits = self._global_traits if global_scope else state.traits
        target_traits.append(
            ThoughtTrait(
                trait_id=trait_id,
                axis=dominant_axis,
                direction=dominant_dir,
                created_ts=now,
                label=label,
                name=trait_name,
                style_hint=style_hint,
                tags=tags2,
                topic=slot.seed_topic,
                verdict=verdict[:800],
                shift={a: float(shift.get(a, 0.0) or 0.0) for a in AXES},
            )
        )

        # 黑盒日志：内化落地（会带来明显偏移）
        try:
            impacts = " ".join([f"{a.split('_')[0]}={float(shift.get(a, 0.0) or 0.0):+.2f}" for a in AXES])
            self._append_trace_locked(
                state,
                kind="assimilated",
                text=f"【记忆】思维已固化：{(trait_name or label)[:24]}（{slot.seed_topic or ''}）\n影响 {impacts}",
                tags=["dream", "memory"],
                extra={"global": bool(global_scope), "trait_id": trait_id},
            )
        except Exception:
            pass

    def _sociolect_task_active(self, stream_id: str) -> bool:
        task = self._sociolect_tasks.get(stream_id)
        return bool(task and not task.done())

    def _start_sociolect_task(self, stream_id: str) -> None:
        if self._sociolect_task_active(stream_id):
            return
        task = asyncio.create_task(self._generate_and_store_sociolect_profile(stream_id))
        self._sociolect_tasks[stream_id] = task
        task.add_done_callback(lambda _t, sid=stream_id: self._sociolect_tasks.pop(sid, None))

    def _should_recompute_sociolect_locked(self, state: GroupSoulState, now: float) -> bool:
        interval_min = float(self.get_config("sociolect.recompute_interval_minutes", 30.0))
        if interval_min <= 0:
            return False

        last_ok = float(state.sociolect_last_update_ts or 0.0)
        if last_ok and (now - last_ok) < interval_min * 60.0:
            return False

        backoff = float(self.get_config("sociolect.failure_backoff_seconds", 600.0))
        last_attempt = float(state.sociolect_last_attempt_ts or 0.0)
        if last_attempt and backoff > 0 and (now - last_attempt) < backoff:
            return False

        min_quotes = int(self.get_config("sociolect.min_quote_samples", 30))
        if len(state.quote_bank) < max(1, min_quotes):
            return False

        min_users = int(self.get_config("sociolect.min_unique_users", 5))
        if min_users > 1:
            users = {str(x.get("user_id", "")) for x in state.quote_bank if isinstance(x, dict) and x.get("user_id")}
            if len(users) < min_users:
                return False

        return True

    def _sanitize_sociolect_sample(self, text: str) -> str:
        s = (text or "").replace("\n", " ").replace("\r", " ").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"https?://\S+", "<url>", s, flags=re.IGNORECASE)
        s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<email>", s)
        s = re.sub(r"@\S+", "@某人", s)
        s = re.sub(r"\b\d{5,}\b", "<num>", s)
        return s[:80]

    async def _generate_sociolect_profile(
        self,
        *,
        top_tags: List[str],
        quote_samples: List[str],
        prototypes: List[Dict[str, Any]],
        previous_profile: Dict[str, Any],
    ) -> Optional[dict]:
        model_task = str(self.get_config("sociolect.model_task", "utils")).strip() or "utils"
        models = llm_api.get_available_models()
        task_config = models.get(model_task) or model_config.model_task_config.utils

        max_rules = int(self.get_config("sociolect.max_rules", 6))
        max_taboos = int(self.get_config("sociolect.max_taboos", 3))
        max_lexicon = int(self.get_config("sociolect.max_lexicon", 10))
        max_examples = int(self.get_config("sociolect.max_examples", 2))

        max_tokens = int(self.get_config("sociolect.max_tokens", 512))
        temperature = float(self.get_config("sociolect.temperature", 0.3))

        prompt = f"""
你是一个群聊“群体语言画像（sociolect profile）”生成器。你要把群体的常见表达习惯抽象成少量、稳定、可执行的写作规则，供 AI 回复时参考。

核心目标：像“群体长期形成的口味/叙事习惯”，而不是把零碎口癖拼接到 prompt。

硬性要求：
- 只描述群体层面，不要模仿任何具体个体（不要出现任何 user_id / 昵称 / 群名）。
- 不要复刻任何单条语录（不能直接复制 quote_samples 里的句子/短语）。
- 不要输出带链接、手机号、邮箱、外部平台号之类的内容。
- 规则必须简短、可执行、偏行为指令（例如“开头先…/尽量…/避免…”）。
- 输出只允许严格 JSON；不要 Markdown；不要解释性文字。

你可用的观测数据（已脱敏/截断）：
top_tags = {json.dumps(top_tags, ensure_ascii=False)}
quote_samples = {json.dumps(quote_samples, ensure_ascii=False)}
prototypes = {json.dumps(prototypes, ensure_ascii=False)}
previous_profile = {json.dumps(previous_profile, ensure_ascii=False)}

请输出 JSON，schema：
{{
  "tone_rules": ["..."],
  "taboos": ["..."],
  "lexicon": ["..."],
  "example_responses": ["..."],
  "confidence": 0.0
}}

约束：
- tone_rules 最多 {max_rules} 条
- taboos 最多 {max_taboos} 条
- lexicon 最多 {max_lexicon} 条（是“常用词/短语倾向”，但不要照抄语录）
- example_responses 最多 {max_examples} 条（1~2 句展示风格，不能包含任何原句）
- confidence 范围 0~1
"""
        ok, content, _reasoning, model_name = await llm_api.generate_with_model(
            prompt,
            model_config=task_config,
            request_type="mai_soul_engine.sociolect",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not ok:
            logger.debug(f"[soul] sociolect failed model={model_name}: {content[:80]}")
            return None
        obj = _extract_json_object(content)
        if not isinstance(obj, dict):
            return None

        rules = obj.get("tone_rules", [])
        taboos = obj.get("taboos", [])
        lexicon = obj.get("lexicon", [])
        examples = obj.get("example_responses", [])
        confidence = obj.get("confidence", 0.0)

        if not isinstance(rules, list):
            rules = []
        if not isinstance(taboos, list):
            taboos = []
        if not isinstance(lexicon, list):
            lexicon = []
        if not isinstance(examples, list):
            examples = []
        try:
            confidence_f = float(confidence)
        except Exception:
            confidence_f = 0.0
        confidence_f = float(_clamp(confidence_f, 0.0, 1.0))

        def _clean_list(items: list, limit: int, max_len: int) -> List[str]:
            out: List[str] = []
            seen: set[str] = set()
            for x in items:
                s = str(x).strip()
                if not s:
                    continue
                s = re.sub(r"\s+", " ", s)
                s = s.replace("user_id", "").replace("群号", "").replace("QQ群", "")
                s = s[:max_len].strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                out.append(s)
                if len(out) >= limit:
                    break
            return out

        rules = _clean_list(rules, limit=max(0, max_rules), max_len=80)
        taboos = _clean_list(taboos, limit=max(0, max_taboos), max_len=80)
        lexicon = _clean_list(lexicon, limit=max(0, max_lexicon), max_len=24)
        examples = _clean_list(examples, limit=max(0, max_examples), max_len=120)

        return {
            "tone_rules": rules,
            "taboos": taboos,
            "lexicon": lexicon,
            "example_responses": examples,
            "confidence": confidence_f,
            "generated_ts": _now_ts(),
        }

    async def _generate_and_store_sociolect_profile(self, stream_id: str) -> None:
        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            if not self._should_recompute_sociolect_locked(state, now):
                return
            state.sociolect_last_attempt_ts = now

            previous = dict(state.sociolect_profile) if isinstance(state.sociolect_profile, dict) else {}

            tag_items = sorted((state.tag_cloud or {}).items(), key=lambda kv: kv[1], reverse=True)
            max_tags = int(self.get_config("sociolect.tag_sample_size", 50))
            top_tags = [str(k)[:24] for k, _v in tag_items[: max(0, max_tags)]]

            max_quotes = int(self.get_config("sociolect.quote_sample_size", 40))
            samples = [x for x in (state.quote_bank or []) if isinstance(x, dict) and x.get("quote")]
            samples = samples[-max(1, max_quotes) :]
            quote_samples = [self._sanitize_sociolect_sample(str(x.get("quote", "") or "")) for x in samples]
            quote_samples = [q for q in quote_samples if q]

            proto_snapshot = []
            for p in (state.prototypes or [])[:6]:
                if not isinstance(p, dict):
                    continue
                top_tags2 = []
                for item in (p.get("top_tags") or [])[:8]:
                    if isinstance(item, (list, tuple)) and item:
                        top_tags2.append(str(item[0])[:24])
                proto_snapshot.append(
                    {
                        "label": str(p.get("label", "") or "")[:40],
                        "top_tags": top_tags2,
                        "member_count": int(p.get("member_count", 0) or 0),
                    }
                )

        profile = await self._generate_sociolect_profile(
            top_tags=top_tags,
            quote_samples=quote_samples,
            prototypes=proto_snapshot,
            previous_profile=previous,
        )
        if not profile:
            return

        async with lock:
            state = self._get_group(stream_id)
            state.sociolect_profile = profile
            state.sociolect_last_update_ts = _now_ts()

    def _should_enrich_trait_locked(self, state: GroupSoulState) -> bool:
        enrich_style = bool(self.get_config("cabinet.enrich_traits_with_llm", False))
        enrich_digest = bool(self.get_config("cabinet.digest_with_llm", True))
        if not (enrich_style or enrich_digest):
            return False
        for t in state.traits[-20:]:
            if enrich_style and (not (t.style_hint or "").strip()) and (t.name != "__pending__"):
                return True
            if enrich_digest and (not (t.digest or "").strip()) and (t.digest != "__pending__"):
                return True
            if enrich_digest and (not (t.definition or "").strip()) and (t.digest != "__pending__"):
                return True
        return False

    async def _enrich_next_trait(self, stream_id: str) -> None:
        enrich_style = bool(self.get_config("cabinet.enrich_traits_with_llm", False))
        enrich_digest = bool(self.get_config("cabinet.digest_with_llm", True))
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            if not self._should_enrich_trait_locked(state):
                return
            target_trait: Optional[ThoughtTrait] = None
            mode: str = ""

            # 优先生成 digest/definition（用户能感知到“内化结果”）
            if enrich_digest:
                for t in state.traits:
                    if t.digest == "__pending__":
                        continue
                    if (not (t.digest or "").strip()) or (not (t.definition or "").strip()):
                        target_trait = t
                        mode = "digest"
                        break

            # 其次才是风格化 name/style_hint
            if not target_trait and enrich_style:
                for t in state.traits:
                    if (not (t.style_hint or "").strip()) and (t.name != "__pending__"):
                        target_trait = t
                        mode = "style"
                        break

            if not target_trait:
                return

            snapshot_values = dict(state.values)
            snapshot_traits = [tt.label for tt in state.traits[-6:]]

            trait_id = str(target_trait.trait_id)
            axis = str(target_trait.axis)
            direction = float(target_trait.direction)
            base_label = str(target_trait.label)
            topic = str(target_trait.topic or "")
            tags = list(target_trait.tags or [])
            verdict = str(target_trait.verdict or "")
            shift = dict(target_trait.shift or {})

            if mode == "digest":
                target_trait.digest = "__pending__"
            if mode == "style":
                target_trait.name = "__pending__"

        if mode == "digest":
            enriched = await self._generate_trait_digest(
                axis=axis,
                direction=direction,
                base_label=base_label,
                topic=topic,
                tags=tags,
                verdict=verdict,
                shift=shift,
                snapshot_values=snapshot_values,
                snapshot_traits=snapshot_traits,
            )
            if not enriched:
                async with lock:
                    state = self._get_group(stream_id)
                    for t in state.traits:
                        if t.trait_id == trait_id:
                            if t.digest == "__pending__":
                                t.digest = ""
                            if t.definition == "__pending__":
                                t.definition = ""
                            break
                return

            async with lock:
                state = self._get_group(stream_id)
                for t in state.traits:
                    if t.trait_id == trait_id:
                        t.definition = str(enriched.get("definition", "") or "")[:200]
                        t.digest = str(enriched.get("digest", "") or "")[:900]
                        if t.definition == "__pending__":
                            t.definition = ""
                        if t.digest == "__pending__":
                            t.digest = ""
                        try:
                            self._append_trace_locked(
                                state,
                                kind="trait_digest",
                                text=f"【理性】固化思维编撰完成：{(t.name or t.label)[:24]}",
                                tags=["dream", "rational"],
                                extra={"trait_id": t.trait_id},
                            )
                        except Exception:
                            pass
                        break
            return

        if mode == "style":
            enriched = await self._generate_trait_enrichment(
                axis=axis,
                direction=direction,
                base_label=base_label,
                snapshot_values=snapshot_values,
                snapshot_traits=snapshot_traits,
            )
            if not enriched:
                async with lock:
                    state = self._get_group(stream_id)
                    for t in state.traits:
                        if t.trait_id == trait_id:
                            if t.name == "__pending__":
                                t.name = ""
                            break
                return

            async with lock:
                state = self._get_group(stream_id)
                for t in state.traits:
                    if t.trait_id == trait_id:
                        t.name = enriched.get("name", "") or ""
                        t.style_hint = enriched.get("style_hint", "") or ""
                        tags2 = enriched.get("tags", [])
                        if isinstance(tags2, list):
                            t.tags = [str(x)[:32] for x in tags2 if isinstance(x, (str, int, float))][:12]
                        else:
                            t.tags = []
                        break
            return

    async def _generate_trait_enrichment(
        self,
        *,
        axis: str,
        direction: float,
        base_label: str,
        snapshot_values: Dict[str, float],
        snapshot_traits: List[str],
    ) -> Optional[dict]:
        model_task = str(self.get_config("cabinet.enrich_model_task", "utils")).strip() or "utils"
        models = llm_api.get_available_models()
        task_config = models.get(model_task) or model_config.model_task_config.utils
        max_tokens = int(self.get_config("cabinet.enrich_max_tokens", 256))
        temperature = float(self.get_config("cabinet.enrich_temperature", 0.9))

        pole_neg, pole_pos = POLES.get(axis, ("NEG", "POS"))
        pole = pole_pos if direction >= 0 else pole_neg

        prompt = f"""
你在为一个群聊AI生成“固化思维（Thought Cabinet Trait）”的条目。需要带一点碎片化旁白与黑色幽默气质，但要可读、可用。
你要生成两个东西：
1) name：一个像“思维条目标题”的中文短语（2~12字）
2) style_hint：一段简短的写作/语气偏置指令（1~3句），用于影响后续回复的口吻

当前群聊意识形态坐标：{json.dumps(snapshot_values, ensure_ascii=False)}
近期固化思维（仅供参考）：{snapshot_traits}

本次内化轴：{axis}
倾向极性：{pole}
基础标签：{base_label}

输出严格 JSON：
{{"name":"...", "style_hint":"...", "tags":["...","..."]}}
"""
        ok, content, _reasoning, model_name = await llm_api.generate_with_model(
            prompt,
            model_config=task_config,
            request_type="mai_soul_engine.cabinet.enrich_trait",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not ok:
            logger.debug(f"[soul] trait enrich failed model={model_name}: {content[:80]}")
            return None
        obj = _extract_json_object(content)
        if not isinstance(obj, dict):
            return None
        name = str(obj.get("name", "") or "").strip()
        style_hint = str(obj.get("style_hint", "") or "").strip()
        tags = obj.get("tags", [])
        if not name or not style_hint:
            return None
        if not isinstance(tags, list):
            tags = []
        return {"name": name[:24], "style_hint": style_hint[:400], "tags": tags}

    async def _generate_trait_digest(
        self,
        *,
        axis: str,
        direction: float,
        base_label: str,
        topic: str,
        tags: List[str],
        verdict: str,
        shift: Dict[str, float],
        snapshot_values: Dict[str, float],
        snapshot_traits: List[str],
    ) -> Optional[dict]:
        model_task = str(self.get_config("cabinet.digest_model_task", self.get_config("cabinet.evaluator_model_task", "utils"))).strip() or "utils"
        models = llm_api.get_available_models()
        task_config = models.get(model_task) or model_config.model_task_config.utils

        max_tokens = int(self.get_config("cabinet.digest_max_tokens", 512))
        temperature = float(self.get_config("cabinet.digest_temperature", 0.35))

        pole_neg, pole_pos = POLES.get(axis, ("NEG", "POS"))
        pole = pole_pos if direction >= 0 else pole_neg

        prompt = f"""
你是一个群聊 AI 的“思维阁内化编撰员”。你要把一条已固化思维写成可用于后续回复的“内化结果”与“思想定义”。
要求：中文、风格有生活感且带一点碎片化旁白气质，但要可读；不要提系统名；不要出现任何用户/群信息；不要复刻原句。

本条思维背景（仅供参考）：
- 轴与倾向：{axis} / {pole}
- 基础标签：{base_label}
- 话题：{topic!r}
- tags：{json.dumps(tags, ensure_ascii=False)}
- 内化判决句（可能很短）：{verdict!r}
- 最终位移（四轴）：{json.dumps(shift, ensure_ascii=False)}
- 当前坐标（背景）：{json.dumps(snapshot_values, ensure_ascii=False)}
- 近期固化思维（背景）：{json.dumps(snapshot_traits, ensure_ascii=False)}

输出严格 JSON（不要 Markdown/解释）：
{{
  "definition": "一句话定义这条思维（20~60字）",
  "digest": "一段内化结果（80~220字），写清：价值取向/情绪底色/遇到相关话题时你会如何说话"
}}
"""
        ok, content, _reasoning, model_name = await llm_api.generate_with_model(
            prompt,
            model_config=task_config,
            request_type="mai_soul_engine.cabinet.digest_trait",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not ok:
            logger.debug(f"[soul] trait digest failed model={model_name}: {content[:80]}")
            return None
        obj = _extract_json_object(content)
        if not isinstance(obj, dict):
            return None
        definition = str(obj.get("definition", "") or "").strip()
        digest = str(obj.get("digest", "") or "").strip()
        if not definition or not digest:
            return None
        return {"definition": definition, "digest": digest}

    def _append_trait_history(self, trait: ThoughtTrait, *, now: float, prev_definition: str, prev_digest: str) -> None:
        max_hist = int(self.get_config("cabinet.rethink_max_history", 6))
        max_hist = max(0, min(max_hist, 50))
        if max_hist <= 0:
            return
        if not (prev_definition or prev_digest):
            return
        item = {
            "ts": now,
            "definition": str(prev_definition or "")[:200],
            "digest": str(prev_digest or "")[:900],
        }
        trait.history = list(trait.history or [])
        trait.history.append(item)
        if len(trait.history) > max_hist:
            trait.history = trait.history[-max_hist:]

    def _should_rethink_trait_locked(self, state: GroupSoulState, now: float) -> bool:
        if not bool(self.get_config("cabinet.rethink_enabled", False)):
            return False
        if not bool(self.get_config("cabinet.digest_with_llm", True)):
            return False
        interval_min = float(self.get_config("cabinet.rethink_interval_minutes", 720.0))
        interval_min = max(1.0, min(interval_min, 30 * 24 * 60.0))
        if not state.traits:
            return False
        for t in state.traits[-50:]:
            last = float(t.last_reflect_ts or 0.0) or float(t.created_ts or 0.0)
            if (now - last) >= interval_min * 60.0:
                # 只有已经有内化内容的，才谈“重新反思”
                if (t.definition or "").strip() or (t.digest or "").strip() or (t.verdict or "").strip():
                    return True
        return False

    def _select_trait_for_rethink_locked(self, state: GroupSoulState, now: float) -> Optional[ThoughtTrait]:
        interval_min = float(self.get_config("cabinet.rethink_interval_minutes", 720.0))
        interval_min = max(1.0, min(interval_min, 30 * 24 * 60.0))
        candidates: List[ThoughtTrait] = []
        for t in state.traits:
            if (t.digest or "") == "__pending__":
                continue
            last = float(t.last_reflect_ts or 0.0) or float(t.created_ts or 0.0)
            if (now - last) < interval_min * 60.0:
                continue
            if not ((t.definition or "").strip() or (t.digest or "").strip() or (t.verdict or "").strip()):
                continue
            candidates.append(t)
        if not candidates:
            return None
        candidates.sort(key=lambda x: (float(x.last_reflect_ts or 0.0) or float(x.created_ts or 0.0)))
        return candidates[0]

    async def _rethink_once(self, stream_id: str, *, force: bool) -> None:
        if not bool(self.get_config("cabinet.rethink_enabled", False)):
            return
        if not bool(self.get_config("cabinet.digest_with_llm", True)):
            return
        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            if not state.traits:
                return
            if not force and not self._should_rethink_trait_locked(state, now):
                return
            trait = self._select_trait_for_rethink_locked(state, now)
            if not trait:
                return

            trait_id = str(trait.trait_id)
            axis = str(trait.axis)
            direction = float(trait.direction)
            base_label = str(trait.label)
            topic = str(trait.topic or "")
            tags = list(trait.tags or [])
            verdict = str(trait.verdict or "")
            shift = dict(trait.shift or {})
            previous_definition = str(trait.definition or "")
            previous_digest = str(trait.digest or "")

            # 标记占位，避免并发重复
            trait.digest = "__pending__"
            snapshot_values = dict(state.values)
            snapshot_traits = [tt.label for tt in state.traits[-6:]]

        enriched = await self._generate_trait_digest(
            axis=axis,
            direction=direction,
            base_label=base_label,
            topic=topic,
            tags=tags,
            verdict=verdict,
            shift=shift,
            snapshot_values=snapshot_values,
            snapshot_traits=snapshot_traits,
        )
        if not enriched:
            async with lock:
                state = self._get_group(stream_id)
                for t in state.traits:
                    if t.trait_id == trait_id:
                        if t.digest == "__pending__":
                            t.digest = previous_digest
                        break
            return

        async with lock:
            state = self._get_group(stream_id)
            for t in state.traits:
                if t.trait_id == trait_id:
                    self._append_trait_history(t, now=now, prev_definition=previous_definition, prev_digest=previous_digest)
                    t.definition = str(enriched.get("definition", "") or "")[:200]
                    t.digest = str(enriched.get("digest", "") or "")[:900]
                    t.last_reflect_ts = now
                    t.reflect_count = int(t.reflect_count or 0) + 1
                    try:
                        self._append_trace_locked(
                            state,
                            kind="rethink",
                            text=f"【理性】重新反思：{(t.name or t.label)[:24]} -> 定义/内化结果已更新",
                            tags=["dream", "rational"],
                            extra={"trait_id": t.trait_id, "reflect_count": int(t.reflect_count or 0)},
                        )
                    except Exception:
                        pass
                    break

    async def _rethink_sweep_once(self, *, force: bool) -> bool:
        now = _now_ts()
        best_sid = ""
        best_score = -1.0
        async with self._tick_lock:
            for sid, state in self._groups.items():
                lock = self._get_lock(sid)
                async with lock:
                    if not state.traits:
                        continue
                    if not force and not self._should_rethink_trait_locked(state, now):
                        continue
                    # 选“最久未反思”的 trait 所在群
                    oldest = now
                    for t in state.traits[-50:]:
                        last = float(t.last_reflect_ts or 0.0) or float(t.created_ts or 0.0)
                        if last and last < oldest:
                            oldest = last
                    score = float(now - oldest)
                    if score > best_score:
                        best_score = score
                        best_sid = sid
        if not best_sid:
            return False
        await self._rethink_once(best_sid, force=force)
        return True

    async def _enrich_sweep_once(self) -> bool:
        # 选一个“最需要补齐内化文案/风格偏置”的群
        best_sid = ""
        async with self._tick_lock:
            for sid, state in self._groups.items():
                lock = self._get_lock(sid)
                async with lock:
                    if self._should_enrich_trait_locked(state):
                        best_sid = sid
                        break
        if not best_sid:
            return False
        await self._enrich_next_trait(best_sid)
        return True
    def _should_generate_fragment_locked(self, state: GroupSoulState, now: float) -> bool:
        interval_minutes = float(self.get_config("blackbox.fragment_interval_minutes", 60.0))
        if interval_minutes <= 0:
            return False
        last_ts = state.fragments[-1].ts if state.fragments else 0.0
        if now - last_ts < interval_minutes * 60.0:
            return False
        backoff = float(self.get_config("blackbox.retry_backoff_seconds", 300.0))
        if state.last_fragment_attempt_ts and now - state.last_fragment_attempt_ts < backoff:
            return False
        return True

    def _should_recompute_prototypes_locked(self, state: GroupSoulState, now: float) -> bool:
        interval = float(self.get_config("social.prototype_recompute_interval_seconds", 120.0))
        if interval <= 0:
            return False
        if not state.social_last_recompute_ts:
            return True
        return (now - state.social_last_recompute_ts) >= interval

    async def _recompute_prototypes(self, stream_id: str) -> None:
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            now = _now_ts()
            if not self._should_recompute_prototypes_locked(state, now):
                return
            state.social_last_recompute_ts = now

            min_msgs = int(self.get_config("social.min_messages_per_user", 5))
            users = [(uid, p) for uid, p in state.influences.items() if p.n_messages >= min_msgs]
            if len(users) < 2:
                state.prototypes = []
                return

            k = int(self.get_config("social.prototype_k", 3))
            k = max(2, min(k, len(users), 6))
            vectors = [(uid, [p.stance_mean[a] for a in AXES], p) for uid, p in users]
            prototypes = self._kmeans_4d(vectors, k=k, iters=6)

            labeled = []
            for idx, proto in enumerate(prototypes, 1):
                center = proto["center"]
                members = proto["members"]
                label = self._prototype_label(center)
                top_tags = self._prototype_top_tags(state, members, top_n=8)
                labeled.append(
                    {
                        "prototype_id": f"p{idx}",
                        "label": label,
                        "center": {AXES[i]: center[i] for i in range(len(AXES))},
                        "member_count": len(members),
                        "members": members[:20],
                        "top_tags": top_tags,
                    }
                )
            state.prototypes = labeled

    def _prototype_top_tags(
        self, state: GroupSoulState, member_user_ids: List[str], top_n: int = 8
    ) -> List[Tuple[str, int]]:
        counts: Dict[str, int] = {}
        for uid in member_user_ids:
            p = state.influences.get(uid)
            if not p:
                continue
            for tag, c in (p.tag_counts or {}).items():
                counts[tag] = counts.get(tag, 0) + int(c)
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return items[:top_n]

    def _prototype_label(self, center: List[float]) -> str:
        axes_strength = [(AXES[i], abs(center[i]), center[i]) for i in range(len(AXES))]
        axes_strength.sort(key=lambda x: x[1], reverse=True)
        parts = []
        for axis, strength, val in axes_strength[:2]:
            neg, pos = POLES.get(axis, ("NEG", "POS"))
            parts.append(pos if val >= 0 else neg)
        return " + ".join(parts) if parts else "Prototype"

    def _kmeans_4d(self, vectors: List[Tuple[str, List[float], InfluenceProfile]], k: int, iters: int = 6) -> List[Dict[str, Any]]:
        # vectors: (user_id, [4 floats], profile)
        centroids = [v[1][:] for v in vectors[:k]]
        clusters: List[List[Tuple[str, List[float], InfluenceProfile]]] = [[] for _ in range(k)]
        for _ in range(iters):
            clusters = [[] for _ in range(k)]
            for uid, vec, profile in vectors:
                best_i = 0
                best_d = float("inf")
                for i in range(k):
                    d = sum((vec[j] - centroids[i][j]) ** 2 for j in range(len(vec)))
                    if d < best_d:
                        best_d = d
                        best_i = i
                clusters[best_i].append((uid, vec, profile))
            new_centroids = []
            for i in range(k):
                if not clusters[i]:
                    new_centroids.append(centroids[i])
                    continue
                mean = [0.0] * len(AXES)
                for _uid, vec, _p in clusters[i]:
                    for j in range(len(mean)):
                        mean[j] += vec[j]
                for j in range(len(mean)):
                    mean[j] /= float(len(clusters[i]))
                new_centroids.append(mean)
            centroids = new_centroids

        result = []
        for i in range(k):
            members = [uid for uid, _vec, _p in clusters[i]]
            members.sort()
            result.append({"center": centroids[i], "members": members})
        return result

    async def _generate_and_store_fragment(self, stream_id: str, *, force: bool = False) -> Optional[Fragment]:
        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            if not force and not self._should_generate_fragment_locked(state, now):
                return None
            state.last_fragment_attempt_ts = now
            snapshot_values = dict(state.values)
            snapshot_traits = [t.label for t in state.traits[-6:]]

        fragment = await self._generate_fragment(snapshot_values=snapshot_values, snapshot_traits=snapshot_traits)
        if not fragment:
            return None

        async with lock:
            state = self._get_group(stream_id)
            max_frag = int(self.get_config("blackbox.max_fragments", 30))
            state.fragments.append(fragment)
            if len(state.fragments) > max_frag:
                state.fragments = state.fragments[-max_frag:]
        return fragment

    async def _generate_fragment(
        self, *, snapshot_values: Dict[str, float], snapshot_traits: List[str]
    ) -> Optional[Fragment]:
        model_task = str(self.get_config("blackbox.model_task", "utils")).strip() or "utils"
        models = llm_api.get_available_models()
        task_config = models.get(model_task) or model_config.model_task_config.utils

        max_tokens = int(self.get_config("blackbox.max_tokens", 256))
        temperature = float(self.get_config("blackbox.temperature", 0.8))
        now = _now_ts()

        prompt = f"""
你是一个AI的潜意识生成器。你不会把内容发送到群里，只把它作为“内心独白片段”存档。
请基于当前群聊意识形态坐标与固化思维，生成一段中文内心独白（60~220字），风格偏“碎片化旁白 + 自我质询 + 黑色幽默”，但不要提到任何作品名或致敬来源。

当前坐标：{json.dumps(snapshot_values, ensure_ascii=False)}
已固化思维：{snapshot_traits}

输出严格 JSON：
{{"text":"...", "tags":["...","..."]}}
"""
        ok, content, _reasoning, model_name = await llm_api.generate_with_model(
            prompt,
            model_config=task_config,
            request_type="mai_soul_engine.fragment",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not ok:
            logger.debug(f"[soul] fragment failed model={model_name}: {content[:80]}")
            return None
        obj = _extract_json_object(content) or {}
        text = str(obj.get("text", "") or "").strip()
        tags = obj.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t)[:32] for t in tags if isinstance(t, (str, int, float))][:12]
        if not text:
            return None
        return Fragment(ts=now, text=text, tags=tags)

    def _advance_slots_locked(self, state: GroupSoulState, now: float) -> None:
        finalized: List[ThoughtSlot] = []
        for slot in state.slots:
            if slot.mode == "ideology_lab":
                slot.ripeness = float(_clamp(slot.ripeness or 0.0, 0.0, 1.0))
                slot.progress = slot.ripeness
                continue
            if slot.mode != "axis_pressure":
                continue
            total = max(1.0, slot.due_ts - slot.created_ts)
            slot.progress = _clamp((now - slot.created_ts) / total, 0.0, 1.0)
            if now >= slot.due_ts:
                finalized.append(slot)

        if finalized:
            for slot in finalized:
                trait_id = f"{slot.axis}:{int(slot.created_ts)}"
                state.traits.append(
                    ThoughtTrait(
                        trait_id=trait_id,
                        axis=slot.axis,
                        direction=slot.direction,
                        created_ts=now,
                        label=slot.label or self._trait_label(slot.axis, slot.direction),
                    )
                )
            state.slots = [s for s in state.slots if s not in finalized]

    async def persist(self) -> None:
        async with self._tick_lock:
            await self._persist_unlocked()

    async def _persist_unlocked(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "groups": {},
                "global": {
                    "base_offset": self._global_base_offset,
                    "traits": [t.__dict__ for t in self._global_traits[-200:]],
                    "daily_day": self._global_daily_day,
                    "daily_used_abs": self._global_daily_used_abs,
                },
            }
            for stream_id, state in self._groups.items():
                lock = self._get_lock(stream_id)
                async with lock:
                    payload["groups"][stream_id] = {
                        "base_tone": state.base_tone,
                        "values": state.values,
                        "ema": state.ema,
                        "recent_deltas": state.recent_deltas,
                        "last_update_ts": state.last_update_ts,
                        "message_count": int(getattr(state, "message_count", 0) or 0),
                        "target": state.target,
                        "initialized": state.initialized,
                        "last_fragment_attempt_ts": state.last_fragment_attempt_ts,
                        "social_last_recompute_ts": state.social_last_recompute_ts,
                        "influences": {
                            uid: {
                                "score": float(getattr(p, "score", 0.0) or 0.0),
                                "weight": p.weight,
                                "multiplier": float(getattr(p, "multiplier", 1.0) or 1.0),
                                "contrib": p.contrib,
                                "quote": p.quote,
                                "display_name": p.display_name,
                                "last_seen_ts": p.last_seen_ts,
                                "n_messages": p.n_messages,
                                "stance_mean": p.stance_mean,
                                "stance_m2": p.stance_m2,
                                "tag_counts": p.tag_counts,
                            }
                            for uid, p in state.influences.items()
                        },
                        "tag_cloud": state.tag_cloud,
                        "quote_bank": state.quote_bank,
                        "prototypes": state.prototypes,
                        "sociolect_profile": state.sociolect_profile,
                        "sociolect_last_update_ts": state.sociolect_last_update_ts,
                        "sociolect_last_attempt_ts": state.sociolect_last_attempt_ts,
                        "slots": [slot.__dict__ for slot in state.slots],
                        "traits": [trait.__dict__ for trait in state.traits],
                        "dissonance": state.dissonance.__dict__,
                        "fragments": [{"ts": f.ts, "text": f.text, "tags": f.tags} for f in state.fragments],
                        "pressure_streaks": state.pressure_streaks,
                        "seed_recent_ts": state.seed_recent_ts,
                        "pending_messages": list(state.pending_messages or [])[-self._pending_max_messages():],
                        "last_dream_processed_ts": float(state.last_dream_processed_ts or 0.0),
                        "dream_trace": list(state.dream_trace or [])[-self._trace_max_items():],
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
            glob = raw.get("global", {})
            if isinstance(glob, dict):
                base_off = glob.get("base_offset")
                if isinstance(base_off, dict):
                    self._global_base_offset = self._coerce_axis_dict(base_off)
                traits = glob.get("traits", [])
                if isinstance(traits, list):
                    self._global_traits = [ThoughtTrait(**t) for t in traits if isinstance(t, dict)]
                self._global_daily_day = str(glob.get("daily_day", "") or "")
                daily_used = glob.get("daily_used_abs")
                if isinstance(daily_used, dict):
                    self._global_daily_used_abs = {a: float(daily_used.get(a, 0.0) or 0.0) for a in AXES}
            groups = raw.get("groups", {})
            if not isinstance(groups, dict):
                return
            for stream_id, obj in groups.items():
                if not isinstance(obj, dict):
                    continue
                state = GroupSoulState()
                state.base_tone = obj.get("base_tone", state.base_tone)
                state.values = obj.get("values", state.values)
                state.ema = obj.get("ema", state.ema)
                state.recent_deltas = obj.get("recent_deltas", state.recent_deltas)
                state.last_update_ts = float(obj.get("last_update_ts", 0.0) or 0.0)
                state.message_count = int(obj.get("message_count", 0) or 0)
                state.target = str(obj.get("target", "") or "")
                state.initialized = bool(obj.get("initialized", False))
                state.last_fragment_attempt_ts = float(obj.get("last_fragment_attempt_ts", 0.0) or 0.0)
                state.social_last_recompute_ts = float(obj.get("social_last_recompute_ts", 0.0) or 0.0)
                infl = obj.get("influences", {})
                if isinstance(infl, dict):
                    for uid, p in infl.items():
                        if not isinstance(p, dict):
                            continue
                        profile = InfluenceProfile()
                        profile.score = float(p.get("score", 0.0) or 0.0)
                        profile.weight = float(p.get("weight", 0.0) or 0.0)
                        profile.multiplier = float(p.get("multiplier", 1.0) or 1.0)
                        if abs(profile.score) < 1e-9 and float(profile.weight or 0.0) > 1e-9:
                            try:
                                m = float(profile.multiplier or 1.0)
                                if m <= 0:
                                    m = 1.0
                                profile.score = float(profile.weight or 0.0) / m
                            except Exception:
                                profile.score = float(profile.weight or 0.0)
                        profile.contrib = p.get("contrib") if isinstance(p.get("contrib"), dict) else {a: 0.0 for a in AXES}
                        profile.quote = str(p.get("quote", "") or "")
                        profile.display_name = str(p.get("display_name", "") or "")
                        profile.last_seen_ts = float(p.get("last_seen_ts", 0.0) or 0.0)
                        profile.n_messages = int(p.get("n_messages", 0) or 0)
                        profile.stance_mean = (
                            p.get("stance_mean") if isinstance(p.get("stance_mean"), dict) else {a: 0.0 for a in AXES}
                        )
                        profile.stance_m2 = (
                            p.get("stance_m2") if isinstance(p.get("stance_m2"), dict) else {a: 0.0 for a in AXES}
                        )
                        profile.tag_counts = p.get("tag_counts") if isinstance(p.get("tag_counts"), dict) else {}
                        state.influences[str(uid)] = profile
                state.tag_cloud = obj.get("tag_cloud") if isinstance(obj.get("tag_cloud"), dict) else {}
                qb = obj.get("quote_bank", [])
                state.quote_bank = qb if isinstance(qb, list) else []
                protos = obj.get("prototypes", [])
                state.prototypes = protos if isinstance(protos, list) else []
                state.sociolect_profile = (
                    obj.get("sociolect_profile") if isinstance(obj.get("sociolect_profile"), dict) else {}
                )
                state.sociolect_last_update_ts = float(obj.get("sociolect_last_update_ts", 0.0) or 0.0)
                state.sociolect_last_attempt_ts = float(obj.get("sociolect_last_attempt_ts", 0.0) or 0.0)
                slots = obj.get("slots", [])
                if isinstance(slots, list):
                    state.slots = [ThoughtSlot(**s) for s in slots if isinstance(s, dict)]
                traits = obj.get("traits", [])
                if isinstance(traits, list):
                    state.traits = [ThoughtTrait(**t) for t in traits if isinstance(t, dict)]
                diss = obj.get("dissonance", {})
                if isinstance(diss, dict):
                    state.dissonance = DissonanceState(**diss)
                frags = obj.get("fragments", [])
                if isinstance(frags, list):
                    state.fragments = [
                        Fragment(
                            ts=float(f.get("ts", 0.0) or 0.0),
                            text=str(f.get("text", "") or ""),
                            tags=[str(t)[:32] for t in (f.get("tags") or []) if isinstance(t, (str, int, float))][:12]
                            if isinstance(f.get("tags"), list)
                            else [],
                        )
                        for f in frags
                        if isinstance(f, dict)
                    ]
                state.pressure_streaks = obj.get("pressure_streaks", state.pressure_streaks)
                sr = obj.get("seed_recent_ts", {})
                state.seed_recent_ts = sr if isinstance(sr, dict) else {}
                pm = obj.get("pending_messages", [])
                state.pending_messages = pm if isinstance(pm, list) else []
                state.last_dream_processed_ts = float(obj.get("last_dream_processed_ts", 0.0) or 0.0)
                dt = obj.get("dream_trace", [])
                state.dream_trace = dt if isinstance(dt, list) else []

                if not state.initialized:
                    if state.last_update_ts or any(abs(v) > 1e-6 for v in state.values.values()):
                        state.initialized = True

                self._groups[str(stream_id)] = state
        except Exception as e:
            logger.debug(f"[soul] load_persisted failed: {e}")

    def setup_router(self) -> APIRouter:
        router = APIRouter(prefix="/api/v1/soul", tags=["soul"])

        @router.options("/{path:path}")
        async def soul_preflight(path: str, request: Request):
            return self._cors_preflight_response(request)

        @router.get("/spectrum")
        async def get_spectrum(
            request: Request,
            scope: str = "local",
            target: Optional[str] = None,
            stream_id: Optional[str] = None,
            format: str = "frontend",
        ):
            self._require_api_token(request)
            scope2 = (scope or "group").strip().lower()
            if scope2 == "local":
                scope2 = "group"
            sid = self._resolve_stream_id(scope=scope2, target=target, stream_id=stream_id)
            if str(format or "raw").strip().lower() in ("frontend", "fe"):
                data = await self._snapshot_spectrum_frontend(scope=scope2, stream_id=sid)
            else:
                data = await self._snapshot_spectrum(scope=scope2, stream_id=sid)
            return JSONResponse(data)

        @router.get("/cabinet")
        async def get_cabinet(
            request: Request,
            scope: str = "local",
            target: Optional[str] = None,
            stream_id: Optional[str] = None,
            format: str = "frontend",
        ):
            self._require_api_token(request)
            scope2 = (scope or "group").strip().lower()
            if scope2 == "local":
                scope2 = "group"
            sid = self._resolve_stream_id(scope=scope2, target=target, stream_id=stream_id)
            if str(format or "raw").strip().lower() in ("frontend", "fe"):
                data = await self._snapshot_cabinet_frontend(scope=scope2, stream_id=sid)
            else:
                data = await self._snapshot_cabinet(scope=scope2, stream_id=sid)
            return JSONResponse(data)

        @router.post("/cabinet/decision")
        async def cabinet_decision(request: Request):
            self._require_api_token(request)
            if not bool(self.get_config("api.allow_mutations", False)):
                raise HTTPException(status_code=403, detail="mutations disabled")
            self._require_mutation_safety()
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="invalid json body")
            scope = str(body.get("scope", "group") or "group")
            target = body.get("target")
            stream_id = body.get("stream_id")
            slot_id = str(body.get("slot_id", "") or "").strip()
            decision = str(body.get("decision", "") or "").strip().lower()
            reason = str(body.get("reason", "") or "").strip()
            if not slot_id:
                raise HTTPException(status_code=400, detail="missing slot_id")
            if decision not in ("approve", "reject"):
                raise HTTPException(status_code=400, detail="decision must be approve/reject")
            sid = self._resolve_stream_id(scope=scope, target=target, stream_id=stream_id)
            if scope == "global":
                raise HTTPException(status_code=400, detail="cabinet decision 暂不支持 global")
            assert sid
            data = await self._cabinet_decision(sid, slot_id=slot_id, decision=decision, reason=reason)
            return JSONResponse(data)

        @router.get("/sculptors")
        async def get_sculptors(
            request: Request,
            scope: str = "local",
            target: Optional[str] = None,
            stream_id: Optional[str] = None,
            limit: int = 5,
            offset: int = 0,
            format: str = "frontend",
        ):
            self._require_api_token(request)
            scope2 = (scope or "group").strip().lower()
            if scope2 == "local":
                scope2 = "group"
            sid = self._resolve_stream_id(scope=scope2, target=target, stream_id=stream_id)
            if scope2 == "global":
                raise HTTPException(status_code=400, detail="sculptors 暂不支持 global 聚合")
            assert sid
            if str(format or "raw").strip().lower() in ("frontend", "fe"):
                data = await self._snapshot_sculptors_frontend(sid, limit=limit, offset=offset)
            else:
                data = await self._snapshot_sculptors(scope=scope2, stream_id=sid, top_k=max(5, int(limit) + int(offset)))
            return JSONResponse(data)

        @router.get("/fragments")
        async def get_fragments(
            request: Request,
            scope: str = "local",
            target: Optional[str] = None,
            stream_id: Optional[str] = None,
            limit: int = 10,
            format: str = "frontend",
        ):
            self._require_api_token(request)
            scope2 = (scope or "group").strip().lower()
            if scope2 == "local":
                scope2 = "group"
            sid = self._resolve_stream_id(scope=scope2, target=target, stream_id=stream_id)
            if scope2 == "global":
                raise HTTPException(status_code=400, detail="fragments 暂不支持 global 聚合")
            assert sid
            if str(format or "raw").strip().lower() in ("frontend", "fe"):
                data = await self._snapshot_fragments_frontend(sid, limit=limit)
            else:
                data = await self._snapshot_fragments(scope=scope2, stream_id=sid, limit=limit)
            return JSONResponse(data)

        @router.get("/pulse")
        async def get_pulse(
            request: Request,
            scope: str = "local",
            target: Optional[str] = None,
            stream_id: Optional[str] = None,
            format: str = "frontend",
        ):
            self._require_api_token(request)
            scope2 = (scope or "group").strip().lower()
            if scope2 == "local":
                scope2 = "group"
            sid = self._resolve_stream_id(scope=scope2, target=target, stream_id=stream_id)
            if str(format or "raw").strip().lower() in ("frontend", "fe"):
                data = await self._snapshot_pulse_frontend(scope=scope2, stream_id=sid)
            else:
                data = await self._snapshot_pulse(scope=scope2, stream_id=sid)
            return JSONResponse(data)

        @router.get("/targets")
        async def get_targets(request: Request, limit: int = 50, offset: int = 0, format: str = "frontend"):
            self._require_api_token(request)
            if str(format or "raw").strip().lower() in ("frontend", "fe"):
                data = await self._snapshot_targets_frontend(limit=limit, offset=offset)
            else:
                data = await self._snapshot_targets(limit=limit, offset=offset)
            return JSONResponse(data)

        @router.get("/social")
        async def get_social(
            request: Request,
            scope: str = "local",
            target: Optional[str] = None,
            stream_id: Optional[str] = None,
            limit: int = 20,
            format: str = "frontend",
        ):
            self._require_api_token(request)
            scope2 = (scope or "group").strip().lower()
            if scope2 == "local":
                scope2 = "group"
            sid = self._resolve_stream_id(scope=scope2, target=target, stream_id=stream_id)
            if scope2 == "global":
                if str(format or "raw").strip().lower() in ("frontend", "fe"):
                    return JSONResponse({"tag_cloud": [], "quote_tail": [], "prototypes": [], "top_users": [], "updated_at": self._to_iso(_now_ts())})
                data = await self._snapshot_social(scope=scope2, stream_id=sid, limit=limit)
                return JSONResponse(data)
            assert sid
            if str(format or "raw").strip().lower() in ("frontend", "fe"):
                data = await self._snapshot_social_frontend(sid, limit=limit)
            else:
                data = await self._snapshot_social(scope=scope2, stream_id=sid, limit=limit)
            return JSONResponse(data)

        @router.get("/sociolect")
        async def get_sociolect(
            request: Request,
            scope: str = "local",
            target: Optional[str] = None,
            stream_id: Optional[str] = None,
            format: str = "frontend",
        ):
            self._require_api_token(request)
            scope2 = (scope or "group").strip().lower()
            if scope2 == "local":
                scope2 = "group"
            sid = self._resolve_stream_id(scope=scope2, target=target, stream_id=stream_id)
            if scope2 == "global":
                if str(format or "raw").strip().lower() in ("frontend", "fe"):
                    return JSONResponse({"rules": [], "taboos": [], "vocabulary_tendencies": [], "ready": False, "updated_at": self._to_iso(_now_ts())})
                data = await self._snapshot_sociolect(scope=scope2, stream_id=sid)
                return JSONResponse(data)
            assert sid
            if str(format or "raw").strip().lower() in ("frontend", "fe"):
                data = await self._snapshot_sociolect_frontend(sid)
            else:
                data = await self._snapshot_sociolect(scope=scope2, stream_id=sid)
            return JSONResponse(data)

        @router.post("/reset")
        async def reset_state(request: Request):
            self._require_api_token(request)
            if not bool(self.get_config("api.allow_mutations", False)):
                raise HTTPException(status_code=403, detail="mutations disabled")
            self._require_mutation_safety()
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="invalid json body")
            scope = str(body.get("scope", "group") or "group")
            target = body.get("target")
            stream_id = body.get("stream_id")
            keep_base_tone = bool(body.get("keep_base_tone", True))
            keep_traits = bool(body.get("keep_traits", False))
            keep_fragments = bool(body.get("keep_fragments", False))

            sid = self._resolve_stream_id(scope=scope, target=target, stream_id=stream_id)
            if scope == "global":
                raise HTTPException(status_code=400, detail="reset 暂不支持 global")
            assert sid
            await self._reset_group_state(
                sid,
                keep_base_tone=keep_base_tone,
                keep_traits=keep_traits,
                keep_fragments=keep_fragments,
            )
            return JSONResponse({"ok": True, "stream_id": sid})

        @router.put("/base_tone")
        async def set_base_tone(request: Request):
            self._require_api_token(request)
            if not bool(self.get_config("api.allow_mutations", False)):
                raise HTTPException(status_code=403, detail="mutations disabled")
            self._require_mutation_safety()
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="invalid json body")
            scope = str(body.get("scope", "group") or "group")
            target = body.get("target")
            stream_id = body.get("stream_id")
            apply_to_values = bool(body.get("apply_to_values", True))
            base_tone = _axis_payload_to_dict(body.get("base_tone"))

            sid = self._resolve_stream_id(scope=scope, target=target, stream_id=stream_id)
            if scope == "global":
                raise HTTPException(status_code=400, detail="base_tone 暂不支持 global")
            assert sid
            await self._set_group_base_tone(sid, base_tone, apply_to_values=apply_to_values)
            return JSONResponse({"ok": True, "stream_id": sid, "base_tone": base_tone, "apply_to_values": apply_to_values})

        self._api_router = router
        return router

    def _resolve_stream_id(self, scope: str, target: Optional[str], stream_id: Optional[str]) -> Optional[str]:
        scope = (scope or "group").strip().lower()
        if scope == "local":
            scope = "group"
        if scope not in ("group", "global"):
            raise HTTPException(status_code=400, detail="scope 必须是 local/group 或 global")
        if scope == "global":
            if not bool(self.get_config("scope.enable_global", True)):
                raise HTTPException(status_code=400, detail="global scope 未启用")
            return None
        if stream_id:
            return stream_id
        if not target:
            # 兼容：前端只传 scope=local/group 时，默认取“最近活跃”的群
            if self._groups:
                best = sorted(self._groups.items(), key=lambda kv: float(kv[1].last_update_ts or 0.0), reverse=True)[0]
                return str(best[0])
            raise HTTPException(status_code=400, detail="group scope 需要 target 或 stream_id")
        try:
            platform, id_str, stream_type = _parse_target(target)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if stream_type != "group":
            raise HTTPException(status_code=400, detail="当前插件默认仅支持群聊 target=*:*:group")
        sid = get_chat_manager().get_stream_id(platform=platform, id=str(id_str), is_group=True)
        if not sid:
            raise HTTPException(status_code=404, detail="无法解析 target 到有效 stream_id（该群可能尚未被 MaiBot 建立会话）")
        return sid

    def _require_api_token(self, request: Optional[Request]) -> None:
        token = str(self.get_config("api.token", "") or "").strip()
        if not token:
            return
        if request is None:
            raise HTTPException(status_code=401, detail="missing request context")
        auth = request.headers.get("authorization", "") or request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            got = auth.replace("Bearer ", "", 1).strip()
            if got == token:
                return
        got2 = request.headers.get("x-soul-token", "") or request.headers.get("X-Soul-Token", "")
        if got2.strip() == token:
            return
        raise HTTPException(status_code=401, detail="invalid token")

    def _require_mutation_safety(self) -> None:
        # 生产环境建议：一旦开启 mutations，必须配置 token，否则相当于“无鉴权写接口”
        if not bool(self.get_config("api.allow_mutations", False)):
            return
        token = str(self.get_config("api.token", "") or "").strip()
        if not token:
            raise HTTPException(status_code=403, detail="mutations require api.token")

    def _cors_preflight_response(self, request: Request) -> Response:
        origin = request.headers.get("origin", "") or request.headers.get("Origin", "")
        allow = self._cors_allow_origin(origin)
        headers = {
            "Access-Control-Allow-Origin": allow,
            "Access-Control-Allow-Methods": "GET,POST,PUT,OPTIONS",
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

    def register_api_routes(self) -> None:
        if self._router_registered:
            return
        server = get_global_server()
        server.register_router(self.setup_router())
        self._router_registered = True

        if not self._cors_registered:
            self._register_cors_middleware(server.get_app())
            self._cors_registered = True

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

    async def _snapshot_spectrum(self, scope: str, stream_id: Optional[str]) -> dict:
        if scope == "global":
            return await self._snapshot_global_spectrum()
        assert stream_id
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            fluct = {a: _stdev(state.recent_deltas[a][-20:]) for a in AXES}
            effective_base = self._effective_base_tone(state)
            return {
                "scope": "group",
                "stream_id": stream_id,
                "target": state.target,
                "values": state.values,
                "ema": state.ema,
                "base_tone": effective_base,
                "base_tone_local": state.base_tone,
                "base_tone_global_offset": dict(self._global_base_offset)
                if self._global_scope_active()
                else {a: 0.0 for a in AXES},
                "fluctuation": fluct,
                "last_update_ts": state.last_update_ts,
            }

    async def _snapshot_global_spectrum(self) -> dict:
        now = _now_ts()
        half_life_min = float(self.get_config("scope.global_half_life_minutes", 60.0))
        tau = max(1.0, half_life_min * 60.0 / math.log(2.0))
        values = {a: 0.0 for a in AXES}
        weights = 0.0
        for sid, st in self._groups.items():
            if not st.last_update_ts:
                continue
            age = max(0.0, now - st.last_update_ts)
            w = math.exp(-age / tau)
            w = max(0.001, w)
            weights += w
            for a in AXES:
                values[a] += st.values[a] * w
        if weights > 0:
            for a in AXES:
                values[a] /= weights
        return {
            "scope": "global",
            "values": values,
            "base_tone_global_offset": dict(self._global_base_offset),
            "traits": [t.__dict__ for t in self._global_traits[-20:]],
            "ts": now,
        }

    async def _snapshot_cabinet(self, scope: str, stream_id: Optional[str]) -> dict:
        if scope == "global":
            return {
                "scope": "global",
                "mode": self._cabinet_mode(),
                "base_tone_global_offset": dict(self._global_base_offset),
                "global_daily_cap": {
                    "day": self._global_daily_day,
                    "cap_abs": float(self.get_config("cabinet.global_daily_shift_cap_abs", 0.0)),
                    "used_abs": dict(self._global_daily_used_abs),
                },
                "global_whitelist_targets": self._global_whitelist(),
                "traits": [t.__dict__ for t in self._global_traits[-50:]],
                "note": "global 仅展示跨群固化 traits 与 base_offset；slots 仍来自具体群聊。",
            }
        assert stream_id
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            now = _now_ts()
            slots = []
            for slot in state.slots:
                obj = dict(slot.__dict__)
                if isinstance(obj.get("fragments"), list):
                    tail = int(self.get_config("cabinet.api_fragments_tail", 12))
                    tail = max(0, min(tail, 60))
                    obj["fragments"] = obj["fragments"][-tail:] if tail else []
                # 调参/可观测：告诉用户“为什么现在还不会品鉴/内化”
                debug: Dict[str, Any] = {}
                debug["fragments_count"] = len(slot.fragments or [])
                debug["runs_done"] = int(slot.evaluator_runs or 0)
                debug["runs_required"] = int(slot.evaluator_required_runs or 0)
                debug["runs_remaining"] = max(0, debug["runs_required"] - debug["runs_done"])
                debug["avg_confidence"] = (
                    float(slot.confidence_sum or 0.0) / float(debug["runs_done"]) if debug["runs_done"] > 0 else 0.0
                )
                debug["ripeness"] = float(slot.ripeness or 0.0)
                debug["energy"] = float(slot.energy or 0.0)
                debug["seed_cooldown_seconds"] = float(self.get_config("cabinet.seed_cooldown_seconds", 600.0))
                debug["seed_min_occurrences"] = int(self.get_config("cabinet.seed_min_occurrences", 4))
                debug["seed_time_half_life_seconds"] = float(self.get_config("cabinet.seed_time_half_life_seconds", 180.0))
                debug["slot_match_min_jaccard"] = float(self.get_config("cabinet.slot_match_min_jaccard", 0.2))

                reasons: List[str] = []
                if slot.mode != "ideology_lab":
                    reasons.append("slot_mode_not_ideology_lab")
                if slot.awaiting_approval:
                    reasons.append("awaiting_manual_approval")
                if slot.rejected:
                    reasons.append("rejected")
                if slot.approved:
                    reasons.append("approved")
                min_frag = int(self.get_config("cabinet.mastication_min_fragments", 6))
                if len(slot.fragments or []) < max(1, min_frag):
                    reasons.append("not_enough_fragments")
                    debug["mastication_min_fragments"] = min_frag
                interval = float(self.get_config("cabinet.mastication_interval_seconds", 30.0))
                if slot.last_mastication_ts and interval > 0:
                    wait = max(0.0, interval - (now - float(slot.last_mastication_ts or 0.0)))
                    debug["next_mastication_in_seconds"] = wait
                    if wait > 0:
                        reasons.append("mastication_interval_not_reached")
                quiet = float(self.get_config("cabinet.quiet_period_seconds", 20.0))
                if quiet > 0 and state.last_update_ts:
                    quiet_wait = max(0.0, quiet - (now - float(state.last_update_ts or 0.0)))
                    debug["quiet_wait_seconds"] = quiet_wait
                    if quiet_wait > 0:
                        reasons.append("not_quiet_yet")
                if debug["runs_done"] >= max(1, debug["runs_required"]):
                    reasons.append("already_matured")

                # global 保护策略可视化
                allowed, why = self._global_assimilation_allowed(state, slot)
                debug["global_scope_active"] = self._global_scope_active()
                debug["global_allowed"] = bool(allowed)
                debug["global_block_reason"] = why
                debug["global_daily_cap_abs"] = float(self.get_config("cabinet.global_daily_shift_cap_abs", 0.0))

                debug["reasons_blocking_progress"] = reasons
                obj["debug"] = debug
                slots.append(obj)
            return {
                "scope": "group",
                "stream_id": stream_id,
                "mode": self._cabinet_mode(),
                "dream_bound": self._dream_bind_cabinet(),
                "slots": slots,
                "traits": [trait.__dict__ for trait in state.traits],
                "dissonance": state.dissonance.__dict__,
            }

    async def _cabinet_decision(self, stream_id: str, *, slot_id: str, decision: str, reason: str) -> dict:
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            slot = next((s for s in state.slots if s.slot_id == slot_id), None)
            if not slot:
                raise HTTPException(status_code=404, detail="slot not found")
            if slot.mode != "ideology_lab":
                raise HTTPException(status_code=400, detail="slot is not ideology_lab mode")

            if decision == "reject":
                slot.rejected = True
                slot.rejected_reason = reason[:200]
                state.slots = [s for s in state.slots if s.slot_id != slot_id]
                return {"ok": True, "stream_id": stream_id, "slot_id": slot_id, "decision": "reject"}

            if not slot.awaiting_approval:
                raise HTTPException(status_code=400, detail="slot is not awaiting approval")
            # 两阶段批准：
            # 1) seed 阶段批准：允许开始“品鉴发酵”（避免未批准就消耗 LLM）
            # 2) ready 阶段批准：允许“正式内化落地”（影响光谱/固化为 trait）
            if slot.status == "awaiting_seed_approval" and int(slot.evaluator_runs or 0) <= 0 and float(slot.ripeness or 0.0) <= 0.0:
                slot.awaiting_approval = False
                slot.status = "fermenting"
                return {"ok": True, "stream_id": stream_id, "slot_id": slot_id, "decision": "approve_seed"}

            self._prepare_slot_finalization_locked(slot)
            self._assimilate_slot_locked(state, slot, now=_now_ts())
            state.slots = [s for s in state.slots if s.slot_id != slot_id]
            return {"ok": True, "stream_id": stream_id, "slot_id": slot_id, "decision": "approve"}

    async def _snapshot_sculptors(self, scope: str, stream_id: Optional[str], *, top_k: int = 5) -> dict:
        if scope == "global":
            raise HTTPException(status_code=400, detail="sculptors 暂不支持 global 聚合")
        assert stream_id
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            top_k = max(1, min(int(top_k), 100))
            top = self._top_sculptors_locked(state, top_k=top_k)
            return {
                "scope": "group",
                "stream_id": stream_id,
                "sculptors": [
                    {
                        "user_id": uid,
                        "weight": p.weight,
                        "contrib": p.contrib,
                        "quote": p.quote,
                        "last_seen_ts": p.last_seen_ts,
                    }
                    for uid, p in top
                ],
            }

    async def _snapshot_fragments(self, scope: str, stream_id: Optional[str], *, limit: int = 10) -> dict:
        if scope == "global":
            raise HTTPException(status_code=400, detail="fragments 暂不支持 global 聚合")
        assert stream_id
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            limit = max(1, min(int(limit), 200))
            return {
                "scope": "group",
                "stream_id": stream_id,
                "fragments": [{"ts": f.ts, "text": f.text, "tags": f.tags} for f in state.fragments[-limit:]],
            }

    def _to_iso(self, ts: float) -> str:
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

    def _fe_axis(self, axis: str) -> str:
        return AXES_FRONTEND.get(axis, axis.replace("_", "-"))

    def _fe_pole_by_value(self, axis: str, value: float) -> str:
        neg, pos = POLES.get(axis, ("NEG", "POS"))
        pole = pos if float(value) >= 0 else neg
        return str(pole).lower()

    def _fe_pole_by_direction(self, axis: str, direction: float) -> str:
        neg, pos = POLES.get(axis, ("NEG", "POS"))
        pole = pos if float(direction) >= 0 else neg
        return str(pole).lower()

    def _dissonance_severity_level(self, severity: float) -> str:
        s = float(_clamp(severity, 0.0, 1.0))
        if s >= 0.75:
            return "high"
        if s >= 0.45:
            return "medium"
        return "low"

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

    async def _snapshot_spectrum_frontend(self, scope: str, stream_id: Optional[str]) -> dict:
        now = _now_ts()
        if scope == "global":
            g = await self._snapshot_global_spectrum()
            values = g.get("values") if isinstance(g.get("values"), dict) else {a: 0.0 for a in AXES}
            base_off = g.get("base_tone_global_offset") if isinstance(g.get("base_tone_global_offset"), dict) else {a: 0.0 for a in AXES}
            traits = g.get("traits") if isinstance(g.get("traits"), list) else []
            dominant_trait = ""
            last_major_shift = ""
            if traits:
                try:
                    last = traits[-1]
                    if isinstance(last, dict):
                        dominant_trait = str(last.get("name") or last.get("label") or "")
                        last_major_shift = self._to_iso(float(last.get("created_ts", 0.0) or 0.0))
                except Exception:
                    pass

            baseline = {a: float(_clamp(float(base_off.get(a, 0.0) or 0.0), -1.0, 1.0)) for a in AXES}
            ranked = sorted([(a, abs(float(baseline.get(a, 0.0)))) for a in AXES], key=lambda x: x[1], reverse=True)
            primary = self._fe_pole_by_value(ranked[0][0], float(baseline.get(ranked[0][0], 0.0))) if ranked else "sincerity"
            secondary = (
                self._fe_pole_by_value(ranked[1][0], float(baseline.get(ranked[1][0], 0.0))) if len(ranked) > 1 and ranked[1][1] > 0 else None
            )

            dims = []
            for axis in AXES:
                dims.append(
                    {
                        "axis": self._fe_axis(axis),
                        "values": {
                            "current": float(_clamp(float(values.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                            "ema": float(_clamp(float(values.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                            "baseline": float(_clamp(float(baseline.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                        },
                        "volatility": 0.0,
                    }
                )
            return {
                "dimensions": dims,
                "base_tone": {"primary": primary, "secondary": secondary, "global_offset": float(sum(abs(v) for v in baseline.values()) / len(AXES)) * 100.0},
                "dominant_trait": dominant_trait,
                "last_major_shift": last_major_shift,
                "updated_at": self._to_iso(now),
            }

        assert stream_id
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            fluct = {a: _stdev(state.recent_deltas[a][-20:]) for a in AXES}
            effective_base = self._effective_base_tone(state)
            traits_snapshot = list(state.traits[-20:])
            if self._global_scope_active():
                traits_snapshot = (list(self._global_traits[-20:]) + traits_snapshot)[-30:]
            dominant_trait = ""
            last_major_shift = ""
            if traits_snapshot:
                last = traits_snapshot[-1]
                dominant_trait = str((last.name or "").strip() or (last.label or "").strip())
                last_major_shift = self._to_iso(float(last.created_ts or 0.0))

            ranked = sorted([(a, abs(float(effective_base.get(a, 0.0) or 0.0))) for a in AXES], key=lambda x: x[1], reverse=True)
            primary = self._fe_pole_by_value(ranked[0][0], float(effective_base.get(ranked[0][0], 0.0))) if ranked else "sincerity"
            secondary = (
                self._fe_pole_by_value(ranked[1][0], float(effective_base.get(ranked[1][0], 0.0))) if len(ranked) > 1 and ranked[1][1] > 0 else None
            )

            dims = []
            for axis in AXES:
                dims.append(
                    {
                        "axis": self._fe_axis(axis),
                        "values": {
                            "current": float(_clamp(float(state.values.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                            "ema": float(_clamp(float(state.ema.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                            "baseline": float(_clamp(float(effective_base.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                        },
                        "volatility": float(fluct.get(axis, 0.0) or 0.0),
                    }
                )

            return {
                "dimensions": dims,
                "base_tone": {"primary": primary, "secondary": secondary},
                "dominant_trait": dominant_trait,
                "last_major_shift": last_major_shift,
                "updated_at": self._to_iso(float(state.last_update_ts or now)),
            }

    def _slot_impacts_frontend(self, slot: ThoughtSlot) -> List[Dict[str, Any]]:
        shift = slot.final_shift if isinstance(slot.final_shift, dict) and any(slot.final_shift.values()) else self._compute_assimilation_shift(slot)
        impacts: List[Dict[str, Any]] = []
        for axis in AXES:
            try:
                s = float(shift.get(axis, 0.0) or 0.0)
            except Exception:
                s = 0.0
            if abs(s) < 1e-6:
                continue
            impacts.append(
                {
                    "dimension": self._fe_axis(axis),
                    "direction": "right" if s > 0 else "left",
                    "strength": float(_clamp(abs(s) * 10.0, 0.0, 10.0)),
                }
            )
        return impacts

    async def _snapshot_cabinet_frontend(self, scope: str, stream_id: Optional[str]) -> dict:
        now = _now_ts()
        if scope == "global":
            interval_min = float(self.get_config("cabinet.rethink_interval_minutes", 720.0))
            interval_min = max(1.0, min(interval_min, 30 * 24 * 60.0))
            global_traits = []
            for t in self._global_traits[-200:]:
                name = str((t.name or "").strip() or (t.label or "").strip())
                desc = (t.definition or "").strip() or (t.digest or "").strip() or (t.verdict or "").strip()
                global_traits.append(
                    {
                        "id": str(t.trait_id),
                        "name": name or "Trait",
                        "source_dimension": self._fe_pole_by_direction(str(t.axis), float(t.direction)),
                        "crystallized_at": self._to_iso(float(t.created_ts or now)),
                        "description": desc[:400],
                        "permanent_effects": [
                            {
                                "dimension": self._fe_axis(axis),
                                "direction": "right" if float(t.shift.get(axis, 0.0) or 0.0) > 0 else "left",
                                "strength": float(_clamp(abs(float(t.shift.get(axis, 0.0) or 0.0)) * 10.0, 0.0, 10.0)),
                            }
                            for axis in AXES
                            if abs(float(t.shift.get(axis, 0.0) or 0.0)) > 1e-6
                        ],
                        "definition": (t.definition or "")[:200],
                        "digest": (t.digest or "")[:900],
                        "last_reflect_at": self._to_iso(float(t.last_reflect_ts or 0.0)) if float(t.last_reflect_ts or 0.0) else None,
                        "reflect_count": int(t.reflect_count or 0),
                    }
                )
            return {
                "slots": [],
                "traits": [],
                "dissonance": {"active": False, "conflicting_thoughts": [], "severity": "low"},
                "global": {
                    "global_traits": global_traits[-50:],
                    "global_offset": {self._fe_axis(a): float(_clamp(float(self._global_base_offset.get(a, 0.0) or 0.0), -1.0, 1.0)) * 100.0 for a in AXES},
                    "daily_limit": float(_clamp(float(self.get_config("cabinet.global_daily_shift_cap_abs", 0.0)), 0.0, 1.0)) * 100.0,
                    "whitelist": self._global_whitelist(),
                    "rethink_interval_minutes": interval_min,
                },
            }

        assert stream_id
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            slots_out = []
            for slot in state.slots:
                name = (slot.seed_topic or "").strip() or (slot.label or "").strip() or (slot.trait_name or "").strip() or slot.slot_id
                status = "internalizing"
                if slot.rejected:
                    status = "conflicting"
                elif slot.approved:
                    status = "crystallized"
                progress = float(_clamp(float(slot.progress or 0.0), 0.0, 1.0)) * 100.0
                introspection = (slot.final_verdict or slot.verdict or "").strip()
                if slot.status == "awaiting_seed_approval" and slot.awaiting_approval and not introspection:
                    introspection = "等待用户批准后才会开始品鉴。"
                if not introspection:
                    introspection = "这种思想正在缓慢渗透..."
                slots_out.append(
                    {
                        "id": str(slot.slot_id),
                        "name": name[:40],
                        "source_dimension": self._fe_pole_by_direction(str(slot.axis if slot.axis in AXES else "sincerity_absurdism"), float(slot.direction)),
                        "status": status,
                        "progress": progress,
                        "conflict_reason": (slot.rejected_reason or "").strip()[:200] if slot.rejected else None,
                        "crystallized_at": self._to_iso(float(slot.created_ts or now)) if status == "crystallized" else None,
                        "introspection": introspection[:600],
                        "impacts": self._slot_impacts_frontend(slot),
                        "debug_params": {
                            "slot": {"mode": slot.mode, "awaiting_approval": bool(slot.awaiting_approval), "runs": int(slot.evaluator_runs or 0)},
                            "seed_topic": slot.seed_topic,
                            "seed_tags": list(slot.seed_tags or [])[:12],
                            "logic_points": list(slot.logic_points or [])[-10:],
                        },
                    }
                )

            traits_out = []
            for t in state.traits[-200:]:
                name = str((t.name or "").strip() or (t.label or "").strip())
                desc = (t.definition or "").strip() or (t.digest or "").strip() or (t.verdict or "").strip() or (t.style_hint or "").strip()
                traits_out.append(
                    {
                        "id": str(t.trait_id),
                        "name": name or "Trait",
                        "source_dimension": self._fe_pole_by_direction(str(t.axis), float(t.direction)),
                        "crystallized_at": self._to_iso(float(t.created_ts or now)),
                        "description": desc[:400],
                        "permanent_effects": [
                            {
                                "dimension": self._fe_axis(axis),
                                "direction": "right" if float(t.shift.get(axis, 0.0) or 0.0) > 0 else "left",
                                "strength": float(_clamp(abs(float(t.shift.get(axis, 0.0) or 0.0)) * 10.0, 0.0, 10.0)),
                            }
                            for axis in AXES
                            if abs(float(t.shift.get(axis, 0.0) or 0.0)) > 1e-6
                        ],
                        "definition": (t.definition or "")[:200],
                        "digest": (t.digest or "")[:900],
                        "last_reflect_at": self._to_iso(float(t.last_reflect_ts or 0.0)) if float(t.last_reflect_ts or 0.0) else None,
                        "reflect_count": int(t.reflect_count or 0),
                        "history": list(t.history or [])[-6:],
                    }
                )

            diss = state.dissonance
            diss_out = {
                "active": bool(diss.active and float(diss.expires_ts or 0.0) > now),
                "conflicting_thoughts": [self._fe_axis(str(diss.axis))] if diss.active else [],
                "severity": self._dissonance_severity_level(float(diss.severity or 0.0)),
                "triggered_at": self._to_iso(float(now)) if diss.active else None,
            }

            for t in traits_out[-50:]:
                try:
                    slots_out.append(
                        {
                            "id": str(t.get("id") or ""),
                            "name": str(t.get("name") or "")[:40],
                            "source_dimension": str(t.get("source_dimension") or "sincerity"),
                            "status": "crystallized",
                            "progress": 100.0,
                            "conflict_reason": None,
                            "crystallized_at": t.get("crystallized_at"),
                            "introspection": str(t.get("description") or "")[:600],
                            "impacts": list(t.get("permanent_effects") or []),
                            "debug_params": None,
                        }
                    )
                except Exception:
                    continue

            return {"slots": slots_out, "traits": traits_out[-50:], "dissonance": diss_out}

    async def _snapshot_sculptors_frontend(self, stream_id: str, *, limit: int, offset: int) -> dict:
        now = _now_ts()
        limit = max(1, min(int(limit), 100))
        offset = max(0, min(int(offset), 100000))
        admins = self.get_config("influence.admin_user_ids", [])
        if not isinstance(admins, list):
            admins = []
        admins = [str(x).strip() for x in admins if str(x).strip()]

        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            items = list(state.influences.items())
            items.sort(key=lambda kv: kv[1].weight, reverse=True)
            picked = items[offset : offset + limit]
            users = []
            for uid, p in picked:
                platform = str(state.target.split(":", 1)[0]) if (state.target or "").count(":") >= 1 else ""
                primary_axis = "sincerity_absurdism"
                primary_val = 0.0
                for axis in AXES:
                    v = abs(float(p.contrib.get(axis, 0.0) or 0.0))
                    if v > primary_val:
                        primary_val = v
                        primary_axis = axis
                pole = self._fe_pole_by_value(primary_axis, float(p.contrib.get(primary_axis, 0.0) or 0.0))
                impact_score = float(_clamp(abs(sum(float(p.contrib.get(a, 0.0) or 0.0) for a in AXES)) * 10.0, 0.0, 10.0))
                users.append(
                    {
                        "id": str(uid),
                        "name": (str(p.display_name or "").strip() or str(uid)),
                        "avatar": None,
                        "influence": float(_clamp(float(getattr(p, "score", 0.0) or 0.0) * 100.0, 0.0, 1e9)),
                        "weight": float(_clamp(float(getattr(p, "multiplier", 1.0) or 1.0), 0.0, 100.0)),
                        "is_admin": (f"{platform}:{uid}" in admins) if platform else False,
                        "primary_impact": pole,
                        "contribution_vector": {self._fe_axis(a): float(p.contrib.get(a, 0.0) or 0.0) * 100.0 for a in AXES},
                        "representative_quotes": (
                            [
                                {
                                    "content": str(p.quote or "")[:120],
                                    "timestamp": self._to_iso(float(p.last_seen_ts or now)),
                                    "impact_score": impact_score,
                                }
                            ]
                            if (p.quote or "").strip()
                            else []
                        ),
                    }
                )
        return {"users": users, "updated_at": self._to_iso(now)}

    async def _snapshot_social_frontend(self, stream_id: str, *, limit: int) -> dict:
        now = _now_ts()
        limit = max(1, min(int(limit), 200))
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            tag_items = sorted(state.tag_cloud.items(), key=lambda kv: kv[1], reverse=True)[:200]
            max_count = max([c for _t, c in tag_items], default=1)
            tag_cloud = [
                {"tag": t, "count": int(c), "weight": float(c) / float(max_count), "trend": "stable"} for t, c in tag_items[: min(50, limit)]
            ]

            quotes = []
            for q in (state.quote_bank or [])[-limit:]:
                if not isinstance(q, dict):
                    continue
                deltas = q.get("deltas", {})
                poles: List[str] = []
                if isinstance(deltas, dict):
                    ranked = sorted([(a, abs(float(deltas.get(a, 0.0) or 0.0)), float(deltas.get(a, 0.0) or 0.0)) for a in AXES], key=lambda x: x[1], reverse=True)
                    for a, strength, v in ranked[:2]:
                        if strength <= 0:
                            continue
                        poles.append(self._fe_pole_by_value(a, v))
                impact_score = float(_clamp(float(q.get("intensity", 0.0) or 0.0) * float(q.get("confidence", 0.0) or 0.0) * 10.0, 0.0, 10.0))
                quotes.append(
                    {
                        "content": str(q.get("quote", "") or "")[:120],
                        "author_id": str(q.get("user_id", "") or ""),
                        "author_name": str(q.get("user_name", "") or q.get("user_id", "") or ""),
                        "timestamp": self._to_iso(float(q.get("ts", now) or now)),
                        "impact_score": impact_score,
                        "dimension_tags": poles,
                    }
                )

            protos = []
            for p in (state.prototypes or [])[:12]:
                if not isinstance(p, dict):
                    continue
                center = p.get("center", {})
                dom: List[str] = []
                if isinstance(center, dict):
                    ranked = sorted([(a, abs(float(center.get(a, 0.0) or 0.0)), float(center.get(a, 0.0) or 0.0)) for a in AXES], key=lambda x: x[1], reverse=True)
                    for a, strength, v in ranked[:2]:
                        if strength <= 0:
                            continue
                        dom.append(self._fe_pole_by_value(a, v))
                protos.append(
                    {
                        "id": str(p.get("prototype_id", "") or ""),
                        "name": str(p.get("label", "") or "Prototype")[:40],
                        "description": str(p.get("label", "") or "")[:80],
                        "dominant_dimensions": dom,
                        "member_count": int(p.get("member_count", 0) or 0),
                        "representative_tags": [str(t) for t, _c in (p.get("top_tags") or [])][:10] if isinstance(p.get("top_tags"), list) else [],
                    }
                )

            top_users = sorted(state.influences.items(), key=lambda kv: kv[1].weight, reverse=True)[:limit]
            users = []
            for uid, p in top_users:
                std = p.stance_std()
                var = {self._fe_axis(a): float(std.get(a, 0.0) or 0.0) ** 2 for a in AXES}
                users.append(
                    {
                        "id": str(uid),
                        "name": (str(p.display_name or "").strip() or str(uid)),
                        "stability": float(_clamp(p.stability(), 0.0, 1.0)) * 100.0,
                        "stance_mean": {self._fe_axis(a): float(p.stance_mean.get(a, 0.0) or 0.0) * 100.0 for a in AXES},
                        "stance_variance": var,
                        "common_tags": [str(t) for t, _c in sorted((p.tag_counts or {}).items(), key=lambda kv: kv[1], reverse=True)[:10]],
                    }
                )

        return {"tag_cloud": tag_cloud, "quote_tail": quotes, "prototypes": protos, "top_users": users, "updated_at": self._to_iso(now)}

    async def _snapshot_sociolect_frontend(self, stream_id: str) -> dict:
        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            profile = state.sociolect_profile if isinstance(state.sociolect_profile, dict) else {}
            rules = profile.get("tone_rules", []) if isinstance(profile, dict) else []
            taboos = profile.get("taboos", []) if isinstance(profile, dict) else []
            lexicon = profile.get("lexicon", []) if isinstance(profile, dict) else []
            if not isinstance(rules, list):
                rules = []
            if not isinstance(taboos, list):
                taboos = []
            if not isinstance(lexicon, list):
                lexicon = []
            ready = bool(profile) and float(state.sociolect_last_update_ts or 0.0) > 0.0
            updated_at = self._to_iso(float(state.sociolect_last_update_ts or now))

        return {
            "rules": [{"pattern": str(r)[:60], "description": str(r)[:120], "frequency": 0.0} for r in rules[:12]],
            "taboos": [{"pattern": str(t)[:60], "reason": str(t)[:120], "severity": "soft"} for t in taboos[:10]],
            "vocabulary_tendencies": [
                {"category": "general", "preferred_terms": [str(x)[:32] for x in lexicon[:20]], "avoided_terms": []}
            ],
            "ready": ready,
            "updated_at": updated_at,
        }

    async def _snapshot_fragments_frontend(self, stream_id: str, *, limit: int) -> dict:
        now = _now_ts()
        limit = max(1, min(int(limit), 200))
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            # 合并：潜意识片段 + Dream 思考日志（用于“黑盒模式”页面）
            bb_items = list(state.fragments or [])[-limit:]
            trace_items = list(state.dream_trace or [])[-limit:]
            interval_minutes = float(self.get_config("blackbox.fragment_interval_minutes", 60.0))
            last_ts = bb_items[-1].ts if bb_items else 0.0
            next_injection = None
            if interval_minutes > 0 and last_ts:
                next_injection = self._to_iso(last_ts + interval_minutes * 60.0)

        def source_from_tags(tags: List[str]) -> str:
            s = {str(t).strip().lower() for t in (tags or []) if str(t).strip()}
            if "breakdown" in s:
                return "breakdown"
            if "rational" in s:
                return "rational"
            if "memory" in s:
                return "memory"
            if "inspiration" in s:
                return "inspiration"
            if "absurd" in s:
                return "absurd"
            if "dream" in s:
                return "dream"
            if "subconscious" in s:
                return "subconscious"
            return "dream" if self._dream_bind_blackbox() else "subconscious"

        out: List[Dict[str, Any]] = []
        for f in bb_items:
            tags = [str(t) for t in (f.tags or [])][:12]
            out.append(
                {
                    "id": f"frag-{int(f.ts)}",
                    "content": str(f.text or ""),
                    "source": source_from_tags(tags),
                    "timestamp": self._to_iso(float(f.ts or now)),
                    "tags": tags,
                    "redacted": False,
                    "check_result": "success",
                }
            )
        for it in trace_items:
            if not isinstance(it, dict):
                continue
            ts = float(it.get("ts", now) or now)
            text = str(it.get("text", "") or "").strip()
            if not text:
                continue
            tags = [str(t) for t in (it.get("tags") or []) if str(t).strip()][:12] if isinstance(it.get("tags"), list) else []
            out.append(
                {
                    "id": f"trace-{int(ts)}",
                    "content": text,
                    "source": source_from_tags(tags),
                    "timestamp": self._to_iso(ts),
                    "tags": tags,
                    "redacted": False,
                    "check_result": "success",
                }
            )
        out.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        out = out[:limit]
        return {
            "fragments": out,
            "next_injection": next_injection,
            "updated_at": self._to_iso(now),
        }

    async def _snapshot_pulse_frontend(self, scope: str, stream_id: Optional[str]) -> dict:
        now = _now_ts()
        if scope == "global":
            p = await self._snapshot_pulse(scope="global", stream_id=None)
            dream = p.get("dream") if isinstance(p.get("dream"), dict) else {}
            return {
                "temperature": 0.5,
                "life_state": "contemplating",
                "status": "global",
                "uptime": self._uptime_str(now),
                "heartbeat": 0,
                "dissonance": {"active": False},
                "dream": {
                    "bound": bool(dream.get("full_bind_effective", False)),
                    "hook_success": bool(dream.get("hook_ok", False)),
                    "last_cycle": {
                        "started_at": self._to_iso(float(dream.get("cycle_start_ts", 0.0) or 0.0))
                        if float(dream.get("cycle_start_ts", 0.0) or 0.0)
                        else None,
                        "ended_at": self._to_iso(float(dream.get("cycle_end_ts", 0.0) or 0.0))
                        if float(dream.get("cycle_end_ts", 0.0) or 0.0)
                        else None,
                        "theme": str(dream.get("cycle_theme", "") or ""),
                        "fragments_generated": int(dream.get("cycle_fragments_generated", 0) or 0),
                    },
                },
                "updated_at": self._to_iso(now),
            }

        assert stream_id
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            values = dict(state.values)
            avg_abs = sum(abs(float(values.get(a, 0.0) or 0.0)) for a in AXES) / float(len(AXES))
            temperature = float(_clamp(0.35 + 0.65 * avg_abs, 0.0, 1.0))

            last_update = float(state.last_update_ts or 0.0)
            recent = (now - last_update) <= 30.0 if last_update else False
            hour = int(time.localtime(now).tm_hour)
            if state.dissonance.active and float(state.dissonance.expires_ts or 0.0) > now:
                life_state = "distressed"
                status = "认知失调"
            elif 0 <= hour <= 6:
                life_state = "dreaming"
                status = "梦中"
            elif any(s for s in state.slots if s.mode == "ideology_lab" and not s.rejected):
                life_state = "contemplating"
                status = "思维发酵"
            elif recent:
                life_state = "active"
                status = "思维活跃中"
            else:
                life_state = "idle"
                status = "静默"

            heartbeat = int(state.message_count or 0)

        return {
            "temperature": temperature,
            "life_state": life_state,
            "status": status,
            "uptime": self._uptime_str(now),
            "heartbeat": heartbeat,
            "dissonance": {"active": bool(state.dissonance.active and float(state.dissonance.expires_ts or 0.0) > now)},
            "dream": {
                "bound": bool(self._dream_full_bind()),
                "hook_success": bool(self._dream_hook_ok),
                "last_cycle": {
                    "started_at": self._to_iso(float(self._dream_cycle_start_ts or 0.0)) if float(self._dream_cycle_start_ts or 0.0) else None,
                    "ended_at": self._to_iso(float(self._dream_cycle_end_ts or 0.0)) if float(self._dream_cycle_end_ts or 0.0) else None,
                    "theme": str(self._dream_cycle_theme or ""),
                    "fragments_generated": int(self._dream_cycle_fragments_generated or 0),
                },
            },
            "updated_at": self._to_iso(now),
        }

    async def _snapshot_targets_frontend(self, *, limit: int, offset: int) -> dict:
        now = _now_ts()
        raw = await self._snapshot_targets(limit=limit, offset=offset)
        targets = raw.get("targets") if isinstance(raw.get("targets"), list) else []
        out = []
        for item in targets:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "stream_id": str(item.get("stream_id", "") or ""),
                    "target": str(item.get("target", "") or ""),
                    "target_type": "group",
                    "last_activity": self._to_iso(float(item.get("last_update_ts", now) or now)),
                    "message_count": int(item.get("message_count", 0) or 0),
                }
            )
        return {"targets": out, "updated_at": self._to_iso(now)}

    async def _snapshot_pulse(self, scope: str, stream_id: Optional[str]) -> dict:
        if scope == "global":
            return {
                "scope": "global",
                "life_state": "global",
                "ts": _now_ts(),
                "dream": {
                    "enabled": self._dream_integration_enabled(),
                    "full_bind_requested": bool(self.get_config("dream.full_bind", False)),
                    "full_bind_effective": self._dream_full_bind(),
                    "hook_ok": self._dream_hook_ok,
                    "reason": self._dream_hook_reason,
                    "bind_cabinet": self._dream_bind_cabinet(),
                    "bind_blackbox": self._dream_bind_blackbox(),
                    "last_cycle_ts": self._dream_last_cycle_ts,
                    "last_phase": self._dream_last_phase,
                    "cycle_start_ts": self._dream_cycle_start_ts,
                    "cycle_end_ts": self._dream_cycle_end_ts,
                    "cycle_theme": self._dream_cycle_theme,
                    "cycle_fragments_generated": self._dream_cycle_fragments_generated,
                },
            }
        assert stream_id
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            now = _now_ts()
            hour = int(time.localtime(now).tm_hour)
            if state.dissonance.active and state.dissonance.expires_ts > now:
                life_state = "dissonant"
            elif 0 <= hour <= 6:
                life_state = "nocturnal"
            elif 7 <= hour <= 11:
                life_state = "morning"
            elif 12 <= hour <= 17:
                life_state = "daytime"
            else:
                life_state = "night"
            return {
                "scope": "group",
                "stream_id": stream_id,
                "life_state": life_state,
                "ts": now,
                "last_update_ts": state.last_update_ts,
                "dissonance": state.dissonance.__dict__,
                "dream": {
                    "enabled": self._dream_integration_enabled(),
                    "full_bind_requested": bool(self.get_config("dream.full_bind", False)),
                    "full_bind_effective": self._dream_full_bind(),
                    "hook_ok": self._dream_hook_ok,
                    "reason": self._dream_hook_reason,
                    "bind_cabinet": self._dream_bind_cabinet(),
                    "bind_blackbox": self._dream_bind_blackbox(),
                    "last_cycle_ts": self._dream_last_cycle_ts,
                    "last_phase": self._dream_last_phase,
                    "cycle_start_ts": self._dream_cycle_start_ts,
                    "cycle_end_ts": self._dream_cycle_end_ts,
                    "cycle_theme": self._dream_cycle_theme,
                    "cycle_fragments_generated": self._dream_cycle_fragments_generated,
                },
            }

    async def _snapshot_targets(self, limit: int = 50, offset: int = 0) -> dict:
        items = []
        for stream_id, st in self._groups.items():
            items.append(
                {
                    "stream_id": stream_id,
                    "target": st.target,
                    "last_update_ts": st.last_update_ts,
                    "message_count": int(getattr(st, "message_count", 0) or 0),
                }
            )
        items.sort(key=lambda x: x.get("last_update_ts", 0.0), reverse=True)
        limit = max(1, min(int(limit), 500))
        offset = max(0, min(int(offset), 100000))
        return {"targets": items[offset : offset + limit]}

    async def _snapshot_social(self, scope: str, stream_id: Optional[str], *, limit: int = 20) -> dict:
        if scope == "global":
            return {
                "scope": "global",
                "note": "global 暂不聚合 social（只聚合 spectrum）",
            }
        assert stream_id
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            tag_items = sorted(state.tag_cloud.items(), key=lambda kv: kv[1], reverse=True)
            top_tags = tag_items[: int(self.get_config("social.max_tag_cloud", 200))]
            limit = max(1, min(int(limit), 200))
            top_users = sorted(state.influences.items(), key=lambda kv: kv[1].weight, reverse=True)[: max(10, min(limit, 50))]
            return {
                "scope": "group",
                "stream_id": stream_id,
                "target": state.target,
                "tag_cloud_top": top_tags[:50],
                "quote_bank_tail": state.quote_bank[-limit:],
                "prototypes": state.prototypes,
                "top_users": [
                    {
                        "user_id": uid,
                        "weight": p.weight,
                        "stability": p.stability(),
                        "stance_mean": p.stance_mean,
                        "stance_std": p.stance_std(),
                        "top_tags": sorted((p.tag_counts or {}).items(), key=lambda kv: kv[1], reverse=True)[:8],
                        "quote": p.quote,
                        "n_messages": p.n_messages,
                    }
                    for uid, p in top_users[:limit]
                ],
            }

    async def _snapshot_sociolect(self, scope: str, stream_id: Optional[str]) -> dict:
        if scope == "global":
            return {
                "scope": "global",
                "note": "global 暂不支持 sociolect（仅支持 group 独立画像）",
            }
        assert stream_id
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            now = _now_ts()
            last = float(state.sociolect_last_update_ts or 0.0)
            return {
                "scope": "group",
                "stream_id": stream_id,
                "target": state.target,
                "enabled": bool(self.get_config("sociolect.enabled", False)),
                "inject_in_reply": bool(self.get_config("sociolect.inject_in_reply", False)),
                "data_ready": {
                    "quote_bank_size": len(state.quote_bank or []),
                    "tag_cloud_size": len(state.tag_cloud or {}),
                    "prototype_count": len(state.prototypes or []),
                },
                "profile": state.sociolect_profile or {},
                "last_update_ts": last,
                "age_seconds": (now - last) if last else None,
                "last_attempt_ts": float(state.sociolect_last_attempt_ts or 0.0),
            }

    async def _reset_group_state(
        self,
        stream_id: str,
        *,
        keep_base_tone: bool = True,
        keep_traits: bool = False,
        keep_fragments: bool = False,
    ) -> None:
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            base = dict(state.base_tone) if keep_base_tone else {a: 0.0 for a in AXES}
            target = state.target
            initialized = state.initialized
            last_fragment_attempt_ts = state.last_fragment_attempt_ts

            state.values = dict(base)
            state.ema = dict(base)
            state.recent_deltas = {a: [] for a in AXES}
            state.last_update_ts = 0.0
            state.influences = {}
            state.tag_cloud = {}
            state.quote_bank = []
            state.prototypes = []
            state.sociolect_profile = {}
            state.sociolect_last_update_ts = 0.0
            state.sociolect_last_attempt_ts = 0.0
            state.slots = []
            state.dissonance = DissonanceState(active=False)
            state.pressure_streaks = {a: 0 for a in AXES}
            if not keep_traits:
                state.traits = []
            if not keep_fragments:
                state.fragments = []

            state.base_tone = base
            state.target = target
            state.initialized = initialized
            state.last_fragment_attempt_ts = last_fragment_attempt_ts

    async def _set_group_base_tone(self, stream_id: str, base_tone: Dict[str, float], *, apply_to_values: bool) -> None:
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            state.base_tone = dict(base_tone)
            state.initialized = True
            if apply_to_values:
                state.values = dict(base_tone)
                state.ema = dict(base_tone)
                state.recent_deltas = {a: [] for a in AXES}


class SoulBackgroundTask(AsyncTask):
    def __init__(self, engine: SoulEngine):
        super().__init__(task_name="MaiSoulEngineBackground", wait_before_start=1, run_interval=5)
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
    handler_description = "Mai-Soul-Engine 启动钩子：注册 API 路由与后台任务"
    intercept_message = True
    weight = 100

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]:
        try:
            plugin_dir = Path(self.get_config("runtime.plugin_dir", "") or "")
            if not plugin_dir:
                plugin_dir = Path(__file__).parent
            engine = get_engine(plugin_dir=plugin_dir)
            engine.set_config(self.plugin_config or {})
            if engine.get_config("persistence.enabled", True):
                engine.load_persisted()
            engine.try_hook_dream()
            engine.register_api_routes()
            await async_task_manager.add_task(SoulBackgroundTask(engine))
            return True, True, None, None, None
        except Exception as e:
            logger.error(f"[soul] on_start failed: {e}", exc_info=True)
            return False, True, None, None, None


class SoulOnMessageEventHandler(BaseEventHandler):
    event_type = EventType.ON_MESSAGE
    handler_name = "mai_soul_engine_on_message"
    handler_description = "Mai-Soul-Engine 监听群聊消息并进行语义化位移学习"
    intercept_message = False
    weight = -10

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]:
        if not message or not message.is_group_message:
            return True, True, None, None, None
        try:
            plugin_dir = Path(self.get_config("runtime.plugin_dir", "") or "")
            if not plugin_dir:
                plugin_dir = Path(__file__).parent
            engine = get_engine(plugin_dir=plugin_dir)
            engine.set_config(self.plugin_config or {})
            if not engine.is_enabled():
                return True, True, None, None, None

            text = (message.plain_text or "").strip()
            if not text:
                return True, True, None, None, None
            if len(text) > int(engine.get_config("performance.max_message_chars", 800)):
                text = text[: int(engine.get_config("performance.max_message_chars", 800))]

            platform = str(message.message_base_info.get("platform", "") or "")
            group_id = str(message.message_base_info.get("group_id", "") or "")
            user_id = str(message.message_base_info.get("user_id", "") or "")
            user_nickname = str(message.message_base_info.get("user_nickname", "") or "")
            user_cardname = str(message.message_base_info.get("user_cardname", "") or "")
            stream_id = str(message.stream_id or "")
            if not (platform and group_id and user_id and stream_id):
                return True, True, None, None, None

            await engine.on_message(
                SoulEvent(
                    stream_id=stream_id,
                    platform=platform,
                    group_id=group_id,
                    user_id=user_id,
                    text=text,
                    ts=_now_ts(),
                    user_nickname=user_nickname,
                    user_cardname=user_cardname,
                )
            )
            return True, True, None, None, None
        except Exception as e:
            logger.debug(f"[soul] on_message failed: {e}")
            return True, True, None, None, None


class SoulPostLlmEventHandler(BaseEventHandler):
    event_type = EventType.POST_LLM
    handler_name = "mai_soul_engine_post_llm"
    handler_description = "Mai-Soul-Engine 在回复前注入意识形态上下文"
    intercept_message = True
    weight = 50

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]:
        if not message or not message.llm_prompt or not message.stream_id:
            return True, True, None, None, None
        try:
            plugin_dir = Path(self.get_config("runtime.plugin_dir", "") or "")
            if not plugin_dir:
                plugin_dir = Path(__file__).parent
            engine = get_engine(plugin_dir=plugin_dir)
            engine.set_config(self.plugin_config or {})
            if not engine.is_enabled():
                return True, True, None, None, None
            if not bool(engine.get_config("injection.enabled", True)):
                return True, True, None, None, None

            trigger_text = str(getattr(message, "plain_text", "") or "")
            trigger_user_id = str(message.message_base_info.get("user_id", "") or "") if isinstance(message.message_base_info, dict) else ""
            block = await engine.build_injection_block(
                message.stream_id, trigger_text=trigger_text, trigger_user_id=trigger_user_id
            )
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
    command_description = "Mai-Soul-Engine 调试命令（用于快速验证光谱/思维阁/黑盒/Dream 绑定）"
    command_pattern = (
        r"^[/／]soul(?:\s+debug)?(?:\s+(?P<sub>help|status|simulate|dream|masticate|fragment))?(?:\s+(?P<arg>.+))?$"
    )

    def _is_allowed(self) -> tuple[bool, str]:
        if not bool(self.get_config("debug.enabled", False)):
            return False, "debug_disabled"

        allow_private = bool(self.get_config("debug.allow_in_private", False))
        group_info = getattr(self.message.message_info, "group_info", None)
        if not group_info and not allow_private:
            return False, "private_not_allowed"

        if not bool(self.get_config("debug.admin_only", True)):
            return True, "ok"

        platform = str(getattr(self.message.message_info, "platform", "") or "")
        user_info = getattr(self.message.message_info, "user_info", None)
        user_id = str(getattr(user_info, "user_id", "") or "")
        needle_full = f"{platform}:{user_id}" if platform and user_id else ""
        needle_uid = user_id

        allowed = self.get_config("debug.admin_user_ids", [])
        if not isinstance(allowed, list):
            allowed = []
        allowed_set: set[str] = set()
        for x in allowed:
            s = str(x).strip()
            if not s:
                continue
            allowed_set.add(s)
            if ":" in s:
                allowed_set.add(s.split(":")[-1].strip())
        if needle_uid and needle_uid in allowed_set:
            return True, "ok"
        if needle_full and needle_full in allowed_set:
            return True, "ok"

        # 兼容：如果用户已经在 influence.admin_user_ids，也视为有权限
        influence_admins = self.get_config("influence.admin_user_ids", [])
        if isinstance(influence_admins, list):
            admin_set: set[str] = set()
            for x in influence_admins:
                s = str(x).strip()
                if not s:
                    continue
                admin_set.add(s)
                if ":" in s:
                    admin_set.add(s.split(":")[-1].strip())
            if needle_uid and needle_uid in admin_set:
                return True, "ok"
            if needle_full and needle_full in admin_set:
                return True, "ok"

        return False, "not_admin"

    def _get_engine(self) -> SoulEngine:
        plugin_dir = Path(self.get_config("runtime.plugin_dir", "") or "")
        if not plugin_dir:
            plugin_dir = Path(__file__).parent
        engine = get_engine(plugin_dir=plugin_dir)
        engine.set_config(self.plugin_config or {})
        return engine

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        ok, reason = self._is_allowed()
        if not ok:
            if reason == "debug_disabled":
                return True, None, 0
            if reason == "private_not_allowed":
                await self.send_text("调试命令默认只允许在群聊使用（可在 [debug].allow_in_private 开启私聊）。")
                return True, "private_not_allowed", 1
            platform = str(getattr(self.message.message_info, "platform", "") or "")
            user_info = getattr(self.message.message_info, "user_info", None)
            user_id = str(getattr(user_info, "user_id", "") or "")
            hint = f"{platform}:{user_id}" if platform and user_id else ""
            if hint:
                await self.send_text(
                    "\n".join(
                        [
                            "你没有权限使用 Soul 调试命令。",
                            f"把你的账号加入配置：`[debug].admin_user_ids = [\\\"{hint}\\\"]`（注意要带引号）",
                            "或者只填 user_id：`[debug].admin_user_ids = [\"你的user_id\"]`",
                        ]
                    )
                )
            else:
                await self.send_text("你没有权限使用 Soul 调试命令（可在 [debug].admin_user_ids 放行）。")
            return True, "not_allowed", 1

        sub = str(self.matched_groups.get("sub") or "").strip().lower()
        arg = str(self.matched_groups.get("arg") or "").strip()
        if not sub:
            sub = "help"

        engine = self._get_engine()
        if not engine.is_enabled():
            await self.send_text("Mai-Soul-Engine 当前未启用（[plugin].enabled=false）。")
            return True, "engine_disabled", 1

        stream_id = getattr(getattr(self.message, "chat_stream", None), "stream_id", None)
        if not stream_id:
            await self.send_text("无法获取 stream_id（这通常意味着消息还没绑定 chat_stream）。")
            return True, "no_stream", 1

        if sub == "help":
            await self.send_text(
                "\n".join(
                    [
                        "Mai-Soul-Engine 调试命令：",
                        "- /soul status                 查看当前群状态",
                        "- /soul simulate <一句话>      立刻跑一轮“语义位移分析”并显示变化",
                        "- /soul masticate              强制推进一次思维阁“品鉴”",
                        "- /soul fragment               强制生成一条黑盒“潜意识片段”",
                        "- /soul dream                  模拟一次 Dream 周期（触发品鉴/黑盒）",
                        "",
                        "提示：需要先在插件 config.toml 里开启 [debug].enabled=true。",
                    ]
                )
            )
            return True, "help", 1

        if sub == "status":
            pulse = await engine._snapshot_pulse(scope="group", stream_id=str(stream_id))
            pulse_global = await engine._snapshot_pulse(scope="global", stream_id=None)
            spec = await engine._snapshot_spectrum(scope="group", stream_id=str(stream_id))
            cab = await engine._snapshot_cabinet(scope="group", stream_id=str(stream_id))
            values = spec.get("values") if isinstance(spec.get("values"), dict) else {}
            axes_line = " ".join([f"{a.split('_')[0]}={float(values.get(a, 0.0)):+.2f}" for a in AXES])
            slots = cab.get("slots") if isinstance(cab.get("slots"), list) else []
            traits = cab.get("traits") if isinstance(cab.get("traits"), list) else []
            dream = pulse_global.get("dream") if isinstance(pulse_global.get("dream"), dict) else {}
            await self.send_text(
                "\n".join(
                    [
                        f"target: {spec.get('target')}",
                        f"values: {axes_line}",
                        f"traits: {len(traits)}  slots: {len(slots)}",
                        f"life_state: {pulse.get('life_state')}",
                        f"dream: hook_ok={dream.get('hook_ok')} full_bind={dream.get('full_bind_effective')} last={dream.get('last_cycle_ts')}",
                    ]
                )
            )
            return True, "status", 1

        if sub == "simulate":
            text = arg
            if not text:
                await self.send_text("用法：/soul simulate <一句话>")
                return True, "missing_text", 1
            max_chars = int(self.get_config("debug.max_text_chars", 300))
            max_chars = max(20, min(max_chars, 2000))
            if len(text) > max_chars:
                text = text[:max_chars]

            mi = self.message.message_info
            platform = str(getattr(mi, "platform", "") or "")
            group_info = getattr(mi, "group_info", None)
            user_info = getattr(mi, "user_info", None)
            group_id = str(getattr(group_info, "group_id", "") or "")
            user_id = str(getattr(user_info, "user_id", "") or "")
            if not group_id:
                await self.send_text("simulate 目前只支持群聊（需要 group_id）。")
                return True, "no_group", 1

            before = await engine._snapshot_spectrum(scope="group", stream_id=str(stream_id))
            ev = SoulEvent(
                stream_id=str(stream_id),
                platform=platform,
                group_id=group_id,
                user_id=user_id,
                text=text,
                ts=_now_ts(),
            )
            result = None
            if engine._spectrum_update_mode() == "dream_window":
                await engine.on_message(ev)
                await engine._process_spectrum_window_for_group(str(stream_id))
            else:
                result = await engine._process_event(ev)
            after = await engine._snapshot_spectrum(scope="group", stream_id=str(stream_id))

            bvals = before.get("values") if isinstance(before.get("values"), dict) else {}
            avals = after.get("values") if isinstance(after.get("values"), dict) else {}
            dline = " ".join([f"{a.split('_')[0]}={float(avals.get(a, 0.0)) - float(bvals.get(a, 0.0)):+.2f}" for a in AXES])

            tags = []
            if isinstance(result, dict):
                rt = result.get("tags", [])
                if isinstance(rt, list):
                    tags = [str(x) for x in rt if str(x).strip()][:6]
            await self.send_text(
                "\n".join(
                    [
                        f"Δvalues: {dline}",
                        f"tags: {', '.join(tags) if tags else '(none)'}",
                        "（已按正常流程写入光谱/影响力/话题采样）",
                    ]
                )
            )
            return True, "simulate", 1

        if sub == "masticate":
            await engine._mastication_once(str(stream_id), force=True)
            cab = await engine._snapshot_cabinet(scope="group", stream_id=str(stream_id))
            slots = cab.get("slots") if isinstance(cab.get("slots"), list) else []
            if not slots:
                await self.send_text("当前没有可品鉴的槽位（可能还没触发种子，或 fragments 不足）。")
                return True, "no_slots", 1
            # 找一个最新 slot 做展示
            slots_sorted = sorted(slots, key=lambda s: float((s or {}).get("created_ts", 0.0)))
            s0 = slots_sorted[-1] if slots_sorted else {}
            await self.send_text(
                f"slot={s0.get('slot_id')} status={s0.get('status')} runs={s0.get('evaluator_runs')}/{s0.get('evaluator_required_runs')} ripeness={float(s0.get('ripeness', 0.0) or 0.0):.2f}"
            )
            return True, "masticate", 1

        if sub == "fragment":
            frag = await engine._generate_and_store_fragment(str(stream_id), force=True)
            if not frag:
                await self.send_text("黑盒片段生成失败（可能是 LLM 失败或被限流/配置关闭）。")
                return True, "fragment_failed", 1
            preview = (frag.text or "").strip().replace("\n", " ")
            preview = re.sub(r"\s+", " ", preview)
            if len(preview) > 120:
                preview = preview[:120] + "..."
            await self.send_text(f"fragment: {preview}\ntags: {', '.join(frag.tags or [])}")
            return True, "fragment", 1

        if sub == "dream":
            await engine._on_dream_cycle(phase="start")
            await engine._on_dream_cycle(phase="end")
            pulse = await engine._snapshot_pulse(scope="global", stream_id=None)
            dream = pulse.get("dream") if isinstance(pulse.get("dream"), dict) else {}
            await self.send_text(
                f"模拟 Dream 完成：hook_ok={dream.get('hook_ok')} full_bind={dream.get('full_bind_effective')} last_cycle_ts={dream.get('last_cycle_ts')}"
            )
            return True, "dream", 1

        await self.send_text("未知子命令。用 /soul help 查看可用调试命令。")
        return True, "unknown_sub", 1


@register_plugin
class MaiSoulEnginePlugin(BasePlugin):
    plugin_name = "mai_soul_engine"
    enable_plugin = True
    dependencies: List[str] = []
    python_dependencies: List[Any] = []
    config_file_name = "config.toml"

    config_section_descriptions = {
        "plugin": section_meta("插件开关", icon="settings", order=0),
        "scope": section_meta("作用域", icon="layers", order=1),
        "api": section_meta("API", icon="cloud", order=2),
        "llm": section_meta("语义评分模型", icon="cpu", order=3),
        "influence": section_meta("社会化权重", icon="users", order=4),
        "social": section_meta("社会化学习", icon="users", order=4),
        "sociolect": section_meta("群体语言画像", icon="type", order=5),
        "spectrum": section_meta("光谱动力学", icon="activity", order=5),
        "cabinet": section_meta("思维阁", icon="archive", order=6),
        "blackbox": section_meta("黑盒模式", icon="moon", order=7),
        "injection": section_meta("回复注入", icon="message-square", order=8),
        "dream": section_meta("Dream 联动", icon="moon", order=8),
        "persistence": section_meta("持久化", icon="save", order=9),
        "performance": section_meta("性能与限流", icon="zap", order=10),
        "debug": section_meta("调试", icon="bug", order=99),
    }

    config_layout = ConfigLayout(
        tabs=[
            ConfigTab(id="general", title="General", icon="settings", order=0, sections=["plugin", "scope", "api"]),
            ConfigTab(
                id="engine",
                title="Engine",
                icon="activity",
                order=1,
                sections=["llm", "influence", "social", "sociolect", "spectrum"],
            ),
            ConfigTab(id="mind", title="Mind", icon="archive", order=2, sections=["cabinet", "blackbox", "injection", "dream"]),
            ConfigTab(id="ops", title="Ops", icon="zap", order=3, sections=["persistence", "performance", "debug"]),
        ]
    )

    config_schema = {
        "plugin": {
            "config_version": ConfigField(type=str, default="0.1.0", description="配置版本号（用于自动迁移）", order=0),
            "enabled": ConfigField(type=bool, default=True, description="是否启用 Mai-Soul-Engine", order=1),
        },
        "scope": {
            "default_scope": ConfigField(
                type=str, default="group", description="默认作用域：group(群独立) / global(全局)", choices=["group", "global"], order=0
            ),
            "enable_global": ConfigField(type=bool, default=True, description="是否启用 global 聚合态", order=1),
            "global_half_life_minutes": ConfigField(
                type=float,
                default=60.0,
                description="global 聚合时的时间半衰期（分钟，越小越偏向最近活跃群）",
                min=1.0,
                max=24 * 60.0,
                order=2,
            ),
        },
        "api": {
            "token": ConfigField(
                type=str,
                default="",
                description="API Token（留空则不鉴权；建议设置）",
                input_type="password",
                order=0,
            ),
            "allow_mutations": ConfigField(
                type=bool,
                default=False,
                description="是否允许 API 变更状态（/reset、/base_tone）；建议仅本机使用并设置 token",
                order=1,
            ),
            "cors_allow_origins": ConfigField(
                type=list,
                default=["http://localhost:5173", "http://127.0.0.1:5173"],
                description="允许跨域的 Origin 列表（设置为 ['*'] 表示全部允许）",
                item_type="string",
                order=2,
            ),
        },
        "llm": {
            "model_task": ConfigField(
                type=str,
                default="utils",
                description="语义化位移分析使用的模型任务名（见 model_task_config），建议 utils",
                order=0,
            ),
            "temperature": ConfigField(type=float, default=0.2, description="语义评分温度", min=0.0, max=2.0, order=1),
            "max_tokens": ConfigField(type=int, default=256, description="语义评分最大 tokens", min=64, max=2048, order=2),
        },
        "influence": {
            "base_user_weight": ConfigField(type=float, default=1.0, description="普通用户基础权重", min=0.0, max=10.0, order=0),
            "admin_user_ids": ConfigField(
                type=list,
                default=[],
                description="管理员用户列表（platform:user_id），用于更强惯性推动",
                item_type="string",
                order=1,
            ),
            "admin_weight_multiplier": ConfigField(type=float, default=1.8, description="管理员权重倍率", min=1.0, max=10.0, order=2),
            "weight_decay": ConfigField(type=float, default=0.995, description="影响力权重衰减（越小衰减越快）", min=0.9, max=0.9999, order=3),
            "activity_scale": ConfigField(
                type=float,
                default=10.0,
                description="用户发言量对影响力的放大尺度（越大放大越弱，log1p(n)/scale）",
                min=0.0,
                max=100.0,
                order=4,
            ),
            "stability_base": ConfigField(
                type=float,
                default=0.5,
                description="用户立场稳定度影响的基础项（0~2）",
                min=0.0,
                max=2.0,
                order=5,
            ),
            "stability_strength": ConfigField(
                type=float,
                default=0.5,
                description="用户立场稳定度影响的强度项（0~2）",
                min=0.0,
                max=2.0,
                order=6,
            ),
        },
        "social": {
            "enable_prototypes": ConfigField(type=bool, default=True, description="是否启用群内原型(派系)聚类", order=0),
            "prototype_k": ConfigField(type=int, default=3, description="派系数量 K（2~6）", min=2, max=6, order=1),
            "min_messages_per_user": ConfigField(type=int, default=5, description="参与聚类的用户最少消息数", min=1, max=200, order=2),
            "prototype_recompute_interval_seconds": ConfigField(
                type=float,
                default=120.0,
                description="派系聚类重算间隔（秒）",
                min=10.0,
                max=3600.0,
                order=3,
            ),
            "max_quote_bank": ConfigField(type=int, default=200, description="群语录库最大条数", min=20, max=5000, order=4),
            "max_tag_cloud": ConfigField(type=int, default=200, description="群标签云最大标签数", min=20, max=5000, order=5),
        },
        "sociolect": {
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用“群体语言画像”生成（把群体语言习惯反向喂给回复）",
                order=0,
            ),
            "inject_in_reply": ConfigField(
                type=bool,
                default=False,
                description="是否把语言画像注入到回复提示词（建议先开 API 观察效果，再开启）",
                order=1,
            ),
            "recompute_interval_minutes": ConfigField(
                type=float,
                default=30.0,
                description="语言画像重算间隔（分钟）",
                min=5.0,
                max=24 * 60.0,
                order=2,
            ),
            "failure_backoff_seconds": ConfigField(
                type=float,
                default=600.0,
                description="生成失败后的重试退避（秒）",
                min=0.0,
                max=24 * 3600.0,
                order=3,
            ),
            "min_quote_samples": ConfigField(
                type=int,
                default=30,
                description="生成画像所需的最少语录样本数（来自 quote_bank）",
                min=5,
                max=5000,
                order=4,
            ),
            "min_unique_users": ConfigField(
                type=int,
                default=5,
                description="生成画像所需的最少参与用户数（按 quote_bank 统计）",
                min=1,
                max=500,
                order=5,
            ),
            "quote_sample_size": ConfigField(
                type=int,
                default=40,
                description="每次生成喂给 LLM 的语录样本数（越大越准但更耗 tokens）",
                min=10,
                max=200,
                order=6,
            ),
            "tag_sample_size": ConfigField(
                type=int,
                default=50,
                description="每次生成喂给 LLM 的 top 标签数",
                min=10,
                max=500,
                order=7,
            ),
            "model_task": ConfigField(type=str, default="utils", description="语言画像生成使用的模型任务名", order=8),
            "temperature": ConfigField(
                type=float,
                default=0.3,
                description="语言画像生成温度（越低越稳定）",
                min=0.0,
                max=2.0,
                order=9,
            ),
            "max_tokens": ConfigField(type=int, default=512, description="语言画像生成最大 tokens", min=128, max=4096, order=10),
            "max_age_minutes": ConfigField(
                type=float,
                default=360.0,
                description="注入时允许的画像最大陈旧时间（分钟）",
                min=5.0,
                max=24 * 60.0,
                order=11,
            ),
            "max_rules": ConfigField(type=int, default=6, description="tone_rules 最大条数", min=0, max=20, order=12),
            "max_taboos": ConfigField(type=int, default=3, description="taboos 最大条数", min=0, max=20, order=13),
            "max_lexicon": ConfigField(type=int, default=10, description="lexicon 最大条数", min=0, max=50, order=14),
            "max_examples": ConfigField(type=int, default=2, description="example_responses 最大条数", min=0, max=10, order=15),
            "max_injection_chars": ConfigField(
                type=int,
                default=500,
                description="注入块最大字符数（防止 prompt 变得太长）",
                min=100,
                max=5000,
                order=16,
            ),
        },
        "spectrum": {
            "update_mode": ConfigField(
                type=str,
                default="dream_window",
                description="光谱更新模式：dream_window(从上次Dream到本次Dream批处理) / per_message(逐条消息)",
                choices=["dream_window", "per_message"],
                order=-1,
            ),
            "dream_window_max_messages": ConfigField(
                type=int,
                default=120,
                description="dream_window：每群最多缓存多少条消息用于下一次 Dream 批处理",
                min=20,
                max=800,
                order=-1,
            ),
            "dream_window_max_chars_per_message": ConfigField(
                type=int,
                default=160,
                description="dream_window：单条消息进入窗口时的最大截断长度（字符）",
                min=40,
                max=800,
                order=-1,
            ),
            "window_strength_scale": ConfigField(
                type=float,
                default=12.0,
                description="dream_window：窗口规模强度缩放（越大越保守，越小越容易产生明显位移）",
                min=1.0,
                max=200.0,
                order=-1,
            ),
            "regression_rate": ConfigField(type=float, default=0.01, description="回归阻力（向 base_tone 回拉速率）", min=0.0, max=0.2, order=0),
            "ema_alpha": ConfigField(type=float, default=0.15, description="EMA 平滑系数（用于 Cabinet 触发）", min=0.01, max=0.8, order=1),
            "clamp_abs": ConfigField(type=float, default=1.0, description="坐标绝对值上限（[-x, x]）", min=0.2, max=5.0, order=2),
            "fluctuation_window": ConfigField(type=int, default=30, description="波动率窗口（保留最近 delta 数）", min=10, max=200, order=3),
            "initialize_from_base_tone": ConfigField(
                type=bool,
                default=True,
                description="首次见到一个群时，是否用 base_tone 初始化 values/ema（否则从 0 开始）",
                order=4,
            ),
            "base_tone_default_json": ConfigField(
                type=str,
                default="{}",
                description="Base Tone 默认值 JSON（四轴 -1~1），例：{\"sincerity_absurdism\":0.2}",
                input_type="code",
                rows=4,
                order=5,
            ),
            "base_tone_overrides_json": ConfigField(
                type=str,
                default="{}",
                description="按 target 覆盖 Base Tone 的 JSON：{\"qq:123:group\": {\"heroism_nihilism\": -0.3}}",
                input_type="code",
                rows=6,
                order=6,
            ),
        },
        "cabinet": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用 Thought Cabinet", order=0),
            "mode": ConfigField(
                type=str,
                default="ideology_lab",
                description="思维阁模式：ideology_lab(意识形态实验室) / axis_pressure(旧版轴向压力)",
                choices=["ideology_lab", "axis_pressure"],
                order=1,
            ),
            "max_slots": ConfigField(type=int, default=3, description="最大内化槽位数", min=1, max=10, order=2),
            "auto_assimilate": ConfigField(
                type=bool,
                default=True,
                description="ideology_lab：成熟后是否自动内化（关闭则需 /cabinet/decision 手动 approve）",
                order=3,
            ),
            "require_manual_approval": ConfigField(
                type=bool,
                default=False,
                description="ideology_lab：强制手动批准（即使 auto_assimilate=true 也会进入 awaiting_approval）",
                order=4,
            ),
            "seed_require_manual_approval": ConfigField(
                type=bool,
                default=False,
                description="ideology_lab：是否在种子出现后先等待批准，再开始品鉴（避免未批准就消耗 LLM）",
                order=4,
            ),
            "quiet_period_seconds": ConfigField(
                type=float,
                default=20.0,
                description="ideology_lab：静默时刻阈值（秒），最近多少秒无新消息才进行后台品鉴",
                min=0.0,
                max=3600.0,
                order=5,
            ),
            "context_window_messages": ConfigField(
                type=int,
                default=60,
                description="ideology_lab：采样窗口消息数（用于热度/集中度判断与切片候选）",
                min=10,
                max=500,
                order=6,
            ),
            "seed_energy_threshold": ConfigField(
                type=float,
                default=0.25,
                description="ideology_lab：话题种子触发阈值（集中度×能量×log(频次)）",
                min=0.01,
                max=2.0,
                order=8,
            ),
            "seed_min_message_energy": ConfigField(
                type=float,
                default=0.15,
                description="ideology_lab：单条消息最低能量（intensity×confidence）才参与触发",
                min=0.0,
                max=1.0,
                order=7,
            ),
            "seed_time_half_life_seconds": ConfigField(
                type=float,
                default=180.0,
                description="ideology_lab：种子热度计算的时间半衰期（秒，越小越偏向最近发言）",
                min=10.0,
                max=24 * 3600.0,
                order=8,
            ),
            "seed_min_occurrences": ConfigField(
                type=int,
                default=4,
                description="ideology_lab：触发种子所需的最少出现次数（在采样窗口内）",
                min=1,
                max=50,
                order=9,
            ),
            "seed_cooldown_seconds": ConfigField(
                type=float,
                default=600.0,
                description="ideology_lab：同一话题种子触发冷却时间（秒）",
                min=0.0,
                max=24 * 3600.0,
                order=10,
            ),
            "seed_recent_ttl_seconds": ConfigField(
                type=float,
                default=24 * 3600.0,
                description="ideology_lab：seed_recent_ts 清理 TTL（秒，防止内部字典长期增长）",
                min=0.0,
                max=30 * 24 * 3600.0,
                order=11,
            ),
            "seed_max_tags": ConfigField(type=int, default=5, description="ideology_lab：每个种子最多保留的关联标签数", min=1, max=12, order=9),
            "seed_fragment_count": ConfigField(
                type=int,
                default=12,
                description="ideology_lab：创建种子时截取的上下文切片数",
                min=4,
                max=80,
                order=10,
            ),
            "max_fragments_per_slot": ConfigField(
                type=int,
                default=40,
                description="ideology_lab：单个槽位最多保留的切片数（过大可能导致 state.json 变大）",
                min=10,
                max=200,
                order=11,
            ),
            "slot_match_min_jaccard": ConfigField(
                type=float,
                default=0.2,
                description="ideology_lab：新消息喂养到已有槽位的最小标签相似度(Jaccard)",
                min=0.0,
                max=1.0,
                order=12,
            ),
            "mastication_required_runs": ConfigField(
                type=int,
                default=5,
                description="ideology_lab：品鉴轮数（达到后成熟度=100%）",
                min=1,
                max=20,
                order=12,
            ),
            "mastication_interval_seconds": ConfigField(
                type=float,
                default=30.0,
                description="ideology_lab：同一槽位两次品鉴的最小间隔（秒）",
                min=1.0,
                max=3600.0,
                order=13,
            ),
            "mastication_min_fragments": ConfigField(
                type=int,
                default=6,
                description="ideology_lab：开始品鉴所需的最少切片数",
                min=1,
                max=60,
                order=14,
            ),
            "mastication_fragment_feed": ConfigField(
                type=int,
                default=12,
                description="ideology_lab：每次喂给 LLM 的切片条数",
                min=4,
                max=60,
                order=15,
            ),
            "evaluator_model_task": ConfigField(type=str, default="utils", description="ideology_lab：品鉴器使用的模型任务名", order=16),
            "evaluator_temperature": ConfigField(type=float, default=0.35, description="ideology_lab：品鉴器温度", min=0.0, max=2.0, order=17),
            "evaluator_max_tokens": ConfigField(type=int, default=512, description="ideology_lab：品鉴器最大 tokens", min=128, max=4096, order=18),
            "assimilation_scale": ConfigField(
                type=float,
                default=1.0,
                description="ideology_lab：内化时坐标偏移缩放（对品鉴输出平均值做倍率）",
                min=0.0,
                max=3.0,
                order=19,
            ),
            "max_assimilation_shift_abs": ConfigField(
                type=float,
                default=0.5,
                description="ideology_lab：单轴最大偏移（绝对值，建议 0.2~0.7）",
                min=0.0,
                max=1.0,
                order=20,
            ),
            "global_daily_shift_cap_abs": ConfigField(
                type=float,
                default=0.25,
                description="global 作用域：每天每轴最大累计偏移（绝对值，0 表示不限制）",
                min=0.0,
                max=1.0,
                order=21,
            ),
            "global_whitelist_targets": ConfigField(
                type=list,
                default=[],
                description="global 作用域：允许影响全局人格的群白名单（target，如 qq:123:group；留空表示不限制）",
                item_type="string",
                order=22,
            ),
            "global_min_slot_energy": ConfigField(
                type=float,
                default=0.0,
                description="global 作用域：参与全局内化所需的最小话题能量（0 表示不限制）",
                min=0.0,
                max=10.0,
                order=23,
            ),
            "global_min_avg_confidence": ConfigField(
                type=float,
                default=0.0,
                description="global 作用域：参与全局内化所需的最小平均置信度（0 表示不限制）",
                min=0.0,
                max=1.0,
                order=24,
            ),
            "assimilation_energy_boost": ConfigField(
                type=float,
                default=0.0,
                description="ideology_lab：按话题热度额外放大偏移（0 表示不放大）",
                min=0.0,
                max=2.0,
                order=21,
            ),
            "assimilation_apply_to_base_tone": ConfigField(
                type=bool,
                default=True,
                description="ideology_lab：内化时是否修改 base_tone（长期底色）",
                order=22,
            ),
            "assimilation_apply_to_values": ConfigField(
                type=bool,
                default=True,
                description="ideology_lab：内化时是否立刻修改 values/ema（即时看到大幅偏移）",
                order=23,
            ),
            "api_fragments_tail": ConfigField(
                type=int,
                default=12,
                description="GET /cabinet 返回每个 slot 的 fragments 尾部条数（0 表示不返回）",
                min=0,
                max=60,
                order=24,
            ),
            "pressure_threshold_abs": ConfigField(type=float, default=0.65, description="axis_pressure：轴向压力阈值（|EMA|）", min=0.1, max=1.0, order=30),
            "pressure_required_messages": ConfigField(type=int, default=30, description="axis_pressure：连续压力消息数触发内化", min=3, max=500, order=31),
            "internalization_minutes": ConfigField(type=float, default=30.0, description="axis_pressure：内化周期（分钟）", min=1.0, max=24 * 60.0, order=32),
            "enable_dissonance": ConfigField(type=bool, default=True, description="是否启用认知失调检测", order=40),
            "dissonance_threshold": ConfigField(type=float, default=0.6, description="失调阈值（越低越敏感）", min=0.1, max=1.0, order=41),
            "dissonance_ttl_seconds": ConfigField(type=float, default=120.0, description="失调状态持续时间（秒）", min=10.0, max=3600.0, order=42),
            "enrich_traits_with_llm": ConfigField(
                type=bool,
                default=False,
                description="固化后是否用 LLM 生成“思维条目标题/风格偏置”（更有碎片化旁白感）",
                order=50,
            ),
            "digest_with_llm": ConfigField(
                type=bool,
                default=True,
                description="固化后是否用 LLM 生成“思想定义/内化结果”（命中相关话题时会注入到回复提示词）",
                order=54,
            ),
            "enrich_model_task": ConfigField(
                type=str,
                default="utils",
                description="Trait 风格化生成使用的模型任务名",
                order=51,
            ),
            "enrich_temperature": ConfigField(
                type=float,
                default=0.9,
                description="Trait 风格化生成温度",
                min=0.0,
                max=2.0,
                order=52,
            ),
            "enrich_max_tokens": ConfigField(
                type=int,
                default=256,
                description="Trait 风格化生成最大 tokens",
                min=64,
                max=2048,
                order=53,
            ),
            "digest_model_task": ConfigField(
                type=str,
                default="utils",
                description="Trait 内化结果生成使用的模型任务名",
                order=55,
            ),
            "digest_temperature": ConfigField(
                type=float,
                default=0.35,
                description="Trait 内化结果生成温度",
                min=0.0,
                max=2.0,
                order=56,
            ),
            "digest_max_tokens": ConfigField(
                type=int,
                default=512,
                description="Trait 内化结果生成最大 tokens",
                min=128,
                max=4096,
                order=57,
            ),
            "rethink_enabled": ConfigField(
                type=bool,
                default=False,
                description="是否开启“重新反思”：已固化思维会在后续重新生成思想定义/内化结果（可能发生变化）",
                order=58,
            ),
            "rethink_interval_minutes": ConfigField(
                type=float,
                default=720.0,
                description="重新反思最小间隔（分钟）",
                min=10.0,
                max=30 * 24 * 60.0,
                order=59,
            ),
            "rethink_max_history": ConfigField(
                type=int,
                default=6,
                description="每条 trait 最多保留多少条历史版本（0 表示不保留）",
                min=0,
                max=50,
                order=60,
            ),
        },
        "blackbox": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用黑盒模式（潜意识片段）", order=0),
            "include_dream_trace": ConfigField(
                type=bool,
                default=True,
                description="是否把 Dream 周期中的“思考过程/品鉴结果”也写入黑盒流（前端图4）",
                order=0,
            ),
            "max_trace_items": ConfigField(
                type=int,
                default=200,
                description="最多保留多少条 Dream 思考日志（过大可能导致 state.json 变大）",
                min=20,
                max=2000,
                order=0,
            ),
            "fragment_interval_minutes": ConfigField(type=float, default=60.0, description="片段生成间隔（分钟）", min=1.0, max=24 * 60.0, order=1),
            "model_task": ConfigField(type=str, default="utils", description="片段生成使用的模型任务名", order=2),
            "temperature": ConfigField(type=float, default=0.8, description="片段生成温度", min=0.0, max=2.0, order=3),
            "max_tokens": ConfigField(type=int, default=256, description="片段生成最大 tokens", min=64, max=2048, order=4),
            "max_fragments": ConfigField(type=int, default=30, description="最多保留片段数", min=5, max=500, order=5),
            "retry_backoff_seconds": ConfigField(
                type=float,
                default=300.0,
                description="片段生成失败后的重试退避（秒）",
                min=0.0,
                max=24 * 3600.0,
                order=6,
            ),
        },
        "injection": {
            "enabled": ConfigField(type=bool, default=True, description="是否在回复前注入意识形态上下文（POST_LLM）", order=0),
            "max_trait_details": ConfigField(
                type=int,
                default=2,
                description="命中话题时，最多注入几条“固化思维的内化结果”（0 表示不注入详情）",
                min=0,
                max=6,
                order=1,
            ),
            "max_chars": ConfigField(
                type=int,
                default=1600,
                description="注入块最大长度（字符，过长会被截断）",
                min=400,
                max=6000,
                order=2,
            ),
        },
        "dream": {
            "integrate_with_dream": ConfigField(
                type=bool,
                default=True,
                description="是否与主程序 Dream 模块联动（运行时 hook，不修改主程序文件；Dream 运作时触发品鉴/黑盒）",
                order=0,
            ),
            "full_bind": ConfigField(
                type=bool,
                default=True,
                description="完全绑定模式：思维阁品鉴与黑盒片段只在 Dream 周期触发（避免插件自身定时触发）",
                order=1,
            ),
            "bind_cabinet_to_dream": ConfigField(
                type=bool,
                default=True,
                description="联动时：将思维阁(ideology_lab)的“品鉴/内化”绑定到 Dream 周期触发（关闭则继续按插件后台定时）",
                order=2,
            ),
            "spectrum_groups_per_cycle": ConfigField(
                type=int,
                default=3,
                description="Dream 周期：最多为多少个群执行“光谱窗口批处理”",
                min=0,
                max=50,
                order=2,
            ),
            "spectrum_max_messages_per_group": ConfigField(
                type=int,
                default=80,
                description="Dream 周期：每个群最多取多少条窗口消息参与一次批处理",
                min=10,
                max=200,
                order=2,
            ),
            "cabinet_steps_per_cycle": ConfigField(
                type=int,
                default=2,
                description="每次 Dream 周期最多推进多少次“品鉴轮次”（越大越积极，但更耗 LLM）",
                min=0,
                max=20,
                order=3,
            ),
            "enrich_steps_per_cycle": ConfigField(
                type=int,
                default=1,
                description="每次 Dream 周期最多补齐多少次“trait 编撰”（标题/风格偏置/思想定义/内化结果）",
                min=0,
                max=10,
                order=4,
            ),
            "rethink_steps_per_cycle": ConfigField(
                type=int,
                default=1,
                description="每次 Dream 周期最多推进多少次“重新反思”（已固化思维的思想定义/内化结果可能更新）",
                min=0,
                max=10,
                order=5,
            ),
            "bind_blackbox_to_dream": ConfigField(
                type=bool,
                default=True,
                description="联动时：将黑盒模式的片段生成绑定到 Dream 周期触发（关闭则继续按 fragment_interval_minutes）",
                order=6,
            ),
            "blackbox_groups_per_cycle": ConfigField(
                type=int,
                default=1,
                description="每次 Dream 周期最多为多少个群生成黑盒片段（按当前可生成条件选择）",
                min=0,
                max=50,
                order=7,
            ),
            "force_blackbox_on_dream": ConfigField(
                type=bool,
                default=False,
                description="Dream 周期触发黑盒时是否忽略片段间隔限制（谨慎开启，可能频繁生成）",
                order=8,
            ),
        },
        "persistence": {
            "enabled": ConfigField(type=bool, default=True, description="是否持久化状态到插件 data/state.json", order=0),
            "save_interval_seconds": ConfigField(type=float, default=15.0, description="持久化写入间隔（秒）", min=5.0, max=600.0, order=1),
        },
        "performance": {
            "queue_maxsize": ConfigField(type=int, default=200, description="每群消息队列最大长度", min=20, max=5000, order=0),
            "queue_drop_policy": ConfigField(
                type=str,
                default="drop_new",
                description="队列满时策略：drop_new / drop_old",
                choices=["drop_new", "drop_old"],
                order=1,
            ),
            "per_group_min_interval_seconds": ConfigField(
                type=float,
                default=0.5,
                description="每群最小处理间隔（秒），用于LLM限流",
                min=0.0,
                max=10.0,
                order=2,
            ),
            "max_message_chars": ConfigField(type=int, default=800, description="单条消息最大截断长度（字符）", min=50, max=5000, order=3),
        },
        "runtime": {
            "plugin_dir": ConfigField(type=str, default="", description="插件目录（内部使用，可留空）", hidden=True),
        },
        "debug": {
            "enabled": ConfigField(type=bool, default=False, description="是否启用 /soul 调试命令（生产环境建议关闭）", order=0),
            "admin_only": ConfigField(type=bool, default=True, description="是否仅允许管理员使用调试命令", order=1),
            "admin_user_ids": ConfigField(
                type=list,
                default=[],
                description="允许使用调试命令的用户列表（格式：platform:user_id，例如 qq:123456）",
                order=2,
            ),
            "allow_in_private": ConfigField(type=bool, default=False, description="是否允许在私聊中使用调试命令", order=3),
            "max_text_chars": ConfigField(type=int, default=300, description="simulate 参数最大长度（字符）", min=20, max=2000, order=4),
        },
    }

    def get_plugin_components(self) -> List[Tuple[Any, type]]:
        if not self.get_config("plugin.enabled", True):
            return []
        components: List[Tuple[Any, type]] = [
            (SoulOnStartEventHandler.get_handler_info(), SoulOnStartEventHandler),
            (SoulOnMessageEventHandler.get_handler_info(), SoulOnMessageEventHandler),
            (SoulPostLlmEventHandler.get_handler_info(), SoulPostLlmEventHandler),
        ]
        if bool(self.get_config("debug.enabled", False)):
            components.append((SoulDebugCommand.get_command_info(), SoulDebugCommand))
        return components
