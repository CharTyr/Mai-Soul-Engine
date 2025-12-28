import asyncio
import hashlib
import json
import math
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from src.common.logger import get_logger
from src.common.server import get_global_server
from src.config.config import global_config, model_config
from src.config.api_ada_configs import TaskConfig
from src.manager.async_task_manager import AsyncTask, async_task_manager
from src.plugin_system import (
    BaseCommand,
    BaseEventHandler,
    BasePlugin,
    ComponentInfo,
    ConfigField,
    ConfigLayout,
    ConfigTab,
    EventType,
    register_plugin,
)
from src.plugin_system.apis import llm_api
from src.plugin_system.base.config_types import section_meta
from src.plugin_system.base.component_types import CustomEventHandlerResult, MaiMessages
from src.memory_system.retrieval_tools.tool_registry import register_memory_retrieval_tool

logger = get_logger("mai_soul_engine")

STATE_SCHEMA_VERSION = 3

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

POLES_ENUM: Dict[str, Tuple[str, str]] = {
    "sincerity_absurdism": ("sincerity", "absurdism"),
    "normies_otakuism": ("normies", "otakuism"),
    "traditionalism_radicalism": ("traditionalism", "radicalism"),
    "heroism_nihilism": ("heroism", "nihilism"),
}

POLES_CN: Dict[str, str] = {
    "sincerity": "严肃建构",
    "absurdism": "抽象解构",
    "normies": "现充主义",
    "otakuism": "二次元沉溺",
    "traditionalism": "传统保守",
    "radicalism": "赛博激进",
    "heroism": "热血英雄主义",
    "nihilism": "虚无主义",
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
    attempts: int = 0
    last_attempt_ts: float = 0.0


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

    last_major_shift_ts: float = 0.0
    last_injection_ts: float = 0.0
    last_injection: Dict[str, Any] = field(default_factory=dict)
    introspection_runs_total: int = 0
    introspection_failures_total: int = 0
    last_introspection_duration_ms: int = 0
    last_introspection_ok: bool = True
    last_introspection_error: str = ""


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
        self._audit_file = self.plugin_dir / "data" / "audit.jsonl"
        self._audit_lock = asyncio.Lock()
        self._last_persist_ts = 0.0
        self._salt = secrets.token_hex(16)
        self._memory_tools_registered = False
        self._startup_logged = False
        self._api_seen_non_preflight = False
        self._log_rate: Dict[str, float] = {}

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

    def _sid_hint(self, stream_id: str) -> str:
        try:
            return f"s_{self._sha256_short(f'{self._salt}|sid|{stream_id}', n=10)}"
        except Exception:
            return "s_?"

    def _should_log(self, key: str, *, interval_sec: float) -> bool:
        now = _now_ts()
        last = float(self._log_rate.get(key, 0.0) or 0.0)
        if (now - last) >= float(interval_sec):
            self._log_rate[key] = float(now)
            return True
        return False

    def log_startup_once(self) -> None:
        if self._startup_logged:
            return
        self._startup_logged = True

        enabled = self.is_enabled()
        token_set = bool(str(self.get_config("api.token", "") or "").strip())
        persistence = bool(self.get_config("persistence.enabled", True))
        injection = bool(self.get_config("injection.enabled", True))
        apply_to_planner = bool(self.get_config("injection.apply_to_planner", True))

        intro_interval = float(self.get_config("introspection.interval_minutes", 20.0) or 20.0)
        intro_window = float(self.get_config("introspection.window_minutes", 30.0) or 30.0)
        intro_rounds = int(self.get_config("introspection.rounds", 4) or 4)
        intro_min_msgs = int(self.get_config("introspection.min_messages_to_run", 30) or 30)

        groups_loaded = len(self._groups)
        state_hint = str(self._state_file)

        logger.info(
            "[soul] Mai-Soul-Engine started: enabled=%s groups=%d api=/api/v1/soul token=%s persistence=%s injection=%s planner_injection=%s",
            enabled,
            groups_loaded,
            "set" if token_set else "empty",
            persistence,
            injection,
            apply_to_planner,
        )
        logger.info(
            "[soul] Introspection: interval=%.1fm window=%.1fm rounds=%d min_messages=%d",
            intro_interval,
            intro_window,
            intro_rounds,
            intro_min_msgs,
        )
        logger.info("[soul] State file: %s", state_hint)

        if not token_set:
            logger.warning("[soul] api.token is empty: /api/v1/soul is unauthenticated (production not recommended)")

        if bool(self.get_config("debug.log_api_calls", False)):
            logger.info("[soul] debug.log_api_calls=true: will log every /api/v1/soul request (WebUI polls /pulse frequently)")

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
    def _sha256_short(self, text: str, *, n: int = 10) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return h[: max(6, min(int(n), 32))]

    def _hash_user_id(self, platform: str, user_id: str) -> str:
        key = f"{self._salt}|u|{platform}:{user_id}"
        return f"u_{self._sha256_short(key, n=12)}"

    def _hash_group_id(self, platform: str, group_id: str) -> str:
        key = f"{self._salt}|g|{platform}:{group_id}"
        return f"g_{self._sha256_short(key, n=12)}"

    def _sanitize_text(self, text: str, *, max_chars: int) -> str:
        s = (text or "").replace("\n", " ").replace("\r", " ").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"https?://\\S+", "<url>", s, flags=re.IGNORECASE)
        s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", "<email>", s)
        s = re.sub(r"@\\S+", "@某人", s)
        # 电话（含可选 +86 / 空格 / 连字符）
        s = re.sub(r"(?:\\+?86[-\\s]?)?1[3-9]\\d{9}", "<phone>", s)
        # 身份证（18 位，末位可能是 X）
        s = re.sub(r"\\b\\d{17}[\\dXx]\\b", "<id>", s)
        # 显式 QQ/群号标记
        s = re.sub(r"(?:QQ群|群号|群|QQ|qq)\\s*[:：]?\\s*([1-9]\\d{4,11})", "<qq>", s)
        # 通用长数字（QQ/群号/手机号以外）
        s = re.sub(r"\\b\\d{5,}\\b", "<num>", s)
        max_chars = max(40, min(int(max_chars), 800))
        return s[:max_chars]

    def _tokenize(self, text: str, *, max_tokens: int = 80) -> List[str]:
        s = (text or "").lower()
        parts = re.findall(r"[\\u4e00-\\u9fff]+|[a-z0-9]+", s)
        out: List[str] = []
        for p in parts:
            if not p:
                continue
            if re.fullmatch(r"[\\u4e00-\\u9fff]+", p):
                # 中文：用 2-gram 提升区分度，并保留原片段
                if len(p) == 1:
                    out.append(p)
                else:
                    out.append(p)
                    for i in range(0, len(p) - 1):
                        out.append(p[i : i + 2])
            else:
                out.append(p)
            if len(out) >= max_tokens:
                break
        return out[:max_tokens]

    def _jaccard(self, a: List[str], b: List[str]) -> float:
        sa = {x for x in a if str(x).strip()}
        sb = {x for x in b if str(x).strip()}
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return float(inter) / float(union or 1)

    def _seed_similarity(self, *, topic: str, tags: List[str], other: ThoughtSeed) -> float:
        t1 = self._tokenize((topic or "") + " " + " ".join([str(x) for x in (tags or [])]), max_tokens=120)
        t2 = self._tokenize((other.topic or "") + " " + " ".join([str(x) for x in (other.tags or [])]), max_tokens=120)
        sim = self._jaccard(t1, t2)
        if {str(x).strip() for x in (tags or []) if str(x).strip()} & {str(x).strip() for x in (other.tags or []) if str(x).strip()}:
            sim += 0.12
        return float(_clamp(sim, 0.0, 1.0))

    def _bm25_scores(self, *, query: str, docs: List[Tuple[str, str]]) -> Dict[str, float]:
        # docs: [(doc_id, doc_text)]
        k1 = 1.2
        b = 0.75
        q_tokens = self._tokenize(query, max_tokens=80)
        if not q_tokens:
            return {}

        doc_terms: Dict[str, Dict[str, int]] = {}
        doc_lens: Dict[str, int] = {}
        df: Dict[str, int] = {}

        for doc_id, text in docs:
            tf: Dict[str, int] = {}
            tokens = self._tokenize(text, max_tokens=240)
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            doc_terms[doc_id] = tf
            doc_lens[doc_id] = len(tokens)
            seen = set(tokens)
            for t in seen:
                df[t] = df.get(t, 0) + 1

        n = max(1, len(docs))
        avgdl = sum(doc_lens.values()) / float(len(doc_lens) or 1)

        scores: Dict[str, float] = {}
        for doc_id, _text in docs:
            tf = doc_terms.get(doc_id, {})
            dl = float(doc_lens.get(doc_id, 0) or 0)
            score = 0.0
            for term in q_tokens:
                freq = float(tf.get(term, 0) or 0)
                if freq <= 0:
                    continue
                dfi = float(df.get(term, 0) or 0)
                # BM25+ 常见写法：log(1 + (N - df + 0.5)/(df + 0.5))
                idf = math.log(1.0 + (n - dfi + 0.5) / (dfi + 0.5))
                denom = freq + k1 * (1.0 - b + b * (dl / max(1e-6, avgdl)))
                score += idf * (freq * (k1 + 1.0) / max(1e-6, denom))
            if score > 0:
                scores[doc_id] = float(score)
        return scores

    def _thought_doc_text(self, t: Thought) -> str:
        return "\n".join(
            [
                str(t.topic or ""),
                str(t.name or ""),
                " ".join([str(x) for x in (t.tags or []) if str(x).strip()]),
                str(t.definition or ""),
                str(t.digest or ""),
                str(t.style_hint or ""),
            ]
        ).strip()

    def _score_thoughts(self, *, thoughts: List[Thought], query: str, now: float) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        docs: List[Tuple[str, str]] = []
        meta: Dict[str, Thought] = {}
        for t in thoughts:
            docs.append((t.thought_id, self._thought_doc_text(t)))
            meta[t.thought_id] = t

        scores = self._bm25_scores(query=q, docs=docs)
        if not scores:
            return []

        half_life_days = float(self.get_config("cabinet.thought_half_life_days", 30.0))
        half_life_days = max(0.0, min(half_life_days, 3650.0))
        min_decay = float(self.get_config("cabinet.thought_min_decay", 0.25))
        min_decay = float(_clamp(min_decay, 0.0, 1.0))

        out: List[Dict[str, Any]] = []
        for tid, s in scores.items():
            t = meta.get(tid)
            if not t:
                continue
            w = 1.0
            if half_life_days > 0:
                age_days = max(0.0, (now - float(t.created_ts or now)) / 86400.0)
                decay = 2 ** (-age_days / max(1e-6, half_life_days))
                w = float(_clamp(decay, min_decay, 1.0))
            out.append({"thought": t, "score": float(s) * w, "raw_score": float(s), "decay": w})
        out.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return out

    async def _search_thoughts(self, *, stream_id: Optional[str], query: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        top_k = max(1, min(int(top_k), 10))
        min_score = float(_clamp(float(min_score), 0.0, 999.0))

        now = _now_ts()
        groups: List[Tuple[str, List[Thought]]] = []
        if stream_id:
            lock = self._get_lock(stream_id)
            async with lock:
                st = self._get_group(stream_id)
                groups.append((stream_id, list(st.thoughts or [])))
        else:
            for sid in list(self._groups.keys()):
                lock = self._get_lock(sid)
                async with lock:
                    st = self._get_group(sid)
                    groups.append((sid, list(st.thoughts or [])))

        hits: List[Dict[str, Any]] = []
        for sid, thoughts in groups:
            scored = self._score_thoughts(thoughts=thoughts, query=q, now=now)
            for h in scored:
                final_score = float(h.get("score", 0.0) or 0.0)
                if final_score < min_score:
                    continue
                hits.append({"stream_id": sid, **h})

        hits.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return hits[:top_k]

    def register_memory_tools(self) -> None:
        if self._memory_tools_registered:
            return
        self._memory_tools_registered = True

        async def soul_search_thoughts(query: str, top_k: int = 3, min_score: float = 0.0, chat_id: str = "") -> str:
            q = str(query or "").strip()
            if not q:
                return "query 为空。"
            hits = await self._search_thoughts(stream_id=str(chat_id or "").strip() or None, query=q, top_k=int(top_k), min_score=float(min_score))
            if not hits:
                return "未找到相关固化思想。"
            lines: List[str] = []
            for h in hits:
                t: Thought = h["thought"]
                score = float(h.get("score", 0.0) or 0.0)
                digest = self._sanitize_text(str(t.digest or ""), max_chars=240)
                definition = self._sanitize_text(str(t.definition or ""), max_chars=180)
                lines.append(f"- id={t.thought_id} score={score:.2f} 思想={t.name} 定义={definition} 结论={digest}")
            return "\n".join(lines)[:1600]

        async def soul_get_thought(thought_id: str, chat_id: str = "") -> str:
            tid = str(thought_id or "").strip()
            if not tid:
                return "thought_id 为空。"
            candidates: List[Tuple[str, Thought]] = []
            sid = str(chat_id or "").strip()
            if sid:
                lock = self._get_lock(sid)
                async with lock:
                    st = self._get_group(sid)
                    for t in (st.thoughts or []):
                        if t.thought_id == tid:
                            candidates.append((sid, t))
                            break
            if not candidates:
                for sid2 in list(self._groups.keys()):
                    lock = self._get_lock(sid2)
                    async with lock:
                        st = self._get_group(sid2)
                        for t in (st.thoughts or []):
                            if t.thought_id == tid:
                                candidates.append((sid2, t))
                                break
                    if candidates:
                        break
            if not candidates:
                return "未找到该 thought_id 对应的固化思想。"
            sid3, t = candidates[0]
            definition = self._sanitize_text(str(t.definition or ""), max_chars=600)
            digest = self._sanitize_text(str(t.digest or ""), max_chars=900)
            tags = [str(x)[:32] for x in (t.tags or []) if str(x).strip()][:12]
            impact = {a: round(float(t.impact_points.get(a, 0.0) or 0.0), 2) for a in AXES} if isinstance(t.impact_points, dict) else {}
            return (
                "\n".join(
                    [
                        f"thought_id: {t.thought_id}",
                        f"name: {t.name}",
                        f"topic: {t.topic}",
                        f"tags: {json.dumps(tags, ensure_ascii=False)}",
                        f"created_at: {_to_iso(float(t.created_ts or _now_ts()))}",
                        f"impact_points: {json.dumps(impact, ensure_ascii=False)}",
                        f"definition: {definition}",
                        f"digest: {digest}",
                    ]
                )[:2400]
            )

        register_memory_retrieval_tool(
            name="soul_search_thoughts",
            description="检索麦麦在当前群聊已固化的思想/观点/立场（用于回答“麦麦怎么看/麦麦的观点/你之前的立场是什么”一类问题）。优先用于群内长期形成的‘固化思想’，不要用来复刻原句。",
            parameters=[
                {"name": "query", "type": "string", "description": "要检索的问题/关键词（尽量短）", "required": True},
                {"name": "top_k", "type": "integer", "description": "返回条数（1~10）", "required": False},
                {"name": "min_score", "type": "float", "description": "最低相关分（可选）", "required": False},
            ],
            execute_func=soul_search_thoughts,
        )
        register_memory_retrieval_tool(
            name="soul_get_thought",
            description="获取某条固化思想的详细定义与内化结论（给出 thought_id 后返回详情）。",
            parameters=[
                {"name": "thought_id", "type": "string", "description": "固化思想 id（来自 soul_search_thoughts 的结果）", "required": True}
            ],
            execute_func=soul_get_thought,
        )
        logger.info("[soul] Registered memory retrieval tools: soul_search_thoughts, soul_get_thought")

    def _pending_cap(self) -> int:
        n = int(self.get_config("introspection.pending_max_messages", 600))
        return max(50, min(n, 2000))

    def _log_cap(self) -> int:
        n = int(self.get_config("introspection.max_log_items", 300))
        return max(50, min(n, 5000))

    def _fluctuation_window(self) -> int:
        n = int(self.get_config("spectrum.fluctuation_window", 30))
        return max(10, min(n, 500))

    def _dominant_poles(self, state: GroupState) -> Tuple[str, Optional[str], float]:
        # 使用 EMA 作为“底色”更稳定
        axis_scores: List[Tuple[str, float]] = []
        for axis in AXES:
            v = float(state.ema.get(axis, 0.0) or 0.0)
            axis_scores.append((axis, abs(v)))
        axis_scores.sort(key=lambda x: x[1], reverse=True)
        if not axis_scores:
            return "sincerity", None, 0.0

        def pole(axis: str, v: float) -> str:
            left, right = POLES_ENUM.get(axis, ("sincerity", "absurdism"))
            return left if v < 0 else right

        primary_axis, primary_abs = axis_scores[0]
        primary_pole = pole(primary_axis, float(state.ema.get(primary_axis, 0.0) or 0.0))
        secondary_pole: Optional[str] = None
        if len(axis_scores) > 1 and axis_scores[1][1] >= 0.05:
            secondary_axis, _sec_abs = axis_scores[1]
            secondary_pole = pole(secondary_axis, float(state.ema.get(secondary_axis, 0.0) or 0.0))
        return primary_pole, secondary_pole, float(primary_abs)

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

        uid = self._hash_user_id(str(platform or ""), str(user_id or ""))
        gid = self._hash_group_id(str(platform or ""), str(group_id or ""))

        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            state.target = f"{platform}:group:{gid}"
            state.message_count = int(state.message_count or 0) + 1
            state.last_activity_ts = _now_ts()
            state.pending_messages.append(
                {
                    "ts": state.last_activity_ts,
                    "uid": uid,
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
                    if ":" in s:
                        p, uid = s.split(":", 1)
                        p = p.strip()
                        uid = uid.strip()
                        if p and uid:
                            ids.add(self._hash_user_id(p, uid))
                    # 未带平台前缀的情况无法安全映射，忽略
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

    def _estimate_prompt_tokens(self, text: str) -> int:
        if not text:
            return 0
        # 保守估算：中文基本 ~1 char/token，混合内容取偏大
        return max(len(text), int(len(text.encode("utf-8")) / 3))

    def _effective_max_tokens(self, *, requested: int, task_config: TaskConfig, prompt: str) -> int:
        req = int(requested)
        req = max(128, min(req, 200_000))
        task_cap = int(getattr(task_config, "max_tokens", 4096) or 4096)
        task_cap = max(128, min(task_cap, 200_000))
        base = min(req, task_cap)

        # 提示词长度预算（纯启发式）：避免 prompt 太大时还给超大输出，导致超上下文失败
        prompt_chars = len(prompt)
        total_budget_chars = int(self.get_config("performance.prompt_budget_chars", 120_000))
        total_budget_chars = max(20_000, min(total_budget_chars, 800_000))
        remain_chars = max(1, total_budget_chars - prompt_chars)
        # 以 2 chars/token 粗略换算（偏保守）
        by_budget = max(256, int(remain_chars / 2))

        return max(128, min(base, by_budget))

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

        temp = float(_clamp(float(temperature), 0.0, 2.0))
        effective = self._effective_max_tokens(requested=int(max_tokens), task_config=task_config, prompt=prompt)

        # 自动降级：若解析失败，逐步收紧约束（更严格 JSON + 更小输出 + 更低温度）
        attempts = [
            {"prompt": prompt, "max_tokens": effective, "temperature": temp},
            {
                "prompt": (prompt + "\n\n再次强调：只输出严格 JSON（不要 Markdown 代码块、不要解释文字）。").strip(),
                "max_tokens": max(256, min(effective, 2048)),
                "temperature": temp,
            },
            {
                "prompt": (
                    "只输出一个 JSON object（必须以 { 开头以 } 结尾），不要输出代码块，不要输出任何解释文字。\n"
                    "如果字段无法确定，请给出保守默认值，但必须满足 schema。\n\n"
                    + prompt
                ).strip(),
                "max_tokens": max(256, min(effective, 1024)),
                "temperature": min(temp, 0.3),
            },
        ]

        last_model = ""
        last_content = ""
        for a in attempts:
            p = str(a.get("prompt", "") or "")
            mt = int(a.get("max_tokens", effective) or effective)
            ttemp = float(_clamp(float(a.get("temperature", temp) or temp), 0.0, 2.0))
            ok, content, _reasoning, model_name = await llm_api.generate_with_model(
                p,
                model_config=task_config,
                request_type=request_type,
                temperature=ttemp,
                max_tokens=int(mt),
            )
            last_model = model_name or last_model
            last_content = content or last_content
            if not ok:
                continue
            obj = _extract_json_object(content)
            if isinstance(obj, dict):
                return obj

        logger.debug(f"[soul] llm_json failed model={last_model}: {str(last_content)[:120]}")
        return None

    async def _summarize_chat_window(
        self,
        *,
        stream_id: str,
        context_lines: List[str],
        temperature: float,
        max_tokens: int,
        request_type: str,
    ) -> Optional[dict]:
        # 分段总结：chunk -> summaries -> merge
        if not context_lines:
            return None

        chunk_char_budget = int(self.get_config("performance.summary_chunk_chars", 7000))
        chunk_char_budget = max(2000, min(chunk_char_budget, 50_000))

        chunks: List[List[str]] = []
        cur: List[str] = []
        cur_len = 0
        for line in context_lines:
            ln = len(line) + 1
            if cur and (cur_len + ln > chunk_char_budget):
                chunks.append(cur)
                cur = []
                cur_len = 0
            cur.append(line)
            cur_len += ln
        if cur:
            chunks.append(cur)

        chunk_summaries: List[dict] = []
        for idx, ch in enumerate(chunks, start=1):
            schema = {
                "chunk": idx,
                "summary": ["要点（1~8条，抽象概括，禁止复刻原句）"],
                "topics": ["热议话题（0~6）"],
                "conflicts": ["争议点（0~6）"],
            }
            prompt = (
                "你是“麦麦的思维内省”，请把一段群聊记录做脱敏、抽象化的总结。\n"
                "硬性要求：\n"
                "- 不要输出任何可识别个人信息、链接/号码/邮箱。\n"
                "- 不要复刻聊天原句（必须转述）。\n"
                "- 只输出严格 JSON。\n\n"
                f"输入（第 {idx}/{len(chunks)} 段）：\n"
                + "\n".join(ch[:300])
                + "\n\nJSON schema 示例：\n"
                + json.dumps(schema, ensure_ascii=False)
            )
            obj = await self._llm_json(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max(512, min(int(max_tokens), 8000)),
                request_type=f"{request_type}.chunk.{idx}",
            )
            if not isinstance(obj, dict):
                return None
            chunk_summaries.append(obj)

        merge_schema = {
            "summary": "总体摘要（200~600字，抽象概括，禁止复刻原句）",
            "topics": ["热议话题（0~10）"],
            "conflicts": ["争议点（0~10）"],
            "participants": ["参与者代号（如 U1/U2，仅用于区分，不要真实姓名）"],
        }
        merge_prompt = (
            "你是“麦麦的思维内省”，请把多段摘要合并为一个总体摘要。\n"
            "硬性要求：\n"
            "- 不要输出任何可识别个人信息、链接/号码/邮箱。\n"
            "- 不要复刻聊天原句（必须转述）。\n"
            "- 只输出严格 JSON。\n\n"
            f"分段摘要：\n{json.dumps(chunk_summaries, ensure_ascii=False)}\n\n"
            f"JSON schema 示例：\n{json.dumps(merge_schema, ensure_ascii=False)}"
        )
        merged = await self._llm_json(
            prompt=merge_prompt,
            temperature=temperature,
            max_tokens=max(512, min(int(max_tokens), 8000)),
            request_type=f"{request_type}.merge",
        )
        if not isinstance(merged, dict):
            return None

        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            self._append_log_locked(
                state,
                kind="context_summary",
                content="【聊天窗口分段总结】\n" + str(merged.get("summary", "") or "")[:1800],
                tags=["introspection", "summary"],
                extra={"chunks": len(chunks)},
            )
        return merged

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
                    "rationale": ["固化理由（1~4条，抽象概括，不复刻原句）"],
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
 - rationale 必须抽象概括，不要带任何可识别信息与原句。

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

        rationale = final.get("rationale", [])
        if isinstance(rationale, list):
            reasons = [self._sanitize_text(str(x), max_chars=140) for x in rationale if str(x).strip()]
            reasons = [x for x in reasons if x][:6]
            if reasons:
                async with lock:
                    state = self._get_group(stream_id)
                    self._append_log_locked(
                        state,
                        kind="cabinet_rationale",
                        content="【思想固化理由】\n" + "\n".join([f"- {x}" for x in reasons]),
                        tags=["introspection", "cabinet"],
                        extra={"topic": seed.topic},
                    )

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

        t0 = _now_ts()
        sid_hint = self._sid_hint(stream_id)
        internalized_ok = 0
        internalized_fail = 0
        seeds_added = 0
        seeds_merged = 0
        major_shift = False
        applied_points: Dict[str, float] = {a: 0.0 for a in AXES}

        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            from_ts, window_items = self._extract_window_locked(state, now=now)
            window_count = len(window_items)
            seed_queue_before = len(state.seed_queue or [])
            if not window_items and not state.seed_queue:
                state.last_introspection_ts = max(float(state.last_introspection_ts or 0.0), now)
                dur_ms = int(max(0.0, (_now_ts() - t0) * 1000.0))
                logger.info("[soul] Introspection noop: sid=%s dur=%dms (empty window & no seeds)", sid_hint, dur_ms)
                return False

        logger.info(
            "[soul] Introspection start%s: sid=%s window_msgs=%d seed_queue=%d",
            " (forced)" if force else "",
            sid_hint,
            int(window_count),
            int(seed_queue_before),
        )

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
                seed.attempts = int(getattr(seed, "attempts", 0) or 0)
                seed.last_attempt_ts = float(_now_ts())
                max_attempts = int(self.get_config("cabinet.max_internalize_attempts", 3))
                max_attempts = max(1, min(max_attempts, 20))
                logger.info(
                    "[soul] Internalize start: sid=%s seed_id=%s attempt=%d/%d",
                    sid_hint,
                    str(getattr(seed, "seed_id", "") or ""),
                    int(seed.attempts) + 1,
                    int(max_attempts),
                )
                thought = await self._internalize_seed(stream_id, seed)
                if not thought:
                    async with lock:
                        state = self._get_group(stream_id)
                        self._append_log_locked(
                            state,
                            kind="cabinet_failed",
                            content=f"【思想内化失败】topic={seed.topic} attempts={seed.attempts + 1}",
                            tags=["introspection", "cabinet"],
                        )
                        seed.attempts = int(seed.attempts) + 1
                        if seed.attempts < max_attempts:
                            state.seed_queue.append(seed)
                            internalized_fail += 1
                            logger.warning(
                                "[soul] Internalize failed: sid=%s seed_id=%s attempt=%d/%d action=retry",
                                sid_hint,
                                str(getattr(seed, "seed_id", "") or ""),
                                int(seed.attempts),
                                int(max_attempts),
                            )
                        else:
                            internalized_fail += 1
                            logger.warning(
                                "[soul] Internalize failed: sid=%s seed_id=%s attempt=%d/%d action=drop",
                                sid_hint,
                                str(getattr(seed, "seed_id", "") or ""),
                                int(seed.attempts),
                                int(max_attempts),
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
                    state.last_major_shift_ts = max(float(state.last_major_shift_ts or 0.0), now)
                    internalized_ok += 1
                    try:
                        impact = {a: round(float(thought.impact_points.get(a, 0.0) or 0.0), 2) for a in AXES}
                    except Exception:
                        impact = {}
                    logger.info(
                        "[soul] Thought crystallized: sid=%s thought_id=%s impact=%s",
                        sid_hint,
                        str(getattr(thought, "thought_id", "") or ""),
                        json.dumps(impact, ensure_ascii=False),
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
        speaker_map: Dict[str, str] = {}
        speaker_seq = 0
        for m in window_items:
            uid = str(m.get("uid", "") or m.get("user_id", "") or "").strip()
            role = "self" if (uid and uid in self_ids) else "other"
            if uid not in speaker_map:
                speaker_seq += 1
                speaker_map[uid] = f"U{speaker_seq}"
            speaker = speaker_map.get(uid, "U?")
            context.append({"role": role, "speaker": speaker, "text": str(m.get("text", "") or "")})

        # 提示词预算：context 过大时先做分段总结，再把摘要喂给多轮内省
        context_lines = [f"{c['speaker']}<{c['role']}>: {c['text']}" for c in context if str(c.get("text", "")).strip()]
        context_json = json.dumps(context, ensure_ascii=False)
        prompt_budget_chars = int(self.get_config("performance.prompt_budget_chars", 120_000))
        prompt_budget_chars = max(20_000, min(prompt_budget_chars, 800_000))
        need_summary = (len(context) >= 180) or (len(context_json) >= int(prompt_budget_chars * 0.55))

        summary_obj: Optional[dict] = None
        if need_summary:
            summary_obj = await self._summarize_chat_window(
                stream_id=stream_id,
                context_lines=context_lines,
                temperature=min(temperature, 0.5),
                max_tokens=max_tokens,
                request_type="mai_soul_engine.introspection.summary",
            )

        if isinstance(summary_obj, dict):
            tail = context_lines[-40:] if context_lines else []
            context_block = json.dumps(
                {
                    "summary": str(summary_obj.get("summary", "") or "")[:2000],
                    "topics": summary_obj.get("topics", []) if isinstance(summary_obj.get("topics"), list) else [],
                    "conflicts": summary_obj.get("conflicts", []) if isinstance(summary_obj.get("conflicts"), list) else [],
                    "recent": tail,
                },
                ensure_ascii=False,
            )
        else:
            context_block = context_json

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
                    "shift_rationale": ["本次偏移的关键理由（1~4条，抽象概括，不复刻原句）"],
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
聊天记录（已脱敏/截断；可能已被分段总结；禁止复刻原句）：{context_block}

上一轮累计要点：{prev}

硬性要求：
- 不要输出任何可识别个人信息、链接/号码/邮箱。
- 所有内容不得复刻聊天原句。
- 偏移以“点”为单位（-100~100），日常偏移应偏小（通常每轴<5），除非你确信出现明显变化。
- 种子 fragments 必须抽象转述。
 - shift_rationale 必须抽象概括，不要带任何可识别信息与原句。

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
            dur_ms = int(max(0.0, (_now_ts() - t0) * 1000.0))
            logger.warning(
                "[soul] Introspection failed: sid=%s dur=%dms window_msgs=%d internalized=%d/%d (LLM output missing/invalid)",
                sid_hint,
                dur_ms,
                int(window_count),
                int(internalized_ok),
                int(internalized_ok + internalized_fail),
            )
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
            applied_points = {a: float(applied.get(a, 0.0) or 0.0) for a in AXES}
            if any(abs(float(applied.get(a, 0.0) or 0.0)) >= 10.0 for a in AXES):
                state.last_major_shift_ts = max(float(state.last_major_shift_ts or 0.0), now)
                major_shift = True

            rationale = final.get("shift_rationale", [])
            if isinstance(rationale, list):
                reasons = [self._sanitize_text(str(x), max_chars=140) for x in rationale if str(x).strip()]
                reasons = [x for x in reasons if x][:6]
                if reasons:
                    self._append_log_locked(
                        state,
                        kind="spectrum_rationale",
                        content="【偏移理由】\n" + "\n".join([f"- {x}" for x in reasons]),
                        tags=["introspection", "spectrum"],
                    )

            # 生成候选思想种子：进入队列，等下一次内省内化
            seed_threshold = float(self.get_config("cabinet.seed_energy_threshold", 0.5))
            seed_threshold = float(_clamp(seed_threshold, 0.0, 1.0))
            seed_cap = int(self.get_config("cabinet.max_seed_queue", 30))
            seed_cap = max(0, min(seed_cap, 200))
            seed_merge_threshold = float(self.get_config("cabinet.seed_merge_threshold", 0.6))
            seed_merge_threshold = float(_clamp(seed_merge_threshold, 0.0, 1.0))
            seed_frag_cap = int(self.get_config("cabinet.max_seed_fragments", 12))
            seed_frag_cap = max(0, min(seed_frag_cap, 50))

            for tp in seeds[:10]:
                if not isinstance(tp, dict):
                    continue
                topic = str(tp.get("topic", "") or "").strip()
                if not topic:
                    continue
                energy = float(tp.get("energy", 0.0) or 0.0)
                if energy < seed_threshold:
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
                new_seed = ThoughtSeed(
                    seed_id=seed_id,
                    topic=topic[:60],
                    tags=ttags2,
                    energy=float(_clamp(energy, 0.0, 1.0)),
                    fragments=frags2,
                    created_ts=now,
                    attempts=0,
                    last_attempt_ts=0.0,
                )

                # 去重/合并：topic/标签相近则合并能量与 fragments，避免 seed_queue 被同一话题灌爆
                best_i = -1
                best_sim = 0.0
                for i, s in enumerate(state.seed_queue):
                    if not isinstance(s, ThoughtSeed):
                        continue
                    if s.topic == new_seed.topic:
                        best_i = i
                        best_sim = 1.0
                        break
                    sim = self._seed_similarity(topic=new_seed.topic, tags=new_seed.tags, other=s)
                    if sim > best_sim:
                        best_sim = sim
                        best_i = i

                if best_i >= 0 and best_sim >= seed_merge_threshold:
                    s = state.seed_queue[best_i]
                    s.energy = float(max(float(s.energy or 0.0), float(new_seed.energy or 0.0)))
                    merged_tags = [str(x)[:32] for x in list(dict.fromkeys((s.tags or []) + (new_seed.tags or []))) if str(x).strip()]
                    s.tags = merged_tags[:12]
                    merged_frags = [str(x) for x in list(dict.fromkeys((s.fragments or []) + (new_seed.fragments or []))) if str(x).strip()]
                    s.fragments = merged_frags[:seed_frag_cap] if seed_frag_cap else []
                    if float(s.created_ts or 0.0) <= 0:
                        s.created_ts = float(new_seed.created_ts or now)
                    seeds_merged += 1
                else:
                    state.seed_queue.append(new_seed)
                    seeds_added += 1

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
        dur_ms = int(max(0.0, (_now_ts() - t0) * 1000.0))
        logger.info(
            "[soul] Introspection done: sid=%s ok=%s dur=%dms window_msgs=%d internalized=%d/%d new_seeds=%d merged=%d major_shift=%s shift=%s",
            sid_hint,
            True,
            dur_ms,
            int(window_count),
            int(internalized_ok),
            int(internalized_ok + internalized_fail),
            int(seeds_added),
            int(seeds_merged),
            bool(major_shift),
            json.dumps({a: round(float(applied_points.get(a, 0.0) or 0.0), 2) for a in AXES}, ensure_ascii=False),
        )
        return True

    # -------------------------
    # Injection
    # -------------------------
    def _is_utility_query(self, text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return False
        s_low = s.lower()
        if "```" in s or "`" in s:
            return True
        if any(k in s_low for k in ["traceback", "exception", "error", "stack", "npm", "pip", "import ", "http ", "https "]):
            return True
        if any(k in s for k in ["报错", "异常", "堆栈", "怎么装", "怎么部署", "怎么配置", "命令", "日志"]):
            return True
        return False

    def _wants_opinion(self, text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return False
        return any(
            k in s
            for k in [
                "怎么看",
                "你觉得",
                "你认为",
                "如何评价",
                "立场",
                "观点",
                "意识形态",
                "政治",
                "道德",
                "意义",
                "价值",
                "虚无",
                "英雄",
                "二次元",
                "现充",
                "宅",
                "AI",
                "人工智能",
                "赛博",
                "传统",
                "保守",
                "激进",
            ]
        )

    def _injection_policy(self, *, trigger_text: str) -> str:
        conf = str(self.get_config("injection.policy", "auto") or "auto").strip().lower()
        if conf in {"full", "spectrum_only", "auto"}:
            pass
        else:
            conf = "auto"
        if conf != "auto":
            return conf
        if self._is_utility_query(trigger_text):
            return "spectrum_only"
        if self._wants_opinion(trigger_text):
            return "full"
        return "auto"

    async def build_injection_block(self, stream_id: str, *, trigger_text: str) -> str:
        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            values = dict(state.values)
            thoughts = list(state.thoughts[-20:])

        spectrum_block = self._spectrum_instruction_block(values)
        policy = self._injection_policy(trigger_text=trigger_text)
        max_details = int(self.get_config("injection.max_thought_details", 2))
        max_details = max(0, min(max_details, 6))
        if policy == "spectrum_only":
            max_details = 0

        min_score = float(self.get_config("injection.min_thought_score", 0.0))
        scored = self._score_thoughts(thoughts=thoughts, query=(trigger_text or "").strip(), now=now)
        picked_scored = [x for x in scored if float(x.get("score", 0.0) or 0.0) >= min_score][:max_details]
        details = []
        for item in picked_scored:
            t: Thought = item["thought"]
            style_hint = str(t.style_hint or "").strip()
            style_hint = self._sanitize_text(style_hint, max_chars=120) if style_hint else ""
            details.append(
                "\n".join(
                    [
                        f"- 思想：{t.name}",
                        f"  定义：{(t.definition or '').strip()[:240]}",
                        f"  内化结论：{(t.digest or '').strip()[:520]}",
                        f"  风格提示：{style_hint}" if style_hint else "",
                    ]
                )
            )
        details = [d for d in details if d.strip()]

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

        async with lock:
            state = self._get_group(stream_id)
            state.last_injection_ts = now
            state.last_injection = {
                "ts": _to_iso(now),
                "policy": policy,
                "picked": [
                    {
                        "thought_id": str(item["thought"].thought_id),
                        "name": str(item["thought"].name),
                        "score": round(float(item.get("score", 0.0) or 0.0), 4),
                        "decay": round(float(item.get("decay", 1.0) or 1.0), 4),
                        "raw_score": round(float(item.get("raw_score", 0.0) or 0.0), 4),
                    }
                    for item in picked_scored
                ],
            }
        return block

    async def build_planner_injection_block(self, stream_id: str) -> str:
        lock = self._get_lock(stream_id)
        async with lock:
            state = self._get_group(stream_id)
            values = dict(state.values)

        spectrum_block = self._spectrum_instruction_block(values)

        block = (
            "\n\n"
            "[Mai‑Soul Planner]\n"
            "你正在进行动作规划。以下是你当前“思想光谱”的倾向（按档位强度）。在选择动作与回复策略时，请让你的立场判断与这些倾向一致。\n"
            "要求：不要在最终输出中显式提及系统名/数值；不要改变你被要求的输出格式。\n"
            f"{spectrum_block}\n"
        ).strip("\n")

        max_chars = int(self.get_config("injection.planner_max_chars", 900))
        max_chars = max(200, min(max_chars, 4000))
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

        async def runner() -> bool:
            t0 = _now_ts()
            ok = False
            err = ""
            try:
                ok = await self._introspection_run(stream_id, force=False)
                return bool(ok)
            except Exception as e:
                err = str(e)
                logger.debug(f"[soul] introspection task crashed: {e}")
                return False
            finally:
                dur_ms = int(max(0.0, (_now_ts() - t0) * 1000.0))
                lock = self._get_lock(stream_id)
                async with lock:
                    st = self._get_group(stream_id)
                    st.introspection_runs_total = int(st.introspection_runs_total or 0) + 1
                    if not ok:
                        st.introspection_failures_total = int(st.introspection_failures_total or 0) + 1
                    st.last_introspection_duration_ms = int(dur_ms)
                    st.last_introspection_ok = bool(ok)
                    st.last_introspection_error = str(err or "")[:300]

        task = asyncio.create_task(runner())
        self._introspection_tasks[stream_id] = task
        task.add_done_callback(lambda _t, sid=stream_id: self._introspection_tasks.pop(sid, None))

    async def tick_background(self) -> None:
        if not self.is_enabled():
            return
        now = _now_ts()
        candidates: List[Tuple[str, float]] = []

        async with self._tick_lock:
            debug_sched = bool(self.get_config("debug.log_scheduler", False))
            interval_min = float(self.get_config("introspection.interval_minutes", 20.0))
            interval_min = max(1.0, min(interval_min, 24 * 60.0))
            quiet = float(self.get_config("introspection.quiet_period_seconds", 20.0))
            quiet = max(0.0, min(quiet, 3600.0))
            min_msgs = int(self.get_config("introspection.min_messages_to_run", 30))
            min_msgs = max(1, min(min_msgs, 2000))
            for sid, st in self._groups.items():
                lock = self._get_lock(sid)
                async with lock:
                    last = float(st.last_introspection_ts or 0.0)
                    due = (now - last) >= interval_min * 60.0
                    quiet_ok = (quiet <= 0.0) or ((now - float(st.last_activity_ts or 0.0)) >= quiet)
                    has_seeds = bool(st.seed_queue)

                    _from_ts, window_items = self._extract_window_locked(st, now=now)
                    enough_msgs = len(window_items) >= min_msgs

                    if (has_seeds or enough_msgs) and due and quiet_ok and not self._introspection_task_active(sid):
                        candidates.append((sid, float(st.last_activity_ts or 0.0)))
                    elif debug_sched and due:
                        reason = ""
                        if self._introspection_task_active(sid):
                            reason = "task_active"
                        elif (has_seeds or enough_msgs) and not quiet_ok:
                            reason = "not_quiet"
                        elif (not has_seeds) and (not enough_msgs) and quiet_ok:
                            reason = "not_enough_msgs"
                        if reason and self._should_log(f"sched:{sid}:{reason}", interval_sec=60.0):
                            logger.info(
                                "[soul] Scheduler skip: sid=%s reason=%s window_msgs=%d/%d seeds=%d quiet_ok=%s",
                                self._sid_hint(sid),
                                reason,
                                int(len(window_items)),
                                int(min_msgs),
                                int(len(st.seed_queue or [])),
                                bool(quiet_ok),
                            )

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
    def _state_version(self, raw: dict) -> int:
        try:
            v = raw.get("schema_version", None)
            if v is None:
                v = raw.get("version", 0)
            return int(v or 0)
        except Exception:
            return 0

    def _migrate_state_payload(self, raw: dict) -> dict:
        # 迁移框架：把旧版本 payload 逐步转换为最新 schema_version。
        # 目标：保持向后兼容；无法兼容时宁可降级到“尽量可用”，不要阻止插件启动。
        if not isinstance(raw, dict):
            return {"schema_version": STATE_SCHEMA_VERSION, "salt": self._salt, "groups": {}}

        v = self._state_version(raw)

        def ensure_base(obj: dict) -> dict:
            obj.setdefault("salt", self._salt)
            obj.setdefault("groups", {})
            if not isinstance(obj.get("groups"), dict):
                obj["groups"] = {}
            return obj

        raw = ensure_base(raw)

        # v0/v1：可能没有 schema_version；字段结构以 groups 为主
        if v <= 1:
            raw = ensure_base(raw)
            v = 2

        # v2 -> v3：显式 schema_version，保留兼容字段 version
        if v == 2:
            raw["schema_version"] = STATE_SCHEMA_VERSION
            raw["version"] = STATE_SCHEMA_VERSION
            v = STATE_SCHEMA_VERSION

        # 更高版本：尽量读取已知字段
        raw["schema_version"] = int(raw.get("schema_version", STATE_SCHEMA_VERSION) or STATE_SCHEMA_VERSION)
        raw["version"] = int(raw.get("version", raw["schema_version"]) or raw["schema_version"])
        return raw

    def _parse_group_state(self, obj: dict) -> GroupState:
        st = GroupState()
        if not isinstance(obj, dict):
            return st

        st.target = str(obj.get("target", "") or "")
        st.message_count = int(obj.get("message_count", 0) or 0)
        st.last_activity_ts = float(obj.get("last_activity_ts", 0.0) or 0.0)

        pm = obj.get("pending_messages", [])
        pending: List[Dict[str, Any]] = []
        if isinstance(pm, list):
            for m in pm[-self._pending_cap() :]:
                if not isinstance(m, dict):
                    continue
                ts = float(m.get("ts", 0.0) or 0.0)
                text = str(m.get("text", "") or "").strip()
                if not text:
                    continue
                uid = str(m.get("uid", "") or "").strip()
                if not uid:
                    legacy_uid = str(m.get("user_id", "") or "").strip()
                    if legacy_uid:
                        uid = f"u_legacy_{self._sha256_short(f'{self._salt}|legacy|{legacy_uid}', n=10)}"
                pending.append({"ts": ts, "uid": uid, "text": text[:800]})
        st.pending_messages = pending

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
        st.last_major_shift_ts = float(obj.get("last_major_shift_ts", 0.0) or 0.0)
        st.introspection_runs_total = int(obj.get("introspection_runs_total", 0) or 0)
        st.introspection_failures_total = int(obj.get("introspection_failures_total", 0) or 0)
        st.last_introspection_duration_ms = int(obj.get("last_introspection_duration_ms", 0) or 0)
        st.last_introspection_ok = bool(obj.get("last_introspection_ok", True))
        st.last_introspection_error = str(obj.get("last_introspection_error", "") or "")[:300]

        logs = obj.get("introspection_logs", [])
        if isinstance(logs, list):
            out: List[IntrospectionLogItem] = []
            for it in logs[-self._log_cap() :]:
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

        def safe_dataclass_kwargs(dc, data: dict) -> dict:
            allowed = getattr(dc, "__dataclass_fields__", {}) or {}
            return {k: data.get(k) for k in allowed.keys() if k in data}

        seeds = obj.get("seed_queue", [])
        if isinstance(seeds, list):
            out_seeds: List[ThoughtSeed] = []
            for s in seeds:
                if not isinstance(s, dict):
                    continue
                try:
                    out_seeds.append(ThoughtSeed(**safe_dataclass_kwargs(ThoughtSeed, s)))
                except Exception:
                    continue
            st.seed_queue = out_seeds

        thoughts = obj.get("thoughts", [])
        if isinstance(thoughts, list):
            out_thoughts: List[Thought] = []
            for t in thoughts:
                if not isinstance(t, dict):
                    continue
                try:
                    out_thoughts.append(Thought(**safe_dataclass_kwargs(Thought, t)))
                except Exception:
                    continue
            st.thoughts = out_thoughts

        return st

    async def append_audit_event(self, event: Dict[str, Any]) -> None:
        if not bool(self.get_config("debug.audit_enabled", True)):
            return
        max_bytes = int(self.get_config("debug.audit_max_bytes", 2_000_000))
        max_bytes = max(50_000, min(max_bytes, 50_000_000))

        now = _now_ts()
        payload = dict(event or {})
        payload.setdefault("ts", _to_iso(now))
        payload.setdefault("plugin", "mai_soul_engine")
        payload.setdefault("schema_version", STATE_SCHEMA_VERSION)

        # 尽量避免落盘过长字段
        for k, v in list(payload.items()):
            if isinstance(v, str) and len(v) > 800:
                payload[k] = v[:800] + "..."

        line = json.dumps(payload, ensure_ascii=False)

        async with self._audit_lock:
            try:
                self._audit_file.parent.mkdir(parents=True, exist_ok=True)
                if self._audit_file.exists() and self._audit_file.stat().st_size > max_bytes:
                    bak = self._audit_file.with_suffix(self._audit_file.suffix + ".bak")
                    try:
                        if bak.exists():
                            bak.unlink()
                    except Exception:
                        pass
                    try:
                        self._audit_file.rename(bak)
                    except Exception:
                        pass
                with open(self._audit_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                return

    async def try_import_on_start(self) -> None:
        if not bool(self.get_config("persistence.import_on_start", False)):
            return
        p = str(self.get_config("persistence.import_path", "") or "").strip()
        path = Path(p) if p else (self.plugin_dir / "data" / "import.json")
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            await self.append_audit_event({"action": "import_state", "ok": False, "reason": f"invalid_json:{e}"})
            logger.warning("[soul] Import state failed: invalid json (%s)", str(path))
            return
        if not isinstance(raw, dict):
            await self.append_audit_event({"action": "import_state", "ok": False, "reason": "not_object"})
            logger.warning("[soul] Import state failed: not a JSON object (%s)", str(path))
            return

        raw = self._migrate_state_payload(raw)
        mode = str(self.get_config("persistence.import_mode", "merge") or "merge").strip().lower()
        if mode not in {"merge", "overwrite"}:
            mode = "merge"

        imported_salt = str(raw.get("salt", "") or "").strip()
        if mode == "merge" and imported_salt and imported_salt != self._salt and self._groups:
            await self.append_audit_event({"action": "import_state", "ok": False, "reason": "salt_mismatch"})
            logger.warning("[soul] Import state blocked: salt mismatch (mode=merge)")
            return
        if mode == "overwrite" and imported_salt:
            self._salt = imported_salt
        if mode == "merge" and imported_salt and not self._groups:
            self._salt = imported_salt

        groups = raw.get("groups", {})
        if not isinstance(groups, dict):
            await self.append_audit_event({"action": "import_state", "ok": False, "reason": "no_groups"})
            logger.warning("[soul] Import state failed: missing groups")
            return

        imported: Dict[str, GroupState] = {}
        for sid, obj in groups.items():
            if not isinstance(obj, dict):
                continue
            imported[str(sid)] = self._parse_group_state(obj)

        if not imported:
            await self.append_audit_event({"action": "import_state", "ok": False, "reason": "empty"})
            logger.warning("[soul] Import state failed: empty groups")
            return

        before_groups = len(self._groups)
        if mode == "overwrite":
            self._groups = imported
        else:
            for sid, st_new in imported.items():
                if sid not in self._groups:
                    self._groups[sid] = st_new
                    continue
                lock = self._get_lock(sid)
                async with lock:
                    st = self._get_group(sid)
                    if float(st_new.last_introspection_ts or 0.0) > float(st.last_introspection_ts or 0.0):
                        st.base_tone = st_new.base_tone
                        st.values = st_new.values
                        st.ema = st_new.ema
                        st.recent_deltas = st_new.recent_deltas
                        st.last_introspection_ts = st_new.last_introspection_ts
                        st.last_major_shift_ts = max(float(st.last_major_shift_ts or 0.0), float(st_new.last_major_shift_ts or 0.0))
                    # 合并 thoughts
                    by_id: Dict[str, Thought] = {t.thought_id: t for t in (st.thoughts or []) if isinstance(t, Thought)}
                    for t in st_new.thoughts or []:
                        if not isinstance(t, Thought):
                            continue
                        cur = by_id.get(t.thought_id)
                        if not cur or float(t.created_ts or 0.0) > float(cur.created_ts or 0.0):
                            by_id[t.thought_id] = t
                    merged_thoughts = list(by_id.values())
                    merged_thoughts.sort(key=lambda x: float(x.created_ts or 0.0))
                    max_slots = int(self.get_config("cabinet.max_slots", 6))
                    max_slots = max(1, min(max_slots, 20))
                    st.thoughts = merged_thoughts[-max_slots:]

                    # 合并 seed_queue（简单拼接 + cap；更复杂策略由内省期合并）
                    merged_seeds = list(st.seed_queue or []) + list(st_new.seed_queue or [])
                    seed_cap = int(self.get_config("cabinet.max_seed_queue", 30))
                    seed_cap = max(0, min(seed_cap, 200))
                    st.seed_queue = merged_seeds[-seed_cap:] if seed_cap else []

                    # 合并日志（按时间保留最后 N 条）
                    merged_logs = list(st.introspection_logs or []) + list(st_new.introspection_logs or [])
                    merged_logs.sort(key=lambda x: float(getattr(x, "ts", 0.0) or 0.0))
                    st.introspection_logs = merged_logs[-self._log_cap() :]

                    st.message_count = max(int(st.message_count or 0), int(st_new.message_count or 0))
                    st.last_activity_ts = max(float(st.last_activity_ts or 0.0), float(st_new.last_activity_ts or 0.0))

        after_groups = len(self._groups)
        try:
            ts = int(_now_ts())
            path.rename(path.with_suffix(path.suffix + f".imported.{ts}"))
        except Exception:
            pass
        await self.append_audit_event({"action": "import_state", "ok": True, "mode": mode, "groups": len(imported)})
        logger.info(
            "[soul] Imported state on start: mode=%s imported_groups=%d total_groups=%d (+%d)",
            mode,
            int(len(imported)),
            int(after_groups),
            int(max(0, after_groups - before_groups)),
        )

    def _fsync_dir(self, dir_path: Path) -> None:
        try:
            fd = os.open(str(dir_path), os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            return

    def _atomic_write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        bak = path.with_suffix(path.suffix + ".bak")

        data = json.dumps(payload, ensure_ascii=False)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            # 先把旧文件挪到 .bak，再原子替换（失败则尝试回滚）
            had_old = path.exists()
            if had_old:
                os.replace(path, bak)
            os.replace(tmp, path)
            self._fsync_dir(path.parent)
        except Exception:
            # 回滚：如果新文件没落下，但 bak 存在，则恢复
            try:
                if not path.exists() and bak.exists():
                    os.replace(bak, path)
            except Exception:
                pass
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise

    async def persist(self) -> None:
        try:
            payload: Dict[str, Any] = {
                "schema_version": STATE_SCHEMA_VERSION,
                "version": STATE_SCHEMA_VERSION,
                "salt": self._salt,
                "updated_at": _to_iso(_now_ts()),
                "groups": {},
            }
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
                        "last_major_shift_ts": float(st.last_major_shift_ts or 0.0),
                        "introspection_runs_total": int(st.introspection_runs_total or 0),
                        "introspection_failures_total": int(st.introspection_failures_total or 0),
                        "last_introspection_duration_ms": int(st.last_introspection_duration_ms or 0),
                        "last_introspection_ok": bool(st.last_introspection_ok),
                        "last_introspection_error": str(st.last_introspection_error or "")[:300],
                        "introspection_logs": [
                            {"ts": it.ts, "kind": it.kind, "content": it.content, "tags": it.tags, "extra": it.extra}
                            for it in (st.introspection_logs or [])[-self._log_cap():]
                        ],
                        "seed_queue": [s.__dict__ for s in (st.seed_queue or [])[-200:]],
                        "thoughts": [t.__dict__ for t in (st.thoughts or [])[-200:]],
                    }
            self._atomic_write_json(self._state_file, payload)
            if bool(self.get_config("debug.log_persistence", False)):
                try:
                    size = int(self._state_file.stat().st_size) if self._state_file.exists() else 0
                except Exception:
                    size = 0
                logger.info("[soul] State saved: groups=%d bytes=%d", int(len(self._groups)), int(size))
        except Exception as e:
            logger.warning(f"[soul] persist failed: {e}")

    def load_persisted(self) -> None:
        if not self._state_file.exists() and not self._state_file.with_suffix(self._state_file.suffix + ".bak").exists():
            return

        def try_load(path: Path) -> Optional[dict]:
            try:
                if not path.exists():
                    return None
                obj = json.loads(path.read_text(encoding="utf-8"))
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None

        loaded_from = self._state_file
        raw = try_load(self._state_file)
        if raw is None:
            loaded_from = self._state_file.with_suffix(self._state_file.suffix + ".bak")
            raw = try_load(loaded_from)
        if raw is None:
            # 两份都坏掉：保留现场但不阻止启动
            try:
                ts = int(_now_ts())
                for p in [self._state_file, self._state_file.with_suffix(self._state_file.suffix + ".bak")]:
                    if p.exists():
                        p.rename(p.with_suffix(p.suffix + f".corrupt.{ts}"))
            except Exception:
                pass
            logger.warning("[soul] Failed to load persisted state: both state.json and backup are corrupt")
            return

        raw = self._migrate_state_payload(raw)

        try:
            salt = str(raw.get("salt", "") or "").strip()
            if salt:
                self._salt = salt
            groups = raw.get("groups", {})
            if not isinstance(groups, dict):
                return
            for sid, obj in groups.items():
                if not isinstance(obj, dict):
                    continue
                self._groups[str(sid)] = self._parse_group_state(obj)
            logger.info("[soul] Loaded persisted state: groups=%d from=%s", int(len(self._groups)), str(loaded_from.name))
        except Exception as e:
            logger.warning(f"[soul] load_persisted failed: {e}")

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
            "Access-Control-Allow-Methods": "GET,OPTIONS",
            "Access-Control-Allow-Headers": "Authorization,Content-Type,X-Soul-Token",
            "Access-Control-Max-Age": "600",
            "Vary": "Origin",
        }
        if allow:
            headers["Access-Control-Allow-Origin"] = allow
        return Response(status_code=204, headers=headers)

    def _cors_allow_origin(self, origin: str) -> Optional[str]:
        # 为了减少配置复杂度：不提供 CORS 白名单配置项。
        # - 如果 api.token 为空：仅允许本机 Origin（localhost/127.0.0.1/[::1]）跨域访问，避免误暴露
        # - 如果 api.token 非空：允许任意 Origin（反射 Origin），便于前端跨域部署
        origin = str(origin or "").strip()
        if not origin or origin.lower() == "null":
            return None

        o = origin.lower()
        if not (o.startswith("http://") or o.startswith("https://")):
            return None

        token = str(self.get_config("api.token", "") or "").strip()
        if token:
            return origin

        if (
            o.startswith("http://localhost")
            or o.startswith("https://localhost")
            or o.startswith("http://127.0.0.1")
            or o.startswith("https://127.0.0.1")
            or o.startswith("http://[::1]")
            or o.startswith("https://[::1]")
        ):
            return origin
        return None

    def _origin_for_log(self, request: Request) -> str:
        origin = request.headers.get("origin", "") or request.headers.get("Origin", "")
        origin = str(origin or "").strip()
        if origin and len(origin) > 120:
            origin = origin[:117] + "..."
        return origin

    def _log_api_call(self, request: Request, *, status_code: int, dur_ms: int) -> None:
        path = str(getattr(request.url, "path", "") or "")
        method = str(getattr(request, "method", "") or "").upper()
        if not path.startswith("/api/v1/soul"):
            return
        if method == "OPTIONS":
            return

        log_all = bool(self.get_config("debug.log_api_calls", False))
        origin = self._origin_for_log(request)
        origin_hint = f" origin={origin}" if origin else ""

        if not self._api_seen_non_preflight:
            self._api_seen_non_preflight = True
            if not log_all and status_code < 400:
                logger.info("[soul] API first request: %s %s -> %d (%dms)%s", method, path, status_code, dur_ms, origin_hint)

        if status_code >= 500:
            logger.error("[soul] API %s %s -> %d (%dms)%s", method, path, status_code, dur_ms, origin_hint)
        elif status_code >= 400:
            logger.warning("[soul] API %s %s -> %d (%dms)%s", method, path, status_code, dur_ms, origin_hint)
        elif log_all:
            logger.info("[soul] API %s %s -> %d (%dms)%s", method, path, status_code, dur_ms, origin_hint)

    def _register_cors_middleware(self, app) -> None:
        @app.middleware("http")
        async def soul_cors_middleware(request: Request, call_next):
            path = str(getattr(request.url, "path", "") or "")
            method = str(getattr(request, "method", "") or "").upper()

            if path.startswith("/api/v1/soul") and method == "OPTIONS":
                resp = self._cors_preflight_response(request)
                self._log_api_call(request, status_code=int(resp.status_code), dur_ms=0)
                return resp

            t0 = _now_ts()
            try:
                response = await call_next(request)
            except HTTPException as e:
                response = JSONResponse({"detail": e.detail}, status_code=int(e.status_code))

            dur_ms = int(max(0.0, (_now_ts() - t0) * 1000.0))

            if path.startswith("/api/v1/soul"):
                origin = request.headers.get("origin", "") or request.headers.get("Origin", "")
                allow = self._cors_allow_origin(origin)
                if allow:
                    response.headers["Access-Control-Allow-Origin"] = allow
                    response.headers["Vary"] = "Origin"
                self._log_api_call(request, status_code=int(getattr(response, "status_code", 0) or 0), dur_ms=dur_ms)
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
                # 用 recent_deltas 简单估计波动：RMS 后缩放到 0~1
                ds = [float(x) for x in (st.recent_deltas.get(axis, []) or []) if isinstance(x, (int, float))]
                if ds:
                    rms = math.sqrt(sum(x * x for x in ds) / float(len(ds)))
                    volatility = float(_clamp(rms * 6.0, 0.0, 1.0))
                else:
                    volatility = 0.0
                dims.append(
                    {
                        "axis": AXES_FRONTEND.get(axis, axis),
                        "values": {
                            "current": float(_clamp(float(st.values.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                            "ema": float(_clamp(float(st.ema.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                            "baseline": float(_clamp(float(st.base_tone.get(axis, 0.0) or 0.0), -1.0, 1.0)) * 100.0,
                        },
                        "volatility": volatility,
                    }
                )
            primary, secondary, primary_abs = self._dominant_poles(st)
            primary_points = float(_clamp(primary_abs * 100.0, 0.0, 100.0))
            tier = self._tier_by_points_abs(primary_points)
            tier_cn = {0: "中立", 1: "轻微", 2: "偏向", 3: "明显", 4: "强烈"}.get(int(tier), "中立")
            dominant_trait = "中立" if tier == 0 else f"{POLES_CN.get(primary, primary)}（{tier_cn}）"
            last_major_shift = _to_iso(float(st.last_major_shift_ts or 0.0)) if float(st.last_major_shift_ts or 0.0) > 0 else ""

        return {
            "dimensions": dims,
            "base_tone": {"primary": primary, **({"secondary": secondary} if secondary else {})},
            "dominant_trait": dominant_trait,
            "last_major_shift": last_major_shift,
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

    async def _snapshot_health(self) -> dict:
        now = _now_ts()
        group_count = len(self._groups)
        pending_total = 0
        seed_total = 0
        thought_total = 0
        active_tasks = 0
        last_activity = 0.0
        last_introspection = 0.0
        runs_total = 0
        failures_total = 0
        latest_intro_duration_ms = 0
        for sid in list(self._groups.keys()):
            lock = self._get_lock(sid)
            async with lock:
                st = self._get_group(sid)
                pending_total += len(st.pending_messages or [])
                seed_total += len(st.seed_queue or [])
                thought_total += len(st.thoughts or [])
                last_activity = max(last_activity, float(st.last_activity_ts or 0.0))
                ts = float(st.last_introspection_ts or 0.0)
                if ts >= last_introspection:
                    last_introspection = ts
                    latest_intro_duration_ms = int(st.last_introspection_duration_ms or 0)
                runs_total += int(st.introspection_runs_total or 0)
                failures_total += int(st.introspection_failures_total or 0)
            if self._introspection_task_active(sid):
                active_tasks += 1

        state_file_ok = self._state_file.exists()
        state_bytes = int(self._state_file.stat().st_size) if state_file_ok else 0
        audit_bytes = int(self._audit_file.stat().st_size) if self._audit_file.exists() else 0

        return {
            "ok": True,
            "enabled": self.is_enabled(),
            "schema_version": STATE_SCHEMA_VERSION,
            "uptime": self._uptime_str(now),
            "groups": {
                "count": group_count,
                "pending_messages_total": pending_total,
                "seed_queue_total": seed_total,
                "thoughts_total": thought_total,
                "active_introspection_tasks": active_tasks,
                "last_activity": _to_iso(last_activity) if last_activity else None,
                "last_introspection": _to_iso(last_introspection) if last_introspection else None,
                "introspection_runs_total": runs_total,
                "introspection_failures_total": failures_total,
                "introspection_failure_rate": round((failures_total / runs_total) if runs_total else 0.0, 4),
                "last_introspection_duration_ms": latest_intro_duration_ms,
            },
            "files": {"state_json": {"exists": state_file_ok, "bytes": state_bytes}, "audit_log": {"bytes": audit_bytes}},
            "updated_at": _to_iso(now),
        }

    async def _snapshot_injection_frontend(self, stream_id: str) -> dict:
        now = _now_ts()
        lock = self._get_lock(stream_id)
        async with lock:
            st = self._get_group(stream_id)
            last = dict(st.last_injection or {})
        return {"last_injection": last, "updated_at": _to_iso(now)}

    async def _export_state_sanitized(self, stream_id: Optional[str]) -> dict:
        now = _now_ts()
        out: Dict[str, Any] = {
            "schema_version": STATE_SCHEMA_VERSION,
            "exported_at": _to_iso(now),
            "salt": self._salt,
            "groups": {},
        }

        sids = [str(stream_id)] if stream_id else list(self._groups.keys())
        for sid in sids:
            lock = self._get_lock(sid)
            async with lock:
                st = self._get_group(sid)
                out["groups"][sid] = {
                    "target": st.target,
                    "message_count": int(st.message_count or 0),
                    "last_activity_ts": float(st.last_activity_ts or 0.0),
                    "pending_messages_count": len(st.pending_messages or []),
                    "base_tone": st.base_tone,
                    "values": st.values,
                    "ema": st.ema,
                    "recent_deltas": st.recent_deltas,
                    "last_introspection_ts": float(st.last_introspection_ts or 0.0),
                    "last_major_shift_ts": float(st.last_major_shift_ts or 0.0),
                    "introspection_logs": [
                        {"ts": it.ts, "kind": it.kind, "content": it.content, "tags": it.tags, "extra": it.extra}
                        for it in (st.introspection_logs or [])[-self._log_cap() :]
                    ],
                    "seed_queue": [s.__dict__ for s in (st.seed_queue or [])[-200:]],
                    "thoughts": [t.__dict__ for t in (st.thoughts or [])[-200:]],
                }
        return out

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

        @router.get("/health")
        async def get_health(request: Request):
            self._require_api_token(request)
            return JSONResponse(await self._snapshot_health())

        @router.get("/injection")
        async def get_injection(request: Request, stream_id: Optional[str] = None, target: Optional[str] = None):
            self._require_api_token(request)
            sid = self._resolve_stream_id(stream_id=stream_id, target=target)
            return JSONResponse(await self._snapshot_injection_frontend(sid))

        @router.get("/export")
        async def export_state(request: Request, stream_id: Optional[str] = None, target: Optional[str] = None):
            self._require_api_token(request)
            sid = self._resolve_stream_id(stream_id=stream_id, target=target) if (stream_id or target) else None
            return JSONResponse(await self._export_state_sanitized(sid))

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
        logger.info("[soul] API routes registered: /api/v1/soul/* (CORS auto, token=%s)", "set" if str(self.get_config("api.token", "") or "").strip() else "empty")


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
            engine.register_memory_tools()
            if engine.get_config("persistence.enabled", True):
                engine.load_persisted()
            await engine.try_import_on_start()
            engine.register_api_routes()
            engine.log_startup_once()
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


class SoulOnPlanEventHandler(BaseEventHandler):
    event_type = EventType.ON_PLAN
    handler_name = "mai_soul_engine_on_plan"
    handler_description = "Mai-Soul: 在 planner 阶段注入思想光谱（联动动作规划）"
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
            if not bool(engine.get_config("injection.apply_to_planner", True)):
                return True, True, None, None, None

            prompt = str(message.llm_prompt)

            block = await engine.build_planner_injection_block(str(message.stream_id))

            # 优先插入到“动作选择要求”之前，避免影响末尾的输出格式示例
            marker = "**动作选择要求**"
            idx = prompt.find(marker)
            if idx != -1:
                new_prompt = prompt[:idx].rstrip() + "\n\n" + block + "\n\n" + prompt[idx:]
            else:
                new_prompt = prompt.rstrip() + "\n\n" + block

            message.modify_llm_prompt(new_prompt, suppress_warning=True)
            if bool(engine.get_config("debug.log_injection", False)):
                logger.info(
                    "[soul] Planner injection applied: sid=%s chars=%d",
                    engine._sid_hint(str(message.stream_id)),
                    int(len(block)),
                )
            return True, True, None, None, message
        except Exception as e:
            logger.debug(f"[soul] on_plan injection failed: {e}")
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

            prompt = str(message.llm_prompt)

            # 回复提示词预算：避免注入把 prompt 推爆
            reply_budget = int(engine.get_config("performance.reply_prompt_budget_chars", 0) or 0)
            if reply_budget <= 0:
                reply_budget = int(engine.get_config("performance.prompt_budget_chars", 120_000) or 120_000)
            reply_budget = max(10_000, min(reply_budget, 800_000))
            safety = int(engine.get_config("performance.reply_injection_safety_chars", 2000) or 2000)
            safety = max(0, min(safety, 20_000))
            allow = reply_budget - len(prompt) - safety
            if allow <= 200:
                lock = engine._get_lock(str(message.stream_id))
                async with lock:
                    st = engine._get_group(str(message.stream_id))
                    st.last_injection_ts = _now_ts()
                    st.last_injection = {"ts": _to_iso(st.last_injection_ts), "skipped": True, "reason": "prompt_budget"}
                if engine._should_log(f"injection_skip_budget:{str(message.stream_id)}", interval_sec=60.0):
                    logger.warning(
                        "[soul] Reply injection skipped: sid=%s reason=prompt_budget prompt_chars=%d budget=%d safety=%d allow=%d",
                        engine._sid_hint(str(message.stream_id)),
                        int(len(prompt)),
                        int(reply_budget),
                        int(safety),
                        int(allow),
                    )
                return True, True, None, None, message

            trigger_text = str(getattr(message, "plain_text", "") or "")
            block = await engine.build_injection_block(message.stream_id, trigger_text=trigger_text)
            trimmed = False
            if len(block) > allow:
                trimmed = True
                if engine._should_log(f"injection_trim_budget:{str(message.stream_id)}", interval_sec=60.0):
                    logger.warning(
                        "[soul] Reply injection trimmed: sid=%s block_chars=%d allow=%d",
                        engine._sid_hint(str(message.stream_id)),
                        int(len(block)),
                        int(allow),
                    )
                block = block[: max(0, allow - 3)].rstrip() + "..."

            marker = "现在，你说："
            idx = prompt.rfind(marker)
            if idx != -1:
                new_prompt = prompt[:idx].rstrip() + "\n\n" + block + "\n\n" + prompt[idx:]
            else:
                new_prompt = prompt.rstrip() + "\n\n" + block

            message.modify_llm_prompt(new_prompt, suppress_warning=True)
            if bool(engine.get_config("debug.log_injection", False)):
                lock = engine._get_lock(str(message.stream_id))
                async with lock:
                    st = engine._get_group(str(message.stream_id))
                    inj = dict(st.last_injection or {}) if isinstance(st.last_injection, dict) else {}
                picked = inj.get("picked", [])
                if not isinstance(picked, list):
                    picked = []
                thought_ids = []
                for it in picked:
                    if not isinstance(it, dict):
                        continue
                    tid = str(it.get("thought_id", "") or "").strip()
                    if tid:
                        thought_ids.append(tid)
                logger.info(
                    "[soul] Reply injection applied: sid=%s policy=%s picked=%d chars=%d allow=%d trimmed=%s thought_ids=%s",
                    engine._sid_hint(str(message.stream_id)),
                    str(inj.get("policy", "") or ""),
                    int(len(picked)),
                    int(len(block)),
                    int(allow),
                    bool(trimmed),
                    ",".join(thought_ids[:3]),
                )
            return True, True, None, None, message
        except Exception as e:
            logger.debug(f"[soul] post_llm injection failed: {e}")
            return True, True, None, None, None


class SoulDebugCommand(BaseCommand):
    command_name = "soul_debug"
    command_description = "Mai-Soul 命令（查看状态/光谱/思维阁；可选强制内省）"
    command_pattern = r"^[/／]soul(?:\s+(?P<sub>help|status|introspect|spectrum|cabinet|logs|targets|injection|光谱|思维阁|日志|目标|注入))?(?:\s+(?P<arg>.+))?$"

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
        engine = self._get_engine()
        ok, reason = self._is_allowed()
        sub_raw = str(self.matched_groups.get("sub") or "").strip() or "help"
        sub = {
            "光谱": "spectrum",
            "思维阁": "cabinet",
            "日志": "logs",
            "目标": "targets",
            "注入": "injection",
        }.get(sub_raw, sub_raw).strip().lower() or "help"
        arg = str(self.matched_groups.get("arg") or "").strip()

        platform = str(getattr(self.message.message_info, "platform", "") or "")
        user_info = getattr(self.message.message_info, "user_info", None)
        user_id = str(getattr(user_info, "user_id", "") or "")
        uid_hash = engine._hash_user_id(platform, user_id) if (platform and user_id) else ""
        stream_id = getattr(getattr(self.message, "chat_stream", None), "stream_id", None)

        await engine.append_audit_event(
            {
                "action": "debug_command",
                "sub": sub,
                "ok": ok,
                "reason": reason,
                "uid": uid_hash,
                "stream_id": str(stream_id or ""),
            }
        )

        if not ok:
            # 非白名单/未启用：保持静默（不回复），但拦截命令，避免被主流程当普通聊天处理
            return True, reason, 2
        if not engine.is_enabled():
            await self.send_text("Mai-Soul 当前未启用（[plugin].enabled=false）。")
            return True, "engine_disabled", 2
        if not stream_id:
            await self.send_text("无法获取 stream_id。")
            return True, "no_stream", 2

        def _pick_sid(arg_text: str) -> tuple[Optional[str], str]:
            # 解析优先级：
            # 1) sid/stream_id 显式指定（sid=xxx / sid:xxx）
            # 2) target 显式指定（target=xxx / target:xxx）
            # 3) 直接给一个 token（视为 stream_id 或 target）
            # 4) fallback：如果当前会话 stream_id 已有群状态则用它；否则仅当只有 1 个群时自动选择；否则要求用户指定
            sids = list(engine._groups.keys())
            arg_text = str(arg_text or "").strip()
            if arg_text:
                token0 = arg_text.split()[0].strip()
                m_sid = re.match(r"(?i)^(?:sid|stream_id|stream)[:=](.+)$", token0)
                if m_sid:
                    return str(m_sid.group(1)).strip(), ""
                m_target = re.match(r"(?i)^(?:target)[:=](.+)$", token0)
                if m_target:
                    needle = str(m_target.group(1)).strip()
                    for sid2, st2 in engine._groups.items():
                        if str(st2.target or "") == needle:
                            return sid2, ""
                    return None, "target_not_found"

                if token0 in engine._groups:
                    return token0, ""
                for sid2, st2 in engine._groups.items():
                    if str(st2.target or "") == token0:
                        return sid2, ""
                # 允许用户直接提供一个 stream_id（即便当前未记录到 groups，也可用于查看空状态）
                return token0, ""

            if str(stream_id) in engine._groups:
                return str(stream_id), ""
            if len(sids) == 1:
                return sids[0], ""
            if len(sids) > 1:
                return None, "need_sid"
            return str(stream_id), ""

        async def _send_targets() -> None:
            snap = await engine._snapshot_targets_frontend(limit=20, offset=0)
            targets = snap.get("targets", [])
            if not isinstance(targets, list) or not targets:
                await self.send_text("Mai-Soul：暂无已记录的群/会话（还没有收到任何群消息）。")
                return
            lines = ["Mai-Soul targets（最近活跃在前，最多 20）："]
            for t in targets[:20]:
                if not isinstance(t, dict):
                    continue
                sid2 = str(t.get("stream_id", "") or "")
                last = str(t.get("last_activity", "") or "")
                mc = int(t.get("message_count", 0) or 0)
                tgt = str(t.get("target", "") or "")
                lines.append(f"- sid={sid2} msgs={mc} last={last} target={tgt}")
            await self.send_text("\n".join(lines)[:1800])

        async def _send_spectrum(sid2: str) -> None:
            snap = await engine._snapshot_spectrum_frontend(sid2)
            dims = snap.get("dimensions", [])
            if not isinstance(dims, list):
                dims = []
            lines = ["Mai-Soul 思想光谱："]
            for d in dims[:10]:
                if not isinstance(d, dict):
                    continue
                axis = str(d.get("axis", "") or "")
                vals = d.get("values", {}) if isinstance(d.get("values"), dict) else {}
                cur = vals.get("current", 0.0)
                ema = vals.get("ema", 0.0)
                base = vals.get("baseline", 0.0)
                vol = d.get("volatility", 0.0)
                lines.append(f"- {axis}: current={cur:.1f} ema={ema:.1f} baseline={base:.1f} volatility={float(vol):.2f}")
            dominant = str(snap.get("dominant_trait", "") or "")
            base_tone = snap.get("base_tone", {}) if isinstance(snap.get("base_tone"), dict) else {}
            primary = str(base_tone.get("primary", "") or "")
            secondary = str(base_tone.get("secondary", "") or "")
            last_shift = str(snap.get("last_major_shift", "") or "")
            if dominant:
                lines.append(f"dominant_trait: {dominant}")
            if primary:
                lines.append(f"base_tone: {primary}" + (f" / {secondary}" if secondary else ""))
            if last_shift:
                lines.append(f"last_major_shift: {last_shift}")
            await self.send_text("\n".join(lines)[:1800])

        async def _send_cabinet(sid2: str) -> None:
            lock = engine._get_lock(sid2)
            async with lock:
                st = engine._get_group(sid2)
                seeds = list(st.seed_queue or [])
                thoughts = list(st.thoughts or [])

            lines = ["Mai-Soul 思维阁："]
            if thoughts:
                lines.append(f"固化思想（{len(thoughts)}）：")
                for t in thoughts[-10:]:
                    if not isinstance(t, Thought):
                        continue
                    impact = {a: round(float((t.impact_points or {}).get(a, 0.0) or 0.0), 1) for a in AXES}
                    lines.append(f"- {t.name} ({t.thought_id}) impact={json.dumps(impact, ensure_ascii=False)}")
            else:
                lines.append("固化思想：0")

            if seeds:
                lines.append(f"待内化种子（{len(seeds)}，展示前 10）：")
                for s in seeds[:10]:
                    if not isinstance(s, ThoughtSeed):
                        continue
                    lines.append(f"- {s.topic} ({s.seed_id}) energy={float(s.energy):.2f} tags={','.join(list(s.tags or [])[:6])}")
            else:
                lines.append("待内化种子：0")

            await self.send_text("\n".join(lines)[:1800])

        async def _send_logs(sid2: str, *, limit: int) -> None:
            snap = await engine._snapshot_introspection_logs_frontend(sid2, limit=limit)
            frags = snap.get("fragments", [])
            if not isinstance(frags, list) or not frags:
                await self.send_text("Mai-Soul：暂无内省日志。")
                return
            lines = [f"Mai-Soul 内省日志（最近 {min(len(frags), limit)} 条）："]
            for it in frags[-limit:]:
                if not isinstance(it, dict):
                    continue
                ts = str(it.get("timestamp", "") or "")
                content = str(it.get("content", "") or "").replace("\r", "").strip()
                content_one = content.replace("\n", " ")
                if len(content_one) > 180:
                    content_one = content_one[:177] + "..."
                lines.append(f"- [{ts}] {content_one}")
            await self.send_text("\n".join(lines)[:1800])

        async def _send_injection(sid2: str) -> None:
            snap = await engine._snapshot_injection_frontend(sid2)
            last = snap.get("last_injection", {}) if isinstance(snap.get("last_injection"), dict) else {}
            await self.send_text(("Mai-Soul 最近一次注入：\n" + json.dumps(last, ensure_ascii=False, indent=2))[:1800])

        if sub == "help":
            await self.send_text(
                "\n".join(
                    [
                        "Mai-Soul 命令：",
                        "- /soul status                  查看状态（类似 WebUI /pulse）",
                        "- /soul spectrum [sid=<sid>]    查看思想光谱（默认当前会话）",
                        "- /soul cabinet [sid=<sid>]     查看思维阁（固化思想/种子队列）",
                        "- /soul logs [N]                查看最近 N 条内省日志（默认 12）",
                        "- /soul targets                 列出已记录的群/会话 stream_id",
                        "- /soul injection [sid=<sid>]   查看最近一次注入命中信息",
                        "- /soul introspect              强制执行一次思维内省（需另开 allow_force_introspect）",
                    ]
                )
            )
            return True, "help", 2

        if sub == "targets":
            await _send_targets()
            return True, "targets", 2

        sid, sid_reason = _pick_sid(arg)
        if not sid:
            if sid_reason == "need_sid":
                await self.send_text("Mai-Soul：私聊里无法自动确定要查看哪个群。先用 `/soul targets` 获取 sid，然后：`/soul spectrum sid=<sid>`。")
                return True, "need_sid", 2
            if sid_reason == "target_not_found":
                await self.send_text("Mai-Soul：未找到该 target。先用 `/soul targets` 查看可用 target/sid。")
                return True, "target_not_found", 2
            await self.send_text("Mai-Soul：无法解析要查看的 stream_id。")
            return True, "no_sid", 2

        if sub == "status":
            pulse = await engine._snapshot_pulse_frontend(str(sid))
            await self.send_text(json.dumps(pulse, ensure_ascii=False, indent=2)[:1500])
            return True, "status", 2

        if sub == "spectrum":
            await _send_spectrum(str(sid))
            return True, "spectrum", 2

        if sub == "cabinet":
            await _send_cabinet(str(sid))
            return True, "cabinet", 2

        if sub == "logs":
            lim = 12
            if arg:
                if arg.isdigit():
                    lim = int(arg)
            lim = max(1, min(int(lim), 60))
            await _send_logs(str(sid), limit=lim)
            return True, "logs", 2

        if sub == "injection":
            await _send_injection(str(sid))
            return True, "injection", 2

        if sub == "introspect":
            if not bool(engine.get_config("debug.allow_force_introspect", False)):
                await engine.append_audit_event(
                    {"action": "debug_introspect", "ok": False, "reason": "force_disabled", "uid": uid_hash, "stream_id": str(stream_id)}
                )
                await self.send_text("已禁用强制思维内省（debug.allow_force_introspect=false）。")
                return True, "force_disabled", 2
            t0 = _now_ts()
            ok2 = await engine._introspection_run(str(sid), force=True)
            dur_ms = int(max(0.0, (_now_ts() - t0) * 1000.0))
            lock = engine._get_lock(str(sid))
            async with lock:
                st = engine._get_group(str(sid))
                st.introspection_runs_total = int(st.introspection_runs_total or 0) + 1
                if not ok2:
                    st.introspection_failures_total = int(st.introspection_failures_total or 0) + 1
                st.last_introspection_duration_ms = int(dur_ms)
                st.last_introspection_ok = bool(ok2)
            await engine.append_audit_event(
                {
                    "action": "debug_introspect",
                    "ok": bool(ok2),
                    "uid": uid_hash,
                    "stream_id": str(sid),
                }
            )
            await self.send_text("思维内省执行完成。" if ok2 else "思维内省未执行（窗口为空或 LLM 失败）。")
            return True, "introspect", 2

        await self.send_text("Mai-Soul：未知子命令。发送 `/soul help` 查看用法。")
        return True, "unknown", 2


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
            "seed_merge_threshold": ConfigField(type=float, default=0.6, description="思想种子合并阈值（相似度>=阈值则合并）", min=0.0, max=1.0, order=3),
            "max_seed_fragments": ConfigField(type=int, default=12, description="每个思想种子最多保留 fragments 条数", min=0, max=50, order=4),
            "internalize_seeds_per_run": ConfigField(type=int, default=1, description="每次内省最多内化几个种子", min=0, max=5, order=5),
            "max_internalize_attempts": ConfigField(type=int, default=3, description="思想种子内化失败最多重试次数", min=1, max=20, order=6),
            "internalization_rounds": ConfigField(type=int, default=3, description="内化轮数（>=2）", min=2, max=6, order=7),
            "temperature": ConfigField(type=float, default=0.35, description="内化温度", min=0.0, max=2.0, order=8),
            "max_tokens": ConfigField(type=int, default=60000, description="内化最大 tokens", min=128, max=200_000, order=9),
            "thought_half_life_days": ConfigField(type=float, default=30.0, description="固化思想检索衰减半衰期（天；0=不衰减）", min=0.0, max=3650.0, order=10),
            "thought_min_decay": ConfigField(type=float, default=0.25, description="固化思想最低衰减权重（0~1）", min=0.0, max=1.0, order=11),
        },
        "introspection": {
            "interval_minutes": ConfigField(type=float, default=20.0, description="内省触发间隔（分钟）", min=1.0, max=24 * 60.0, order=0),
            "window_minutes": ConfigField(type=float, default=30.0, description="回看聊天时间窗（分钟）", min=1.0, max=24 * 60.0, order=1),
            "rounds": ConfigField(type=int, default=4, description="内省轮数（>=2）", min=2, max=8, order=2),
            "min_messages_to_run": ConfigField(type=int, default=30, description="触发一次内省所需最少消息条数（有种子则可绕过）", min=1, max=2000, order=3),
            "max_messages_per_group": ConfigField(type=int, default=500, description="单次内省最多读取消息条数", min=20, max=2000, order=4),
            "quiet_period_seconds": ConfigField(type=float, default=20.0, description="静默窗口（秒）：避免边聊边回想", min=0.0, max=3600.0, order=5),
            "max_groups_per_tick": ConfigField(type=int, default=1, description="每次 tick 最多启动几个群的内省任务", min=1, max=50, order=6),
            "temperature": ConfigField(type=float, default=0.7, description="内省温度", min=0.0, max=2.0, order=7),
            "max_tokens": ConfigField(type=int, default=700, description="内省最大 tokens", min=128, max=4096, order=8),
            "window_strength_scale": ConfigField(type=float, default=12.0, description="窗口规模强度缩放（越大越保守）", min=1.0, max=200.0, order=9),
            "use_main_personality": ConfigField(type=bool, default=True, description="是否读取主程序人格设定参与内省/内化", order=10),
            "pending_max_messages": ConfigField(type=int, default=600, description="每群最多缓存多少条待回想消息", min=50, max=2000, order=11),
            "max_log_items": ConfigField(type=int, default=300, description="最多保留多少条内省日志", min=50, max=5000, order=12),
        },
        "injection": {
            "enabled": ConfigField(type=bool, default=True, description="是否在回复前注入光谱与固化思想（POST_LLM）", order=0),
            "policy": ConfigField(type=str, default="auto", description="注入策略：auto/full/spectrum_only", order=1),
            "min_thought_score": ConfigField(type=float, default=0.0, description="注入固化思想的最低相关分（0 表示只要命中就可注入）", min=0.0, max=50.0, order=2),
            "max_thought_details": ConfigField(type=int, default=2, description="最多注入几条相关固化思想详情", min=0, max=6, order=3),
            "max_chars": ConfigField(type=int, default=1400, description="注入块最大字符数", min=400, max=6000, order=4),
            "apply_to_planner": ConfigField(type=bool, default=True, description="是否在 planner 阶段注入思想光谱（联动动作规划）", order=5),
            "planner_max_chars": ConfigField(type=int, default=900, description="planner 注入块最大字符数", min=200, max=4000, order=6),
        },
        "persistence": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用持久化（state.json）", order=0),
            "save_interval_seconds": ConfigField(type=float, default=15.0, description="保存间隔（秒）", min=3.0, max=3600.0, order=1),
            "import_on_start": ConfigField(type=bool, default=False, description="启动时从 import 文件导入（仅本地文件，默认关闭）", order=2),
            "import_path": ConfigField(type=str, default="", description="导入文件路径（留空则使用 data/import.json）", order=3),
            "import_mode": ConfigField(type=str, default="merge", description="导入模式：merge/overwrite", order=4),
        },
        "performance": {
            "max_message_chars": ConfigField(type=int, default=800, description="单条消息最大字符数（入库前截断）", min=100, max=5000, order=0),
            "prompt_budget_chars": ConfigField(type=int, default=120_000, description="提示词总字符预算（超出会分段总结/降级）", min=20_000, max=800_000, order=1),
            "summary_chunk_chars": ConfigField(type=int, default=7000, description="分段总结：每段输入字符预算", min=2000, max=50_000, order=2),
            "reply_prompt_budget_chars": ConfigField(type=int, default=120_000, description="回复阶段提示词预算（用于限制注入；默认同 prompt_budget_chars）", min=10_000, max=800_000, order=3),
            "reply_injection_safety_chars": ConfigField(type=int, default=2000, description="回复注入预留安全字符数（避免推爆）", min=0, max=20_000, order=4),
        },
        "debug": {
            "enabled": ConfigField(type=bool, default=False, description="是否启用 /soul 命令（查看状态/光谱/思维阁；生产建议关闭）", order=0),
            "admin_only": ConfigField(type=bool, default=True, description="调试命令是否仅管理员可用", order=1),
            "admin_user_ids": ConfigField(type=list, default=[], description="允许调试命令的用户列表（platform:user_id 或 user_id）", item_type="string", order=2),
            "allow_force_introspect": ConfigField(type=bool, default=False, description="是否允许 /soul introspect 强制内省（写状态；默认关闭）", order=3),
            "audit_enabled": ConfigField(type=bool, default=True, description="是否记录调试/导入等操作的审计日志（data/audit.jsonl）", order=4),
            "audit_max_bytes": ConfigField(type=int, default=2_000_000, description="审计日志最大字节数（超出自动轮转）", min=50_000, max=50_000_000, order=5),
            "log_api_calls": ConfigField(type=bool, default=False, description="是否记录 /api/v1/soul API 调用日志（可能很频繁）", order=6),
            "log_scheduler": ConfigField(type=bool, default=False, description="是否记录内省调度跳过原因（用于排查为何不触发；可能较频繁）", order=7),
            "log_injection": ConfigField(type=bool, default=False, description="是否记录注入摘要日志（planner/replyer；可能较频繁）", order=8),
            "log_persistence": ConfigField(type=bool, default=False, description="是否记录每次状态保存日志（可能较频繁）", order=9),
        },
        "runtime": {"plugin_dir": ConfigField(type=str, default="", description="运行时注入：插件目录（一般不用填）", order=0)},
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        return [
            (SoulOnStartEventHandler.get_handler_info(), SoulOnStartEventHandler),
            (SoulOnMessageEventHandler.get_handler_info(), SoulOnMessageEventHandler),
            (SoulOnPlanEventHandler.get_handler_info(), SoulOnPlanEventHandler),
            (SoulPostLlmEventHandler.get_handler_info(), SoulPostLlmEventHandler),
            (SoulDebugCommand.get_command_info(), SoulDebugCommand),
        ]
