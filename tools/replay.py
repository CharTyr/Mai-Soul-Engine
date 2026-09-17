"""离线回放：把一段固定输入跑过「候选 → 接纳 → 注入」三阶段，产出**可复现记录**。

**用途**：回归时回答「这条输入会不会被接纳？会注入什么？为什么？」——不连宿主、
不联网、不写真实库。

**必须诚实的地方**（方案 T20 原文）：

- 回放里的 LLM 响应是**固定 fixture**，输出记录里逐条标明
  ``llm.kind = "fixture"`` 与 ``llm.disclaimer``。
  **不得**把 fixture 的输出说成「真实模型表现」——那只能靠线上实测。
- 记录必须**可复现**：同一份输入跑两次得到逐字节相同的记录
  （时间戳、随机 id 一律不进记录）。

**三阶段**（各自只用真实代码路径，不另写一套逻辑）：

1. ``candidate``：`thought/candidate.py` 的结构化校验——合法则通过，非法给机器可读原因
2. ``acceptance``：真实 `InternalizationEngine.internalize_seed`——写了什么（trait/光谱）或为什么没写
3. ``injection``：真实 `ideology_injector._select_traits`——选中谁、按什么理由、谁被冷却挡住
4. ``pairing``：真实 `claim_snapshot_for_response`——注入快照与本轮回复的配对是否**可唯一确定**
   （同会话多条未认领快照 → 歧义，下游不得据此改人格）

用法::

    python -m plugins.CharTyr_Mai-Soul-Engine.tools.replay fixtures/garbage.json
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

__all__ = ["REPLAY_FORMAT_VERSION", "run_replay", "summarize"]

REPLAY_FORMAT_VERSION = 1

# 固定声明：写进每一份记录，防止 fixture 输出被当成真实模型表现
LLM_DISCLAIMER = (
    "本记录中的 LLM 响应为**固定 fixture**，用于验证代码路径与决策原因；"
    "它**不代表**真实模型的质量、稳定性或倾向。真实表现须以线上实测为准。"
)


def _import(name: str) -> Any:
    """相对导入（本模块既可被 pytest 载入，也可当脚本跑）。"""
    import importlib

    pkg = __package__.rsplit(".", 1)[0] if __package__ and "." in __package__ else ""
    if pkg:
        return importlib.import_module(f"{pkg}.{name}")
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root.parent.parent))
    return importlib.import_module(f"{root.name}.{name}")


def _fixture_plugin(config: Any, workdir: Path, llm_responses: list[Any]) -> Any:
    """最小插件桩：LLM 按顺序吐 fixture 响应，其余走真实代码。"""

    state: dict[str, Any] = {"calls": 0}

    class _Ctx:
        class _Chat:
            @staticmethod
            async def get_stream_by_user_id(**k: Any) -> str:
                return "fixture-stream"

        # 宿主的能力调用接口（**不是** ctx.llm.generate）：
        # generate_soul_text 走 plugin.ctx.call_capability("llm.generate", ...)。
        # 桩写错接口会让「接纳」阶段一步都跑不到 LLM——测试必须能发现这一点。
        async def call_capability(self, capability: str, timeout_ms: int = 0, **kwargs: Any) -> dict[str, str]:
            if capability != "llm.generate":
                return {"success": False}
            idx = state["calls"]
            state["calls"] = idx + 1
            if not llm_responses:
                return {"response": ""}
            payload = llm_responses[min(idx, len(llm_responses) - 1)]
            if isinstance(payload, str):
                return {"response": payload}
            return {"response": json.dumps(payload, ensure_ascii=False)}

        chat = _Chat()

    ctx = _Ctx()
    return SimpleNamespace(
        config=config, ctx=ctx, _plugin_dir=workdir, _wv_config_view=None,
        _wv_service=None,
    )


def _stage_candidate(candidate_mod: Any, raw: Any, max_delta: int) -> dict[str, Any]:
    """阶段 1：结构化校验。"""
    c = candidate_mod.build_trait_candidate(raw, max_delta=max_delta)
    out: dict[str, Any] = {"valid": bool(c.valid)}
    if c.valid:
        out["thought"] = c.thought
        out["layer"] = c.layer
        out["spectrum_deltas"] = dict(c.spectrum_deltas)
        out["warnings"] = list(c.warnings)
    else:
        out["rejection_reason"] = c.rejection_reason
    return out


def _stage_acceptance(
    engine_mod: Any, plugin: Any, seed: dict[str, Any], dedup: dict[str, Any]
) -> dict[str, Any]:
    """阶段 2：真实内化路径（写入 or 明确拒绝）。"""
    engine = engine_mod.InternalizationEngine(plugin)
    result = asyncio.run(engine.internalize_seed(seed, dedup=dedup))
    if result.get("success"):
        return {
            "accepted": True,
            "trait_id": result.get("trait_id", ""),
            "spectrum_impact": result.get("spectrum_impact", {}),
        }
    return {
        "accepted": False,
        "rejected_as_candidate": bool(result.get("candidate_rejected")),
        "blocked_by_mode": bool(result.get("blocked_by_mode")),
        "error": result.get("error", ""),
    }


def _stage_injection(
    injector_mod: Any, traits: list[Any], text: str, stream_id: str, max_traits: int
) -> dict[str, Any]:
    """阶段 3：注入选择（干跑，不真注入）。"""
    selected, mode, picked = injector_mod._select_traits(
        traits, text, stream_id, max_traits, True, 0.0,
    )
    return {
        "selection_mode": mode,
        "picked": [
            {
                "trait_id": p.get("thought_id", ""),
                "activation_reason": p.get("activation_reason", ""),
                "score": round(float(p.get("score", 0.0) or 0.0), 4),
            }
            for p in picked
        ],
        "count": len(selected),
    }


def run_replay(fixture: dict[str, Any], *, workdir: Path | None = None) -> dict[str, Any]:
    """跑一次回放，返回可复现记录（不含时间戳/随机 id）。"""
    candidate_mod = _import("thought.candidate")
    engine_mod = _import("thought.internalization_engine")
    injector_mod = _import("components.ideology_injector")
    seed_mod = _import("models.ideology_model")
    schema_mod = _import("plugin_ui_schema")

    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="soul-replay-"))
    tmp.mkdir(parents=True, exist_ok=True)
    db_path = tmp / "soul.db"

    seed_mod.close_db()
    seed_mod.init_db(db_path)

    record: dict[str, Any] = {
        "replay_format_version": REPLAY_FORMAT_VERSION,
        "scenario": fixture.get("scenario", "unnamed"),
        "llm": {
            "kind": "fixture",
            "fixture_responses": len(fixture.get("llm_responses", []) or []),
            "disclaimer": LLM_DISCLAIMER,
        },
        "stages": {},
    }

    max_delta = int(fixture.get("max_delta", 10))

    # ── 阶段 1：候选校验（每个 raw 一条结论）──
    record["stages"]["candidate"] = [
        _stage_candidate(candidate_mod, raw, max_delta)
        for raw in fixture.get("candidates", [])
    ]

    # 预置 trait（供注入阶段选中；也让接纳阶段有对手可比）
    for preset in fixture.get("preset_traits", []):
        seed_mod.create_crystallized_trait(
            trait_id=preset["trait_id"], stream_id=preset.get("stream_id", "group-A"),
            seed_id="preset", name=preset.get("name", "预置"), question=preset.get("question", ""),
            thought=preset.get("thought", ""), tags_json=json.dumps(preset.get("tags", []), ensure_ascii=False),
            confidence=int(preset.get("confidence", 80)), evidence_json="[]",
            spectrum_impact_json="{}",
        )

    # ── 阶段 2：接纳（走真实内化引擎）──
    cfg = schema_mod.MaiSoulEngineConfig()
    cfg.plugin.mode = str(fixture.get("mode", "apply"))
    cfg.thought_cabinet.auto_dedup_enabled = bool(fixture.get("dedup_enabled", True))
    plugin = _fixture_plugin(cfg, tmp, fixture.get("llm_responses", []))
    spectrum = seed_mod.get_or_create_spectrum("global")
    spectrum.initialized = True
    spectrum.save()
    before = {
        "sincerity": spectrum.sincerity, "engagement": spectrum.engagement,
        "closeness": spectrum.closeness, "directness": spectrum.directness,
    }

    acceptance: list[dict[str, Any]] = []
    for seed_fixture in fixture.get("seeds", []):
        seed = dict(seed_fixture)
        seed.setdefault("seed_id", seed.get("id", "fixture-seed"))
        seed.setdefault("id", seed["seed_id"])
        seed.setdefault("stream_id", "group-A")
        seed.setdefault("type", "fixture")
        seed.setdefault("event", "fixture event")
        seed.setdefault("reasoning", "fixture reasoning")
        seed.setdefault("intensity", 0.8)
        seed.setdefault("seed_confidence", 0.8)
        seed.setdefault("evidence", [])
        seed.setdefault("context", [])
        seed.setdefault("created_at", None)
        acceptance.append(
            _stage_acceptance(
                engine_mod, plugin, seed,
                {"enabled": bool(fixture.get("dedup_enabled", True)), "threshold": 0.78},
            )
        )
    record["stages"]["acceptance"] = acceptance

    def _dump_scope(scope_id: str) -> dict[str, int]:
        s = seed_mod.get_or_create_spectrum(scope_id)
        return {
            "sincerity": s.sincerity, "engagement": s.engagement,
            "closeness": s.closeness, "directness": s.directness,
        }

    # 只记 global 会**误导**：局部优先（local_first_evolution 默认开）把 delta
    # 写进来源群作用域，global 看起来「没变化」其实是设计如此。
    # 因此把两个作用域都记下来，让「局部隔离」这件事在记录里可见。
    after_global = _dump_scope("global")
    seed_scopes = sorted({str(s.get("stream_id", "group-A")) for s in fixture.get("seeds", [])} or {"group-A"})
    record["stages"]["spectrum"] = {
        "before": before,
        "after": after_global,
        "by_scope": {scope: _dump_scope(scope) for scope in seed_scopes},
    }

    # ── 阶段 4：配对（多账号/并发歧义场景）──
    pairing_fixture = fixture.get("pairing")
    if pairing_fixture:
        sr_mod = _import("models.self_reflection")
        session = str(pairing_fixture.get("session_id", "fixture-session"))
        for i in range(int(pairing_fixture.get("snapshots", 1))):
            sr_mod.create_injection_snapshot(
                "group-A", session, f'["trait-{i}"]', "{}", "{}", "tag_hit",
                bot_identity=str(pairing_fixture.get("bot_identity", "")),
            )
        claimed = sr_mod.claim_snapshot_for_response(
            session, str(pairing_fixture.get("reply_message_id", "reply-1"))
        )
        record["stages"]["pairing"] = {
            "snapshots": int(pairing_fixture.get("snapshots", 1)),
            "claimed": claimed is not None,
            "ambiguous": bool(claimed.pairing_ambiguous) if claimed else False,
            "bot_identity": (claimed.bot_identity if claimed else ""),
        }

    # ── 阶段 3：注入选择 ──
    stream_id = str(fixture.get("inject_stream_id", "group-A"))
    traits = seed_mod.query_active_traits_for_injection(stream_id=stream_id, limit=40)
    record["stages"]["injection"] = _stage_injection(
        injector_mod, traits, str(fixture.get("inject_text", "")), stream_id,
        int(fixture.get("max_traits", 3)),
    )

    seed_mod.close_db()

    # 可复现：随机生成的 id 一律换成稳定别名（uuid 不得进记录）
    keep_ids = {str(p.get("trait_id", "")) for p in fixture.get("preset_traits", [])}
    _normalize_generated_ids(record, keep_ids)

    # 包一层统计，方便对拍
    record["summary"] = summarize(record)
    return record


_GENERATED_ID_RE = re.compile(r"\btrait_[0-9a-f]{6,}\b")


def _normalize_generated_ids(record: dict[str, Any], keep_ids: set[str]) -> None:
    """把**运行期生成的**随机 id 换成稳定别名，保证记录可复现。

    `trait_<hex>` 是 uuid 派生的——直接进记录会让两次回放永远不同。
    别名按首次出现顺序分配（顺序是确定的：记录按固定阶段顺序构建）。
    预置 id 是 fixture 给的，保持原样。
    """
    alias: dict[str, str] = {}

    def _alias(match: re.Match[str]) -> str:
        value = match.group(0)
        if value in keep_ids:
            return value
        if value not in alias:
            alias[value] = f"<generated-trait#{len(alias) + 1}>"
        return alias[value]

    def _walk(node: Any) -> Any:
        if isinstance(node, str):
            return _GENERATED_ID_RE.sub(_alias, node)
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        return node

    normalized = _walk(record)
    record.clear()
    record.update(normalized)


def summarize(record: dict[str, Any]) -> dict[str, Any]:
    """把记录压成几个可断言的数字（用于两次运行对拍）。"""
    candidates = record["stages"]["candidate"]
    acceptance = record["stages"]["acceptance"]
    return {
        "candidates_total": len(candidates),
        "candidates_valid": sum(1 for c in candidates if c.get("valid")),
        "accepted": sum(1 for a in acceptance if a.get("accepted")),
        "injected": int(record["stages"]["injection"].get("count", 0)),
        "selection_mode": record["stages"]["injection"].get("selection_mode", ""),
        "pairing_ambiguous": bool(
            record["stages"].get("pairing", {}).get("ambiguous", False)
        ),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("用法: replay <fixture.json> [out.json]")
        return 2
    fixture = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    record = run_replay(fixture)
    text = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    if len(argv) > 2:
        Path(argv[2]).write_text(text, encoding="utf-8")
        print(f"记录已写入 {argv[2]}")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
