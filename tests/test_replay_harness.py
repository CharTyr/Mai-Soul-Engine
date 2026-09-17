"""T20 离线回放：候选 / 接纳 / 注入 / 配对的**可复现记录**。

四条硬要求（方案原文）：
1. 给出可复现记录 —— 同一份输入跑两次，记录逐字节相同；
2. 覆盖反例、刷屏、多账号歧义、长期无证据；
3. **不得把固定 fixture 响应伪称真实模型表现** —— 记录里必须显式标明；
4. 反例必须给机器可读原因，而不是静默通过。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .conftest import _import_soul_submodule

_VALID_INTERNALIZATION = {
    "thought": "该直说的时候就直接说，不要绕弯子",
    "confidence": 0.8,
    "tags": ["直说"],
    "ideology_layer": "conduct",
    "spectrum_deltas": {"directness": 2},
    "reasoning": "fixture",
}


def _replay_mod() -> Any:
    return _import_soul_submodule("tools.replay")


def _run(fixture: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    return _replay_mod().run_replay(fixture, workdir=tmp_path)


# ─── 1. 可复现 ──────────────────────────────────────────────────────


def test_same_fixture_produces_byte_identical_records(tmp_path: Path) -> None:
    """同输入 → 逐字节相同记录（时间戳/随机 id 不得进记录）。"""
    fixture = {
        "scenario": "determinism",
        "candidates": [{"thought": "观点", "confidence": 0.5, "spectrum_deltas": {"sincerity": 1}}],
        "preset_traits": [
            {"trait_id": "trait-preset", "stream_id": "group-A", "name": "预置",
             "question": "要不要直说", "thought": "该直说时直接说", "tags": ["直说"]},
        ],
        "llm_responses": [_VALID_INTERNALIZATION],
        "seeds": [{"seed_id": "seed-det", "event": "讨论要不要直说"}],
        "inject_text": "要不要直说这件事",
        "inject_stream_id": "group-A",
    }
    first = json.dumps(_run(fixture, tmp_path / "a"), ensure_ascii=False, sort_keys=True)
    second = json.dumps(_run(fixture, tmp_path / "b"), ensure_ascii=False, sort_keys=True)
    assert first == second, "同一份 fixture 两次回放结果不同（不可复现）"


# ─── 2. 反例：必须给出机器可读原因 ─────────────────────────────────


def test_negative_candidates_get_machine_readable_reasons(tmp_path: Path) -> None:
    """反例：空观点 / 未知轴 / 置信度越界 / 非映射 → 全部拒绝且有原因。"""
    fixture = {
        "scenario": "negative",
        "candidates": [
            {"thought": "", "confidence": 0.5},
            {"thought": "有观点", "confidence": 0.5, "spectrum_deltas": {"imaginary_axis": 3}},
            {"thought": "有观点", "confidence": 9.0},
            {"thought": "有观点", "confidence": "not-a-number"},
            "not-a-mapping",
        ],
        "llm_responses": [],
    }
    record = _run(fixture, tmp_path)

    rows = record["stages"]["candidate"]
    assert len(rows) == 5
    assert all(not r["valid"] for r in rows), "反例被放行了"
    reasons = [r["rejection_reason"] for r in rows]
    assert any("empty_thought" in r for r in reasons)
    assert any("unknown_axis" in r for r in reasons)
    assert any("confidence_out_of_range" in r for r in reasons)
    assert any("confidence_not_numeric" in r for r in reasons)
    assert any("not_a_mapping" in r for r in reasons)
    # 反例不产生任何接纳
    assert record["summary"]["accepted"] == 0


# ─── 3. 刷屏：注入条数必须有界 ─────────────────────────────────────


def test_spam_input_cannot_exceed_injection_budget(tmp_path: Path) -> None:
    """刷屏：预置再多观点 + 再长的文本，选中条数不得超过 max_traits。"""
    fixture = {
        "scenario": "spam",
        "max_traits": 3,
        "preset_traits": [
            {
                "trait_id": f"trait-{i}",
                "stream_id": "group-A",
                "name": f"观点{i}",
                "question": "要不要直说",
                "thought": "刷屏也要有边界",
                "tags": ["直说"],
                "confidence": 90,
            }
            for i in range(20)
        ],
        "inject_text": "直说 " * 200,
        "inject_stream_id": "group-A",
        "llm_responses": [],
    }
    record = _run(fixture, tmp_path)

    injection = record["stages"]["injection"]
    assert injection["count"] <= 3, f"刷屏突破了注入预算: {injection['count']}"
    assert len(injection["picked"]) <= 3


# ─── 4. 多账号歧义：必须标记而不是猜 ───────────────────────────────


def test_multi_snapshot_pairing_is_marked_ambiguous(tmp_path: Path) -> None:
    """同一会话两条未认领快照 → 无法唯一配对 → 必须标歧义。"""
    fixture = {
        "scenario": "ambiguous-pairing",
        "pairing": {"session_id": "sess-amb", "snapshots": 2, "bot_identity": "qq:12345678"},
        "llm_responses": [],
    }
    record = _run(fixture, tmp_path)

    pairing = record["stages"]["pairing"]
    assert pairing["claimed"] is True
    assert pairing["ambiguous"] is True, "多条未认领快照没有被判为歧义（在猜）"
    assert pairing["bot_identity"] == "qq:12345678", "作用域字段缺失"


def test_single_snapshot_pairing_is_not_ambiguous(tmp_path: Path) -> None:
    """只有一条未认领快照 → 可唯一配对，不该被误标歧义。"""
    fixture = {
        "scenario": "clean-pairing",
        "pairing": {"session_id": "sess-one", "snapshots": 1},
        "llm_responses": [],
    }
    pairing = _run(fixture, tmp_path)["stages"]["pairing"]
    assert pairing["claimed"] is True and pairing["ambiguous"] is False


# ─── 5. 长期无证据：不得凭空产生人格 ───────────────────────────────


def test_no_evidence_never_produces_personality(tmp_path: Path) -> None:
    """长期无证据：没有种子进去，就不得有任何 trait 或光谱变化。"""
    fixture = {
        "scenario": "long-drought",
        "candidates": [],
        "seeds": [],
        "llm_responses": [],
        "inject_text": "随便说点什么",
        "inject_stream_id": "group-A",
    }
    record = _run(fixture, tmp_path)

    assert record["summary"]["accepted"] == 0
    assert record["stages"]["spectrum"]["before"] == record["stages"]["spectrum"]["after"], (
        "无证据却改动了光谱"
    )
    assert record["summary"]["injected"] == 0


def test_rejection_is_reported_not_silently_accepted(tmp_path: Path) -> None:
    """LLM 给出空观点 → 接纳阶段必须明确拒绝（candidate_rejected），不写人格。"""
    fixture = {
        "scenario": "empty-thought",
        "llm_responses": [{"thought": "", "confidence": 0.9, "spectrum_deltas": {"sincerity": 5}}],
        "seeds": [{"seed_id": "seed-empty", "event": "空观点"}],
    }
    record = _run(fixture, tmp_path)

    acceptance = record["stages"]["acceptance"]
    assert len(acceptance) == 1
    assert acceptance[0]["accepted"] is False
    assert acceptance[0]["rejected_as_candidate"] is True
    assert record["stages"]["spectrum"]["before"] == record["stages"]["spectrum"]["after"]


# ─── 6. 诚实标注：fixture 不得被当成真实模型表现 ───────────────────


def test_record_labels_llm_as_fixture(tmp_path: Path) -> None:
    """记录必须显式标明 LLM 是 fixture，并附免责说明。"""
    fixture = {"scenario": "labelling", "llm_responses": [_VALID_INTERNALIZATION]}
    record = _run(fixture, tmp_path)

    assert record["llm"]["kind"] == "fixture", "记录没有标明这是 fixture 响应"
    disclaimer = record["llm"]["disclaimer"]
    assert "fixture" in disclaimer.lower() or "固定" in disclaimer
    assert "不代表" in disclaimer, "缺少「不代表真实模型表现」的说明"
    assert record["llm"]["fixture_responses"] == 1


# ─── 7. CLI：能产出记录文件 ────────────────────────────────────────


def test_cli_writes_record_file(tmp_path: Path, capsys: Any) -> None:
    """CLI 落盘：`replay <fixture> <out>` 必须真的写出可解析的 JSON。"""
    replay = _replay_mod()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps({"scenario": "cli", "llm_responses": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    out = tmp_path / "record.json"

    assert replay.main(["replay", str(fixture_path), str(out)]) == 0
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["scenario"] == "cli"
    assert parsed["llm"]["kind"] == "fixture"


def test_cli_without_args_returns_usage(tmp_path: Path) -> None:
    assert _replay_mod().main(["replay"]) == 2


def test_positive_seed_is_accepted_and_scoped_locally(tmp_path: Path) -> None:
    """正例全链路：候选通过 → 接纳 → 可注入；且**局部优先**在记录里可见。

    只记 global 会误导（看起来"光谱没变"）——实际 delta 按设计写进了来源群，
    记录必须让这件事看得见。
    """
    fixture = {
        "scenario": "positive",
        "candidates": [dict(_VALID_INTERNALIZATION)],
        "llm_responses": [_VALID_INTERNALIZATION],
        "seeds": [{"seed_id": "seed-pos", "stream_id": "group-A", "event": "讨论要不要直说"}],
        "inject_text": "要不要直说这件事",
        "inject_stream_id": "group-A",
    }
    record = _run(fixture, tmp_path)

    assert record["summary"] == {
        "candidates_total": 1,
        "candidates_valid": 1,
        "accepted": 1,
        "injected": 1,
        "selection_mode": "tag_hit",
        "pairing_ambiguous": False,
    }

    spectrum = record["stages"]["spectrum"]
    assert spectrum["before"] == spectrum["after"], "局部优先却改动了全局光谱"
    assert spectrum["by_scope"]["group-A"]["directness"] == 52, (
        "来源群作用域没有收到光谱影响"
    )

    # 注入理由必须可追溯到具体 tag（便于排查"为什么选/没选"）
    picked = record["stages"]["injection"]["picked"]
    assert picked and picked[0]["activation_reason"] == "tag_hit:直说"
