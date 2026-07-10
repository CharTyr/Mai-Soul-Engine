"""0B.4: self_observation 种子独立日上限测试。

验证：
- count_self_observation_seeds_created_today 计数正确（空=0，混合 type 过滤）
- 日上限入口检查：cap=1 时占满配额后 _maybe_create_self_observation_seed 返回 None

从宿主仓根运行：
``uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/test_self_observation_daily_cap.py -q``
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from .conftest import _import_soul_submodule


# ─── count_self_observation_seeds_created_today ─────────────────


def test_count_empty(soul_db: Any) -> None:
    """空库返回 0。"""
    seeds_mod = _import_soul_submodule("models.seeds")
    assert seeds_mod.count_self_observation_seeds_created_today() == 0


def test_count_mixed_types(soul_db: Any) -> None:
    """创建 2 个 self_observation + 1 个其他 type → 返回 2。"""
    seeds_mod = _import_soul_submodule("models.seeds")
    conn_mod = _import_soul_submodule("models._conn")

    today = conn_mod._dt_to_str(datetime.now())

    # self_observation 种子 #1
    sid1 = f"so_{uuid.uuid4().hex[:8]}"
    seeds_mod.create_thought_seed(
        seed_id=sid1,
        stream_id="g1",
        seed_type="self_observation",
        event="test obs 1",
        intensity=50,
        confidence=50,
        evidence_json="[]",
        reasoning="test",
        potential_impact_json="{}",
    )
    # 确保 created_at 在今天
    conn_mod._get_conn().execute(
        "UPDATE soul_thought_seeds SET created_at = ? WHERE seed_id = ?",
        (today, sid1),
    )
    conn_mod._get_conn().commit()

    # self_observation 种子 #2
    sid2 = f"so_{uuid.uuid4().hex[:8]}"
    seeds_mod.create_thought_seed(
        seed_id=sid2,
        stream_id="g2",
        seed_type="self_observation",
        event="test obs 2",
        intensity=50,
        confidence=50,
        evidence_json="[]",
        reasoning="test",
        potential_impact_json="{}",
    )
    conn_mod._get_conn().execute(
        "UPDATE soul_thought_seeds SET created_at = ? WHERE seed_id = ?",
        (today, sid2),
    )
    conn_mod._get_conn().commit()

    # 普通种子（不同 type）
    sid3 = f"n_{uuid.uuid4().hex[:8]}"
    seeds_mod.create_thought_seed(
        seed_id=sid3,
        stream_id="g1",
        seed_type="conflict",
        event="normal seed",
        intensity=50,
        confidence=50,
        evidence_json="[]",
        reasoning="test",
        potential_impact_json="{}",
    )
    conn_mod._get_conn().execute(
        "UPDATE soul_thought_seeds SET created_at = ? WHERE seed_id = ?",
        (today, sid3),
    )
    conn_mod._get_conn().commit()

    assert seeds_mod.count_self_observation_seeds_created_today() == 2


# ─── _maybe_create_self_observation_seed 日上限检查 ────────────


def _mock_plugin_with_cap(cap: int) -> SimpleNamespace:
    """构造带 self_observation_daily_cap 的 plugin 对象。"""
    return SimpleNamespace(
        config=SimpleNamespace(
            self_reflection=SimpleNamespace(
                self_observation_daily_cap=cap,
            ),
            thought_cabinet=SimpleNamespace(
                admin_notification_enabled=False,
            ),
        ),
    )


class _MockPending:
    """最小化 pending 对象供 _maybe_create_self_observation_seed 使用。"""
    def __init__(self) -> None:
        self.stream_id = "g"
        self.response_text = "测试回复文本"


def test_cap_zero_does_not_block(soul_db: Any) -> None:
    """cap=0 时不拦截（0=不限制），即使已有 today 种子。"""
    seeds_mod = _import_soul_submodule("models.seeds")
    conn_mod = _import_soul_submodule("models._conn")

    # 先创建 2 个 today self_observation 种子
    today = conn_mod._dt_to_str(datetime.now())
    for i in range(2):
        sid = f"pre_{uuid.uuid4().hex[:8]}"
        seeds_mod.create_thought_seed(
            seed_id=sid, stream_id=f"g{i}", seed_type="self_observation",
            event=f"pre obs {i}", intensity=50, confidence=50,
            evidence_json="[]", reasoning="test", potential_impact_json="{}",
        )
        conn_mod._get_conn().execute(
            "UPDATE soul_thought_seeds SET created_at = ? WHERE seed_id = ?",
            (today, sid),
        )
    conn_mod._get_conn().commit()

    plugin = _mock_plugin_with_cap(0)
    ev = _import_soul_submodule("components.reflection_evaluator")
    pending = _MockPending()
    # cap=0 不拦截（走 seed_manager 异常回路返回 None）
    result = asyncio.run(ev._maybe_create_self_observation_seed(
        plugin, {"name": "test", "thought": "x", "spectrum_impact": {}, "confidence": 50},
        pending, [], [], "reason text",
    ))
    assert result is None


def test_cap_one_blocks_second_seed(soul_db: Any) -> None:
    """cap=1 且今日已有 1 个 self_observation 种子 → 跳过。"""
    seeds_mod = _import_soul_submodule("models.seeds")
    conn_mod = _import_soul_submodule("models._conn")

    # 今天已有 1 个 self_observation 种子
    today = conn_mod._dt_to_str(datetime.now())
    sid = f"existing_{uuid.uuid4().hex[:8]}"
    seeds_mod.create_thought_seed(
        seed_id=sid, stream_id="g", seed_type="self_observation",
        event="existing", intensity=50, confidence=50,
        evidence_json="[]", reasoning="test", potential_impact_json="{}",
    )
    conn_mod._get_conn().execute(
        "UPDATE soul_thought_seeds SET created_at = ? WHERE seed_id = ?",
        (today, sid),
    )
    conn_mod._get_conn().commit()

    # 验证计数 = 1
    assert seeds_mod.count_self_observation_seeds_created_today() == 1

    plugin = _mock_plugin_with_cap(1)
    ev = _import_soul_submodule("components.reflection_evaluator")
    pending = _MockPending()
    result = asyncio.run(ev._maybe_create_self_observation_seed(
        plugin, {"name": "test", "thought": "x", "spectrum_impact": {}, "confidence": 50},
        pending, [], [], "reason text",
    ))
    # 被 cap 拦截直接返回 None（非 seed_manager 异常）
    assert result is None


def test_daily_cap_not_reached_allows(soul_db: Any) -> None:
    """cap=2 且今日只有 1 个 self_observation → 不拦截（走 seed_manager 异常回落）。"""
    seeds_mod = _import_soul_submodule("models.seeds")
    conn_mod = _import_soul_submodule("models._conn")

    today = conn_mod._dt_to_str(datetime.now())
    sid = f"existing_{uuid.uuid4().hex[:8]}"
    seeds_mod.create_thought_seed(
        seed_id=sid, stream_id="g", seed_type="self_observation",
        event="existing", intensity=50, confidence=50,
        evidence_json="[]", reasoning="test", potential_impact_json="{}",
    )
    conn_mod._get_conn().execute(
        "UPDATE soul_thought_seeds SET created_at = ? WHERE seed_id = ?",
        (today, sid),
    )
    conn_mod._get_conn().commit()

    assert seeds_mod.count_self_observation_seeds_created_today() == 1

    plugin = _mock_plugin_with_cap(2)
    ev = _import_soul_submodule("components.reflection_evaluator")
    pending = _MockPending()
    result = asyncio.run(ev._maybe_create_self_observation_seed(
        plugin, {"name": "test", "thought": "x", "spectrum_impact": {}, "confidence": 50},
        pending, [], [], "reason text",
    ))
    # cap=2, today=1, 不拦截 → 继续到 seed_manager 异常 → None
    assert result is None
