"""分用途投递 + 提示词预算（方案 §4.1）。

- Planner 视图：立场光谱 + 分层/情绪/图谱 + 固化观点 + 自评自查
- Replyer 视图：只给本次相关观点与表达倾向；**不落快照、不打冷却**
- 体积控制走估算 token 预算（顺序即优先级，从尾部丢，丢弃数记日志）
- 同一块重复调用不得叠加（宿主 replyer 每次重试都会调 hook）
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .conftest import _import_soul_submodule

_TS = "2026-09-17T10:00:00+00:00"


def _sys_item(text: str) -> dict:
    return {
        "item_type": "SystemMessageItem",
        "meta": {"item_id": "sys1", "logical_turn_id": None, "timestamp": _TS},
        "parts": [{"type": "text", "text": text}],
    }


def _user_item(text: str) -> dict:
    return {
        "item_type": "UserMessageItem",
        "meta": {"item_id": "u1", "logical_turn_id": None, "timestamp": _TS},
        "parts": [{"type": "text", "text": text}],
    }


def _host_kwargs(**extra: Any) -> dict:
    kw = {
        "hook_name": "maisaka.planner.before_request",
        "items": [_sys_item("你是 Mai，一个友善的群聊助手。\n"), _user_item("要不要直说这件事")],
        "item_schema_version": 3,
        "session_id": "qq-123-group",
    }
    kw.update(extra)
    return kw


def _block_text(kwargs: dict) -> str:
    parts = kwargs["items"][0].get("parts") or []
    return "\n".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))


class _WVStub:
    """P1 桩：分层摘要/情绪行可控，用来区分「planner 该有的」与「replyer 不该有的」。"""

    def build_layer_trait_summary(self, *a: Any, **k: Any) -> str:
        return "【层摘要】价值观层：真诚偏高"

    def mood_prompt_lines(self) -> list[str]:
        return ["【情绪】语气可稍偏积极"]

    def build_graph_hint(self, *a: Any, **k: Any) -> str:
        return "【图谱】相关观点：X → Y"


class _Plugin:
    """最小插件桩：只满足注入器读取的配置与属性。"""

    def __init__(self, **injection_overrides: Any) -> None:
        schema = _import_soul_submodule("plugin_ui_schema")
        cfg = schema.MaiSoulEngineConfig()
        cfg.plugin.mode = "apply"
        cfg.injection.scope = "global"
        for key, value in injection_overrides.items():
            setattr(cfg.injection, key, value)
        self.config = cfg
        self._plugin_dir = Path("/tmp/soul-replyer-view-test")
        self._data_dir = self._plugin_dir / "data"
        self._wv_service = _WVStub()
        self._wv_config_view = None

    class _Ctx:
        class _Chat:
            @staticmethod
            async def get_group_streams(**kw: Any) -> list[str]:
                return ["qq-123-group"]

            @staticmethod
            async def get_private_streams(**kw: Any) -> list[str]:
                return []

        chat = _Chat()

    ctx = _Ctx()


def _seed_traits(im: Any) -> None:
    s = im.get_or_create_spectrum("global")
    s.initialized = True  # 未初始化光谱会走「未初始化 → 不注入」的合法早退
    s.save()
    im.create_crystallized_trait(
        trait_id="trait-1", stream_id="global", seed_id="s1",
        name="直率", question="讨论要不要直说", thought="该直说时就直接说",
        tags_json='["直说"]', confidence=80, evidence_json="[]", spectrum_impact_json="{}",
    )


# ─── 分用途：内容差异 ───────────────────────────────────────────────


def test_planner_and_replyer_views_differ(soul_db: Any) -> None:
    """同一份数据下，两个视图必须不同：replyer 不含分层摘要/图谱，且预算更小。"""
    inj = _import_soul_submodule("components.ideology_injector")
    _seed_traits(soul_db)

    planner_kwargs = _host_kwargs()
    result_p = asyncio.run(inj.inject_ideology(_Plugin(), **planner_kwargs))
    planner_text = _block_text(result_p["modified_kwargs"])

    replyer_kwargs = _host_kwargs(_purpose="replyer")
    result_r = asyncio.run(inj.inject_ideology(_Plugin(), **replyer_kwargs))
    replyer_text = _block_text(result_r["modified_kwargs"])

    assert "层摘要" in planner_text and "图谱" in planner_text
    assert "层摘要" not in replyer_text, "replyer 视图混进了决策材料（分层摘要）"
    assert "图谱" not in replyer_text, "replyer 视图混进了决策材料（图谱）"
    # 两边都必须有本次观点
    assert "trait-1" in planner_text and "trait-1" in replyer_text


def test_replyer_view_does_not_write_snapshot(soul_db: Any) -> None:
    """replyer 视图不得落注入快照（快照锚点属于 planner，多写会让歧义判定恒真）。"""
    inj = _import_soul_submodule("components.ideology_injector")
    sr = _import_soul_submodule("models.self_reflection")
    _seed_traits(soul_db)
    plugin = _Plugin()
    plugin.config.self_reflection.enabled = True

    asyncio.run(inj.inject_ideology(plugin, **_host_kwargs(_purpose="replyer")))

    conn = _import_soul_submodule("models._conn")._get_conn()
    n = conn.execute("SELECT COUNT(*) AS n FROM soul_injection_snapshots").fetchone()["n"]
    assert n == 0, "replyer 视图落了快照"


def test_replyer_view_does_not_mark_cooldown(soul_db: Any) -> None:
    """replyer 视图不得消耗 planner 的冷却状态。"""
    inj = _import_soul_submodule("components.ideology_injector")
    inj._RECENT_TRAIT_INJECTION.clear()  # 模块级状态：先清掉其他测试的残留
    _seed_traits(soul_db)
    plugin = _Plugin()

    asyncio.run(inj.inject_ideology(plugin, **_host_kwargs(_purpose="replyer")))

    assert not inj._RECENT_TRAIT_INJECTION.get("qq-123-group"), (
        "replyer 视图把 trait 打进了冷却（会挤掉 planner 的选择）"
    )


# ─── 预算 ───────────────────────────────────────────────────────────


def test_replyer_budget_truncates_to_configured_limit(soul_db: Any) -> None:
    """预算生效：超出预算的观点被丢弃（顺序即优先级，从尾部丢）。"""
    inj = _import_soul_submodule("components.ideology_injector")
    im = soul_db
    _spectrum = im.get_or_create_spectrum("global")
    _spectrum.initialized = True
    _spectrum.save()
    for i in range(10):
        im.create_crystallized_trait(
            trait_id=f"trait-{i}", stream_id="global", seed_id=f"s{i}",
            name=f"观点{i}", question=f"问题{i}", thought=f"这是第{i}条观点内容",
            tags_json='["直说"]', confidence=80, evidence_json="[]", spectrum_impact_json="{}",
        )

    plugin = _Plugin(replyer_token_budget=60, max_traits=10)
    result = asyncio.run(inj.inject_ideology(plugin, **_host_kwargs(_purpose="replyer")))
    text = _block_text(result["modified_kwargs"])

    kept = text.count("- (trait-")
    assert 0 < kept < 10, f"预算没有裁剪（保留 {kept} 条）"


def test_budget_zero_keeps_at_least_one_line(soul_db: Any) -> None:
    """预算为 0 时至少保留第一条：把「有内容」裁成「没内容」更难排查。"""
    inj = _import_soul_submodule("components.ideology_injector")
    _seed_traits(soul_db)
    plugin = _Plugin(replyer_token_budget=0)
    result = asyncio.run(inj.inject_ideology(plugin, **_host_kwargs(_purpose="replyer")))
    assert "trait-1" in _block_text(result["modified_kwargs"])


# ─── 幂等：同一块不叠加 ─────────────────────────────────────────────


def test_repeated_append_does_not_stack_blocks() -> None:
    """同一块重复合并不得叠加（宿主 replyer 每次重试都会重新调用 hook）。"""
    hp = _import_soul_submodule("utils.host_prompt_items")
    block = "【动态层】这是同一份注入块，内容足够长以通过探针匹配。"
    kwargs = _host_kwargs()

    once, strategy1 = hp.append_block_to_first_system(kwargs, block)
    assert strategy1 == hp.STRATEGY_APPENDED

    twice, strategy2 = hp.append_block_to_first_system(once, block)
    assert strategy2 == hp.STRATEGY_ALREADY_PRESENT
    assert _block_text(twice).count(block) == 1, "同一块被叠加了多次"


def test_replyer_hook_respects_switch(soul_db: Any) -> None:
    """关闭 replyer 注入时不改提示项。"""
    plugin_mod = _import_soul_submodule("plugin")
    p = plugin_mod.MaiSoulEnginePlugin()
    p.set_plugin_config(_config_dict(replyer_injection_enabled=False))
    result = asyncio.run(p.soul_replyer_injector(**_host_kwargs()))
    assert result.get("modified_kwargs") is None


def _config_dict(**injection_overrides: Any) -> dict:
    schema = _import_soul_submodule("plugin_ui_schema")
    cfg = schema.MaiSoulEngineConfig()
    for key, value in injection_overrides.items():
        setattr(cfg.injection, key, value)
    return cfg.model_dump()
