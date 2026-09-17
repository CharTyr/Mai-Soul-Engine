"""运行模式对「改写人格」类命令的闸门。

observe / off 模式下不得通过命令改写正式人格（trait 生命周期、槽位、全局提升、
种子接纳）。否则「观察模式」只是名义上的——一次误操作照样改人格。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .conftest import _import_soul_submodule


def _plugin(mode: str) -> Any:
    sent: list[str] = []

    class _Ctx:
        class _Send:
            async def text(self, text: str, stream_id: str = "") -> None:
                sent.append(text)

        send = _Send()

    return SimpleNamespace(
        config=SimpleNamespace(
            plugin=SimpleNamespace(enabled=mode == "apply", mode=mode),
            admin=SimpleNamespace(admin_user_id="qq:admin123"),
            thought_cabinet=SimpleNamespace(
                enabled=True,
                fermentation_enabled=False,
                auto_dedup_enabled=True,
                auto_dedup_threshold=0.78,
            ),
        ),
        ctx=_Ctx(),
        _plugin_dir="/tmp",
        _sent=sent,
    )


def _kwargs(text: str) -> dict:
    return {
        "platform": "qq",
        "user_id": "admin123",
        "text": text,
        "message": {
            "platform": "qq",
            "user_info": {"user_id": "admin123"},
            "processed_plain_text": text,
        },
    }


def test_check_mutation_mode_allows_apply() -> None:
    """apply 模式放行。"""
    su = _import_soul_submodule("utils.spectrum_utils")
    ok, _err = su.check_mutation_mode(_plugin("apply"), "审核思维种子")
    assert ok is True


def test_check_mutation_mode_blocks_observe() -> None:
    """observe 模式拒绝改写人格，并说明如何开启。"""
    su = _import_soul_submodule("utils.spectrum_utils")
    ok, err = su.check_mutation_mode(_plugin("observe"), "审核思维种子")

    assert ok is False
    assert "observe" in err
    assert "apply" in err, "提示里必须给出解决办法"


def test_check_mutation_mode_blocks_off() -> None:
    """off 模式同样拒绝。"""
    su = _import_soul_submodule("utils.spectrum_utils")
    ok, _err = su.check_mutation_mode(_plugin("off"), "审核思维种子")
    assert ok is False


def test_seed_approve_blocked_in_observe_mode(soul_db: Any) -> None:
    """observe 模式下 /soul_approve 不得触发内化。"""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    tc = _import_soul_submodule("components.thought_commands")
    plugin = _plugin("observe")

    manager = MagicMock()
    manager.get_seed_by_id = AsyncMock(return_value={"seed_id": "s1", "status": "pending"})
    engine = MagicMock()
    engine.internalize_seed = AsyncMock(return_value={"success": True})

    seed_manager_mod = _import_soul_submodule("thought.seed_manager")
    engine_mod = _import_soul_submodule("thought.internalization_engine")
    with patch.object(seed_manager_mod, "ThoughtSeedManager") as mgr_cls, \
         patch.object(engine_mod, "InternalizationEngine", return_value=engine):
        mgr_cls.from_plugin_config.return_value = manager
        asyncio.run(tc.handle_seed_approve(plugin, "g", **_kwargs("/soul_approve s1")))

    assert engine.internalize_seed.await_count == 0, "observe 模式不得内化"
    assert any("apply" in m for m in plugin._sent), "必须明确提示需要 apply 模式"


def test_trait_slot_blocked_in_observe_mode(soul_db: Any) -> None:
    """observe 模式下 /soul_slot 不得改槽位。"""
    import asyncio

    tc = _import_soul_submodule("components.thought_commands")
    plugin = _plugin("observe")

    result = asyncio.run(tc.handle_trait_slot(plugin, "g", **_kwargs("/soul_slot t1 3")))

    assert result[0] is True
    assert any("apply" in m for m in plugin._sent)


def test_trait_disable_blocked_in_observe_mode(soul_db: Any) -> None:
    """observe 模式下 /soul_trait_disable 不得改生命周期。"""
    import asyncio

    tc = _import_soul_submodule("components.thought_commands")
    plugin = _plugin("observe")

    asyncio.run(tc.handle_trait_disable(plugin, "g", **_kwargs("/soul_trait_disable t1")))

    assert any("apply" in m for m in plugin._sent)


def test_reset_blocked_in_off_mode(soul_db: Any) -> None:
    """off 模式下 /soul_reset 不得清光谱。"""
    import asyncio

    rc = _import_soul_submodule("components.reset_command")
    plugin = _plugin("off")
    plugin._reset_confirm_ts = {}

    asyncio.run(rc.handle_reset(plugin, "g", **_kwargs("/soul_reset")))

    assert any("apply" in m for m in plugin._sent)
    assert plugin._reset_confirm_ts == {}, "被拦下时不应记录确认状态"


def test_readonly_commands_still_work_in_observe_mode(soul_db: Any) -> None:
    """只读命令在 observe 模式下必须照常可用（观察就是要看）。"""
    import asyncio

    tc = _import_soul_submodule("components.thought_commands")
    plugin = _plugin("observe")

    asyncio.run(tc.handle_traits_list(plugin, "g", **_kwargs("/soul_traits")))

    assert plugin._sent, "只读命令应有输出"
    assert not any("apply" in m for m in plugin._sent), "只读命令不应被模式闸门拦"
