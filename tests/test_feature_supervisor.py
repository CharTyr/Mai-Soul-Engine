"""测试 _compute_desired_tasks 纯函数（后台任务期望状态计算）。

覆盖：总开关、逐任务开关、发酵开关组合。
"""

from __future__ import annotations

from typing import Any

from .conftest import _import_soul_submodule


def _build_config(**overrides: Any) -> Any:
    """用 SimpleNamespace 构建模拟配置对象。"""
    from types import SimpleNamespace

    def _b(v: Any) -> bool:
        return bool(v)

    # 默认值：所有子开关关闭，plugin.enabled=True
    defaults = {
        "plugin": SimpleNamespace(enabled=True),
        "evolution": SimpleNamespace(evolution_enabled=False),
        "notion": SimpleNamespace(enabled=False),
        "self_reflection": SimpleNamespace(enabled=False),
        "thought_cabinet": SimpleNamespace(enabled=False, fermentation_enabled=False),
    }
    # 逐层应用 overrides
    for key, val in overrides.items():
        parts = key.split(".", 1)
        if len(parts) == 2:
            section, attr = parts
            if section in defaults:
                if isinstance(val, dict):
                    for k2, v2 in val.items():
                        setattr(defaults[section], k2, v2)
                else:
                    setattr(defaults[section], attr, val)
            else:
                defaults[section] = SimpleNamespace(**{attr: val})
        elif len(parts) == 1:
            if isinstance(val, dict):
                defaults[key] = SimpleNamespace(**val)
            else:
                defaults[key] = SimpleNamespace(enabled=val)

    return SimpleNamespace(**defaults)


def _get_compute() -> Any:
    pkg = _import_soul_submodule("plugin")
    return pkg.MaiSoulEnginePlugin._compute_desired_tasks


# ─── 全关（默认） ──────────────────────────────────────────


def test_all_disabled() -> None:
    """所有子开关默认关闭 → 四个 desired 全 false。"""
    fn = _get_compute()
    cfg = _build_config()
    result = fn(cfg)
    assert result == {"evolution": False, "notion": False, "reflection": False, "fermentation": False}


# ─── 全开 ──────────────────────────────────────────────────


def test_all_enabled() -> None:
    """所有子开关开启 + plugin.enabled=True → 四个 desired 全 true。"""
    fn = _get_compute()
    cfg = _build_config(
        plugin={"enabled": True},
        evolution={"evolution_enabled": True},
        notion={"enabled": True},
        self_reflection={"enabled": True},
        thought_cabinet={"enabled": True, "fermentation_enabled": True},
    )
    result = fn(cfg)
    assert result == {"evolution": True, "notion": True, "reflection": True, "fermentation": True}


# ─── 总开关 ────────────────────────────────────────────────


def test_plugin_disabled_kills_all() -> None:
    """plugin.enabled=False → 四个 desired 全 false，即使子开关全开。"""
    fn = _get_compute()
    cfg = _build_config(
        plugin={"enabled": False},
        evolution={"evolution_enabled": True},
        notion={"enabled": True},
        self_reflection={"enabled": True},
        thought_cabinet={"enabled": True, "fermentation_enabled": True},
    )
    result = fn(cfg)
    assert result == {"evolution": False, "notion": False, "reflection": False, "fermentation": False}


# ─── 逐个独立开关 ──────────────────────────────────────────


def test_evolution_only() -> None:
    fn = _get_compute()
    cfg = _build_config(evolution={"evolution_enabled": True})
    result = fn(cfg)
    assert result["evolution"] is True
    assert result["notion"] is False
    assert result["reflection"] is False
    assert result["fermentation"] is False


def test_notion_only() -> None:
    fn = _get_compute()
    cfg = _build_config(notion={"enabled": True})
    result = fn(cfg)
    assert result["evolution"] is False
    assert result["notion"] is True
    assert result["reflection"] is False
    assert result["fermentation"] is False


def test_reflection_only() -> None:
    fn = _get_compute()
    cfg = _build_config(self_reflection={"enabled": True})
    result = fn(cfg)
    assert result["evolution"] is False
    assert result["notion"] is False
    assert result["reflection"] is True
    assert result["fermentation"] is False


# ─── 发酵开关组合 ────────────────────────────────────────


def test_fermentation_requires_both() -> None:
    """发酵需要 thought_cabinet.enabled + fermentation_enabled 同时为 True。"""
    fn = _get_compute()
    # case 1: 只有 thought_cabinet.enabled
    cfg = _build_config(thought_cabinet={"enabled": True, "fermentation_enabled": False})
    assert fn(cfg)["fermentation"] is False
    # case 2: 只有 fermentation_enabled
    cfg = _build_config(thought_cabinet={"enabled": False, "fermentation_enabled": True})
    assert fn(cfg)["fermentation"] is False
    # case 3: 两者都开
    cfg = _build_config(thought_cabinet={"enabled": True, "fermentation_enabled": True})
    assert fn(cfg)["fermentation"] is True


def test_fermentation_default_false_does_not_break() -> None:
    """fermentation_enabled 默认 False 时 getattr fallback 正确。"""
    fn = _get_compute()
    # 不设 fermentation_enabled 属性
    from types import SimpleNamespace
    cfg = _build_config()
    cfg.thought_cabinet = SimpleNamespace(enabled=True)
    result = fn(cfg)
    assert result["fermentation"] is False


# ─── 总开关 + 发酵组合 ────────────────────────────────────


def test_plugin_disabled_fermentation_off() -> None:
    """plugin.enabled=False 时发酵不会启动，即使发酵开关全开。"""
    fn = _get_compute()
    cfg = _build_config(
        plugin={"enabled": False},
        thought_cabinet={"enabled": True, "fermentation_enabled": True},
    )
    assert fn(cfg)["fermentation"] is False
