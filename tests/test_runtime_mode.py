"""运行模式：关闭 / 观察 / 应用。

一个 `enabled` 布尔扛不住三种语义。拆开：

| 模式 | 注入回复 | 后台学习 | 改写正式人格 | 管理员接纳 |
|---|---|---|---|---|
| off（默认） | ✗ | ✗ | ✗ | ✗ |
| observe | ✗ | ✓ | ✗ | ✗ |
| apply | ✓ | ✓ | ✓ | ✓ |

关键约束：**旧配置 `enabled=true` 不得隐式进入 apply**。升级后若没有显式写
`mode`，一律按 observe 处理并给出提示——否则一次升级就会让插件突然开始改写
人格并影响真实回复。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .conftest import _import_soul_submodule


def _mode() -> Any:
    return _import_soul_submodule("utils.runtime_mode")


def _cfg(*, mode: str = "", enabled: bool = False) -> Any:
    return SimpleNamespace(plugin=SimpleNamespace(mode=mode, enabled=enabled))


# ─── 显式模式 ───────────────────────────────────────────────────────


def test_explicit_off_mode_disables_everything() -> None:
    """off：不注入、不学习、不改人格、不接纳。"""
    m = _mode()
    resolved = m.resolve_runtime_mode(_cfg(mode="off", enabled=True))

    assert resolved.mode == m.MODE_OFF
    assert resolved.injection_enabled is False
    assert resolved.learning_enabled is False
    assert resolved.mutation_allowed is False
    assert resolved.acceptance_allowed is False


def test_explicit_observe_mode_learns_without_mutating() -> None:
    """observe：可以学习与生成候选，但不得注入、不得改正式人格。"""
    m = _mode()
    resolved = m.resolve_runtime_mode(_cfg(mode="observe", enabled=True))

    assert resolved.mode == m.MODE_OBSERVE
    assert resolved.injection_enabled is False, "观察模式不得影响真实回复"
    assert resolved.mutation_allowed is False, "观察模式不得改写已接纳人格"
    assert resolved.acceptance_allowed is False
    assert resolved.learning_enabled is True


def test_explicit_apply_mode_enables_all() -> None:
    """apply：四类能力全开。"""
    m = _mode()
    resolved = m.resolve_runtime_mode(_cfg(mode="apply", enabled=False))

    assert resolved.mode == m.MODE_APPLY
    assert resolved.injection_enabled is True
    assert resolved.learning_enabled is True
    assert resolved.mutation_allowed is True
    assert resolved.acceptance_allowed is True


def test_mode_is_case_insensitive_and_trimmed() -> None:
    """模式名大小写与空白不敏感。"""
    m = _mode()
    assert m.resolve_runtime_mode(_cfg(mode="  APPLY ")).mode == m.MODE_APPLY


# ─── 旧配置兼容（关键：不得隐式 apply） ─────────────────────────────


def test_legacy_enabled_true_does_not_become_apply() -> None:
    """旧配置 enabled=true 且未写 mode → observe，不是 apply。"""
    m = _mode()
    resolved = m.resolve_runtime_mode(_cfg(mode="", enabled=True))

    assert resolved.mode == m.MODE_OBSERVE
    assert resolved.injection_enabled is False
    assert resolved.mutation_allowed is False
    assert resolved.migrated_from_legacy is True, "需向操作者提示已降级为观察"


def test_legacy_enabled_false_is_off() -> None:
    """旧配置 enabled=false 且未写 mode → off。"""
    m = _mode()
    resolved = m.resolve_runtime_mode(_cfg(mode="", enabled=False))

    assert resolved.mode == m.MODE_OFF
    assert resolved.migrated_from_legacy is True


def test_unknown_mode_falls_back_conservatively() -> None:
    """未知模式名不得被当成 apply（保守回落到 observe/off，并提示）。"""
    m = _mode()
    resolved = m.resolve_runtime_mode(_cfg(mode="prod", enabled=True))

    assert resolved.mode == m.MODE_OBSERVE
    assert resolved.injection_enabled is False
    assert resolved.invalid_mode is True


def test_missing_config_attributes_do_not_crash() -> None:
    """配置对象缺字段 → 视为 off，而不是抛异常。"""
    m = _mode()
    resolved = m.resolve_runtime_mode(SimpleNamespace(plugin=SimpleNamespace()))
    assert resolved.mode == m.MODE_OFF


# ─── 描述与看板 ─────────────────────────────────────────────────────


def test_describe_reports_mode_and_gates() -> None:
    """看板/健康输出需能区分三种模式与各自闸门。"""
    m = _mode()
    text = m.describe_runtime_mode(_cfg(mode="observe"))

    assert "observe" in text
    assert "注入" in text


def test_allowed_states_are_enumerated() -> None:
    """模式集合显式声明，便于配置校验与文档生成。"""
    m = _mode()
    assert set(m.ALL_MODES) == {m.MODE_OFF, m.MODE_OBSERVE, m.MODE_APPLY}
    assert m.MODE_OFF == m.DEFAULT_MODE, "默认必须是 off"


# ─── 真实默认值的实际效果（文档必须与之一致） ────────────────────────


def test_schema_default_mode_is_off() -> None:
    """schema 默认 mode = "off"：新装/未显式配置的实例什么都不做。

    注意这意味着"旧配置 enabled=true 且 mode 从未写过"的实例（pydantic 会补
    默认值 "off"）实际落在 **off**，而不是 observe。要恢复学习但不注入，
    必须显式写 observe。文档必须按这个真实行为描述。
    """
    schema = _import_soul_submodule("plugin_ui_schema")
    section = schema.PluginSectionConfig()
    assert section.mode == "off"

    resolved = _mode().resolve_runtime_mode(
        SimpleNamespace(plugin=SimpleNamespace(mode=section.mode, enabled=True))
    )
    assert resolved.mode == "off"
    assert resolved.injection_enabled is False
    assert resolved.learning_enabled is False
    assert resolved.mutation_allowed is False


def test_empty_mode_with_legacy_enabled_maps_to_observe() -> None:
    """mode 为空串（配置文件里真的没有该键）且 enabled=true → observe。"""
    resolved = _mode().resolve_runtime_mode(
        SimpleNamespace(plugin=SimpleNamespace(mode="", enabled=True))
    )
    assert resolved.mode == "observe"
    assert resolved.migrated_from_legacy is True
