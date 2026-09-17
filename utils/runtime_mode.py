"""运行模式：关闭 / 观察 / 应用。

一个 `enabled` 布尔承担不了三种语义——「关掉」和「看看就好」和「真的生效」
是三种不同的运行状态。本模块把它们拆开，作为唯一的模式判定入口。

| 模式 | 注入回复 | 后台学习 | 改写正式人格 | 管理员接纳 |
|---|---|---|---|---|
| off      | ✗ | ✗ | ✗ | ✗ |
| observe  | ✗ | ✓ | ✗ | ✗ |
| apply    | ✓ | ✓ | ✓ | ✓ |

**旧配置不隐式进入 apply**：`mode` 未显式配置时，`enabled=true` 只映射到
`observe`（学习继续、但既不注入也不改人格），并标记 `migrated_from_legacy`
供上层提示操作者显式选模式。否则一次升级就会让插件突然改写人格并影响真实回复。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ALL_MODES",
    "DEFAULT_MODE",
    "MODE_APPLY",
    "MODE_OBSERVE",
    "MODE_OFF",
    "RuntimeMode",
    "describe_runtime_mode",
    "resolve_runtime_mode",
]

MODE_OFF = "off"
MODE_OBSERVE = "observe"
MODE_APPLY = "apply"
ALL_MODES: tuple[str, ...] = (MODE_OFF, MODE_OBSERVE, MODE_APPLY)
DEFAULT_MODE = MODE_OFF


@dataclass(frozen=True)
class RuntimeMode:
    """解析后的运行时闸门。"""

    mode: str = DEFAULT_MODE
    injection_enabled: bool = False
    learning_enabled: bool = False
    mutation_allowed: bool = False
    acceptance_allowed: bool = False
    migrated_from_legacy: bool = False
    invalid_mode: bool = False


def _flag(source: Any, name: str, default: Any = None) -> Any:
    value = getattr(source, name, None)
    return default if value is None else value


def resolve_runtime_mode(plugin_config: Any) -> RuntimeMode:
    """把插件配置解析成运行时闸门。

    解析顺序：
      1. ``plugin.mode`` 显式且合法 → 采用
      2. ``plugin.mode`` 非法（如拼错）→ 保守回落 observe/off，并标记 invalid_mode
      3. 未配置 mode → 旧兼容：``enabled=true`` 映射 observe，否则 off
    """
    plugin_section = _flag(plugin_config, "plugin", None)
    if plugin_section is None:
        return RuntimeMode(mode=DEFAULT_MODE)

    raw_mode = str(_flag(plugin_section, "mode", "") or "").strip().lower()
    legacy_enabled = bool(_flag(plugin_section, "enabled", False))

    migrated = False
    invalid = False
    if raw_mode in ALL_MODES:
        mode = raw_mode
    else:
        # 未配置或拼错：保守处理，绝不落 apply
        invalid = bool(raw_mode)
        mode = MODE_OBSERVE if legacy_enabled else MODE_OFF
        migrated = True

    if mode == MODE_APPLY:
        return RuntimeMode(
            mode=mode,
            injection_enabled=True,
            learning_enabled=True,
            mutation_allowed=True,
            acceptance_allowed=True,
            migrated_from_legacy=migrated,
            invalid_mode=invalid,
        )
    if mode == MODE_OBSERVE:
        return RuntimeMode(
            mode=mode,
            injection_enabled=False,
            learning_enabled=True,
            mutation_allowed=False,
            acceptance_allowed=False,
            migrated_from_legacy=migrated,
            invalid_mode=invalid,
        )
    return RuntimeMode(
        mode=MODE_OFF,
        migrated_from_legacy=migrated,
        invalid_mode=invalid,
    )


def describe_runtime_mode(plugin_config: Any) -> str:
    """给看板/健康输出用的一行描述。"""
    resolved = resolve_runtime_mode(plugin_config)
    lines = [
        f"运行模式：{resolved.mode}",
        f"- 注入回复：{'开' if resolved.injection_enabled else '关'}",
        f"- 后台学习：{'开' if resolved.learning_enabled else '关'}",
        f"- 改写人格：{'允许' if resolved.mutation_allowed else '禁止'}",
        f"- 管理员接纳：{'允许' if resolved.acceptance_allowed else '禁止'}",
    ]
    if resolved.invalid_mode:
        lines.append("- ⚠️ 配置里的 mode 值无法识别，已按最小权限回落")
    elif resolved.migrated_from_legacy:
        lines.append(
            "- ⚠️ 未显式配置 mode（旧配置兼容）：当前按 observe 运行；"
            "要真正生效请显式设置 mode = \"apply\""
        )
    return "\n".join(lines)
