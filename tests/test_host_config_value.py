"""宿主 config.get 返回值契约测试。

契约来源（宿主只读参考）：
- SDK `maibot_sdk/context.py:271 _normalize_capability_result` 会按
  `_CAPABILITY_RESULT_KEYS` 把 `config.get` 的返回**先解包**：
  `{"success": True, "value": X}` → 插件直接拿到 `X`（裸值，不是 dict）。
- 故插件侧读取必须同时接受「裸值」与「旧式 success/value 包装」，
  并且不得把「解包后为空的字符串」误判成失败。
"""

from __future__ import annotations

from .conftest import _import_soul_submodule


def test_reads_unwrapped_plain_string() -> None:
    """SDK 2.x 解包后直接是字符串。"""
    cfg = _import_soul_submodule("utils.host_config")
    assert cfg.read_config_value("3659592968") == "3659592968"


def test_reads_unwrapped_empty_string() -> None:
    """解包后是空串 → 返回空串，不得抛错。"""
    cfg = _import_soul_submodule("utils.host_config")
    assert cfg.read_config_value("") == ""


def test_reads_wrapped_success_value() -> None:
    """旧式 {success, value} 包装仍受支持。"""
    cfg = _import_soul_submodule("utils.host_config")
    assert cfg.read_config_value({"success": True, "value": "温柔体贴"}) == "温柔体贴"


def test_reads_wrapped_value_without_success_key() -> None:
    """带 value 但无 success 键的 dict（部分能力形状）也能取值。"""
    cfg = _import_soul_submodule("utils.host_config")
    assert cfg.read_config_value({"value": "段子手"}) == "段子手"


def test_failed_envelope_returns_default() -> None:
    """success=False → 视为无值。"""
    cfg = _import_soul_submodule("utils.host_config")
    assert cfg.read_config_value({"success": False, "error": "boom"}) == ""


def test_none_and_non_string_return_default() -> None:
    """None / 列表 / 未预期类型 → 默认值。"""
    cfg = _import_soul_submodule("utils.host_config")
    assert cfg.read_config_value(None) == ""
    assert cfg.read_config_value([1, 2]) == ""
    assert cfg.read_config_value(None, default="fallback") == "fallback"


def test_numbers_are_stringified() -> None:
    """数字型配置（如 bot.qq_account 被读成 int）需转字符串。"""
    cfg = _import_soul_submodule("utils.host_config")
    assert cfg.read_config_value(3659592968) == "3659592968"
    assert cfg.read_config_value(True) == "True"


def test_whitespace_is_stripped() -> None:
    """值两端空白剥离。"""
    cfg = _import_soul_submodule("utils.host_config")
    assert cfg.read_config_value("  温柔  ") == "温柔"
    assert cfg.read_config_value({"success": True, "value": "  温柔  "}) == "温柔"
