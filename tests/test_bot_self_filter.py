"""P0: bot 自消息过滤测试（防自指泄漏）。

验证演化分析时 bot 自身消息被排除：
- is_bot_self_message 匹配逻辑
- filter_messages_for_evolution 短路优先于 monitored_users 白名单
- 未配置 bot_self_id 时回退到 excluded_users 语义
- 全未配置时泄漏（说明为何要警告）

从宿主仓根运行：``uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/test_bot_self_filter.py -q``
"""

from __future__ import annotations

from .conftest import _import_soul_submodule


def _msg(platform: str, user_id: str, text: str = "hi") -> dict:
    return {
        "user_info": {
            "platform": platform,
            "user_id": user_id,
            "user_nickname": user_id,
        },
        "processed_plain_text": text,
    }


# ─── is_bot_self_message 匹配逻辑 ──────────────────────────────────


def test_is_bot_self_match() -> None:
    su = _import_soul_submodule("utils.spectrum_utils")
    assert su.is_bot_self_message("qq", "999", ["qq:999"]) is True


def test_is_bot_self_no_match() -> None:
    su = _import_soul_submodule("utils.spectrum_utils")
    assert su.is_bot_self_message("qq", "888", ["qq:999"]) is False


def test_is_bot_self_empty_list() -> None:
    su = _import_soul_submodule("utils.spectrum_utils")
    assert su.is_bot_self_message("qq", "999", []) is False
    assert su.is_bot_self_message("qq", "999", None) is False


def test_is_bot_self_multi_platform() -> None:
    su = _import_soul_submodule("utils.spectrum_utils")
    assert su.is_bot_self_message("qq", "999", ["tg:1", "qq:999"]) is True
    assert su.is_bot_self_message("tg", "1", ["tg:1", "qq:999"]) is True
    assert su.is_bot_self_message("qq", "1", ["tg:1", "qq:999"]) is False


# ─── filter_messages_for_evolution 短路与回退 ──────────────────────


def test_filter_excludes_bot_self_when_whitelist_empty() -> None:
    """核心修复：monitored_users 空（全员计入）时，bot 自消息仍被排除。"""
    su = _import_soul_submodule("utils.spectrum_utils")
    msgs = [_msg("qq", "999", "我说句话"), _msg("qq", "111", "用户发言")]
    cfg = {"monitored_users": [], "excluded_users": [], "bot_self_id": ["qq:999"]}
    out = su.filter_messages_for_evolution(msgs, cfg)
    assert len(out) == 1
    assert out[0]["user_info"]["user_id"] == "111"


def test_filter_bot_self_short_circuits_whitelist() -> None:
    """bot 自身检查优先于 monitored_users：即使 bot 在白名单里也排除。"""
    su = _import_soul_submodule("utils.spectrum_utils")
    msgs = [_msg("qq", "999", "bot"), _msg("qq", "111", "用户")]
    cfg = {
        "monitored_users": ["qq:999", "qq:111"],
        "excluded_users": [],
        "bot_self_id": ["qq:999"],
    }
    out = su.filter_messages_for_evolution(msgs, cfg)
    assert len(out) == 1
    assert out[0]["user_info"]["user_id"] == "111"


def test_filter_falls_back_to_excluded_users_when_bot_self_unset() -> None:
    """bot_self_id 未配置时，excluded_users 仍能排除 bot。"""
    su = _import_soul_submodule("utils.spectrum_utils")
    msgs = [_msg("qq", "999", "bot"), _msg("qq", "111", "用户")]
    cfg = {"monitored_users": [], "excluded_users": ["qq:999"], "bot_self_id": []}
    out = su.filter_messages_for_evolution(msgs, cfg)
    assert len(out) == 1
    assert out[0]["user_info"]["user_id"] == "111"


def test_filter_no_bot_config_leaks_self() -> None:
    """回归基线：bot_self_id 与 excluded_users 都空时，bot 自消息会泄漏。

    这正是要在演化任务里发警告的场景——证明过滤函数本身不会"猜"谁是 bot，
    必须由配置显式声明。
    """
    su = _import_soul_submodule("utils.spectrum_utils")
    msgs = [_msg("qq", "999", "bot"), _msg("qq", "111", "用户")]
    cfg = {"monitored_users": [], "excluded_users": [], "bot_self_id": []}
    out = su.filter_messages_for_evolution(msgs, cfg)
    assert len(out) == 2
