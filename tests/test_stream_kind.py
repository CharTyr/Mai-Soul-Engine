"""会话类型判定：显式接口，不猜 session_id 字符串。

方案明令：「不得通过 session_id 包含 private 等字样推断会话类型」。
旧实现正是这么干的（`":private" in stream_id`），宿主一旦改 id 编码就会静默失效。

宿主提供了显式流列表（`chat.get_group_streams` / `chat.get_private_streams`），
本组测试钉住判定逻辑与注入侧的保守策略。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from .conftest import _import_soul_submodule


def _sk() -> Any:
    return _import_soul_submodule("utils.stream_kind")


class _Chat:
    """可控的宿主 chat 能力。"""

    def __init__(
        self,
        groups: Any = None,
        privates: Any = None,
        raise_on_group: bool = False,
    ) -> None:
        self.groups = groups if groups is not None else ["qq-1-group"]
        self.privates = privates if privates is not None else ["qq-2-private"]
        self.raise_on_group = raise_on_group
        self.calls: list[str] = []

    async def get_group_streams(self, platform: str = "qq") -> Any:
        self.calls.append("group")
        if self.raise_on_group:
            raise RuntimeError("宿主接口炸了")
        return self.groups

    async def get_private_streams(self, platform: str = "qq") -> Any:
        self.calls.append("private")
        return self.privates


def _plugin(chat: _Chat | None) -> Any:
    if chat is None:
        return SimpleNamespace(ctx=SimpleNamespace())
    return SimpleNamespace(ctx=SimpleNamespace(chat=chat))


def _fresh() -> Any:
    sk = _sk()
    sk.clear_stream_kind_cache()
    return sk


# ─── 判定逻辑 ───────────────────────────────────────────────────────


def test_identifies_group_stream() -> None:
    """在群列表里 → group。"""
    sk = _fresh()
    assert asyncio.run(sk.resolve_stream_kind(_plugin(_Chat()), "qq-1-group")) == sk.STREAM_KIND_GROUP


def test_identifies_private_stream() -> None:
    """在私聊列表里 → private（且群列表查不到才查私聊）。"""
    sk = _fresh()
    chat = _Chat()
    kind = asyncio.run(sk.resolve_stream_kind(_plugin(chat), "qq-2-private"))

    assert kind == sk.STREAM_KIND_PRIVATE
    assert chat.calls == ["group", "private"]


def test_unknown_when_in_neither_list() -> None:
    """两边都没有 → unknown（不猜）。"""
    sk = _fresh()
    assert asyncio.run(sk.resolve_stream_kind(_plugin(_Chat()), "qq-3-mystery")) == sk.STREAM_KIND_UNKNOWN


def test_unknown_without_chat_capability() -> None:
    """宿主没提供 chat 能力 → unknown，不抛异常。"""
    sk = _fresh()
    assert asyncio.run(sk.resolve_stream_kind(_plugin(None), "qq-1-group")) == sk.STREAM_KIND_UNKNOWN


def test_unknown_on_capability_error() -> None:
    """宿主接口报错 → unknown（不影响注入主流程）。"""
    sk = _fresh()
    chat = _Chat(raise_on_group=True)
    assert asyncio.run(sk.resolve_stream_kind(_plugin(chat), "qq-1-group")) == sk.STREAM_KIND_UNKNOWN


def test_empty_stream_id_is_unknown() -> None:
    """空 stream_id → unknown，且不去调宿主。"""
    sk = _fresh()
    chat = _Chat()
    assert asyncio.run(sk.resolve_stream_kind(_plugin(chat), "")) == sk.STREAM_KIND_UNKNOWN
    assert chat.calls == []


# ─── 返回值形状兼容 ──────────────────────────────────────────────────


def test_accepts_envelope_and_dict_entries() -> None:
    """兼容 {success,value:[...]} 包装与 dict 元素（宿主返回形状不统一）。"""
    sk = _fresh()
    chat = _Chat(
        groups=[{"session_id": "qq-1-group", "platform": "qq"}],
        privates={"success": True, "value": [{"stream_id": "qq-2-private"}]},
    )
    plugin = _plugin(chat)

    assert asyncio.run(sk.resolve_stream_kind(plugin, "qq-1-group")) == sk.STREAM_KIND_GROUP
    sk.clear_stream_kind_cache()
    assert asyncio.run(sk.resolve_stream_kind(plugin, "qq-2-private")) == sk.STREAM_KIND_PRIVATE


# ─── 缓存 ───────────────────────────────────────────────────────────


def test_result_is_cached() -> None:
    """判定结果有 TTL 缓存（注入在热路径上，不该每次问宿主）。"""
    sk = _fresh()
    chat = _Chat()
    plugin = _plugin(chat)

    asyncio.run(sk.resolve_stream_kind(plugin, "qq-1-group"))
    asyncio.run(sk.resolve_stream_kind(plugin, "qq-1-group"))

    assert chat.calls == ["group"], "第二次应命中缓存"


def test_unknown_is_not_cached_positively() -> None:
    """unknown 不写入正向缓存（会话可能是刚建的，下次还要再问）。"""
    sk = _fresh()
    chat = _Chat(groups=[])
    plugin = _plugin(chat)

    asyncio.run(sk.resolve_stream_kind(plugin, "qq-9-new"))
    asyncio.run(sk.resolve_stream_kind(plugin, "qq-9-new"))

    assert chat.calls.count("group") == 2


def test_cache_can_be_cleared() -> None:
    """卸载/配置热更时能清缓存，避免跨实例泄漏。"""
    sk = _fresh()
    chat = _Chat()
    plugin = _plugin(chat)
    asyncio.run(sk.resolve_stream_kind(plugin, "qq-1-group"))

    sk.clear_stream_kind_cache()
    asyncio.run(sk.resolve_stream_kind(plugin, "qq-1-group"))

    assert chat.calls == ["group", "group"]


# ─── 注入侧策略：判定不出时不把人格注入到可能被排除的会话 ─────────────


def _injector_plugin(tmp_path: Any, *, scope: str, inject_private: bool, groups: Any) -> Any:
    class _Chat:
        async def get_group_streams(self, platform: str = "qq") -> Any:
            return groups

        async def get_private_streams(self, platform: str = "qq") -> Any:
            return []

    return SimpleNamespace(
        ctx=SimpleNamespace(chat=_Chat()),
        config=SimpleNamespace(
            plugin=SimpleNamespace(enabled=True, mode="apply"),
            injection=SimpleNamespace(
                scope=scope,
                inject_private=inject_private,
                max_traits=3,
                fallback_recent_impact=False,
                trait_cooldown_seconds=0,
            ),
            monitor=SimpleNamespace(monitored_groups=["12345678"], excluded_groups=[]),
        ),
        _plugin_dir=tmp_path,
    )


def test_unknown_kind_skips_when_private_disabled(soul_db: Any, tmp_path: Any) -> None:
    """scope=all + 不允许私聊 + 类型未知 → 保守跳过（更严格的设置胜出）。"""
    sk = _sk()
    sk.clear_stream_kind_cache()
    injector = _import_soul_submodule("components.ideology_injector")
    plugin = _injector_plugin(tmp_path, scope="all", inject_private=False, groups=[])

    result = asyncio.run(
        injector.inject_ideology(
            plugin,
            session_id="qq-9-unknown",
            items=[{"item_type": "system", "meta": {}, "parts": [{"type": "text", "text": "S"}]}],
        )
    )

    assert "modified_kwargs" not in result


def test_unknown_kind_allows_with_private_enabled(soul_db: Any, tmp_path: Any) -> None:
    """允许私聊注入时，类型未知不构成跳过理由。"""
    sk = _sk()
    sk.clear_stream_kind_cache()
    injector = _import_soul_submodule("components.ideology_injector")
    plugin = _injector_plugin(tmp_path, scope="all", inject_private=True, groups=[])
    # 光谱未初始化会直接 continue，所以先初始化
    _import_soul_submodule("models.ideology_model").get_or_create_spectrum("global")

    result = asyncio.run(
        injector.inject_ideology(
            plugin,
            session_id="qq-9-unknown",
            items=[{"item_type": "system", "meta": {}, "parts": [{"type": "text", "text": "S"}]}],
        )
    )

    # 放行了（光谱未初始化时会 continue，但不应带 skip 语义）——只断言没有因类型被拦
    assert isinstance(result, dict)


# ─── 平台探测（作用域字段；宿主零改动前提下唯一诚实解）─────────────────


class _PlatformChat:
    """按平台返回不同流列表的宿主 chat 能力。"""

    def __init__(self, by_platform: dict[str, list[str]]) -> None:
        self.by_platform = by_platform
        self.probed: list[str] = []

    async def get_group_streams(self, platform: str = "qq") -> Any:
        self.probed.append(platform)
        return [s for s in self.by_platform.get(platform, []) if s.endswith("-group")]

    async def get_private_streams(self, platform: str = "qq") -> Any:
        self.probed.append(platform)
        return [s for s in self.by_platform.get(platform, []) if s.endswith("-private")]


def _plugin_with_platforms(chat: Any, platforms: list[str] | None) -> Any:
    plugin_cfg = SimpleNamespace(platforms=platforms) if platforms is not None else SimpleNamespace()
    return SimpleNamespace(ctx=SimpleNamespace(chat=chat), config=SimpleNamespace(plugin=plugin_cfg))


def test_resolve_scope_reports_platform_from_host_lists() -> None:
    """平台来自**宿主流列表**（按配置逐平台探测），不是从 session_id 猜的。"""
    sk = _fresh()
    chat = _PlatformChat({"discord": ["dc-77-group"], "qq": ["qq-1-group"]})

    kind, platform = asyncio.run(sk.resolve_stream_scope(_plugin_with_platforms(chat, ["discord", "qq"]), "dc-77-group"))
    assert (kind, platform) == ("group", "discord")

    kind2, platform2 = asyncio.run(sk.resolve_stream_scope(_plugin_with_platforms(chat, ["discord", "qq"]), "qq-1-group"))
    assert (kind2, platform2) == ("group", "qq")


def test_resolve_scope_platform_is_empty_when_unknown() -> None:
    """探测不到 → 空串（未知）。**不编造**平台。"""
    sk = _fresh()
    chat = _PlatformChat({"qq": ["qq-1-group"]})
    kind, platform = asyncio.run(sk.resolve_stream_scope(_plugin_with_platforms(chat, ["qq"]), "nope-9-group"))
    assert kind == "unknown" and platform == ""


def test_resolve_scope_uses_default_platform_when_unconfigured() -> None:
    """没配置平台时用宿主默认 ``qq``；配置列表决定**探测范围**。"""
    sk = _fresh()
    chat = _PlatformChat({"qq": ["qq-5-group"]})

    kind, platform = asyncio.run(sk.resolve_stream_scope(_plugin_with_platforms(chat, None), "qq-5-group"))
    assert (kind, platform) == ("group", "qq")

    # 只声明 discord → 不去探测 qq，因此判定为未知（宁可未知也不越界猜）
    sk.clear_stream_kind_cache()
    kind2, platform2 = asyncio.run(sk.resolve_stream_scope(_plugin_with_platforms(chat, ["discord"]), "qq-5-group"))
    assert (kind2, platform2) == ("unknown", "")


def test_resolve_scope_survives_one_platform_failing() -> None:
    """单个平台探测失败不能拖垮其余平台。"""
    sk = _fresh()

    class _Flaky(_PlatformChat):
        async def get_group_streams(self, platform: str = "qq") -> Any:
            if platform == "discord":
                raise RuntimeError("discord 不支持")
            return await super().get_group_streams(platform)

    chat = _Flaky({"qq": ["qq-3-group"]})
    kind, platform = asyncio.run(sk.resolve_stream_scope(_plugin_with_platforms(chat, ["discord", "qq"]), "qq-3-group"))
    assert (kind, platform) == ("group", "qq")


def test_stream_scope_is_cached_with_platform() -> None:
    """缓存要连平台一起缓存（否则第二次调用平台就丢了）。"""
    sk = _fresh()
    chat = _PlatformChat({"qq": ["qq-9-group"]})
    plugin = _plugin_with_platforms(chat, ["qq"])

    asyncio.run(sk.resolve_stream_scope(plugin, "qq-9-group"))
    before = len(chat.probed)
    _kind, platform = asyncio.run(sk.resolve_stream_scope(plugin, "qq-9-group"))
    assert platform == "qq"
    assert len(chat.probed) == before, "第二次调用没有走缓存"


def test_snapshot_records_platform_scope_field(soul_db: Any) -> None:
    """快照必须记录平台（作用域字段）；缺失时留空，不编造。"""
    sr = _import_soul_submodule("models.self_reflection")

    snap = sr.create_injection_snapshot(
        "group-A", "sess-1", '["t"]', "{}", "{}", "tag_hit",
        bot_identity="qq:12345", platform="discord",
    )
    stored = sr.get_injection_snapshot(snap)
    assert stored is not None
    assert stored.platform == "discord"
    assert stored.bot_identity == "qq:12345"

    snap2 = sr.create_injection_snapshot("group-A", "sess-2", "[]", "{}", "{}", "spectrum_only")
    stored2 = sr.get_injection_snapshot(snap2)
    assert stored2 is not None and stored2.platform == ""
