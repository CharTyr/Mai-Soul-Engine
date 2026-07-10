"""Tests for host persona reading and prompt formatting."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from .conftest import _import_soul_submodule


def test_fetch_host_persona_empty_config() -> None:
    """Both config keys empty → source='empty', hash empty."""
    hp = _import_soul_submodule("utils.host_persona")
    hp.invalidate_host_persona_cache()

    class Ctx:
        async def call_capability(self, capability: str, **kwargs):
            assert capability == "config.get"
            return {"success": True, "value": ""}

    snap = asyncio.run(hp.fetch_host_persona(SimpleNamespace(ctx=Ctx())))

    assert snap.source == "empty"
    assert snap.personality_text == ""
    assert snap.reply_style_text == ""
    assert snap.profile_hash == ""


def test_fetch_host_persona_with_personality() -> None:
    """Only personality configured → hash is stable, source='config.get'."""
    hp = _import_soul_submodule("utils.host_persona")
    hp.invalidate_host_persona_cache()  # clear cross-test cache

    class Ctx:
        _call_count = 0

        async def call_capability(self, capability: str, **kwargs):
            self._call_count += 1
            if self._call_count == 1:
                # personality.personality
                return {"success": True, "value": "温柔体贴的大姐姐"}
            # personality.reply_style
            return {"success": True, "value": ""}

    snap = asyncio.run(hp.fetch_host_persona(SimpleNamespace(ctx=Ctx())))

    assert snap.source == "config.get"
    assert snap.personality_text == "温柔体贴的大姐姐"
    assert snap.reply_style_text == ""

    # second call uses cache
    snap2 = asyncio.run(hp.fetch_host_persona(SimpleNamespace(ctx=SimpleNamespace())))
    assert snap2.profile_hash == snap.profile_hash
    assert snap2.personality_text == snap.personality_text
    assert snap2.source == snap.source


def test_fetch_host_persona_both_set() -> None:
    """Both personality and reply_style configured."""
    hp = _import_soul_submodule("utils.host_persona")
    hp.invalidate_host_persona_cache()

    class Ctx:
        _call_count = 0

        async def call_capability(self, capability: str, **kwargs):
            self._call_count += 1
            if self._call_count == 1:
                return {"success": True, "value": "毒舌吐槽"}
            return {"success": True, "value": "段子手，每句都要带梗"}

    snap = asyncio.run(hp.fetch_host_persona(SimpleNamespace(ctx=Ctx())))

    assert snap.source == "config.get"
    assert snap.personality_text == "毒舌吐槽"
    assert snap.reply_style_text == "段子手，每句都要带梗"
    assert len(snap.profile_hash) == 16  # hex[:16]


def test_fetch_host_persona_failure_tolerant() -> None:
    """Config.get raises → treated as empty, no crash."""
    hp = _import_soul_submodule("utils.host_persona")
    hp.invalidate_host_persona_cache()

    class Ctx:
        async def call_capability(self, capability: str, **kwargs):
            raise RuntimeError("config service down")

    snap = asyncio.run(hp.fetch_host_persona(SimpleNamespace(ctx=Ctx())))

    assert snap.source == "empty"
    assert snap.personality_text == ""
    assert snap.reply_style_text == ""


def test_format_persona_for_prompt_empty() -> None:
    """Empty snapshot → empty string."""
    hp = _import_soul_submodule("utils.host_persona")
    snap = hp.HostPersonaSnapshot(source="empty")
    assert hp.format_persona_for_prompt(snap) == ""


def test_format_persona_for_prompt_with_text() -> None:
    """Non-empty snapshot → formatted block with header."""
    hp = _import_soul_submodule("utils.host_persona")
    snap = hp.HostPersonaSnapshot(
        personality_text="温柔可靠",
        reply_style_text="每句带语气词",
        source="config.get",
    )
    result = hp.format_persona_for_prompt(snap)

    assert "【宿主固定人设基底" in result
    assert "人格设定：温柔可靠" in result
    assert "回复风格：每句带语气词" in result


def test_format_persona_for_prompt_truncation() -> None:
    """Long persona is truncated to max_chars."""
    hp = _import_soul_submodule("utils.host_persona")
    long_p = "x" * 900
    long_r = "y" * 900
    snap = hp.HostPersonaSnapshot(
        personality_text=long_p,
        reply_style_text=long_r,
        source="config.get",
    )
    result = hp.format_persona_for_prompt(snap, max_chars=100)
    # block part (after header) should be ≤ 100
    header = "\n\n【宿主固定人设基底（不可与身份事实冲突；可延伸细化）】\n"
    body = result[len(header):]
    assert len(body) <= 100
    assert "人格设定：" in result
    assert "回复风格：" not in body  # truncated before reply_style


def test_normalize_hash_stable() -> None:
    """Same input → same hash."""
    hp = _import_soul_submodule("utils.host_persona")
    h1 = hp._normalize_hash(" 温柔  可靠 ", "每句带语气词")
    h2 = hp._normalize_hash("温柔 可靠", "每句带语气词")
    assert h1 == h2
    assert len(h1) == 16


def test_cache_invalidation() -> None:
    """invalidate_host_persona_cache clears module-level cache."""
    hp = _import_soul_submodule("utils.host_persona")
    hp.invalidate_host_persona_cache()

    class Ctx:
        async def call_capability(self, capability: str, **kwargs):
            return {"success": True, "value": "some"}

    snap = asyncio.run(hp.fetch_host_persona(SimpleNamespace(ctx=Ctx())))
    assert snap.source == "config.get"

    hp.invalidate_host_persona_cache()

    # Now fresh fetch returns empty config
    class CtxEmpty:
        async def call_capability(self, capability: str, **kwargs):
            return {"success": True, "value": ""}

    snap2 = asyncio.run(hp.fetch_host_persona(SimpleNamespace(ctx=CtxEmpty())))
    assert snap2.source == "empty"
    assert snap2.personality_text == ""
