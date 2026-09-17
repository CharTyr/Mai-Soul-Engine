"""T18：日志与产物不含凭据、原始私聊、真实标识；调试追踪有权限与保留期（TTL）。

方案原文把 T18 列成「未验证」——没有测试就等于没有保证。这里把可验证的部分
变成可执行的检查：

1. 注入日志（唯一的调试追踪产物）**不得写入原始对话文本**；
2. 调试追踪必须有**保留期**（只按大小轮转在低频环境下等于永不清理）；
3. 仓库产物不得包含凭据或真实标识（`config.toml` 不入库；示例 ID 用占位符）。
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Any

from .conftest import _import_soul_submodule

# 真实部署里出现过的标识形态（用于「不得出现在仓库产物里」的扫描）
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("bearer/API key", re.compile(r"\b(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{20,})\b")),
    ("私钥块", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("硬编码 token 字段", re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[\"'][A-Za-z0-9._-]{20,}[\"']")),
]

# 已经是占位符的形态，扫描时跳过
_PLACEHOLDER = re.compile(r"(12345678|YOUR_|<[^>]+>|xxx+|\.\.\.|placeholder)", re.IGNORECASE)


def _repo_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    skip_dirs = {".git", "__pycache__", ".pytest_cache", "node_modules", "data", "config_back"}
    out: list[Path] = []
    for path in root.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix in {".py", ".toml", ".md", ".json", ".yaml", ".yml"}:
            out.append(path)
    return out


# ─── 1. 注入日志不得写入原始对话文本 ───────────────────────────────


def test_injection_log_never_contains_raw_message_text(soul_db: Any, tmp_path: Path) -> None:
    """日志只记元数据：拿一段可识别的私聊原文走一遍注入，日志里不得出现它。"""
    inj = _import_soul_submodule("components.ideology_injector")
    inj._injection_log_counter = 0

    secret = "这是不该出现在日志里的私聊原文-7391"

    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    plugin = SimpleNamespace(
        config=soul_db and None,  # 见下方显式构造
    )
    schema = _import_soul_submodule("plugin_ui_schema")
    cfg = schema.MaiSoulEngineConfig()
    cfg.plugin.mode = "apply"
    plugin.config = cfg
    plugin._plugin_dir = tmp_path
    plugin._data_dir = tmp_path / "data"
    plugin._wv_config_view = None

    class _WV:
        def build_layer_trait_summary(self, *a: Any, **k: Any) -> str:
            return ""

        def mood_prompt_lines(self) -> list[str]:
            return []

        def build_graph_hint(self, *a: Any, **k: Any) -> str:
            return ""

    plugin._wv_service = _WV()

    class _Ctx:
        class chat:
            get_group_streams = AsyncMock(return_value=["qq-123-group"])
            get_private_streams = AsyncMock(return_value=[])

    plugin.ctx = _Ctx()

    spectrum = soul_db.get_or_create_spectrum("global")
    spectrum.initialized = True
    spectrum.save()

    items = [
        {
            "item_type": "SystemMessageItem",
            "meta": {"item_id": "s"},
            "parts": [{"type": "text", "text": "你是 Mai\n"}],
        },
        {
            "item_type": "UserMessageItem",
            "meta": {"item_id": "u"},
            "parts": [{"type": "text", "text": secret}],
        },
    ]
    # 采样是按计数取模的：多跑几次确保至少写出一条日志
    for _ in range(inj.INJECTION_LOG_EVERY):
        asyncio.run(
            inj.inject_ideology(
                plugin, items=items, item_schema_version=3, session_id="qq-123-group"
            )
        )

    log = tmp_path / "data" / "injections.jsonl"
    assert log.exists(), "注入日志没有落盘（本测试想验证的就是它的内容）"
    content = log.read_text(encoding="utf-8")

    # 非空校验：日志必须真的写了条目，否则「不含原文」是句废话
    lines = [ln for ln in content.splitlines() if ln.strip()]
    assert lines, "注入日志是空的——那这个测试什么也没验证"
    import json as _json

    entry = _json.loads(lines[-1])
    assert "policy" in entry and "prompt_version" in entry, (
        f"日志条目缺少预期元数据字段: {sorted(entry)}"
    )
    assert secret not in content, "注入日志写入了原始对话文本"
    assert "qq-123-group" not in content, "注入日志写入了会话标识（真实群号）"


# ─── 2. 调试追踪必须有保留期 ───────────────────────────────────────


def test_injection_log_has_retention_ttl(tmp_path: Path) -> None:
    """超过保留期的日志（含轮转文件）必须被清理；新文件保留。"""
    inj = _import_soul_submodule("components.ideology_injector")
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    old = data_dir / "injections.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    rotated_old = data_dir / "injections.1.jsonl"
    rotated_old.write_text("{}\n", encoding="utf-8")

    stale = time.time() - (inj.INJECTION_LOG_RETENTION_DAYS + 1) * 86400
    os.utime(old, (stale, stale))
    os.utime(rotated_old, (stale, stale))

    fresh = data_dir / "injections.2.jsonl"
    fresh.write_text("{}\n", encoding="utf-8")

    removed = inj._purge_expired_injection_logs(data_dir)

    assert sorted(removed) == ["injections.1.jsonl", "injections.jsonl"]
    assert fresh.exists(), "未过期的日志被误删"
    assert inj.INJECTION_LOG_RETENTION_DAYS > 0


def test_retention_is_days_not_unbounded(tmp_path: Path) -> None:
    """保留期必须是有限值（0 或负数等于关掉清理，属配置错误）。"""
    inj = _import_soul_submodule("components.ideology_injector")
    assert 0 < inj.INJECTION_LOG_RETENTION_DAYS <= 365


# ─── 3. 仓库产物不得含凭据 / 真实标识 ──────────────────────────────


def test_repo_has_no_committed_credentials() -> None:
    """扫描入库文件：不得出现 API key / 私钥 / 硬编码 token。"""
    hits: list[str] = []
    for path in _repo_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in _SECRET_PATTERNS:
            for match in pattern.finditer(text):
                if _PLACEHOLDER.search(match.group(0)):
                    continue
                hits.append(f"{path.name}: {label}: {match.group(0)[:40]}")
    assert not hits, "仓库产物里出现了疑似凭据：\n" + "\n".join(hits[:10])


def test_real_config_is_not_tracked() -> None:
    """真实配置（config.toml）与运行数据不得入库。"""
    root = Path(__file__).resolve().parent.parent
    assert not (root / "config.toml").exists(), "真实 config.toml 被提交进仓库"
    assert (root / "config_template.toml").exists(), "缺少脱敏配置模板"
    gitignore = root / ".gitignore"
    assert gitignore.exists(), "缺少 .gitignore"
    rules = gitignore.read_text(encoding="utf-8")
    for must_ignore in ("config.toml", "data/"):
        assert must_ignore in rules, f".gitignore 没有忽略 {must_ignore}"
