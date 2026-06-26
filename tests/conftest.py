"""Mai-Soul-Engine 插件测试共享 fixture 与工具函数。

使用方式：从宿主仓根 `cd /root/seren/rdev-Maibot` 运行
`uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/ -q`。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

PLUGIN_DIR = Path("plugins/CharTyr_Mai-Soul-Engine")


def _import_soul_submodule(name: str) -> Any:
    """通过 importlib 导入插件子模块（因目录名含连字符，不能直接 import）。"""
    if not PLUGIN_DIR.is_dir():
        pytest.skip("CharTyr_Mai-Soul-Engine 插件目录不存在")
    return importlib.import_module(f"plugins.CharTyr_Mai-Soul-Engine.{name}")


@pytest.fixture
def soul_db(tmp_path: Path) -> Any:
    """共享的 soul.db fixture：init_db → yield model shim → close_db。"""
    im = _import_soul_submodule("models.ideology_model")
    im.init_db(tmp_path / "soul.db")
    yield im
    im.close_db()
