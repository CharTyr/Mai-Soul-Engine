"""Mai-Soul-Engine 插件测试共享 fixture 与工具函数。

测试以 ``plugins.CharTyr_Mai-Soul-Engine.<submodule>`` 形式导入（插件目录名含连字符，
不能直接 ``import``）。导入路径由本模块自行推导，因此两种调用方式都可用：

- 从宿主仓根运行：``pytest plugins/CharTyr_Mai-Soul-Engine/tests/ -q``
- 从插件目录运行：``pytest tests/ -q``

路径推导失败时**直接报错**，不得静默 skip —— 历史行为是
``pytest.skip("插件目录不存在")``，会让整套 300+ 测试以「全跳过」假绿通过。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

# tests/conftest.py → tests/ → 插件目录
PLUGIN_DIR = Path(__file__).resolve().parent.parent
PACKAGE_NAME = f"plugins.{PLUGIN_DIR.name}"
# 含 plugins/ 的目录：宿主仓根，或本地测试 scaffold 根
HOST_ROOT = PLUGIN_DIR.parent.parent

# 独立 checkout 时使用的临时脚手架根（进程内复用）
_SCAFFOLD_ROOT: Path | None = None


def _ensure_host_root_on_path() -> None:
    """把宿主根加入 sys.path，使 ``plugins.<dir>`` 可被 importlib 解析。"""
    host_root = str(HOST_ROOT)
    if host_root not in sys.path:
        sys.path.insert(0, host_root)


def _ensure_package_importable() -> None:
    """保证 ``plugins.<插件目录名>`` 这个导入路径存在。

    该包名依赖「上级目录含 ``plugins/``」的布局（宿主仓或测试 scaffold）。
    插件被单独 checkout（如本地开发副本）时没有这层结构，此时在系统临时目录里
    搭一个 ``plugins/<名>`` 符号链接脚手架，既不污染仓库也不改变导入语义。
    """
    global _SCAFFOLD_ROOT

    if (HOST_ROOT / "plugins" / PLUGIN_DIR.name).exists():
        _ensure_host_root_on_path()
        return

    if _SCAFFOLD_ROOT is None or not _SCAFFOLD_ROOT.exists():
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="soul-test-scaffold-"))
        (root / "plugins").mkdir(parents=True, exist_ok=True)
        (root / "plugins" / PLUGIN_DIR.name).symlink_to(PLUGIN_DIR, target_is_directory=True)
        _SCAFFOLD_ROOT = root

    scaffold = str(_SCAFFOLD_ROOT)
    if scaffold not in sys.path:
        sys.path.insert(0, scaffold)


def _import_soul_submodule(name: str) -> Any:
    """通过 importlib 导入插件子模块（因目录名含连字符，不能直接 import）。"""
    if not PLUGIN_DIR.is_dir():
        raise RuntimeError(
            f"插件目录不可用: {PLUGIN_DIR}（测试入口推导失败，不应静默跳过）"
        )
    _ensure_package_importable()
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture
def soul_db(tmp_path: Path) -> Any:
    """共享的 soul.db fixture：init_db → yield model shim → close_db。"""
    im = _import_soul_submodule("models.ideology_model")
    im.init_db(tmp_path / "soul.db")
    # 清理跨测试共享的模块级缓存（reflection_feedback._summary_cache 等）
    _clear_module_caches()
    yield im
    im.close_db()


def _clear_module_caches() -> None:
    """清理可能跨测试污染的模块级缓存。"""
    try:
        fb = _import_soul_submodule("components.reflection_feedback")
        if hasattr(fb, "invalidate_reflection_summary_cache"):
            fb.invalidate_reflection_summary_cache()
    except Exception:
        pass
