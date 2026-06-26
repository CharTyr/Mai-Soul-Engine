"""为兼容历史导入保留的重导出聚合层。

原始单体已按实体拆分为 ``models/_conn.py`` / ``models/spectrum.py`` /
``models/history.py`` / ``models/seeds.py`` / ``models/traits.py`` /
``models/p1.py``。

本文件仅做 ``from .submodule import *`` 重导出，确保所有现有
``from ..models.ideology_model import XXX`` 和
``from ..models import ideology_model as im`` 调用零破坏。
"""

from __future__ import annotations

from . import _conn as _core_conn
from ._conn import *
from .history import *
from .p1 import *
from .seeds import *
from .spectrum import *
from .traits import *


def __getattr__(name: str):
    """兼容动态访问：_db_path / _conn 由 init_db 在 _conn 模块中设定，
    ``from ._conn import *`` 拷贝的值不会随 init_db 更新，
    这里通过 ``__getattr__`` 动态委托到 _conn 模块。
    """
    if name in {"_conn", "_db_path"}:
        return getattr(_core_conn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({"_conn", "_db_path", *dir(type(_core_conn))})
