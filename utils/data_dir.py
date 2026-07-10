"""数据目录解析与安全迁移。

职责：
1. ``resolve_plugin_data_dir(plugin)`` — 优先使用宿主 ``ctx.paths.data_dir``，
   否则回退到 ``plugin_dir/data``。
2. ``resolve_and_prepare_data_dir(plugin)`` — 解析 + 按需迁移，返回包含
   ``data_dir``、``source``、``migrated`` 等信息的 dict。
3. ``migrate_data_dir_if_needed(legacy_data, target_data)`` — 从旧位置
   安全拷贝 soul.db 与附属文件到新位置，写迁移标记。

迁移安全约定：
- 禁止 ``shutil.move`` 正在使用的 WAL 库 → 用 ``sqlite3.backup`` API
  或完整拷贝后 integrity_check。
- 迁移失败 → 完整回退到旧目录，log error，不丢失数据。
- 不删除旧文件，只写 ``migration_marker.json``。
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── 宿主 data_dir 子目录名 ─────────────────────────────────────────

HOST_SUBDIR = "mai_soul_engine"


def resolve_plugin_data_dir(plugin: Any) -> tuple[Path, str]:
    """解析插件数据目录。

    优先级：
    1. 宿主 ``ctx.paths.data_dir`` / ``mai_soul_engine`` 子目录（推荐）
    2. 宿主 ``ctx.paths.data_dir`` 直接作为根（兼容旧配置）
    3. 兜底：``plugin_dir / "data"``

    Returns:
        (data_dir, source) — source 为 ``"host"`` 或 ``"plugin_dir"``。
    """
    host_path = _try_get_host_data_dir(plugin)
    if host_path is not None:
        # 尝试子目录 mai_soul_engine，避免与其它插件冲突
        sub_path = host_path / HOST_SUBDIR
        try:
            sub_path.mkdir(parents=True, exist_ok=True)
            return sub_path, "host"
        except OSError:
            pass
        # 子目录不可写 → 直接用宿主根目录
        try:
            host_path.mkdir(parents=True, exist_ok=True)
            return host_path, "host"
        except OSError:
            pass

    # 兜底：插件目录下的 data/
    plugin_dir = Path(plugin._plugin_dir) if hasattr(plugin, "_plugin_dir") else Path(__file__).parent.parent
    fallback = plugin_dir / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback, "plugin_dir"


def _try_get_host_data_dir(plugin: Any) -> Path | None:
    """尝试从宿主 ctx 获取 data_dir，多种 fallback。"""
    ctx = getattr(plugin, "ctx", None)
    if ctx is None:
        return None

    # 方法1: ctx.paths.data_dir（推荐，SDK 规范）
    paths = getattr(ctx, "paths", None)
    if paths is not None:
        if isinstance(paths, dict):
            d = paths.get("data_dir")
        else:
            d = getattr(paths, "data_dir", None)
        if d:
            p = Path(str(d))
            if p.is_dir() or p.parent.is_dir():
                return p.resolve()

    # 方法2: ctx.data_dir 直接挂载
    d = getattr(ctx, "data_dir", None)
    if d:
        p = Path(str(d))
        if p.is_dir() or p.parent.is_dir():
            return p.resolve()

    return None


def resolve_and_prepare_data_dir(plugin: Any) -> dict[str, Any]:
    """完整的 data_dir 解析与迁移流程，供 ``on_load`` 调用。

    在 ``init_db`` 之前调用——此时旧 soul.db 尚未打开连接，
    可安全操作 legacy 文件。

    Returns:
        dict:
          - ``data_dir`` (Path): 最终使用的数据目录
          - ``source`` (str): ``"host"`` | ``"plugin_dir"``
          - ``migrated`` (bool): 是否执行了迁移
          - ``migration_detail`` (dict | None): 迁移详情
    """
    plugin_dir = Path(plugin._plugin_dir) if hasattr(plugin, "_plugin_dir") else Path(__file__).parent.parent
    legacy_data = plugin_dir / "data"

    target_data, source = resolve_plugin_data_dir(plugin)

    result: dict[str, Any] = {
        "data_dir": target_data,
        "source": source,
        "migrated": False,
        "migration_detail": None,
    }

    # 相同路径 → 无需迁移
    if target_data.resolve() == legacy_data.resolve():
        return result

    # 目标已有 soul.db → 直接用，不覆盖
    if (target_data / "soul.db").exists():
        logger.info(
            "[data_dir] 目标目录已有 soul.db，直接使用: %s (source=%s)",
            target_data, source,
        )
        return result

    # legacy 无 soul.db → 无需迁移
    if not (legacy_data / "soul.db").exists():
        logger.info(
            "[data_dir] 无旧 soul.db，直接使用: %s (source=%s)",
            target_data, source,
        )
        return result

    # 执行迁移
    try:
        detail = migrate_data_dir_if_needed(legacy_data, target_data)
        result["migrated"] = detail["migrated"]
        result["migration_detail"] = detail
        if detail.get("error"):
            logger.error("[data_dir] 迁移失败: %s，回退到 plugin_dir/data", detail["error"])
            result["data_dir"] = legacy_data
            result["source"] = "plugin_dir"
            result["migrated"] = False
        else:
            logger.info(
                "[data_dir] 数据已从 %s 迁移到 %s (source=%s)",
                legacy_data, target_data, source,
            )
    except Exception as exc:
        logger.exception("[data_dir] 迁移异常: %s，回退到 plugin_dir/data", exc)
        result["data_dir"] = legacy_data
        result["source"] = "plugin_dir"

    return result


def migrate_data_dir_if_needed(legacy_data: Path, target_data: Path) -> dict[str, Any]:
    """将 legacy_data 中的 soul.db 及相关文件安全迁移到 target_data。

    安全措施：
    - 迁移前 legacy soul.db 未打开连接（调用方保证在 init_db 前调用）
    - 使用 ``sqlite3.backup`` 确保完整拷贝
    - 拷贝后执行 integrity_check
    - 拷贝 audit.jsonl、injections.jsonl 等附属文件（跳过 WAL/SHM）
    - 在 legacy 写入 migration_marker.json

    Returns:
        dict: {migrated: bool, source: str, target: str, error: str | None}
    """
    result: dict[str, Any] = {
        "migrated": False,
        "source": str(legacy_data.resolve()),
        "target": str(target_data.resolve()),
        "error": None,
    }

    target_data.mkdir(parents=True, exist_ok=True)

    # ── 1. soul.db 用 sqlite3.backup ──────────────────────────────
    src_db = legacy_data / "soul.db"
    dst_db = target_data / "soul.db"

    if src_db.exists():
        try:
            _backup_sqlite(src_db, dst_db)
            logger.info("[data_dir] soul.db 已备份: %s → %s", src_db, dst_db)
        except Exception as exc:
            result["error"] = f"soul.db 备份失败: {exc}"
            return result

        # 完整性检查
        try:
            _verify_sqlite_integrity(dst_db)
        except Exception as exc:
            # 完整性失败 → 删除目标文件，返回错误
            dst_db.unlink(missing_ok=True)
            result["error"] = f"soul.db 完整性检查失败: {exc}"
            return result

    # ── 2. 拷贝其他非 WAL 文件 ────────────────────────────────────
    _copy_auxiliary_files(legacy_data, target_data, exclude_wal=True)

    # ── 3. 在 legacy 写迁移标记 ────────────────────────────────────
    _write_migration_marker(legacy_data, target_data)

    result["migrated"] = True
    return result


# ─── 内部工具 ────────────────────────────────────────────────────────


def _backup_sqlite(src: Path, dst: Path) -> None:
    """使用 sqlite3.backup API 安全拷贝 SQLite 数据库。

    即使原库有 WAL 文件未 checkpoint，backup 也能得到完整快照。
    """
    src_conn = sqlite3.connect(str(src))
    try:
        dst_conn = sqlite3.connect(str(dst))
        try:
            src_conn.backup(dst_conn)
            dst_conn.commit()
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def _verify_sqlite_integrity(db_path: Path) -> None:
    """对新库执行 integrity_check，失败时抛异常。"""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        if row and row[0] != "ok":
            raise RuntimeError(f"integrity_check: {row[0]}")
    finally:
        conn.close()


def _copy_auxiliary_files(src: Path, dst: Path, *, exclude_wal: bool = True) -> None:
    """拷贝 soul.db 外的附属文件（audit.jsonl, injections.jsonl, migration_state.json）。

    跳过 -wal, -shm 等 WAL 附属文件（backup 已固化）。
    """
    WAL_SUFFIXES = frozenset({"-wal", "-shm"})
    for f in src.iterdir():
        if not f.is_file():
            continue
        if exclude_wal and f.suffix in WAL_SUFFIXES:
            continue
        # 跳过 soul.db 本身（已用 backup 拷贝）
        if f.name == "soul.db":
            continue
        # 跳过 migration_marker.json（我们在最后写）
        if f.name == "migration_marker.json":
            continue
        try:
            shutil.copy2(str(f), str(dst / f.name))
        except OSError as exc:
            logger.warning("[data_dir] 拷贝附属文件失败 %s: %s", f.name, exc)


def _write_migration_marker(legacy_data: Path, target_data: Path) -> None:
    """在 legacy data 目录写 migration_marker.json。"""
    marker = {
        "migrated_to": str(target_data.resolve()),
        "at": datetime.now(timezone.utc).isoformat(),
        "integrity": "verified",
        "note": "原始文件保留未删除",
    }
    marker_path = legacy_data / "migration_marker.json"
    try:
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(marker, f, ensure_ascii=False, indent=2)
        logger.info("[data_dir] 迁移标记已写入: %s", marker_path)
    except OSError as exc:
        logger.warning("[data_dir] 写迁移标记失败: %s", exc)
