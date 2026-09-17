"""数据目录盘点与迁移预演（**只读**，不自动选源）。

**为什么需要**：历史上有两份可能同时存在的 soul.db——`plugins/<插件>/data/`
和 `data/plugins/<plugin-id>/mai_soul_engine/`，内容可能不同。选错源会丢掉
真实的种子/观点/自评历史。而"按文件大小或修改时间自动选"是不可靠的：
大文件可能是垃圾堆积，新文件可能是刚初始化的空库。

所以本工具只做三件事：

1. **只读盘点**每份库（用 SQLite 的 ``mode=ro``，绝不建表、绝不写）
2. **给出事实**：schema 版本、迁移记录、各表行数、关键状态是否已初始化
3. **明确指出需要操作者决定**，并在检测到"两份都有实质数据"时报警

选哪一份永远由操作者决定，工具不替人选。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "DatabaseInventory",
    "MigrationPreview",
    "build_migration_preview",
    "inventory_database",
]

# 盘点时统计行数的表（缺失的表记为 None，不报错）
_TRACKED_TABLES: tuple[str, ...] = (
    "soul_spectrum",
    "soul_crystallized_traits",
    "soul_thought_seeds",
    "soul_fermentation_inputs",
    "soul_context_slices",
    "soul_mood_state",
    "soul_thought_edges",
    "soul_injection_snapshots",
    "soul_pending_reflections",
    "soul_self_reflections",
    "soul_seed_operations",
    "soul_notifications",
)


@dataclass
class DatabaseInventory:
    """单个数据库的只读盘点结果。"""

    path: str
    exists: bool = False
    readable: bool = False
    size_bytes: int = 0
    schema_version: int | None = None
    migrations: list[dict[str, Any]] = field(default_factory=list)
    table_counts: dict[str, int | None] = field(default_factory=dict)
    spectrum_initialized: bool | None = None
    trait_enabled_count: int | None = None
    seed_counts_by_status: dict[str, int] = field(default_factory=dict)
    last_activity: str = ""
    error: str = ""
    section_errors: list[str] = field(default_factory=list)

    @property
    def has_substantive_data(self) -> bool:
        """是否含有值得保留的历史（种子/观点/自评）。不用于自动选源，只用于报警。"""
        counts = self.table_counts
        for table in (
            "soul_thought_seeds",
            "soul_crystallized_traits",
            "soul_self_reflections",
            "soul_spectrum",
        ):
            if (counts.get(table) or 0) > 0:
                return True
        return False


def _ro_connect(path: Path) -> sqlite3.Connection:
    """以只读方式打开；绝不创建文件、绝不建表。"""
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def inventory_database(path: str | Path) -> DatabaseInventory:
    """只读盘点一个 soul.db；文件不存在/不可读都返回结构化结果而非抛异常。"""
    p = Path(path)
    inv = DatabaseInventory(path=str(p))
    if not p.exists():
        inv.error = "文件不存在"
        return inv
    inv.exists = True
    try:
        inv.size_bytes = p.stat().st_size
    except OSError as e:
        inv.error = f"stat 失败: {e}"

    try:
        conn = _ro_connect(p)
    except sqlite3.Error as e:
        inv.error = f"以只读方式打开失败: {e}"
        return inv

    try:
        inv.readable = True

        # 探测：sqlite3.connect 不会校验文件内容（拿个文本文件也能"打开"），
        # 所以必须真的查一次才算可读
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.DatabaseError as e:
            inv.readable = False
            inv.error = f"不是可用的 SQLite 数据库: {e}"
            return inv

        def _section(label: str, fn: Any) -> None:
            """逐段隔离：某一节读不出来（列名/结构差异）不该让整份盘点失败。

            这个工具最需要处理的恰恰是**结构异常**的库（旧版本、迁移中途），
            所以"一节失败就全盘失败"是设计错误。
            """
            try:
                fn()
            except sqlite3.Error as e:
                inv.section_errors.append(f"{label}: {e}")

        def _read_schema_version() -> None:
            inv.schema_version = int(conn.execute("PRAGMA user_version").fetchone()[0])

        def _read_migrations() -> None:
            if not _table_exists(conn, "soul_schema_migrations"):
                return
            # 列名在不同版本里有差（如 applied_at），用 SELECT * 再按可用键取值
            rows = conn.execute(
                "SELECT * FROM soul_schema_migrations ORDER BY version"
            ).fetchall()
            inventoried: list[dict[str, Any]] = []
            for r in rows:
                keys = set(r.keys())
                inventoried.append(
                    {
                        "version": r["version"] if "version" in keys else None,
                        "name": r["name"] if "name" in keys else "",
                        "status": r["status"] if "status" in keys else "",
                        "applied_at": (
                            r["applied_at"] if "applied_at" in keys
                            else (r["updated_at"] if "updated_at" in keys else "")
                        ),
                    }
                )
            inv.migrations = inventoried

        def _read_table_counts() -> None:
            for table in _TRACKED_TABLES:
                if not _table_exists(conn, table):
                    inv.table_counts[table] = None
                    continue
                try:
                    inv.table_counts[table] = int(
                        conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    )
                except sqlite3.Error:
                    inv.table_counts[table] = None

        def _read_spectrum() -> None:
            if not _table_exists(conn, "soul_spectrum"):
                return
            row = conn.execute(
                "SELECT * FROM soul_spectrum WHERE scope_id = 'global'"
            ).fetchone()
            if row is None:
                inv.spectrum_initialized = None
                return
            keys = set(row.keys())
            if "initialized" in keys:
                inv.spectrum_initialized = bool(row["initialized"])

        def _read_traits() -> None:
            if not _table_exists(conn, "soul_crystallized_traits"):
                return
            inv.trait_enabled_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM soul_crystallized_traits "
                    "WHERE enabled = 1 AND deleted = 0"
                ).fetchone()[0]
            )

        def _read_seeds() -> None:
            if not _table_exists(conn, "soul_thought_seeds"):
                return
            rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM soul_thought_seeds GROUP BY status"
            ).fetchall()
            inv.seed_counts_by_status = {r["status"] or "?": int(r["cnt"]) for r in rows}

        def _read_last_activity() -> None:
            if not _table_exists(conn, "soul_history"):
                return
            row = conn.execute("SELECT MAX(created_at) AS last FROM soul_history").fetchone()
            inv.last_activity = (row["last"] or "") if row is not None else ""

        _section("schema 版本", _read_schema_version)
        _section("迁移记录", _read_migrations)
        _section("表计数", _read_table_counts)
        _section("光谱状态", _read_spectrum)
        _section("trait 计数", _read_traits)
        _section("种子统计", _read_seeds)
        _section("最近活动", _read_last_activity)
    finally:
        conn.close()

    return inv


@dataclass
class MigrationPreview:
    """多份候选的对比预演。**不做自动选源**。"""

    candidates: list[DatabaseInventory] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    decision_required: bool = True
    note: str = (
        "工具不自动选源：选哪一份会决定历史数据的去留，必须由操作者指定。"
        "建议先确认哪份含真实历史，再用它作为权威目录。"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "note": self.note,
            "decision_required": self.decision_required,
            "warnings": self.warnings,
            "candidates": [asdict(c) for c in self.candidates],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def build_migration_preview(paths: Iterable[str | Path]) -> MigrationPreview:
    """对比多份候选库，输出事实与告警。"""

    preview = MigrationPreview(candidates=[inventory_database(p) for p in paths])

    readable = [c for c in preview.candidates if c.readable]
    missing = [c for c in preview.candidates if not c.exists]
    substantive = [c for c in readable if c.has_substantive_data]

    for c in missing:
        preview.warnings.append(f"候选不存在（忽略）: {c.path}")
    if not readable:
        preview.warnings.append("没有任何可读候选，无法预演")
        return preview

    if len(substantive) > 1:
        preview.warnings.append(
            "⚠️ 有 " + str(len(substantive)) + " 份候选都含实质数据（种子/观点/自评）："
            "选错会丢历史。请逐份核对上面的表计数后再决定。"
        )

    versions = {c.schema_version for c in readable if c.schema_version is not None}
    if len(versions) > 1:
        preview.warnings.append(
            f"⚠️ 候选的 schema 版本不一致：{sorted(versions)}。"
            "低版本会在启动时自动迁移，但这意味着两份数据处于不同阶段。"
        )

    empty_init = [
        c for c in readable
        if not c.has_substantive_data and (c.schema_version or 0) > 0
    ]
    if empty_init and substantive:
        preview.warnings.append(
            "存在「已初始化但无历史数据」的候选：它可能是空壳，"
            "不能因为文件更新时间更近就选它。"
        )

    return preview


def _format_text(preview: MigrationPreview) -> str:
    """人读的预演报告。"""
    lines: list[str] = ["=== 数据目录迁移预演（只读）==="]
    for c in preview.candidates:
        lines.append("")
        lines.append(f"候选: {c.path}")
        if not c.exists:
            lines.append("  状态: 不存在")
            continue
        lines.append(f"  大小: {c.size_bytes} 字节")
        if not c.readable:
            lines.append(f"  状态: 不可读（{c.error or '未知原因'}）")
            continue
        lines.append(f"  schema 版本: {c.schema_version}")
        if c.migrations:
            ok = sum(1 for m in c.migrations if m["status"] == "success")
            lines.append(f"  迁移记录: {len(c.migrations)} 条（success {ok}）")
        lines.append(f"  光谱已初始化: {c.spectrum_initialized}")
        lines.append(f"  启用中的 trait: {c.trait_enabled_count}")
        if c.seed_counts_by_status:
            seeds = "、".join(f"{k}={v}" for k, v in sorted(c.seed_counts_by_status.items()))
            lines.append(f"  种子: {seeds}")
        counts = "、".join(
            f"{t.replace('soul_', '')}={v if v is not None else '—'}"
            for t, v in c.table_counts.items()
        )
        lines.append(f"  表计数: {counts}")
        if c.last_activity:
            lines.append(f"  最近活动: {c.last_activity}")
        lines.append(f"  含实质数据: {'是' if c.has_substantive_data else '否'}")

    if preview.warnings:
        lines.append("")
        lines.append("--- 告警 ---")
        lines.extend(f"⚠️ {w}" for w in preview.warnings)

    lines.append("")
    lines.append(f"说明: {preview.note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI：盘点候选库并打印预演报告。**不写任何文件、不选源**。"""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    paths = [a for a in args if not a.startswith("--")]
    if not paths:
        print("用法: python migration/inventory.py <soul.db> [另一个 soul.db ...] [--json]")
        print("只读盘点，不会修改任何文件；选哪一份由你决定。")
        return 2

    preview = build_migration_preview(paths)
    print(preview.to_json() if as_json else _format_text(preview))
    return 0 if any(c.readable for c in preview.candidates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
