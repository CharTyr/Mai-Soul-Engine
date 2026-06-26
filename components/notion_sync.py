"""Notion 同步组件 — 将 traits 与光谱同步到 Notion 数据库（可选展示）。

周期性地从插件本地数据同步到 Notion 数据库，供公共展示使用。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..utils.notion_frontend import (
    NotionFrontendConfig,
    NotionPropertyMap,
    NotionSpectrumPropertyMap,
    sync_notion_frontend,
)

logger = logging.getLogger(__name__)


async def run_notion_sync_loop(plugin) -> None:
    """Notion 同步循环 — 周期性同步 traits 与光谱到 Notion。

    由 plugin._notion_sync_loop() 以 asyncio.create_task 启动。
    首次同步有可配置的延迟，之后按 sync_interval_seconds 间隔执行。

    Args:
        plugin: MaiSoulEnginePlugin 实例。
    """
    plugin_dir: Path = plugin._plugin_dir
    first_run = True

    while True:
        try:
            cfg = _build_config(plugin)
            if not cfg.enabled:
                await asyncio.sleep(60)
                continue

            if first_run and cfg.first_delay_seconds > 0:
                await asyncio.sleep(cfg.first_delay_seconds)
            first_run = False

            res = await asyncio.to_thread(sync_notion_frontend, plugin_dir=plugin_dir, cfg=cfg)
            logger.debug("[Mai-Soul-Engine] Notion 同步结果: %s", res)

            await asyncio.sleep(cfg.sync_interval_seconds)
        except asyncio.CancelledError:
            logger.info("[Mai-Soul-Engine] Notion 前端同步任务已停止")
            break
        except Exception as e:
            logger.error("[Mai-Soul-Engine] Notion 前端同步任务出错: %s", e, exc_info=True)
            await asyncio.sleep(60)


def _build_config(plugin) -> NotionFrontendConfig:
    """从 plugin.config.notion 构建 NotionFrontendConfig。"""
    import os as _os

    nc = plugin.config.notion

    token = str(nc.token or "").strip() or _os.getenv("MAIBOT_SOUL_NOTION_TOKEN", "").strip()

    return NotionFrontendConfig(
        enabled=bool(nc.enabled),
        token=token,
        database_id=str(nc.database_id or "").replace("-", "").replace(" ", ""),
        sync_spectrum=bool(nc.sync_spectrum),
        spectrum_database_id=str(nc.spectrum_database_id or "").replace("-", "").replace(" ", ""),
        spectrum_scope_id=str(nc.spectrum_scope_id or "global").strip() or "global",
        spectrum_mode=str(nc.spectrum_mode or "dimension_rows").strip() or "dimension_rows",
        sync_interval_seconds=max(60, int(nc.sync_interval_seconds or 600)),
        first_delay_seconds=max(0, int(nc.first_delay_seconds or 5)),
        max_traits=max(0, int(nc.max_traits or 200)),
        visibility_default=str(nc.visibility_default or "Public").strip() or "Public",
        never_overwrite_user_fields=bool(nc.never_overwrite_user_fields),
        max_rich_text_chars=max(200, int(nc.max_rich_text_chars or 1800)),
        property_map=NotionPropertyMap(
            title=str(nc.property_title or "Name"),
            trait_id=str(nc.property_trait_id or "TraitId"),
            tags=str(nc.property_tags or "Tags"),
            question=str(nc.property_question or "Question"),
            thought=str(nc.property_thought or "Thought"),
            confidence=str(nc.property_confidence or "Confidence"),
            impact_score=str(nc.property_impact_score or "ImpactScore"),
            status=str(nc.property_status or "Status"),
            visibility=str(nc.property_visibility or "Visibility"),
            updated_at=str(nc.property_updated_at or "UpdatedAt"),
        ),
        spectrum_property_map=NotionSpectrumPropertyMap(
            title=str(nc.spectrum_property_title or "Name"),
            scope_id=str(nc.spectrum_property_scope_id or "ScopeId"),
            economic=str(nc.spectrum_property_economic or "Sincerity"),
            social=str(nc.spectrum_property_social or "Engagement"),
            diplomatic=str(nc.spectrum_property_diplomatic or "Closeness"),
            progressive=str(nc.spectrum_property_progressive or "Directness"),
            value=str(nc.spectrum_property_value or "Value"),
            initialized=str(nc.spectrum_property_initialized or "Initialized"),
            last_evolution=str(nc.spectrum_property_last_evolution or "LastEvolution"),
            updated_at=str(nc.spectrum_property_updated_at or "UpdatedAt"),
        ),
    )
