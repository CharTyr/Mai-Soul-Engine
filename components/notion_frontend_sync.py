import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

from src.plugin_system import BaseEventHandler, EventType
from src.plugin_system.base.component_types import MaiMessages

logger = logging.getLogger(__name__)


class NotionFrontendSyncTask(BaseEventHandler):
    event_type = EventType.ON_START
    handler_name = "notion_frontend_sync"
    handler_description = "同步 traits 与意识形态光谱到 Notion 数据库（公共展示，可选）"
    weight = 2
    intercept_message = False

    _task: Optional[asyncio.Task] = None

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[dict], Optional[MaiMessages]]:
        if not self.get_config("notion.enabled", False):
            return True, True, None, None, message

        if NotionFrontendSyncTask._task is None or NotionFrontendSyncTask._task.done():
            NotionFrontendSyncTask._task = asyncio.create_task(self._sync_loop())
            logger.info("[Mai-Soul-Engine] Notion 前端同步任务已启动")
        return True, True, None, None, message

    async def _sync_loop(self) -> None:
        from ..utils.notion_frontend import build_notion_frontend_config, sync_notion_frontend

        plugin_dir = Path(__file__).parent.parent
        first_run = True
        while True:
            try:
                cfg = build_notion_frontend_config(self)
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
