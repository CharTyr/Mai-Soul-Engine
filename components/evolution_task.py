import asyncio
import json
import logging
from typing import Optional, Tuple
from datetime import datetime
from src.plugin_system import BaseEventHandler, EventType
from src.common.data_models.mai_messages import MaiMessages

logger = logging.getLogger(__name__)


class EvolutionTaskHandler(BaseEventHandler):
    event_type = EventType.ON_START
    handler_name = "evolution_task"
    handler_description = "启动周期性演化任务"
    weight = 1
    intercept_message = False

    _task: Optional[asyncio.Task] = None

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[dict], Optional[MaiMessages]]:
        if not self.get_config("evolution_enabled", True):
            return True, True, None, None, message

        if EvolutionTaskHandler._task is None or EvolutionTaskHandler._task.done():
            EvolutionTaskHandler._task = asyncio.create_task(self._evolution_loop())
            logger.info("世界观演化任务已启动")

        return True, True, None, None, message

    async def _evolution_loop(self):
        from ..models.ideology_model import get_or_create_spectrum, init_tables

        init_tables()

        while True:
            try:
                interval_hours = self.get_config("evolution_interval_hours", 1.0)
                await asyncio.sleep(interval_hours * 3600)

                if not self.get_config("evolution_enabled", True):
                    continue

                spectrum = get_or_create_spectrum("global")
                if not spectrum.initialized:
                    continue

                evolution_rate = self.get_config("evolution_rate", 5)
                monitored_groups = self.get_config("monitored_groups", [])

                if not monitored_groups:
                    continue

                for group_config_id in monitored_groups:
                    await self._analyze_group(group_config_id, evolution_rate)

            except asyncio.CancelledError:
                logger.info("世界观演化任务已停止")
                break
            except Exception as e:
                logger.error(f"世界观演化任务出错: {e}")
                await asyncio.sleep(60)

    async def _analyze_group(self, group_config_id: str, evolution_rate: int):
        from ..models.ideology_model import (
            get_or_create_spectrum,
            GroupEvolutionRecord,
            EvolutionHistory,
        )
        from ..prompts.ideology_prompts import EVOLUTION_ANALYSIS_PROMPT
        from ..utils.spectrum_utils import update_spectrum_value, parse_chat_id
        from src.llm_models.utils_model import LLMRequest
        from src.plugin_system.apis.message_api import get_messages_by_time_in_chat

        try:
            platform, chat_id, chat_type = parse_chat_id(group_config_id)
            stream_id = f"{platform}:{chat_id}:{chat_type}" if platform else chat_id

            record, _ = GroupEvolutionRecord.get_or_create(group_id=stream_id)
            last_time = record.last_analyzed
            now = datetime.now()

            messages = await get_messages_by_time_in_chat(
                stream_id, last_time.timestamp(), now.timestamp()
            )

            if not messages or len(messages) < 5:
                return

            msg_text = "\n".join(
                [f"{getattr(m, 'sender_nickname', 'user')}: {getattr(m, 'content', '')}" for m in messages[:50]]
            )

            prompt = EVOLUTION_ANALYSIS_PROMPT.format(rate=evolution_rate, messages=msg_text)

            llm = LLMRequest()
            response, _ = await llm.generate_response_async(prompt)

            if not response:
                return

            try:
                response = response.strip()
                if response.startswith("```"):
                    response = response.split("\n", 1)[1].rsplit("```", 1)[0]
                deltas = json.loads(response)
            except json.JSONDecodeError:
                logger.warning(f"无法解析LLM响应: {response}")
                return

            spectrum = get_or_create_spectrum("global")

            economic_delta = max(-evolution_rate, min(evolution_rate, deltas.get("economic", 0)))
            social_delta = max(-evolution_rate, min(evolution_rate, deltas.get("social", 0)))
            diplomatic_delta = max(-evolution_rate, min(evolution_rate, deltas.get("diplomatic", 0)))
            progressive_delta = max(-evolution_rate, min(evolution_rate, deltas.get("progressive", 0)))

            spectrum.economic = update_spectrum_value(spectrum.economic, economic_delta)
            spectrum.social = update_spectrum_value(spectrum.social, social_delta)
            spectrum.diplomatic = update_spectrum_value(spectrum.diplomatic, diplomatic_delta)
            spectrum.progressive = update_spectrum_value(spectrum.progressive, progressive_delta)
            spectrum.last_evolution = now
            spectrum.updated_at = now
            spectrum.save()

            EvolutionHistory.create(
                timestamp=now,
                group_id=stream_id,
                economic_delta=economic_delta,
                social_delta=social_delta,
                diplomatic_delta=diplomatic_delta,
                progressive_delta=progressive_delta,
                reason=f"分析了{len(messages)}条消息",
            )

            record.last_analyzed = now
            record.save()

            logger.info(f"群{stream_id}演化完成: e={economic_delta}, s={social_delta}, d={diplomatic_delta}, p={progressive_delta}")

        except Exception as e:
            logger.error(f"分析群{group_config_id}时出错: {e}")
