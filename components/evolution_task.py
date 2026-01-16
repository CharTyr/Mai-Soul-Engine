import asyncio
import json
import logging
from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path
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
        from ..utils.audit_log import init_audit_log

        if not self.get_config("evolution_enabled", True):
            return True, True, None, None, message

        plugin_dir = Path(__file__).parent.parent
        init_audit_log(plugin_dir)

        if EvolutionTaskHandler._task is None or EvolutionTaskHandler._task.done():
            EvolutionTaskHandler._task = asyncio.create_task(self._evolution_loop())
            logger.info("灵魂光谱演化任务已启动")

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
                logger.info("灵魂光谱演化任务已停止")
                break
            except Exception as e:
                logger.error(f"灵魂光谱演化任务出错: {e}")
                await asyncio.sleep(60)

    async def _analyze_group(self, group_config_id: str, evolution_rate: int):
        from ..models.ideology_model import (
            get_or_create_spectrum,
            GroupEvolutionRecord,
            EvolutionHistory,
        )
        from ..prompts.ideology_prompts import EVOLUTION_ANALYSIS_PROMPT
        from ..utils.spectrum_utils import (
            update_spectrum_value,
            parse_chat_id,
            sanitize_text,
            smooth_delta,
        )
        from ..utils.audit_log import log_evolution
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

            msg_lines = []
            for m in messages[:50]:
                nickname = getattr(m, "sender_nickname", "user")
                content = str(getattr(m, "content", ""))
                sanitized = sanitize_text(content, max_chars=200)
                msg_lines.append(f"{nickname}: {sanitized}")
            msg_text = "\n".join(msg_lines)

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
            
            before = {
                "economic": spectrum.economic,
                "social": spectrum.social,
                "diplomatic": spectrum.diplomatic,
                "progressive": spectrum.progressive,
            }

            ema_alpha = self.get_config("ema_alpha", 0.3)
            
            raw_deltas = {
                "economic": max(-evolution_rate, min(evolution_rate, deltas.get("economic", 0))),
                "social": max(-evolution_rate, min(evolution_rate, deltas.get("social", 0))),
                "diplomatic": max(-evolution_rate, min(evolution_rate, deltas.get("diplomatic", 0))),
                "progressive": max(-evolution_rate, min(evolution_rate, deltas.get("progressive", 0))),
            }
            
            smoothed_deltas = {
                "economic": smooth_delta(spectrum.economic, raw_deltas["economic"], ema_alpha),
                "social": smooth_delta(spectrum.social, raw_deltas["social"], ema_alpha),
                "diplomatic": smooth_delta(spectrum.diplomatic, raw_deltas["diplomatic"], ema_alpha),
                "progressive": smooth_delta(spectrum.progressive, raw_deltas["progressive"], ema_alpha),
            }

            spectrum.economic = update_spectrum_value(spectrum.economic, smoothed_deltas["economic"])
            spectrum.social = update_spectrum_value(spectrum.social, smoothed_deltas["social"])
            spectrum.diplomatic = update_spectrum_value(spectrum.diplomatic, smoothed_deltas["diplomatic"])
            spectrum.progressive = update_spectrum_value(spectrum.progressive, smoothed_deltas["progressive"])
            spectrum.last_evolution = now
            spectrum.updated_at = now
            spectrum.save()

            after = {
                "economic": spectrum.economic,
                "social": spectrum.social,
                "diplomatic": spectrum.diplomatic,
                "progressive": spectrum.progressive,
            }

            EvolutionHistory.create(
                timestamp=now,
                group_id=stream_id,
                economic_delta=smoothed_deltas["economic"],
                social_delta=smoothed_deltas["social"],
                diplomatic_delta=smoothed_deltas["diplomatic"],
                progressive_delta=smoothed_deltas["progressive"],
                reason=f"分析了{len(messages)}条消息",
            )

            await log_evolution(
                group_id=stream_id,
                before=before,
                after=after,
                deltas=smoothed_deltas,
                reason=f"分析了{len(messages)}条消息",
                message_count=len(messages),
            )

            record.last_analyzed = now
            record.save()

            logger.info(
                f"群{stream_id}演化完成: e={smoothed_deltas['economic']}, "
                f"s={smoothed_deltas['social']}, d={smoothed_deltas['diplomatic']}, "
                f"p={smoothed_deltas['progressive']}"
            )

        except Exception as e:
            logger.error(f"分析群{group_config_id}时出错: {e}")
