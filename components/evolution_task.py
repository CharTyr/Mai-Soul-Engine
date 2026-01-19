import asyncio
import json
import logging
from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path
from src.plugin_system import BaseEventHandler, EventType
from src.plugin_system.base.component_types import MaiMessages

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

        evolution_enabled = self.get_config("evolution.evolution_enabled", True)
        logger.debug(f"演化任务检查: evolution_enabled={evolution_enabled}")

        if not evolution_enabled:
            logger.debug("演化任务未启用，跳过")
            return True, True, None, None, message

        plugin_dir = Path(__file__).parent.parent
        init_audit_log(plugin_dir)

        if EvolutionTaskHandler._task is None or EvolutionTaskHandler._task.done():
            EvolutionTaskHandler._task = asyncio.create_task(self._evolution_loop())
            logger.info("灵魂光谱演化任务已启动")
        else:
            logger.debug("演化任务已在运行中")

        return True, True, None, None, message

    async def _evolution_loop(self):
        from ..models.ideology_model import get_or_create_spectrum, init_tables
        from ..utils.spectrum_utils import match_chat, parse_chat_id

        init_tables()
        logger.debug("演化循环已初始化数据库表")

        while True:
            try:
                interval_hours = self.get_config("evolution.evolution_interval_hours", 1.0)
                logger.debug(f"演化循环等待 {interval_hours} 小时")
                await asyncio.sleep(interval_hours * 3600)

                if not self.get_config("evolution.evolution_enabled", True):
                    logger.debug("演化已禁用，跳过本轮")
                    continue

                spectrum = get_or_create_spectrum("global")
                if not spectrum.initialized:
                    logger.debug("光谱未初始化，跳过本轮")
                    continue

                evolution_rate = self.get_config("evolution.evolution_rate", 5)
                monitored_groups = self.get_config("monitor.monitored_groups", []) or []
                excluded_groups = self.get_config("monitor.excluded_groups", []) or []
                logger.debug(
                    f"演化参数: rate={evolution_rate}, groups={monitored_groups}, excluded_groups={excluded_groups}"
                )

                if not monitored_groups:
                    logger.debug("无监控群组，跳过本轮")
                    continue

                groups_to_analyze: list[str] = []
                for group_config_id in monitored_groups:
                    platform, chat_id, chat_type = parse_chat_id(group_config_id)
                    if any(match_chat(platform, chat_id, chat_type, exc) for exc in excluded_groups):
                        logger.debug(f"群组已在排除列表中，跳过: {group_config_id}")
                        continue
                    groups_to_analyze.append(group_config_id)

                if not groups_to_analyze:
                    logger.debug("监控群组全部被排除，跳过本轮")
                    continue

                for group_config_id in groups_to_analyze:
                    logger.debug(f"开始分析群组: {group_config_id}")
                    await self._analyze_group(group_config_id, evolution_rate)

            except asyncio.CancelledError:
                logger.info("灵魂光谱演化任务已停止")
                break
            except Exception as e:
                logger.error(f"灵魂光谱演化任务出错: {e}", exc_info=True)
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
            chat_config_to_stream_id,
            sanitize_text,
            smooth_delta,
            apply_resistance,
            is_user_monitored,
        )
        from ..utils.audit_log import log_evolution
        from src.chat.utils.utils import is_bot_self
        from src.llm_models.utils_model import LLMRequest
        from src.plugin_system.apis.message_api import get_messages_by_time_in_chat

        try:
            stream_id = chat_config_to_stream_id(group_config_id)

            record, _ = GroupEvolutionRecord.get_or_create(group_id=stream_id)
            last_time = record.last_analyzed
            now = datetime.now()

            messages = await get_messages_by_time_in_chat(
                stream_id,
                last_time.timestamp(),
                now.timestamp(),
                filter_command=True,
                filter_intercept_message_level=1,
            )
            logger.debug(f"群{stream_id}获取消息: {len(messages) if messages else 0}条, 时间范围: {last_time} - {now}")

            if not messages or len(messages) < 5:
                logger.debug(f"群{stream_id}消息不足5条，跳过分析")
                return

            max_messages = self.get_config("evolution.max_messages_per_analysis", 200)
            max_chars = self.get_config("evolution.max_chars_per_message", 200)

            monitor_config = {
                "monitored_users": self.get_config("monitor.monitored_users", []) or [],
                "excluded_users": self.get_config("monitor.excluded_users", []) or [],
            }
            filtered_messages = []
            for m in messages:
                user_info = getattr(m, "user_info", None)
                msg_platform = str(getattr(user_info, "platform", "") or "")
                msg_user_id = str(getattr(user_info, "user_id", "") or "")
                if is_bot_self(msg_platform, msg_user_id):
                    continue
                if not is_user_monitored(msg_platform, msg_user_id, monitor_config):
                    continue
                filtered_messages.append(m)

            messages = filtered_messages
            if len(messages) < 5:
                logger.debug(f"群{stream_id}过滤后消息不足5条，跳过分析")
                return

            msg_lines = []
            for m in messages[:max_messages]:
                user_info = getattr(m, "user_info", None)
                nickname = (
                    getattr(user_info, "user_cardname", None)
                    or getattr(user_info, "user_nickname", None)
                    or getattr(user_info, "user_id", None)
                    or "user"
                )
                content = getattr(m, "processed_plain_text", None) or getattr(m, "display_message", None) or ""
                sanitized = sanitize_text(str(content), max_chars=max_chars)
                if sanitized:
                    msg_lines.append(f"{nickname}: {sanitized}")
            msg_text = "\n".join(msg_lines)

            prompt = EVOLUTION_ANALYSIS_PROMPT.format(rate=evolution_rate, messages=msg_text)

            thought_cabinet_enabled = self.get_config("thought_cabinet.enabled", False)
            logger.debug(f"思维阁启用状态: {thought_cabinet_enabled}")
            if thought_cabinet_enabled:
                from ..prompts.thought_prompts import ENHANCED_EVOLUTION_PROMPT

                prompt = ENHANCED_EVOLUTION_PROMPT.format(rate=evolution_rate, messages=msg_text)

            llm = LLMRequest()
            logger.debug(f"发送LLM请求，prompt长度: {len(prompt)}")
            response, _ = await llm.generate_response_async(prompt)
            logger.debug(f"LLM响应长度: {len(response) if response else 0}")

            if not response:
                return

            try:
                response = response.strip()
                if response.startswith("```"):
                    response = response.split("\n", 1)[1].rsplit("```", 1)[0]
                result = json.loads(response)

                if thought_cabinet_enabled and "spectrum_deltas" in result:
                    deltas = result["spectrum_deltas"]
                    thought_seeds = result.get("thought_seeds", [])
                    await self._process_thought_seeds(thought_seeds, stream_id)
                else:
                    deltas = result
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

            ema_alpha = self.get_config("evolution.ema_alpha", 0.3)
            resistance = self.get_config("evolution.direction_resistance", 0.5)

            raw_deltas = {
                "economic": max(-evolution_rate, min(evolution_rate, deltas.get("economic", 0))),
                "social": max(-evolution_rate, min(evolution_rate, deltas.get("social", 0))),
                "diplomatic": max(-evolution_rate, min(evolution_rate, deltas.get("diplomatic", 0))),
                "progressive": max(-evolution_rate, min(evolution_rate, deltas.get("progressive", 0))),
            }

            resisted_deltas = {}
            new_dirs = {}
            for dim in ["economic", "social", "diplomatic", "progressive"]:
                last_dir = getattr(spectrum, f"last_{dim}_dir", 0)
                adj_delta, new_dir = apply_resistance(raw_deltas[dim], last_dir, resistance)
                resisted_deltas[dim] = adj_delta
                new_dirs[dim] = new_dir

            smoothed_deltas = {
                "economic": smooth_delta(spectrum.economic, resisted_deltas["economic"], ema_alpha),
                "social": smooth_delta(spectrum.social, resisted_deltas["social"], ema_alpha),
                "diplomatic": smooth_delta(spectrum.diplomatic, resisted_deltas["diplomatic"], ema_alpha),
                "progressive": smooth_delta(spectrum.progressive, resisted_deltas["progressive"], ema_alpha),
            }

            spectrum.economic = update_spectrum_value(spectrum.economic, smoothed_deltas["economic"])
            spectrum.social = update_spectrum_value(spectrum.social, smoothed_deltas["social"])
            spectrum.diplomatic = update_spectrum_value(spectrum.diplomatic, smoothed_deltas["diplomatic"])
            spectrum.progressive = update_spectrum_value(spectrum.progressive, smoothed_deltas["progressive"])
            spectrum.last_economic_dir = new_dirs["economic"]
            spectrum.last_social_dir = new_dirs["social"]
            spectrum.last_diplomatic_dir = new_dirs["diplomatic"]
            spectrum.last_progressive_dir = new_dirs["progressive"]
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
            logger.error(f"分析群{group_config_id}时出错: {e}", exc_info=True)

    async def _process_thought_seeds(self, seeds: list, stream_id: str):
        from ..thought.seed_manager import ThoughtSeedManager

        logger.debug(f"处理思维种子: 收到 {len(seeds)} 个")
        if not seeds:
            return

        config = {
            "max_seeds": self.get_config("thought_cabinet.max_seeds", 20),
            "min_trigger_intensity": self.get_config("thought_cabinet.min_trigger_intensity", 0.7),
            "admin_user_id": self.get_config("admin.admin_user_id", ""),
        }
        manager = ThoughtSeedManager(config)

        for seed_data in seeds[:2]:
            seed_id = await manager.create_seed(seed_data)
            if seed_id:
                logger.info(f"群{stream_id}创建思维种子: {seed_id}")
                if self.get_config("thought_cabinet.admin_notification_enabled", True):
                    await self._notify_admin_seed(manager, seed_id, seed_data)

    async def _notify_admin_seed(self, manager, seed_id: str, seed_data: dict) -> None:
        """向管理员私聊发送思维种子通知（尽量不影响主流程）。"""
        from maim_message import UserInfo
        from src.chat.message_receive.chat_stream import get_chat_manager
        from ..utils.spectrum_utils import parse_user_id

        admin_config_id = self.get_config("admin.admin_user_id", "")
        if not admin_config_id:
            return

        platform, user_id = parse_user_id(admin_config_id)
        if not platform or not user_id:
            return

        chat_manager = get_chat_manager()
        admin_stream_id = chat_manager.get_stream_id(platform, user_id, is_group=False)
        if not chat_manager.get_stream(admin_stream_id):
            try:
                await chat_manager.get_or_create_stream(
                    platform=platform,
                    user_info=UserInfo(platform=platform, user_id=user_id, user_nickname=user_id),
                    group_info=None,
                )
            except Exception:
                logger.exception("创建管理员私聊聊天流失败，无法发送种子通知")
                return

        try:
            await self.send_text(
                admin_stream_id,
                manager.format_seed_notification(seed_id, seed_data),
                storage_message=False,
            )
        except Exception:
            logger.exception("发送思维种子通知失败")
