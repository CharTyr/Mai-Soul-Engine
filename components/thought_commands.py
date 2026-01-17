from typing import Optional, Tuple
import re
import logging
from src.plugin_system import BaseCommand
from src.plugin_system.apis import send_api

logger = logging.getLogger(__name__)


class SeedListCommand(BaseCommand):
    command_name = "soul_seeds"
    command_description = "查看待审核的思维种子"
    command_pattern = r"^/soul_seeds\s*$"

    async def _send_response(self, text: str):
        """发送响应消息到聊天"""
        if self.message.chat_stream:
            await send_api.text_to_stream(text, self.message.chat_stream.stream_id, typing=False, storage_message=False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..utils.spectrum_utils import match_user
        from ..thought.seed_manager import ThoughtSeedManager

        admin_user_id = self.get_config("admin.admin_user_id", "")
        # 从 message_info 中正确获取平台和用户信息
        platform = self.message.message_info.platform if self.message.message_info else ""
        user_id = str(self.message.message_info.user_info.user_id) if self.message.message_info and self.message.message_info.user_info else ""

        if not match_user(platform, user_id, admin_user_id):
            msg = "只有管理员可以查看思维种子"
            await self._send_response(msg)
            return True, msg, 2

        if not self.get_config("thought_cabinet.enabled", False):
            msg = "思维阁系统未启用"
            await self._send_response(msg)
            return True, msg, 2

        config = {
            "max_seeds": self.get_config("thought_cabinet.max_seeds", 20),
            "min_trigger_intensity": self.get_config("thought_cabinet.min_trigger_intensity", 0.7),
            "admin_user_id": admin_user_id,
        }
        manager = ThoughtSeedManager(config)
        seeds = await manager.get_pending_seeds()

        msg = manager.format_seeds_list(seeds)
        await self._send_response(msg)
        return True, msg, 2


class SeedApproveCommand(BaseCommand):
    command_name = "soul_approve"
    command_description = "批准思维种子内化"
    command_pattern = r"^/soul_approve\s+(\w+)\s*$"

    async def _send_response(self, text: str):
        """发送响应消息到聊天"""
        if self.message.chat_stream:
            await send_api.text_to_stream(text, self.message.chat_stream.stream_id, typing=False, storage_message=False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..utils.spectrum_utils import match_user
        from ..thought.seed_manager import ThoughtSeedManager
        from ..thought.internalization_engine import InternalizationEngine

        admin_user_id = self.get_config("admin.admin_user_id", "")
        # 从 message_info 中正确获取平台和用户信息
        platform = self.message.message_info.platform if self.message.message_info else ""
        user_id = str(self.message.message_info.user_info.user_id) if self.message.message_info and self.message.message_info.user_info else ""

        if not match_user(platform, user_id, admin_user_id):
            msg = "只有管理员可以审核思维种子"
            await self._send_response(msg)
            return True, msg, 2

        if not self.get_config("thought_cabinet.enabled", False):
            msg = "思维阁系统未启用"
            await self._send_response(msg)
            return True, msg, 2

        # 从 processed_plain_text 获取消息内容
        content = self.message.processed_plain_text if hasattr(self.message, "processed_plain_text") else ""

        match = re.match(self.command_pattern, str(content))
        if not match:
            msg = "用法: /soul_approve <种子ID>"
            await self._send_response(msg)
            return True, msg, 2

        seed_id = match.group(1)

        config = {"max_seeds": 20, "min_trigger_intensity": 0.7, "admin_user_id": admin_user_id}
        manager = ThoughtSeedManager(config)
        seed = await manager.get_seed_by_id(seed_id)

        if not seed:
            msg = f"未找到种子 {seed_id}"
            await self._send_response(msg)
            return True, msg, 2

        if seed.get("status") != "pending":
            msg = f"种子 {seed_id} 不在待审核状态"
            await self._send_response(msg)
            return True, msg, 2

        engine = InternalizationEngine()
        result = await engine.internalize_seed(seed)

        if result["success"]:
            await manager.delete_seed(seed_id)
            impact = result["spectrum_impact"]
            impact_str = ", ".join([f"{k}:{v:+d}" for k, v in impact.items() if v != 0])
            msg = f"✅ 种子 {seed_id} 已批准内化\n\n固化观点: {result['thought'][:100]}...\n\n光谱影响: {impact_str or '无'}"
            await self._send_response(msg)
            return True, msg, 2
        else:
            msg = f"❌ 种子 {seed_id} 内化失败: {result['error']}"
            await self._send_response(msg)
            return True, msg, 2


class SeedRejectCommand(BaseCommand):
    command_name = "soul_reject"
    command_description = "拒绝思维种子"
    command_pattern = r"^/soul_reject\s+(\w+)\s*$"

    async def _send_response(self, text: str):
        """发送响应消息到聊天"""
        if self.message.chat_stream:
            await send_api.text_to_stream(text, self.message.chat_stream.stream_id, typing=False, storage_message=False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..utils.spectrum_utils import match_user

        admin_user_id = self.get_config("admin.admin_user_id", "")
        # 从 message_info 中正确获取平台和用户信息
        platform = self.message.message_info.platform if self.message.message_info else ""
        user_id = str(self.message.message_info.user_info.user_id) if self.message.message_info and self.message.message_info.user_info else ""

        if not match_user(platform, user_id, admin_user_id):
            msg = "只有管理员可以审核思维种子"
            await self._send_response(msg)
            return True, msg, 2

        if not self.get_config("thought_cabinet.enabled", False):
            msg = "思维阁系统未启用"
            await self._send_response(msg)
            return True, msg, 2

        # 从 processed_plain_text 获取消息内容
        content = self.message.processed_plain_text if hasattr(self.message, "processed_plain_text") else ""

        match = re.match(self.command_pattern, str(content))
        if not match:
            msg = "用法: /soul_reject <种子ID>"
            await self._send_response(msg)
            return True, msg, 2

        seed_id = match.group(1)

        from ..thought.seed_manager import ThoughtSeedManager

        config = {"max_seeds": 20, "min_trigger_intensity": 0.7, "admin_user_id": admin_user_id}
        manager = ThoughtSeedManager(config)
        seed = await manager.get_seed_by_id(seed_id)

        if not seed:
            msg = f"未找到种子 {seed_id}"
            await self._send_response(msg)
            return True, msg, 2

        if seed.get("status") != "pending":
            msg = f"种子 {seed_id} 不在待审核状态"
            await self._send_response(msg)
            return True, msg, 2

        await manager.delete_seed(seed_id)
        logger.info(f"管理员拒绝思维种子: {seed_id}")
        msg = f"✅ 种子 {seed_id} 已拒绝并删除"
        await self._send_response(msg)
        return True, msg, 2
