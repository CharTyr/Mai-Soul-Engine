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
        from ..models.ideology_model import init_tables

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

        init_tables()

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
        from ..models.ideology_model import init_tables

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

        init_tables()

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
            trait_id = result.get("trait_id", "")
            trait_line = f"\ntrait_id: {trait_id}" if trait_id else ""
            msg = (
                f"✅ 种子 {seed_id} 已批准内化{trait_line}\n\n"
                f"固化观点: {result['thought'][:100]}...\n\n"
                f"光谱影响: {impact_str or '无'}"
            )
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
        from ..models.ideology_model import init_tables

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

        init_tables()

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


class TraitListCommand(BaseCommand):
    command_name = "soul_traits"
    command_description = "查看已固化的 traits（可按群过滤）"
    command_pattern = r"^/soul_traits(?:\s+(\S+))?\s*$"

    async def _send_response(self, text: str):
        if self.message.chat_stream:
            await send_api.text_to_stream(text, self.message.chat_stream.stream_id, typing=False, storage_message=False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..utils.spectrum_utils import match_user
        from ..models.ideology_model import CrystallizedTrait, init_tables

        admin_user_id = self.get_config("admin.admin_user_id", "")
        platform = self.message.message_info.platform if self.message.message_info else ""
        user_id = (
            str(self.message.message_info.user_info.user_id)
            if self.message.message_info and self.message.message_info.user_info
            else ""
        )

        if not match_user(platform, user_id, admin_user_id):
            msg = "只有管理员可以查看 traits"
            await self._send_response(msg)
            return True, msg, 2

        if not self.get_config("thought_cabinet.enabled", False):
            msg = "思维阁系统未启用"
            await self._send_response(msg)
            return True, msg, 2

        content = self.message.processed_plain_text if hasattr(self.message, "processed_plain_text") else ""
        match = re.match(self.command_pattern, str(content))
        stream_id = match.group(1).strip() if match and match.group(1) else None

        init_tables()

        query = CrystallizedTrait.select().where(CrystallizedTrait.deleted == False)  # noqa: E712
        if stream_id and stream_id != "global":
            query = query.where(CrystallizedTrait.stream_id == stream_id)

        traits = list(query.order_by(CrystallizedTrait.created_at.desc()).limit(50))
        if not traits:
            msg = "当前没有已固化的 traits"
            await self._send_response(msg)
            return True, msg, 2

        lines = ["🧠 已固化 traits：", ""]
        if stream_id:
            lines.append(f"过滤 stream_id: {stream_id}")
            lines.append("")

        for t in traits:
            status = "enabled" if t.enabled else "disabled"
            lines.append(f"- {t.trait_id} [{status}] stream={t.stream_id or '-'} name={t.name}")
            q = (getattr(t, "question", "") or "").replace("\n", " ").strip()
            if q:
                if len(q) > 80:
                    q = f"{q[:80]}..."
                lines.append(f"  问: {q}")
            snippet = (t.thought or "").replace("\n", " ").strip()
            if len(snippet) > 80:
                snippet = f"{snippet[:80]}..."
            if snippet:
                lines.append(f"  {snippet}")

        msg = "\n".join(lines)
        await self._send_response(msg)
        return True, msg, 2


class TraitDisableCommand(BaseCommand):
    command_name = "soul_trait_disable"
    command_description = "禁用指定 trait"
    command_pattern = r"^/soul_trait_disable\s+(\w+)\s*$"

    async def _send_response(self, text: str):
        if self.message.chat_stream:
            await send_api.text_to_stream(text, self.message.chat_stream.stream_id, typing=False, storage_message=False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..utils.spectrum_utils import match_user
        from ..models.ideology_model import CrystallizedTrait, init_tables

        admin_user_id = self.get_config("admin.admin_user_id", "")
        platform = self.message.message_info.platform if self.message.message_info else ""
        user_id = (
            str(self.message.message_info.user_info.user_id)
            if self.message.message_info and self.message.message_info.user_info
            else ""
        )

        if not match_user(platform, user_id, admin_user_id):
            msg = "只有管理员可以禁用 trait"
            await self._send_response(msg)
            return True, msg, 2

        if not self.get_config("thought_cabinet.enabled", False):
            msg = "思维阁系统未启用"
            await self._send_response(msg)
            return True, msg, 2

        content = self.message.processed_plain_text if hasattr(self.message, "processed_plain_text") else ""
        match = re.match(self.command_pattern, str(content))
        if not match:
            msg = "用法: /soul_trait_disable <trait_id>"
            await self._send_response(msg)
            return True, msg, 2

        trait_id = match.group(1)
        init_tables()

        trait = CrystallizedTrait.get_or_none(CrystallizedTrait.trait_id == trait_id)
        if not trait or trait.deleted:
            msg = f"未找到 trait {trait_id}"
            await self._send_response(msg)
            return True, msg, 2

        trait.enabled = False
        trait.save()

        msg = f"✅ trait {trait_id} 已禁用"
        await self._send_response(msg)
        return True, msg, 2


class TraitEnableCommand(BaseCommand):
    command_name = "soul_trait_enable"
    command_description = "启用指定 trait"
    command_pattern = r"^/soul_trait_enable\s+(\w+)\s*$"

    async def _send_response(self, text: str):
        if self.message.chat_stream:
            await send_api.text_to_stream(text, self.message.chat_stream.stream_id, typing=False, storage_message=False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..utils.spectrum_utils import match_user
        from ..models.ideology_model import CrystallizedTrait, init_tables

        admin_user_id = self.get_config("admin.admin_user_id", "")
        platform = self.message.message_info.platform if self.message.message_info else ""
        user_id = (
            str(self.message.message_info.user_info.user_id)
            if self.message.message_info and self.message.message_info.user_info
            else ""
        )

        if not match_user(platform, user_id, admin_user_id):
            msg = "只有管理员可以启用 trait"
            await self._send_response(msg)
            return True, msg, 2

        if not self.get_config("thought_cabinet.enabled", False):
            msg = "思维阁系统未启用"
            await self._send_response(msg)
            return True, msg, 2

        content = self.message.processed_plain_text if hasattr(self.message, "processed_plain_text") else ""
        match = re.match(self.command_pattern, str(content))
        if not match:
            msg = "用法: /soul_trait_enable <trait_id>"
            await self._send_response(msg)
            return True, msg, 2

        trait_id = match.group(1)
        init_tables()

        trait = CrystallizedTrait.get_or_none(CrystallizedTrait.trait_id == trait_id)
        if not trait or trait.deleted:
            msg = f"未找到 trait {trait_id}"
            await self._send_response(msg)
            return True, msg, 2

        trait.enabled = True
        trait.save()

        msg = f"✅ trait {trait_id} 已启用"
        await self._send_response(msg)
        return True, msg, 2


class TraitDeleteCommand(BaseCommand):
    command_name = "soul_trait_delete"
    command_description = "删除指定 trait（软删除）"
    command_pattern = r"^/soul_trait_delete\s+(\w+)\s*$"

    async def _send_response(self, text: str):
        if self.message.chat_stream:
            await send_api.text_to_stream(text, self.message.chat_stream.stream_id, typing=False, storage_message=False)

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..utils.spectrum_utils import match_user
        from ..models.ideology_model import CrystallizedTrait, init_tables

        admin_user_id = self.get_config("admin.admin_user_id", "")
        platform = self.message.message_info.platform if self.message.message_info else ""
        user_id = (
            str(self.message.message_info.user_info.user_id)
            if self.message.message_info and self.message.message_info.user_info
            else ""
        )

        if not match_user(platform, user_id, admin_user_id):
            msg = "只有管理员可以删除 trait"
            await self._send_response(msg)
            return True, msg, 2

        if not self.get_config("thought_cabinet.enabled", False):
            msg = "思维阁系统未启用"
            await self._send_response(msg)
            return True, msg, 2

        content = self.message.processed_plain_text if hasattr(self.message, "processed_plain_text") else ""
        match = re.match(self.command_pattern, str(content))
        if not match:
            msg = "用法: /soul_trait_delete <trait_id>"
            await self._send_response(msg)
            return True, msg, 2

        trait_id = match.group(1)
        init_tables()

        trait = CrystallizedTrait.get_or_none(CrystallizedTrait.trait_id == trait_id)
        if not trait or trait.deleted:
            msg = f"未找到 trait {trait_id}"
            await self._send_response(msg)
            return True, msg, 2

        trait.enabled = False
        trait.deleted = True
        trait.save()

        msg = f"✅ trait {trait_id} 已删除"
        await self._send_response(msg)
        return True, msg, 2
