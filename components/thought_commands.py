from typing import Optional, Tuple
import re
import logging
from src.plugin_system import BaseCommand

logger = logging.getLogger(__name__)


class SeedListCommand(BaseCommand):
    command_name = "soul_seeds"
    command_description = "查看待审核的思维种子"
    command_pattern = r"^/soul_seeds\s*$"

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..utils.spectrum_utils import match_user
        from ..thought.seed_manager import ThoughtSeedManager

        admin_user_id = self.get_config("admin_user_id", "")
        platform = getattr(self.message, "platform", "")
        user_id = str(getattr(self.message, "user_id", ""))

        if not match_user(platform, user_id, admin_user_id):
            return True, "只有管理员可以查看思维种子", 2

        if not self.get_config("enabled", False):
            return True, "思维阁系统未启用", 2

        config = {
            "max_seeds": self.get_config("max_seeds", 20),
            "min_trigger_intensity": self.get_config("min_trigger_intensity", 0.7),
            "admin_user_id": admin_user_id,
        }
        manager = ThoughtSeedManager(config)
        seeds = await manager.get_pending_seeds()

        return True, manager.format_seeds_list(seeds), 2


class SeedApproveCommand(BaseCommand):
    command_name = "soul_approve"
    command_description = "批准思维种子内化"
    command_pattern = r"^/soul_approve\s+(\w+)\s*$"

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..utils.spectrum_utils import match_user
        from ..thought.seed_manager import ThoughtSeedManager
        from ..thought.internalization_engine import InternalizationEngine

        admin_user_id = self.get_config("admin_user_id", "")
        platform = getattr(self.message, "platform", "")
        user_id = str(getattr(self.message, "user_id", ""))

        if not match_user(platform, user_id, admin_user_id):
            return True, "只有管理员可以审核思维种子", 2

        if not self.get_config("enabled", False):
            return True, "思维阁系统未启用", 2

        content = getattr(self.message, "content", "")
        if hasattr(content, "get_plain_text"):
            content = content.get_plain_text()

        match = re.match(self.command_pattern, str(content))
        if not match:
            return True, "用法: /soul_approve <种子ID>", 2

        seed_id = match.group(1)

        config = {"max_seeds": 20, "min_trigger_intensity": 0.7, "admin_user_id": admin_user_id}
        manager = ThoughtSeedManager(config)
        seed = await manager.get_seed_by_id(seed_id)

        if not seed:
            return True, f"未找到种子 {seed_id}", 2

        if "待审核" not in seed.get("content", ""):
            return True, f"种子 {seed_id} 不在待审核状态", 2

        engine = InternalizationEngine()
        result = await engine.internalize_seed(seed)

        if result["success"]:
            impact = result["spectrum_impact"]
            impact_str = ", ".join([f"{k}:{v:+d}" for k, v in impact.items() if v != 0])
            return (
                True,
                f"✅ 种子 {seed_id} 已批准内化\n\n固化观点: {result['thought'][:100]}...\n\n光谱影响: {impact_str or '无'}",
                2,
            )
        else:
            return True, f"❌ 种子 {seed_id} 内化失败: {result['error']}", 2


class SeedRejectCommand(BaseCommand):
    command_name = "soul_reject"
    command_description = "拒绝思维种子"
    command_pattern = r"^/soul_reject\s+(\w+)\s*$"

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        from ..utils.spectrum_utils import match_user

        admin_user_id = self.get_config("admin_user_id", "")
        platform = getattr(self.message, "platform", "")
        user_id = str(getattr(self.message, "user_id", ""))

        if not match_user(platform, user_id, admin_user_id):
            return True, "只有管理员可以审核思维种子", 2

        if not self.get_config("enabled", False):
            return True, "思维阁系统未启用", 2

        content = getattr(self.message, "content", "")
        if hasattr(content, "get_plain_text"):
            content = content.get_plain_text()

        match = re.match(self.command_pattern, str(content))
        if not match:
            return True, "用法: /soul_reject <种子ID>", 2

        seed_id = match.group(1)

        from ..thought.seed_manager import ThoughtSeedManager

        config = {"max_seeds": 20, "min_trigger_intensity": 0.7, "admin_user_id": admin_user_id}
        manager = ThoughtSeedManager(config)
        seed = await manager.get_seed_by_id(seed_id)

        if not seed:
            return True, f"未找到种子 {seed_id}", 2

        if "待审核" not in seed.get("content", ""):
            return True, f"种子 {seed_id} 不在待审核状态", 2

        await manager.delete_seed(seed_id)
        logger.info(f"管理员拒绝思维种子: {seed_id}")
        return True, f"✅ 种子 {seed_id} 已拒绝并删除", 2
