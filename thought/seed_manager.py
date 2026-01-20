import json
import uuid
import logging
from datetime import datetime
from typing import Optional, List

logger = logging.getLogger(__name__)

THOUGHT_TYPES = {
    "道德审判": "对是非善恶的判断和立场",
    "权力质疑": "对权力结构和社会秩序的质疑",
    "存在焦虑": "对存在意义和人生价值的思考",
    "集体认同": "对群体归属和身份认同的思考",
    "变革渴望": "对现状改变和未来发展的渴望",
}


class ThoughtSeedManager:
    def __init__(self, config: dict):
        self.max_seeds = config.get("max_seeds", 20)
        self.min_intensity = config.get("min_trigger_intensity", 0.7)
        self.admin_user_id = config.get("admin_user_id", "")

    async def create_seed(self, seed_data: dict, stream_id: str = "") -> Optional[str]:
        """创建思维种子并存入数据库（不写入外部知识库）"""
        from ..models.ideology_model import ThoughtSeed

        logger.debug(f"尝试创建种子, 强度: {seed_data.get('intensity', 0)}, 阈值: {self.min_intensity}")

        if seed_data.get("intensity", 0) < self.min_intensity:
            logger.debug(f"种子强度不足，跳过创建")
            return None

        await self._cleanup_excess_seeds()

        seed_id = f"seed_{uuid.uuid4().hex[:8]}"

        # 存入数据库
        ThoughtSeed.create(
            seed_id=seed_id,
            stream_id=stream_id or "",
            seed_type=seed_data["type"],
            event=seed_data["event"],
            intensity=int(seed_data["intensity"] * 100),  # 转换为0-100整数
            reasoning=seed_data["reasoning"],
            potential_impact_json=json.dumps(seed_data.get("potential_impact", {}), ensure_ascii=False),
            status="pending",
        )

        logger.info(f"创建思维种子: {seed_id} (类型: {seed_data['type']}, 强度: {seed_data['intensity']:.2f})")
        return seed_id

    async def _cleanup_excess_seeds(self):
        """清理超限的旧种子"""
        from ..models.ideology_model import ThoughtSeed

        seeds = list(ThoughtSeed.select().where(ThoughtSeed.status == "pending").order_by(ThoughtSeed.created_at.desc()))
        logger.debug(f"当前待审核种子数: {len(seeds)}, 最大限制: {self.max_seeds}")

        if len(seeds) >= self.max_seeds:
            logger.info(f"种子数超限，清理 {len(seeds) - self.max_seeds + 1} 个旧种子")
            for seed in seeds[self.max_seeds - 1 :]:
                seed.delete_instance()
                logger.debug(f"清理旧种子: {seed.seed_id}")

    async def delete_seed(self, seed_id: str) -> bool:
        """删除种子（从数据库）"""
        from ..models.ideology_model import ThoughtSeed

        logger.debug(f"删除种子: {seed_id}")
        try:
            seed = ThoughtSeed.get(ThoughtSeed.seed_id == seed_id)
            seed.delete_instance()
            logger.info(f"种子已删除: {seed_id}")
            return True
        except ThoughtSeed.DoesNotExist:
            logger.warning(f"删除种子失败: {seed_id} 不存在")
            return False

    async def get_pending_seeds(self, stream_id: str | None = None) -> List[dict]:
        """获取所有待审核种子"""
        from ..models.ideology_model import ThoughtSeed

        query = ThoughtSeed.select().where(ThoughtSeed.status == "pending")
        if stream_id and stream_id != "global":
            query = query.where(ThoughtSeed.stream_id == stream_id)
        seeds = list(query.order_by(ThoughtSeed.created_at.desc()))
        logger.debug(f"查询待审核种子, 找到 {len(seeds)} 个")

        # 转换为字典格式以保持兼容性
        result = []
        for seed in seeds:
            result.append(
                {
                    "seed_id": seed.seed_id,
                    "stream_id": getattr(seed, "stream_id", "") or "",
                    "type": seed.seed_type,
                    "event": seed.event,
                    "intensity": seed.intensity / 100.0,  # 转回0-1范围
                    "reasoning": seed.reasoning,
                    "potential_impact": json.loads(seed.potential_impact_json),
                    "created_at": seed.created_at.isoformat(),
                    "status": seed.status,
                }
            )
        return result

    async def get_seed_by_id(self, seed_id: str) -> Optional[dict]:
        """根据ID获取种子"""
        from ..models.ideology_model import ThoughtSeed

        logger.debug(f"查询种子: {seed_id}")
        try:
            seed = ThoughtSeed.get(ThoughtSeed.seed_id == seed_id)
            logger.debug(f"找到种子: {seed_id}")
            return {
                "seed_id": seed.seed_id,
                "stream_id": getattr(seed, "stream_id", "") or "",
                "type": seed.seed_type,
                "event": seed.event,
                "intensity": seed.intensity / 100.0,
                "reasoning": seed.reasoning,
                "potential_impact": json.loads(seed.potential_impact_json),
                "created_at": seed.created_at.isoformat(),
                "status": seed.status,
            }
        except ThoughtSeed.DoesNotExist:
            logger.debug(f"未找到种子: {seed_id}")
            return None

    def format_seed_notification(self, seed_id: str, seed_data: dict) -> str:
        """格式化种子通知消息"""
        impact = seed_data.get("potential_impact", {})
        impact_str = ", ".join([f"{k}:{v:+d}" for k, v in impact.items() if v != 0])

        return f"""🧠 新思维种子待审核

种子ID: {seed_id}
类型: {seed_data["type"]}
事件: {seed_data["event"][:100]}...
强度: {seed_data["intensity"]:.2f}
预期影响: {impact_str or "无"}

审核命令:
/soul_approve {seed_id} - 批准内化
/soul_reject {seed_id} - 拒绝种子
/soul_seeds - 查看所有待审核种子"""

    def format_seeds_list(self, seeds: list) -> str:
        """格式化种子列表"""
        if not seeds:
            return "当前没有待审核的思维种子"

        result = "🧠 待审核思维种子:\n\n"
        for seed in seeds:
            result += f"ID: {seed['seed_id']}\n"
            if seed.get("stream_id"):
                result += f"来源: {seed['stream_id']}\n"
            result += f"类型: {seed['type']}\n"
            result += f"事件: {seed['event'][:50]}...\n"
            result += f"强度: {seed['intensity']:.2f}\n\n"

        return result
