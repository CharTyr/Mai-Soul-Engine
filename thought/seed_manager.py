import json
import uuid
import logging
from datetime import datetime
from typing import Optional

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

    async def create_seed(self, seed_data: dict) -> Optional[str]:
        from src.chat.knowledge.lpmm_ops import lpmm_ops

        logger.debug(f"尝试创建种子, 强度: {seed_data.get('intensity', 0)}, 阈值: {self.min_intensity}")

        if seed_data.get("intensity", 0) < self.min_intensity:
            logger.debug(f"种子强度不足，跳过创建")
            return None

        await self._cleanup_excess_seeds()

        seed_id = f"seed_{uuid.uuid4().hex[:8]}"

        seed_content = f"""思维种子 - {seed_data["type"]} [待审核]

种子ID: {seed_id}
触发事件: {seed_data["event"]}
检测强度: {seed_data["intensity"]:.2f}
检测原因: {seed_data["reasoning"]}
预期光谱影响: {json.dumps(seed_data.get("potential_impact", {}), ensure_ascii=False)}
创建时间: {datetime.now().isoformat()}
状态: 待审核

这是一个关于{THOUGHT_TYPES.get(seed_data["type"], "未知类型")}的思维种子，需要管理员决定是否内化。"""

        await lpmm_ops.add_content(seed_content, auto_split=False)
        logger.info(f"创建思维种子: {seed_id} (类型: {seed_data['type']}, 强度: {seed_data['intensity']:.2f})")

        return seed_id

    async def _cleanup_excess_seeds(self):
        seeds = await self.get_pending_seeds()
        logger.debug(f"当前种子数: {len(seeds)}, 最大限制: {self.max_seeds}")
        if len(seeds) >= self.max_seeds:
            logger.info(f"种子数超限，清理 {len(seeds) - self.max_seeds + 1} 个旧种子")
            for seed in seeds[self.max_seeds - 1 :]:
                seed_id = self._extract_field(seed.get("content", ""), "种子ID")
                if seed_id:
                    await self.delete_seed(seed_id)
                    logger.debug(f"清理旧种子: {seed_id}")

    async def delete_seed(self, seed_id: str) -> bool:
        from src.chat.knowledge.lpmm_ops import lpmm_ops

        logger.debug(f"删除种子: {seed_id}")
        result = await lpmm_ops.delete(seed_id, exact_match=False)
        deleted = result.get("deleted_count", 0) > 0
        if deleted:
            logger.info(f"种子已删除: {seed_id}")
        else:
            logger.warning(f"删除种子失败: {seed_id}, 结果: {result}")
        return deleted

    async def get_pending_seeds(self) -> list:
        from src.chat.knowledge.lpmm_ops import lpmm_ops

        seeds = await lpmm_ops.search("思维种子 待审核", top_k=20)
        logger.debug(f"查询待审核种子, 找到 {len(seeds)} 个")
        return seeds

    async def get_seed_by_id(self, seed_id: str) -> Optional[dict]:
        from src.chat.knowledge.lpmm_ops import lpmm_ops

        logger.debug(f"查询种子: {seed_id}")
        seeds = await lpmm_ops.search(f"思维种子 {seed_id}", top_k=1)
        if seeds:
            logger.debug(f"找到种子: {seed_id}")
        else:
            logger.debug(f"未找到种子: {seed_id}")
        return seeds[0] if seeds else None

    def format_seed_notification(self, seed_id: str, seed_data: dict) -> str:
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
        if not seeds:
            return "当前没有待审核的思维种子"

        result = "🧠 待审核思维种子:\n\n"
        for seed in seeds:
            content = seed.get("content", "")
            seed_id = self._extract_field(content, "种子ID")
            seed_type = self._extract_field(content, "思维种子 -").split("[")[0].strip()
            event = self._extract_field(content, "触发事件")[:50]
            intensity = self._extract_field(content, "检测强度")

            result += f"ID: {seed_id}\n"
            result += f"类型: {seed_type}\n"
            result += f"事件: {event}...\n"
            result += f"强度: {intensity}\n\n"

        return result

    def _extract_field(self, content: str, field_name: str) -> str:
        for line in content.split("\n"):
            if field_name in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
                return line.replace(field_name, "").strip()
        return ""
