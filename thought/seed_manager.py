import json
import logging
import uuid
from difflib import SequenceMatcher
from typing import List, Optional

logger = logging.getLogger(__name__)

THOUGHT_TYPES = {
    "真诚与虚伪的冲突": "对真实表达与场面伪装的取舍",
    "亲近与边界的拉扯": "对人际距离与自我保护的权衡",
    "热闹与克制的取舍": "对热情投入与精力保存的抉择",
    "直率与含蓄的抉择": "对有话直说与委婉表达的倾向",
    "群体认同与自我坚持": "对融入群体与保持个性的思考",
}

# 上下文窗口：每条 evidence 匹配到原文后取前后各 N 条
_CONTEXT_RADIUS = 2
_CONTEXT_MAX_LINES = 10
_CONTEXT_MAX_LINE_LEN = 200
# fuzzy 匹配阈值：evidence 片段与原文行的相似度需达到此值才采纳
_CONTEXT_MATCH_THRESHOLD = 0.4


def _match_evidence_to_context(evidence: list[str], msg_lines: list[str]) -> list[str]:
    """将 LLM 摘录的 evidence 片段模糊匹配回原始消息行，取上下文窗口。

    Args:
        evidence: LLM 返回的证据片段列表（如 ["A: xxx", "B: yyy"]）。
        msg_lines: 发送给 LLM 的原始消息行列表（如 ["A: content1", "B: content2", ...]）。

    Returns:
        匹配到的上下文消息行列表（每行截断到 _CONTEXT_MAX_LINE_LEN），最多 _CONTEXT_MAX_LINES 条。
        无匹配时返回空列表。
    """
    if not evidence or not msg_lines:
        return []

    matched_indices: set[int] = set()
    for ev in evidence:
        ev_clean = str(ev).replace("\n", " ").replace("\r", " ").strip()
        if not ev_clean:
            continue
        best_ratio = 0.0
        best_idx = -1
        for i, line in enumerate(msg_lines):
            line_clean = str(line).replace("\n", " ").replace("\r", " ").strip()
            ratio = SequenceMatcher(None, ev_clean.casefold(), line_clean.casefold()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
        if best_idx >= 0 and best_ratio >= _CONTEXT_MATCH_THRESHOLD:
            start = max(0, best_idx - _CONTEXT_RADIUS)
            end = min(len(msg_lines), best_idx + _CONTEXT_RADIUS + 1)
            for j in range(start, end):
                matched_indices.add(j)

    if not matched_indices:
        return []

    sorted_indices = sorted(matched_indices)
    if len(sorted_indices) > _CONTEXT_MAX_LINES:
        sorted_indices = sorted_indices[:_CONTEXT_MAX_LINES]

    result: list[str] = []
    for idx in sorted_indices:
        line = str(msg_lines[idx]).replace("\n", " ").replace("\r", " ").strip()
        while "  " in line:
            line = line.replace("  ", " ")
        if len(line) > _CONTEXT_MAX_LINE_LEN:
            line = f"{line[:_CONTEXT_MAX_LINE_LEN]}..."
        result.append(line)
    return result


class ThoughtSeedManager:
    def __init__(self, config: dict):
        self.max_seeds = config.get("max_seeds", 20)
        self.min_intensity = config.get("min_trigger_intensity", 0.7)
        self.admin_user_id = config.get("admin_user_id", "")
        self.seed_ttl_hours = float(config.get("seed_ttl_hours", 168.0))
        self.reviewed_keep_count = int(config.get("reviewed_keep_count", 200))
        self.seed_dedup_threshold = float(config.get("seed_dedup_threshold", 0.82))

    async def create_seed(
        self,
        seed_data: dict,
        stream_id: str = "",
        context_messages: list[str] | None = None,
    ) -> Optional[str]:
        """创建思维种子并存入数据库。

        Args:
            seed_data: LLM 返回的种子数据。
            stream_id: 来源聊天流 ID。
            context_messages: 发送给 LLM 的原始消息行列表，用于提取上下文窗口。
        """
        from ..models.ideology_model import create_thought_seed
        from ..utils.evidence_utils import dumps_evidence_json, normalize_evidence

        logger.debug(f"尝试创建种子, 强度: {seed_data.get('intensity', 0)}, 阈值: {self.min_intensity}")

        if seed_data.get("intensity", 0) < self.min_intensity:
            logger.debug(f"种子强度不足，跳过创建")
            return None

        # 过期超龄 pending 种子 + 清理过多的已审核种子
        self.expire_old_seeds()
        await self._cleanup_excess_seeds()
        await self._cleanup_old_reviewed_seeds()

        # 对已有 pending 种子语义去重，避免同话题跨周期反复入池
        if self._is_duplicate_pending_seed(seed_data, stream_id):
            logger.info(f"种子与已有 pending 种子语义重复，跳过创建 (类型: {seed_data.get('type', '')})")
            return None

        seed_id = f"seed_{uuid.uuid4().hex[:8]}"

        raw_confidence = seed_data.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        confidence_int = int(round(confidence * 100))

        evidence = seed_data.get("evidence", None)
        if evidence is None:
            evidence = seed_data.get("evidence_quotes", [])
        normalized_evidence = normalize_evidence(evidence, max_items=3, max_len=180)

        # 从原始消息行中提取上下文窗口
        context_lines = _match_evidence_to_context(normalized_evidence, context_messages or [])
        context_json = json.dumps(context_lines, ensure_ascii=False)

        # 存入数据库
        create_thought_seed(
            seed_id=seed_id,
            stream_id=stream_id or "",
            seed_type=seed_data["type"],
            event=seed_data["event"],
            intensity=int(seed_data["intensity"] * 100),  # 转换为0-100整数
            confidence=confidence_int,
            evidence_json=dumps_evidence_json(normalized_evidence),
            reasoning=seed_data["reasoning"],
            potential_impact_json=json.dumps(seed_data.get("potential_impact", {}), ensure_ascii=False),
            context_json=context_json,
            status="pending",
        )

        logger.info(f"创建思维种子: {seed_id} (类型: {seed_data['type']}, 强度: {seed_data['intensity']:.2f}, 上下文{len(context_lines)}条)")
        return seed_id

    async def _cleanup_excess_seeds(self):
        """清理超限的旧种子"""
        from ..models.ideology_model import get_pending_thought_seeds

        seeds = get_pending_thought_seeds()
        logger.debug(f"当前待审核种子数: {len(seeds)}, 最大限制: {self.max_seeds}")

        if len(seeds) >= self.max_seeds:
            logger.info(f"种子数超限，清理 {len(seeds) - self.max_seeds + 1} 个旧种子")
            for seed in seeds[self.max_seeds - 1 :]:
                seed.delete_instance()
                logger.debug(f"清理旧种子: {seed.seed_id}")

    async def delete_seed(self, seed_id: str) -> bool:
        """删除种子（从数据库）"""
        from ..models.ideology_model import get_thought_seed_by_id

        logger.debug(f"删除种子: {seed_id}")
        seed = get_thought_seed_by_id(seed_id)
        if seed is None:
            logger.warning(f"删除种子失败: {seed_id} 不存在")
            return False
        seed.delete_instance()
        logger.info(f"种子已删除: {seed_id}")
        return True

    def mark_seed_status(self, seed_id: str, status: str) -> bool:
        """更新种子状态（approved/rejected/expired），不删除记录以保留审计链。"""
        from ..models.ideology_model import update_seed_status

        ok = update_seed_status(seed_id, status)
        if ok:
            logger.info(f"种子状态更新: {seed_id} → {status}")
        else:
            logger.warning(f"种子状态更新失败: {seed_id} 不存在")
        return ok

    def expire_old_seeds(self) -> int:
        """将超过 TTL 的 pending 种子标记为 expired。"""
        from ..models.ideology_model import expire_old_pending_seeds

        count = expire_old_pending_seeds(self.seed_ttl_hours)
        if count:
            logger.info(f"过期 {count} 个超龄 pending 种子 (TTL={self.seed_ttl_hours}h)")
        return count

    async def _cleanup_old_reviewed_seeds(self) -> None:
        """清理过多的已审核种子（approved/rejected/expired），保留最近 reviewed_keep_count 条。"""
        from ..models.ideology_model import count_reviewed_seeds, delete_oldest_reviewed_seeds

        total = count_reviewed_seeds()
        if total > self.reviewed_keep_count:
            deleted = delete_oldest_reviewed_seeds(self.reviewed_keep_count)
            if deleted:
                logger.info(f"清理 {deleted} 个旧已审核种子 (保留 {self.reviewed_keep_count})")

    def _is_duplicate_pending_seed(self, seed_data: dict, stream_id: str) -> bool:
        """判断新种子是否与同群已有 pending 种子语义重复（本地 difflib，不调 LLM）。

        比较 type + event + reasoning 拼接文本的相似度，超过阈值视为重复。
        """
        if self.seed_dedup_threshold <= 0:
            return False
        from ..models.ideology_model import get_pending_thought_seeds

        new_text = self._seed_signature(
            seed_data.get("type", ""),
            seed_data.get("event", ""),
            seed_data.get("reasoning", ""),
        )
        if not new_text:
            return False

        # 只和同群（含全局）的 pending 种子比较
        existing = get_pending_thought_seeds(stream_id=stream_id or None)
        for seed in existing:
            old_text = self._seed_signature(seed.seed_type, seed.event, seed.reasoning)
            if not old_text:
                continue
            ratio = SequenceMatcher(None, new_text, old_text).ratio()
            if ratio >= self.seed_dedup_threshold:
                logger.debug(f"种子与 {seed.seed_id} 相似度 {ratio:.2f} >= {self.seed_dedup_threshold}")
                return True
        return False

    @staticmethod
    def _seed_signature(seed_type: str, event: str, reasoning: str) -> str:
        """生成用于去重比较的种子签名文本。"""
        parts = [str(seed_type or ""), str(event or ""), str(reasoning or "")]
        s = " ".join(p.strip() for p in parts if p.strip())
        s = s.replace("\n", " ").replace("\r", " ")
        while "  " in s:
            s = s.replace("  ", " ")
        return s.casefold().strip()

    async def get_pending_seeds(self, stream_id: str | None = None) -> List[dict]:
        """获取所有待审核种子"""
        from ..models.ideology_model import get_pending_thought_seeds
        from ..utils.evidence_utils import parse_evidence_json

        seeds = get_pending_thought_seeds(stream_id=stream_id)
        logger.debug(f"查询待审核种子, 找到 {len(seeds)} 个")

        # 转换为字典格式以保持兼容性
        result = []
        for seed in seeds:
            evidence = parse_evidence_json(seed.evidence_json or "[]")
            context = _parse_context_json(seed.context_json or "[]")
            result.append(
                {
                    "seed_id": seed.seed_id,
                    "stream_id": seed.stream_id or "",
                    "type": seed.seed_type,
                    "event": seed.event,
                    "intensity": seed.intensity / 100.0,  # 转回0-1范围
                    "confidence": float(seed.confidence or 0) / 100.0,
                    "evidence": evidence,
                    "context": context,
                    "reasoning": seed.reasoning,
                    "potential_impact": json.loads(seed.potential_impact_json),
                    "created_at": seed.created_at.isoformat(),
                    "status": seed.status,
                }
            )
        return result

    async def get_seed_by_id(self, seed_id: str) -> Optional[dict]:
        """根据ID获取种子"""
        from ..models.ideology_model import get_thought_seed_by_id
        from ..utils.evidence_utils import parse_evidence_json

        logger.debug(f"查询种子: {seed_id}")
        seed = get_thought_seed_by_id(seed_id)
        if seed is None:
            logger.debug(f"未找到种子: {seed_id}")
            return None
        logger.debug(f"找到种子: {seed_id}")
        evidence = parse_evidence_json(seed.evidence_json or "[]")
        context = _parse_context_json(seed.context_json or "[]")
        return {
            "seed_id": seed.seed_id,
            "stream_id": seed.stream_id or "",
            "type": seed.seed_type,
            "event": seed.event,
            "intensity": seed.intensity / 100.0,
            "confidence": float(seed.confidence or 0) / 100.0,
            "evidence": evidence,
            "context": context,
            "reasoning": seed.reasoning,
            "potential_impact": json.loads(seed.potential_impact_json),
            "created_at": seed.created_at.isoformat(),
            "status": seed.status,
        }

    def format_seed_notification(self, seed_id: str, seed_data: dict) -> str:
        """格式化种子通知消息"""
        impact = seed_data.get("potential_impact", {})
        impact_str = ", ".join([f"{k}:{v:+d}" for k, v in impact.items() if v != 0])
        confidence = seed_data.get("confidence", None)
        confidence_line = ""
        try:
            if confidence is not None:
                confidence_line = f"\n置信度: {float(confidence):.2f}"
        except Exception:
            confidence_line = ""
        evidence = seed_data.get("evidence", []) or []
        evidence_block = ""
        if isinstance(evidence, list):
            lines = [str(x).strip() for x in evidence if str(x).strip()]
            if lines:
                evidence_block = "\n证据片段:\n" + "\n".join([f"- {x[:120]}{'...' if len(x) > 120 else ''}" for x in lines[:3]])

        # 上下文窗口：展示原始对话片段
        context = seed_data.get("context", []) or []
        context_block = ""
        if isinstance(context, list) and context:
            context_block = "\n原始对话上下文:\n" + "\n".join([f"│ {x}" for x in context])

        return f"""🧠 新思维种子待审核

种子ID: {seed_id}
类型: {seed_data["type"]}
事件: {seed_data["event"][:100]}...
强度: {seed_data["intensity"]:.2f}
{confidence_line}{evidence_block}{context_block}
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
            result += f"强度: {seed['intensity']:.2f}\n"
            # 上下文预览：取第一条
            context = seed.get("context", []) or []
            if context:
                preview = str(context[0]).replace("\n", " ").strip()
                if len(preview) > 60:
                    preview = preview[:60] + "..."
                result += f"上下文: {preview}\n"
            result += "\n"

        return result


def _parse_context_json(raw: str) -> list[str]:
    """解析上下文 JSON 字符串为字符串列表。"""
    try:
        data = json.loads(raw or "[]")
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if str(x).strip()]
