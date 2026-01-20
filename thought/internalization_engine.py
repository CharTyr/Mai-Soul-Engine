import logging
import uuid
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

INTERNALIZATION_PROMPT = """基于以下思维种子，进行深层的哲学内化：

种子类型: {type}
触发事件: {event}
检测原因: {reasoning}

请完成两个任务：

**任务1：形成深层观点**
以第一人称内心独白的形式，形成一个明确的观点或立场（50-100字）。
这个观点将成为我的一部分，影响我对类似话题的看法。

**任务2：评估光谱影响**
这个观点会如何影响我的意识形态光谱？
- economic: 对公平vs效率的看法 (-10到+10)
- social: 对自由vs秩序的看法 (-10到+10)
- diplomatic: 对开放vs本土的看法 (-10到+10)
- progressive: 对变化vs传统的看法 (-10到+10)

请以JSON格式返回:
{{"thought": "我形成的深层观点...", "spectrum_impact": {{"economic": 0, "social": 0, "diplomatic": 0, "progressive": 0}}, "reasoning": "为什么会产生这样的光谱影响", "tags": ["关键词1", "关键词2"]}}"""


def _compact_line(text: str, limit: int) -> str:
    s = (text or "").replace("\n", " ").replace("\r", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s[:limit]


class InternalizationEngine:
    async def internalize_seed(self, seed: dict) -> dict:
        from src.llm_models.utils_model import LLMRequest
        from ..models.ideology_model import init_tables
        from ..utils.trait_tags import normalize_tags

        try:
            init_tables()
            # 新格式：直接使用数据库返回的结构化字典
            seed_info = {
                "id": seed.get("seed_id", ""),
                "stream_id": seed.get("stream_id", "") or "",
                "type": seed.get("type", "未知"),
                "event": seed.get("event", ""),
                "reasoning": seed.get("reasoning", ""),
                "potential_impact": seed.get("potential_impact", {}),
            }
            logger.debug(f"开始内化种子: {seed_info.get('id', 'unknown')}, 类型: {seed_info.get('type', 'unknown')}")

            prompt = INTERNALIZATION_PROMPT.format(
                type=seed_info.get("type", "未知"),
                event=seed_info.get("event", ""),
                reasoning=seed_info.get("reasoning", ""),
            )

            llm = LLMRequest()
            logger.debug(f"发送内化LLM请求，prompt长度: {len(prompt)}")
            response, _ = await llm.generate_response_async(prompt)
            logger.debug(f"内化LLM响应长度: {len(response) if response else 0}")

            result = self._parse_response(response)
            if not result:
                logger.warning(f"内化响应解析失败: {seed_info.get('id', '')}")
                return {"success": False, "error": "内化响应解析失败"}

            spectrum_impact = await self._apply_spectrum_impact(result["spectrum_impact"])
            logger.debug(f"光谱影响已应用: {spectrum_impact}")

            now = datetime.now()
            trait_id = f"trait_{uuid.uuid4().hex[:8]}"
            question = _compact_line(seed_info.get("event", ""), 140)
            reason = _compact_line(seed_info.get("reasoning", ""), 140)
            question_text = f"{seed_info.get('type', '思维')}: {question}" if question else f"{seed_info.get('type', '思维')}"
            if reason:
                question_text = f"{question_text}\n线索: {reason}"
            seed_info["question"] = question_text
            tags = result.get("tags") or []
            seed_info["tags"] = normalize_tags(tags)
            await self._store_crystallized_trait(seed_info, result, spectrum_impact, trait_id=trait_id, now=now)

            logger.info(f"种子内化成功: {seed_info.get('id', '')}, 观点: {result['thought'][:50]}...")
            return {"success": True, "spectrum_impact": spectrum_impact, "thought": result["thought"], "trait_id": trait_id}

        except Exception as e:
            logger.error(f"内化失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _parse_response(self, response: str) -> Optional[dict]:
        import json

        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(response)
            if isinstance(result, dict) and "tags" in result and not isinstance(result.get("tags"), list):
                result["tags"] = []
            return result
        except json.JSONDecodeError:
            logger.warning(f"无法解析内化响应: {response}")
            return None

    async def _apply_spectrum_impact(self, impact: dict) -> dict:
        from ..models.ideology_model import get_or_create_spectrum
        from ..utils.spectrum_utils import update_spectrum_value

        spectrum = get_or_create_spectrum("global")

        old_values = {
            "economic": spectrum.economic,
            "social": spectrum.social,
            "diplomatic": spectrum.diplomatic,
            "progressive": spectrum.progressive,
        }
        logger.debug(f"应用光谱影响前: {old_values}")

        for dim in ["economic", "social", "diplomatic", "progressive"]:
            delta = max(-10, min(10, impact.get(dim, 0)))
            new_val = update_spectrum_value(getattr(spectrum, dim), delta)
            setattr(spectrum, dim, new_val)

        spectrum.updated_at = datetime.now()
        spectrum.save()

        result = {
            dim: getattr(spectrum, dim) - old_values[dim] for dim in ["economic", "social", "diplomatic", "progressive"]
        }
        logger.debug(f"应用光谱影响后: {result}")
        return result

    async def _store_crystallized_trait(self, seed_info: dict, result: dict, impact: dict, trait_id: str, now: datetime) -> None:
        import json

        from ..models.ideology_model import CrystallizedTrait
        from ..utils.trait_tags import dumps_tags_json

        CrystallizedTrait.create(
            trait_id=trait_id,
            stream_id=seed_info.get("stream_id", "") or "",
            seed_id=seed_info.get("id", "") or "",
            name=seed_info.get("type", "trait"),
            question=seed_info.get("question", "") or "",
            thought=result.get("thought", "") or "",
            tags_json=dumps_tags_json(seed_info.get("tags")),
            spectrum_impact_json=json.dumps(impact or {}, ensure_ascii=False),
            created_at=now,
            enabled=True,
            deleted=False,
        )
        logger.info(f"已创建 trait 记录: {trait_id} (seed={seed_info.get('id', '')})")
