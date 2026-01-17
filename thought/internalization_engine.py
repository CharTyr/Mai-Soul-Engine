import json
import logging
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
{{"thought": "我形成的深层观点...", "spectrum_impact": {{"economic": 0, "social": 0, "diplomatic": 0, "progressive": 0}}, "reasoning": "为什么会产生这样的光谱影响"}}"""


class InternalizationEngine:
    async def internalize_seed(self, seed: dict) -> dict:
        from src.llm_models.utils_model import LLMRequest

        try:
            # 新格式：直接使用数据库返回的结构化字典
            seed_info = {
                "id": seed.get("seed_id", ""),
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

            await self._store_solidified_thought(seed_info, result, spectrum_impact)

            logger.info(f"种子内化成功: {seed_info.get('id', '')}, 观点: {result['thought'][:50]}...")
            return {"success": True, "spectrum_impact": spectrum_impact, "thought": result["thought"]}

        except Exception as e:
            logger.error(f"内化失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _parse_response(self, response: str) -> Optional[dict]:
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(response)
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

    async def _store_solidified_thought(self, seed_info: dict, result: dict, impact: dict):
        from src.chat.knowledge.lpmm_ops import lpmm_ops

        impact_str = ", ".join([f"{k}:{v:+d}" for k, v in impact.items() if v != 0])

        solidified_content = f"""思维固化 - {seed_info.get("type", "未知")}

固化观点: {result["thought"]}

光谱影响: {impact_str or "无"}
影响原因: {result.get("reasoning", "")}
原始种子: {seed_info.get("id", "")}
固化时间: {datetime.now().isoformat()}

这是一个经过深层内化的观点，已经成为我价值观的一部分。"""

        await lpmm_ops.add_content(solidified_content, auto_split=False)
        logger.info(f"思维固化完成: {seed_info.get('id', '')}")
