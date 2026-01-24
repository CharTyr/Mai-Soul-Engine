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
{{"thought": "我形成的深层观点...", "spectrum_impact": {{"economic": 0, "social": 0, "diplomatic": 0, "progressive": 0}}, "reasoning": "为什么会产生这样的光谱影响", "confidence": 0.85, "tags": ["关键词1", "关键词2"]}}"""


def _compact_line(text: str, limit: int) -> str:
    s = (text or "").replace("\n", " ").replace("\r", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s[:limit]


class InternalizationEngine:
    async def internalize_seed(self, seed: dict, dedup: dict | None = None) -> dict:
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
                "intensity": float(seed.get("intensity", 0.0) or 0.0),
                "seed_confidence": float(seed.get("confidence", 0.0) or 0.0),
                "evidence": seed.get("evidence", []) or [],
                "created_at": seed.get("created_at", None),
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

            internalization_confidence = 0.0
            try:
                internalization_confidence = float(result.get("confidence", 0.0) or 0.0)
            except Exception:
                internalization_confidence = 0.0
            internalization_confidence = max(0.0, min(1.0, internalization_confidence))

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
            trait_confidence = max(seed_info["seed_confidence"], seed_info["intensity"], internalization_confidence)
            trait_confidence = max(0.0, min(1.0, float(trait_confidence)))
            evidence_entry = {
                "seed_id": seed_info.get("id", ""),
                "stream_id": seed_info.get("stream_id", ""),
                "type": seed_info.get("type", ""),
                "event": seed_info.get("event", ""),
                "reasoning": seed_info.get("reasoning", ""),
                "created_at": seed_info.get("created_at", None),
                "intensity": seed_info.get("intensity", 0.0),
                "seed_confidence": seed_info.get("seed_confidence", 0.0),
                "internalization_confidence": internalization_confidence,
                "evidence": seed_info.get("evidence", []) or [],
            }

            stored = await self._upsert_crystallized_trait(
                seed_info=seed_info,
                result=result,
                impact=spectrum_impact,
                trait_id=trait_id,
                now=now,
                evidence_entry=evidence_entry,
                confidence=int(round(trait_confidence * 100)),
                dedup=dedup,
            )
            trait_id = stored.get("trait_id", trait_id)

            logger.info(f"种子内化成功: {seed_info.get('id', '')}, 观点: {result['thought'][:50]}...")
            return {
                "success": True,
                "spectrum_impact": spectrum_impact,
                "thought": result["thought"],
                "trait_id": trait_id,
                "merged": bool(stored.get("merged", False)),
                "merged_into": stored.get("merged_into", None),
                "dedup_similarity": stored.get("dedup_similarity", None),
            }

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
            if not isinstance(result, dict):
                return None
            if "tags" in result and not isinstance(result.get("tags"), list):
                result["tags"] = []
            if "confidence" in result:
                try:
                    result["confidence"] = float(result.get("confidence", 0.0) or 0.0)
                except Exception:
                    result["confidence"] = 0.0
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

    async def _upsert_crystallized_trait(
        self,
        seed_info: dict,
        result: dict,
        impact: dict,
        trait_id: str,
        now: datetime,
        evidence_entry: dict,
        confidence: int,
        dedup: dict | None,
    ) -> dict:
        import json

        from ..models.ideology_model import CrystallizedTrait
        from ..utils.trait_tags import dumps_tags_json, parse_tags_json
        from ..utils.trait_evidence import append_evidence_json, dumps_evidence_json

        dedup_enabled = bool((dedup or {}).get("enabled", True))
        try:
            dedup_threshold = float((dedup or {}).get("threshold", 0.78))
        except Exception:
            dedup_threshold = 0.78
        dedup_threshold = max(0.0, min(1.0, dedup_threshold))

        merged_into: str | None = None
        dedup_similarity: float | None = None
        if dedup_enabled:
            target = await self._find_dedup_target(
                stream_id=seed_info.get("stream_id", "") or "",
                new_name=seed_info.get("type", "trait"),
                new_question=seed_info.get("question", "") or "",
                new_thought=result.get("thought", "") or "",
                new_tags=parse_tags_json(dumps_tags_json(seed_info.get("tags"))),
                threshold=dedup_threshold,
            )
            if target:
                merged_into = target.get("trait_id")
                dedup_similarity = target.get("similarity")

        if merged_into:
            trait = CrystallizedTrait.get_or_none(CrystallizedTrait.trait_id == merged_into)
            if trait and (not trait.deleted) and trait.enabled:
                existing_tags = parse_tags_json(getattr(trait, "tags_json", "[]") or "[]")
                merged_tags = list(dict.fromkeys([*existing_tags, *(seed_info.get("tags") or [])]))

                trait.tags_json = dumps_tags_json(merged_tags)
                trait.evidence_json = append_evidence_json(getattr(trait, "evidence_json", "[]") or "[]", evidence_entry)
                trait.confidence = max(int(getattr(trait, "confidence", 0) or 0), int(confidence), int(round((dedup_similarity or 0.0) * 100)))
                trait.spectrum_impact_json = json.dumps(impact or {}, ensure_ascii=False)
                trait.created_at = now
                trait.save()
                logger.info(f"已合并到已有 trait: {merged_into} (seed={seed_info.get('id', '')})")
                return {"trait_id": merged_into, "merged": True, "merged_into": merged_into, "dedup_similarity": dedup_similarity}

        CrystallizedTrait.create(
            trait_id=trait_id,
            stream_id=seed_info.get("stream_id", "") or "",
            seed_id=seed_info.get("id", "") or "",
            name=seed_info.get("type", "trait"),
            question=seed_info.get("question", "") or "",
            thought=result.get("thought", "") or "",
            tags_json=dumps_tags_json(seed_info.get("tags")),
            confidence=int(confidence),
            evidence_json=dumps_evidence_json([evidence_entry]),
            spectrum_impact_json=json.dumps(impact or {}, ensure_ascii=False),
            created_at=now,
            enabled=True,
            deleted=False,
        )
        logger.info(f"已创建 trait 记录: {trait_id} (seed={seed_info.get('id', '')})")
        return {"trait_id": trait_id, "merged": False}

    async def _find_dedup_target(
        self,
        stream_id: str,
        new_name: str,
        new_question: str,
        new_thought: str,
        new_tags: list[str],
        threshold: float,
    ) -> dict | None:
        import json
        import difflib

        from src.llm_models.utils_model import LLMRequest
        from ..models.ideology_model import CrystallizedTrait
        from ..utils.trait_tags import parse_tags_json

        if not new_thought.strip():
            return None

        query = CrystallizedTrait.select().where(
            (CrystallizedTrait.deleted == False)  # noqa: E712
            & (CrystallizedTrait.enabled == True)  # noqa: E712
            & (CrystallizedTrait.stream_id == stream_id)
        )
        candidates = list(query.order_by(CrystallizedTrait.created_at.desc()).limit(40))
        if not candidates:
            return None

        def candidate_score(t: CrystallizedTrait) -> float:
            tags = parse_tags_json(getattr(t, "tags_json", "[]") or "[]")
            overlap = 0
            if tags and new_tags:
                overlap = len(set([x.casefold() for x in tags]) & set([x.casefold() for x in new_tags]))
            ratio = difflib.SequenceMatcher(None, (t.thought or "")[:400], new_thought[:400]).ratio()
            return float(overlap * 2) + float(ratio)

        ranked = sorted(candidates, key=candidate_score, reverse=True)[:12]

        items = []
        for t in ranked:
            tags = parse_tags_json(getattr(t, "tags_json", "[]") or "[]")
            items.append(
                {
                    "trait_id": t.trait_id,
                    "name": t.name,
                    "tags": tags,
                    "question": (getattr(t, "question", "") or "")[:140],
                    "thought": (t.thought or "")[:220],
                }
            )

        prompt = (
            "你是一个“trait 去重合并”助手。判断新 trait 是否与已有 trait 在语义上高度重复/同义（只是表述不同）。\n"
            "若重复，返回 duplicate_of=对应 trait_id；否则 duplicate_of 为空。\n"
            "仅当 similarity >= 阈值 才建议合并。\n\n"
            f"阈值: {threshold}\n\n"
            f"新 trait:\n- name: {new_name}\n- tags: {new_tags}\n- question: {new_question[:180]}\n- thought: {new_thought[:320]}\n\n"
            f"已有 traits（候选）:\n{json.dumps(items, ensure_ascii=False)}\n\n"
            "请只输出 JSON:\n"
            '{"duplicate_of": "", "similarity": 0.0, "reason": ""}'
        )

        llm = LLMRequest()
        response, _ = await llm.generate_response_async(prompt)
        if not response:
            return None
        response = str(response).strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            data = json.loads(response)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        duplicate_of = str(data.get("duplicate_of", "") or "").strip()
        try:
            similarity = float(data.get("similarity", 0.0) or 0.0)
        except Exception:
            similarity = 0.0
        similarity = max(0.0, min(1.0, similarity))
        if not duplicate_of or similarity < threshold:
            return None
        if duplicate_of not in {x["trait_id"] for x in items}:
            return None
        return {"trait_id": duplicate_of, "similarity": similarity}
