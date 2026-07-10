import logging
import uuid
from datetime import datetime
from typing import Optional

from ..utils.runtime_resolution import generate_soul_text
from ..prompts.fermentation_prompts import FERMENTED_INTERNALIZATION_PROMPT

logger = logging.getLogger(__name__)

# Prompt Version: v2.3.0
INTERNALIZATION_PROMPT = """基于以下思维种子，进行深层的哲学内化：

种子类型: {type}
触发事件: {event}
检测原因: {reasoning}
种子强度: {intensity:.2f}
置信度: {confidence:.2f}

证据片段:
{evidence}

原始对话上下文:
{context}

预期光谱影响（参考，可在此基础上调整）:
{potential_impact}

请完成两个任务：

**任务1：形成深层观点**
以第一人称内心独白的形式，形成一个明确的观点或立场（50-100字）。
这个观点将成为我的一部分，影响我对类似话题的看法。
请基于证据片段和原始对话上下文来形成观点，而非凭空发挥。

**任务2：评估光谱影响**
这个观点会如何影响我的群聊社交人格倾向？
- sincerity: 对真实自然vs配合社交表演的看法 (-10到+10)
- engagement: 对克制怕消耗vs热情投入的看法 (-10到+10)
- closeness: 对保持距离vs容易亲近的看法 (-10到+10)
- directness: 对含蓄绕弯vs有话直说的看法 (-10到+10)
sincerity 与 directness 相互独立：sincerity 看"是否违心/配合表演"，directness 看"信息是否绕弯/留余地"，不要把两轴同向联动。
参考上方"预期光谱影响"但不必完全一致——你基于完整上下文的判断更准确。

tags：打 1-3 个能描述"在什么场景下会用到这个观点"的标签，优先用场景词（如接梗、阴阳、劝架、短回复、技术向、拒绝、冷场救、玩梗、吐槽、边界），也可用话题词（如游戏、感情、职场）。

请以JSON格式返回:
{{"thought": "我形成的深层观点...", "ideology_layer": "values|worldview|conduct", "spectrum_deltas": {{"sincerity": 0, "engagement": 0, "closeness": 0, "directness": 0}}, "reasoning": "为什么会产生这样的光谱影响", "confidence": 0.85, "tags": ["关键词1", "关键词2"]}}"""


def _compact_line(text: str, limit: int) -> str:
    s = (text or "").replace("\n", " ").replace("\r", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s[:limit]


class InternalizationEngine:
    def __init__(self, plugin):
        """初始化内化引擎。

        Args:
            plugin: 插件实例，用于访问 plugin.ctx.llm。
        """
        self._plugin = plugin

    async def internalize_seed(self, seed: dict, dedup: dict | None = None, fermentation_inputs: list[str] | None = None) -> dict:
        from ..utils.trait_tags import normalize_tags

        try:
            # 新格式：直接使用数据库返回的结构化字典
            seed_info = {
                "id": seed.get("seed_id", ""),
                "stream_id": seed.get("stream_id", "") or "",
                "type": seed.get("type", "未知"),
                "event": seed.get("event", ""),
                "intensity": float(seed.get("intensity", 0.0) or 0.0),
                "seed_confidence": float(seed.get("confidence", 0.0) or 0.0),
                "evidence": seed.get("evidence", []) or [],
                "context": seed.get("context", []) or [],
                "created_at": seed.get("created_at", None),
                "reasoning": seed.get("reasoning", ""),
                "potential_impact": seed.get("potential_impact", {}),
            }
            logger.debug(f"开始内化种子: {seed_info.get('id', 'unknown')}, 类型: {seed_info.get('type', 'unknown')}")

            # 格式化 evidence 和 context 为可读文本
            evidence_lines = seed_info.get("evidence", [])
            evidence_text = "\n".join([f"- {x}" for x in evidence_lines[:5]]) if evidence_lines else "（无）"
            context_lines = seed_info.get("context", [])
            context_text = "\n".join([f"│ {x}" for x in context_lines[:10]]) if context_lines else "（无）"
            potential_impact = seed_info.get("potential_impact", {})
            impact_text = ", ".join([f"{k}:{v:+d}" for k, v in potential_impact.items() if v != 0]) or "（无）"

            # v2.4.0: 发酵后内化使用专用 prompt + 更大光谱影响
            if fermentation_inputs:
                fermentation_text = "\n".join([f"│ {x}" for x in fermentation_inputs[:20]]) if fermentation_inputs else "（无）"
                fermented_max_delta = 15
                try:
                    fermented_max_delta = int(getattr(self._plugin.config.thought_cabinet, "fermented_max_internalize_delta", 15))
                except (AttributeError, TypeError, ValueError):
                    pass
                prompt = FERMENTED_INTERNALIZATION_PROMPT.format(
                    type=seed_info.get("type", "未知"),
                    event=seed_info.get("event", ""),
                    reasoning=seed_info.get("reasoning", ""),
                    intensity=seed_info.get("intensity", 0.0),
                    confidence=seed_info.get("seed_confidence", 0.0),
                    evidence=evidence_text,
                    context=context_text,
                    fermentation_inputs=fermentation_text,
                    potential_impact=impact_text,
                    max_delta=fermented_max_delta,
                )
                is_fermented = True
            else:
                prompt = INTERNALIZATION_PROMPT.format(
                    type=seed_info.get("type", "未知"),
                    event=seed_info.get("event", ""),
                    reasoning=seed_info.get("reasoning", ""),
                    intensity=seed_info.get("intensity", 0.0),
                    confidence=seed_info.get("seed_confidence", 0.0),
                    evidence=evidence_text,
                    context=context_text,
                    potential_impact=impact_text,
                )
                is_fermented = False

            # H2: 追加宿主固定人设基底（若已配置），让内化 LLM 感知 bot 人格约束
            from ..utils.host_persona import fetch_host_persona, format_persona_for_prompt

            persona_snap = await fetch_host_persona(self._plugin)
            persona_block = format_persona_for_prompt(persona_snap)
            if persona_block:
                prompt += persona_block

            result = await generate_soul_text(self._plugin, prompt)
            response = result.get("response", "")
            logger.debug(f"内化LLM响应长度: {len(response) if response else 0}")

            result = self._parse_response(response)
            if not result:
                logger.warning(f"内化响应解析失败: {seed_info.get('id', '')}")
                return {"success": False, "error": "内化响应解析失败"}

            internalization_confidence = 0.0
            try:
                internalization_confidence = float(result.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                internalization_confidence = 0.0
            internalization_confidence = max(0.0, min(1.0, internalization_confidence))

            # Phase 0C：原子化内化 — 光谱 + trait + graph edges 在同一事务中
            # 先完成 LLM 处理后，再统一进入 DB 写入阶段。BEGIN 必须放在所有 DML 之前，
            # 否则 isolation_level='' 下 conn.execute(UPDATE) 会隐式开启事务。
            from ..models._conn import _get_conn as _get_db_conn

            conn = _get_db_conn()
            conn.execute("BEGIN")
            try:
                spectrum_impact = await self._apply_spectrum_impact(
                    result.get("spectrum_deltas", result.get("spectrum_impact", {})),
                    is_fermented=is_fermented,
                    commit=False,
                )
                logger.debug(f"光谱影响已应用（暂未提交）: {spectrum_impact}")

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
                    commit=False,
                )
                trait_id = stored.get("trait_id", trait_id)

                from ..worldview.service import WorldviewService, config_from_plugin

                wv = WorldviewService(config_from_plugin(self._plugin))
                wv.register_trait_graph(
                    trait_id=trait_id,
                    seed_id=str(seed_info.get("id", "") or ""),
                    merged_into=stored.get("merged_into"),
                    commit=False,
                )

                conn.commit()
            except Exception:
                conn.execute("ROLLBACK")
                raise

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

        # 顶层兜底：捕获所有异常以返回结构化错误，不吞没（已 log+exc_info）
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
                except (TypeError, ValueError):
                    result["confidence"] = 0.0
            return result
        except json.JSONDecodeError:
            logger.warning(f"无法解析内化响应: {response}")
            return None

    async def _apply_spectrum_impact(self, impact: dict, is_fermented: bool = False, commit: bool = True) -> dict:
        from ..models.ideology_model import apply_spectrum_deltas, get_or_create_spectrum

        spectrum = get_or_create_spectrum("global")

        old_values = {
            "sincerity": spectrum.sincerity,
            "engagement": spectrum.engagement,
            "closeness": spectrum.closeness,
            "directness": spectrum.directness,
        }
        logger.debug(f"应用光谱影响前: {old_values}")

        # v2.4.0: 发酵后内化使用 fermented_max_internalize_delta（更大），即时内化用 max_internalize_delta
        if is_fermented:
            max_delta = 15
            try:
                max_delta = int(getattr(self._plugin.config.thought_cabinet, "fermented_max_internalize_delta", 15))
            except (AttributeError, TypeError, ValueError):
                pass
        else:
            max_delta = 10
            try:
                max_delta = int(getattr(self._plugin.config.thought_cabinet, "max_internalize_delta", 10))
            except (AttributeError, TypeError, ValueError):
                pass
        applied = apply_spectrum_deltas(
            "internalize",
            {
                "sincerity": int(impact.get("sincerity", 0) or 0),
                "engagement": int(impact.get("engagement", 0) or 0),
                "closeness": int(impact.get("closeness", 0) or 0),
                "directness": int(impact.get("directness", 0) or 0),
            },
            max_per_axis=max_delta,
            group_id="",
            reason="发酵后 trait 内化光谱影响" if is_fermented else "trait 内化光谱影响",
            commit=commit,
        )

        logger.debug(f"应用光谱影响后: {applied}")
        return applied

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
        commit: bool = True,
    ) -> dict:
        import json

        from ..models.ideology_model import create_crystallized_trait, get_crystallized_trait_by_id, set_trait_lifecycle_state, create_thought_edge
        from ..utils.trait_tags import dumps_tags_json, parse_tags_json
        from ..utils.trait_evidence import append_trait_evidence_json, dumps_trait_evidence_json

        dedup_enabled = bool((dedup or {}).get("enabled", True))
        try:
            dedup_threshold = float((dedup or {}).get("threshold", 0.78))
        except (TypeError, ValueError):
            dedup_threshold = 0.78
        dedup_threshold = max(0.0, min(1.0, dedup_threshold))

        relation_result: dict | None = None
        if dedup_enabled:
            relation_result = await self._classify_trait_relation(
                stream_id=seed_info.get("stream_id", "") or "",
                new_name=seed_info.get("type", "trait"),
                new_question=seed_info.get("question", "") or "",
                new_thought=result.get("thought", "") or "",
                new_tags=parse_tags_json(dumps_tags_json(seed_info.get("tags"))),
                threshold=dedup_threshold,
            )

        if relation_result and relation_result.get("relation") not in ("none", ""):
            target_trait_id = relation_result.get("target_trait_id", "") or ""
            similarity = relation_result.get("similarity", 0.0) or 0.0
            relation = relation_result.get("relation", "none") or "none"
            # reason = relation_result.get("reason", "")  # 暂不使用，保留供后续日志扩展

            # ── duplicate：合并到旧 trait（现有逻辑） ──
            if relation == "duplicate":
                trait = get_crystallized_trait_by_id(target_trait_id)
                if trait and (not trait.deleted) and trait.enabled:
                    existing_tags = parse_tags_json(trait.tags_json or "[]")
                    merged_tags = list(dict.fromkeys([*existing_tags, *(seed_info.get("tags") or [])]))

                    trait.tags_json = dumps_tags_json(merged_tags)
                    trait.evidence_json = append_trait_evidence_json(trait.evidence_json or "[]", evidence_entry)
                    trait.confidence = max(int(trait.confidence or 0), int(confidence), int(round((similarity or 0.0) * 100)))
                    trait.spectrum_impact_json = json.dumps(impact or {}, ensure_ascii=False)
                    trait.lifecycle_state = "strengthened"
                    trait.created_at = now
                    trait.save(commit=commit)
                    logger.info(f"已合并到已有 trait: {target_trait_id} (seed={seed_info.get('id', '')})")
                    return {"trait_id": target_trait_id, "merged": True, "merged_into": target_trait_id, "dedup_similarity": similarity}

            # ── contradicted / weakened / revised ──
            # 顺序：先建新 trait（成功后）再标记旧 trait，避免非原子写入下
            # "旧 trait 已禁用但新 trait 未创建" 的不可逆语义丢失（P0：失败时旧 trait 不受影响）
            if relation in ("contradicted", "weakened", "revised"):
                # 置信度阈值防误报：contradicted >= 0.70，weakened/revised >= 0.60
                threshold_map = {"contradicted": 0.70, "weakened": 0.60, "revised": 0.60}
                min_sim = threshold_map[relation]
                if similarity < min_sim:
                    logger.info(
                        f"关系 {relation} 相似度 {similarity:.2f} 低于阈值 {min_sim}，降级为 none"
                    )
                    relation = "none"
                else:
                    from ..worldview.constants import normalize_ideology_layer
                    from ..worldview.service import WorldviewService, config_from_plugin

                    tags = parse_tags_json(dumps_tags_json(seed_info.get("tags")))
                    layer = normalize_ideology_layer(
                        str(result.get("ideology_layer", "") or ""),
                        default=WorldviewService.infer_layer_from_tags(tags),
                    )

                    # 1) 先创建新 trait（active）
                    from ..worldview.constants import GLOBAL_STREAM

                    origin = seed_info.get("stream_id", "") or ""
                    create_crystallized_trait(
                        trait_id=trait_id,
                        stream_id=GLOBAL_STREAM,  # 思想属于 Bot 全局
                        seed_id=seed_info.get("id", "") or "",
                        name=seed_info.get("type", "trait"),
                        question=seed_info.get("question", "") or "",
                        thought=result.get("thought", "") or "",
                        tags_json=dumps_tags_json(seed_info.get("tags")),
                        confidence=int(confidence),
                        evidence_json=dumps_trait_evidence_json([evidence_entry]),
                        spectrum_impact_json=json.dumps(impact or {}, ensure_ascii=False),
                        created_at=now,
                        enabled=True,
                        deleted=False,
                        ideology_layer=layer,
                        lifecycle_state="active",
                        origin_stream_id=origin,  # 经历来源
                        commit=commit,
                    )

                    # 2) 新 trait 创建成功后，再标记旧 trait（contradicted 同时禁用）
                    if relation == "contradicted":
                        set_trait_lifecycle_state(target_trait_id, "contradicted", enabled=False, commit=commit)
                    elif relation == "weakened":
                        set_trait_lifecycle_state(target_trait_id, "weakened", commit=commit)
                    elif relation == "revised":
                        set_trait_lifecycle_state(target_trait_id, "revised", commit=commit)
                    logger.info(f"旧 trait {target_trait_id} 被标记为 {relation} (seed={seed_info.get('id', '')})")

                    # 3) 写思想图谱边：旧 trait → 新 trait
                    edge_types = {"contradicted": "contradicted_by", "weakened": "weakened_by", "revised": "revised_by"}
                    edge_type = edge_types[relation]
                    create_thought_edge(
                        from_trait_id=target_trait_id,
                        to_trait_id=trait_id,
                        relation_type=edge_type,
                        source_ref=seed_info.get("id", "") or "",
                        commit=commit,
                    )
                    logger.info(f"已创建边 {edge_type}: {target_trait_id} → {trait_id} (seed={seed_info.get('id', '')})")
                    return {"trait_id": trait_id, "merged": False, "relation": relation, "related_to": target_trait_id}

        # none 或未匹配：创建新 trait（现有路径）
        from ..worldview.constants import GLOBAL_STREAM, normalize_ideology_layer
        from ..worldview.service import WorldviewService, config_from_plugin

        tags = parse_tags_json(dumps_tags_json(seed_info.get("tags")))
        layer = normalize_ideology_layer(
            str(result.get("ideology_layer", "") or ""),
            default=WorldviewService.infer_layer_from_tags(tags),
        )

        origin = seed_info.get("stream_id", "") or ""
        create_crystallized_trait(
            trait_id=trait_id,
            stream_id=GLOBAL_STREAM,  # 思想属于 Bot 全局
            seed_id=seed_info.get("id", "") or "",
            name=seed_info.get("type", "trait"),
            question=seed_info.get("question", "") or "",
            thought=result.get("thought", "") or "",
            tags_json=dumps_tags_json(seed_info.get("tags")),
            confidence=int(confidence),
            evidence_json=dumps_trait_evidence_json([evidence_entry]),
            spectrum_impact_json=json.dumps(impact or {}, ensure_ascii=False),
            created_at=now,
            enabled=True,
            deleted=False,
            ideology_layer=layer,
            lifecycle_state="active",
            origin_stream_id=origin,  # 经历来源
            commit=commit,
        )
        logger.info(f"已创建 trait 记录: {trait_id} (seed={seed_info.get('id', '')})")
        return {"trait_id": trait_id, "merged": False}

    async def _classify_trait_relation(
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

        from ..models.ideology_model import query_active_traits_for_injection
        from ..utils.trait_tags import parse_tags_json

        if not new_thought.strip():
            return None

        # 候选集包含本群 trait + 全局 trait（query_active_traits_for_injection 已按
        # stream_id OR GLOBAL_STREAM 匹配且过滤 enabled），确保新 trait 能与全局观点比对矛盾
        candidates = query_active_traits_for_injection(
            stream_id=stream_id,
            limit=40,
        )
        if not candidates:
            return None

        def candidate_score(t) -> float:
            tags = parse_tags_json(t.tags_json or "[]")
            overlap = 0
            if tags and new_tags:
                overlap = len(set([x.casefold() for x in tags]) & set([x.casefold() for x in new_tags]))
            ratio = difflib.SequenceMatcher(None, (t.thought or "")[:400], new_thought[:400]).ratio()
            return float(overlap * 2) + float(ratio)

        ranked = sorted(candidates, key=candidate_score, reverse=True)[:12]
        # id→trait 映射，用于返回前的代码层校验（strengthened 豁免）
        ranked_by_id = {t.trait_id: t for t in ranked}

        items = []
        for t in ranked:
            tags = parse_tags_json(t.tags_json or "[]")
            thought_text = (t.thought or "")[:220]
            # 标记 strengthened 候选：此类旧 trait 仅可判为 duplicate
            if t.lifecycle_state == "strengthened":
                thought_text += "（已被多次证据强化，仅可判为 duplicate）"
            items.append(
                {
                    "trait_id": t.trait_id,
                    "name": t.name,
                    "tags": tags,
                    "question": (t.question or "")[:140],
                    "thought": thought_text,
                }
            )

        prompt = (
            "判断新 trait 与已有 traits 的关系，选择最匹配的一项：\n\n"
            "关系类型：\n"
            "- duplicate: 语义高度重复/同义，仅表述不同 → relation=duplicate\n"
            "- contradicted: 在同一话题上持相反立场（如旧 trait 认为「群聊应真诚直率」，\n"
            "  新 trait 认为「场面话是社交必备」）→ relation=contradicted\n"
            "  注意：仅当立场明确相反时才判为矛盾；部分质疑不算矛盾\n"
            "- weakened: 新证据部分削弱了旧观点的置信度，但不完全推翻 → relation=weakened\n"
            "- revised: 新知是旧知的更精细版本（增加条件、场景限定），语义相似但明显改进 → relation=revised\n"
            "- 若旧 trait 的 lifecycle_state 为 'strengthened'（已被多次证据强化），\n"
            "  不可判为 contradicted/weakened/revised，仅可判为 duplicate\n\n"
            f"阈值: {threshold}\n\n"
            f"新 trait:\n- name: {new_name}\n- tags: {new_tags}\n- question: {new_question[:180]}\n- thought: {new_thought[:320]}\n\n"
            f"已有 traits（候选）:\n{json.dumps(items, ensure_ascii=False)}\n\n"
            "输出 JSON:\n"
            '{"target_trait_id": "", "similarity": 0.0, "relation": "none", "reason": ""}'
        )

        result = await generate_soul_text(self._plugin, prompt)
        response = result.get("response", "")
        if not response:
            return None
        response = str(response).strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            data = json.loads(response)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        target_trait_id = str(data.get("target_trait_id", "") or "").strip()
        try:
            similarity = float(data.get("similarity", 0.0) or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        similarity = max(0.0, min(1.0, similarity))
        relation = str(data.get("relation", "none") or "none").strip().lower()
        reason = str(data.get("reason", "") or "").strip()
        valid_ids = {x["trait_id"] for x in items}

        # duplicate 需要 similarity >= threshold 且 target_trait_id 在候选中
        if relation == "duplicate":
            if similarity < threshold or target_trait_id not in valid_ids:
                return None
        elif relation in ("contradicted", "weakened", "revised"):
            if not target_trait_id or target_trait_id not in valid_ids:
                return None
            # 代码层兜底：strengthened trait 仅可判 duplicate，即使 LLM 不遵守 prompt 约束
            # 也强制降级为 none，避免被多次证据强化的牢固观点被单条新种子推翻
            target = ranked_by_id.get(target_trait_id)
            if target is not None and target.lifecycle_state == "strengthened":
                logger.warning(
                    f"LLM 尝试将 strengthened trait {target_trait_id} 判为 {relation}，已强制降级为 none"
                )
                return {"target_trait_id": "", "similarity": 0.0, "relation": "none", "reason": reason}
        else:
            # none 或其他未预期值 → 不返回 relation（由 caller 建新 trait）
            return {"target_trait_id": "", "similarity": 0.0, "relation": "none", "reason": ""}

        return {
            "target_trait_id": target_trait_id,
            "similarity": similarity,
            "relation": relation,
            "reason": reason,
        }
