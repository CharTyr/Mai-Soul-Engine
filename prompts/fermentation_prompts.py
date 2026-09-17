"""发酵相关 Prompt（v2.4.0 新增）。

包含：
1. 关联度判断 prompt：判断群聊消息与种子的关联度
2. 发酵后内化 prompt：基于种子原始内容 + 发酵期间累积输入，形成明确结论
"""

# Prompt Version: v2.4.0

FERMENTATION_RELEVANCE_PROMPT = """判断以下群聊消息与一个思维种子的关联度。

思维种子摘要:
- 类型: {seed_type}
- 事件: {seed_event}
- 原因: {seed_reasoning}

待判断的群聊消息:
{messages}

请逐条判断每条消息与种子的关联度（0-1）：
- 1.0: 直接讨论相同话题，表达了相关立场
- 0.7: 间接相关，提到了相关概念或类似情境
- 0.4: 弱相关，话题有交集但核心不同
- 0.0: 完全无关

请以JSON格式返回，一个数组，每条消息一个元素:
[{{"index": 0, "score": 0.8, "reason": "简要说明"}}, ...]

只返回JSON，不要其他内容。"""


FERMENTED_INTERNALIZATION_PROMPT = """基于以下思维种子及其发酵期间积累的群聊讨论，形成一个明确的、带结论的思想。

种子信息:
- 类型: {type}
- 触发事件: {event}
- 检测原因: {reasoning}
- 种子强度: {intensity:.2f}
- 置信度: {confidence:.2f}

种子原始证据:
{evidence}

原始对话上下文:
{context}

发酵期间积累的相关群聊讨论（按时间排列）:
{fermentation_inputs}

预期光谱影响（参考，可在此基础上调整）:
{potential_impact}

请完成三个任务：

**任务1：形成明确结论**
以第一人称内心独白的形式，基于种子原始想法和发酵期间的群聊讨论，形成一个**明确的、带结论的思想**。
- 如果发酵期间的讨论支持种子方向，强化结论
- 如果发酵期间的讨论提出反例或挑战，调整或修正结论
- 如果发酵输入与种子方向矛盾，体现思考后的取舍
- 结论应该是清晰的立场，不是模棱两可的"需要继续观察"
- 50-120字

**任务2：评估光谱影响**
这个结论会如何影响我的群聊社交人格倾向？
- sincerity: 对真实自然vs配合社交表演的看法 (-{max_delta}到+{max_delta})
- engagement: 对克制怕消耗vs热情投入的看法 (-{max_delta}到+{max_delta})
- closeness: 对保持距离vs容易亲近的看法 (-{max_delta}到+{max_delta})
- directness: 对含蓄绕弯vs有话直说的看法 (-{max_delta}到+{max_delta})
sincerity 与 directness 相互独立：sincerity 看"是否违心/配合表演"，directness 看"信息是否绕弯/留余地"。
经过发酵的思想更成熟，光谱影响应该比即时内化更明显。

**任务3：标注关键发酵输入**
在 reasoning 中说明哪些发酵输入对结论有实质影响（如有）。

tags：打 1-3 个能描述"在什么场景下会用到这个观点"的标签，优先用场景词（如接梗、阴阳、劝架、短回复、技术向、拒绝、冷场救、玩梗、吐槽、边界），也可用话题词（如游戏、感情、职场）。

请以JSON格式返回:
{{"thought": "我形成的明确结论...", "ideology_layer": "values|worldview|conduct", "spectrum_deltas": {{"sincerity": 0, "engagement": 0, "closeness": 0, "directness": 0}}, "reasoning": "为什么产生这样的结论，哪些发酵输入有实质影响", "confidence": 0.85, "tags": ["关键词1", "关键词2"]}}"""
