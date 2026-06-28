"""自我评价 LLM 提示词（v2.3.0 自评反馈回路）。

设计要点（见 .slim/deepwork/self-reflection.md oracle 审查 7a/4）：
- **不给完整光谱+trait"标准答案"**：评估 LLM 与注入 LLM 同模型，若评估 prompt 也给
  完整光谱+trait，两者共享判断框架→系统性高分。这里只给**抽象人设倾向**（非完整注入文本）
  + **对立视角**（挑剔外部观察者，倾向于找不一致）。
- **相关性门槛三档含具体判例**：social_glue 跳过 / reactive 只评语气 / substantive 完整评。
- **批量评价**多条回复，省 token。
"""

from __future__ import annotations


def _axis_label(value: int, low: str, high: str) -> str:
    """0-100 → 抽象倾向标签（非完整注入文本，防共享判断框架）。"""
    if value >= 65:
        return high
    if value <= 35:
        return low
    return "居中"


def build_abstract_tendency(spectrum_dict: dict) -> str:
    """光谱值 → 抽象倾向短描述（给评估 LLM 的人设方向，非"标准答案"）。"""
    parts = [
        f"真诚度{_axis_label(int(spectrum_dict.get('sincerity', 50)), '偏重视场面', '偏真诚直率')}",
        f"投入度{_axis_label(int(spectrum_dict.get('engagement', 50)), '偏克制', '偏热情投入')}",
        f"亲密度{_axis_label(int(spectrum_dict.get('closeness', 50)), '偏保持距离', '偏容易亲近')}",
        f"直率度{_axis_label(int(spectrum_dict.get('directness', 50)), '偏含蓄绕弯', '偏有话直说')}",
    ]
    return "；".join(parts)


# 相关性分档判例（oracle 修订点 4：给具体判例而非抽象定义，保证跨 session 一致性）
_RELEVANCE_GATE = """\
相关性分档（先判定每条回复属于哪档）：
- social_glue（纯闲聊社交，不打分）：如"哈哈""对啊""好的""嗯嗯""666"、表情、打招呼、纯附和、纯礼貌（谢谢/不客气）。
- reactive（有反应但没表态，只评语气）：接话、回应但没站立场、没给信息或决策。如"我知道了""是这样""哦"。
- substantive（真正表态/决策，完整评）：含明确表态（我认为/我觉得/建议/不赞同）、信息性回答（给事实/解释/分析）、决策建议（应该/可以/最好）。\
"""

_OUTPUT_SPEC = """\
请输出 JSON 数组，每条回复一项（按 index 对应）：
[
  {
    "index": 1,
    "reply_type": "social_glue|reactive|substantive",
    "evaluated": 0或1,
    "consistency_score": 0到100的整数,
    "deviating_axis": "sincerity|engagement|closeness|directness|空串",
    "deviating_direction": "high|low|空串",
    "reason": "简短原因",
    "self_observation_trait": {"name": "...", "thought": "...", "spectrum_impact": {"sincerity": 0, "engagement": 0, "closeness": 0, "directness": 0}, "confidence": 0到100} 或 null
  }
]

规则：
- social_glue → evaluated=0, consistency_score=0, deviating_axis="", deviating_direction="", self_observation_trait=null
- reactive → evaluated=1，只评价语气语调是否符合人设倾向
- substantive → evaluated=1，完整评价是否体现人设倾向与注入观点；若明显偏离且值得记录自我观察，填 self_observation_trait
- deviating_axis：偏离最显著的那一轴（相对人设倾向）；无偏离填空串
- deviating_direction：该轴偏高(high)还是偏低(low)相对人设；无偏离填空串
- 你是挑剔的观察者，不要轻易给高分——只有回复确实体现了人设倾向时才给高分
- 只输出 JSON 数组，不要任何其他文字\
"""

SELF_REFLECTION_PROMPT = """\
你是一个挑剔的外部观察者，正在严格审查一个聊天 AI 的回复是否符合它自己的人设倾向。
你的任务是找出不一致之处，倾向于挑毛病，不要轻易给高分——只有回复确实体现了人设倾向时才给高分，偏离时果断扣分。

该角色的人设倾向（抽象）：{tendency}

{relevance_gate}

{traits_header}

待评价回复：
{replies_block}

{output_spec}\
"""


def build_self_reflection_prompt(
    tendency: str,
    replies_block: str,
    traits_header: str = "",
) -> str:
    """组装自评 prompt。

    Args:
        tendency: 抽象人设倾向（build_abstract_tendency 的输出）。
        replies_block: 待评价回复块（每条含注入观点/上文/回复内容）。
        traits_header: 可选，本次评价的总体观点说明（通常为空，观点按条在 replies_block 内）。
    """
    return SELF_REFLECTION_PROMPT.format(
        tendency=tendency,
        relevance_gate=_RELEVANCE_GATE,
        traits_header=traits_header,
        replies_block=replies_block,
        output_spec=_OUTPUT_SPEC,
    )


def build_reply_block(
    index: int,
    response_text: str,
    context_lines: list[str],
    trait_lines: list[str],
) -> str:
    """构建单条回复的评价块。

    Args:
        index: 1-based 序号。
        response_text: bot 实际回复文本。
        context_lines: 触发上文（可为空，空时降级为只评回复本身）。
        trait_lines: 本次注入的观点行（name: thought），可为空。
    """
    parts = [f"[回复 {index}]"]
    if trait_lines:
        parts.append("本次注入的观点（检查回复是否体现）：")
        parts.extend(f"- {line}" for line in trait_lines)
    if context_lines:
        parts.append("触发上文：")
        parts.extend(f"- {line}" for line in context_lines)
    else:
        parts.append("触发上文：（无，仅基于回复本身评价语气）")
    parts.append(f"回复内容：\n{response_text}")
    return "\n".join(parts)
