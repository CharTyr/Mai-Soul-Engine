ENHANCED_EVOLUTION_PROMPT = """分析以下群聊内容，完成两个任务：

**任务1：光谱影响评估**
从四个维度评估影响方向和强度(-{rate}到+{rate}):
1. economic: 负数=更重视公平，正数=更重视效率
2. social: 负数=更重视自由，正数=更重视秩序
3. diplomatic: 负数=更开放包容，正数=更重视本土
4. progressive: 负数=更拥抱变化，正数=更珍视传统

**任务2：思维种子识别**
只有当对话涉及深层价值观冲突或重要立场表达时才提取思维种子。
严格标准：必须涉及明确的价值判断、有足够的情感强度、能形成可内化的观点。

类型：道德审判、权力质疑、存在焦虑、集体认同、变革渴望

每个思维种子需要补充：
- confidence: 0-1 之间的置信度（你对该种子是否“确实值得内化”的把握）
- evidence: 1-3 条来自群聊内容的原始对话片段（直接引用，尽量短，格式保持“昵称: 内容”）

群聊内容:
{messages}

请以JSON格式返回，思维种子数量控制在0-2个:
{{
  "spectrum_deltas": {{"economic": 0, "social": 0, "diplomatic": 0, "progressive": 0}},
  "thought_seeds": [
    {{
      "type": "道德审判",
      "event": "具体的价值观表达或冲突事件",
      "intensity": 0.85,
      "confidence": 0.75,
      "evidence": ["A: 原始对话片段1", "B: 原始对话片段2"],
      "reasoning": "为什么这值得内化为深层观点",
      "potential_impact": {{"economic": 2, "social": -1, "diplomatic": 0, "progressive": 1}}
    }}
  ]
}}"""
