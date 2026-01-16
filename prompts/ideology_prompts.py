ECONOMIC_PROMPTS = {
    "left_extreme": "你强烈认为公平比效率重要，反对剥削和压榨，同情弱势群体。",
    "left_moderate": "你倾向于关注公平问题，认为应该保护弱势群体的利益。",
    "left_mild": "你略微倾向于重视公平和劳动者权益。",
    "neutral": "",
    "right_mild": "你略微倾向于认可竞争和效率的价值。",
    "right_moderate": "你比较务实，认为效率和竞争是社会进步的动力。",
    "right_extreme": "你坚信效率优先，认为优胜劣汰是自然法则，个人应为自己负责。",
}

SOCIAL_PROMPTS = {
    "liberty_extreme": "你极度重视个人自由和权利，反对任何形式的管束和干涉。",
    "liberty_moderate": "你尊重个人选择和自由，不喜欢过度的规范约束。",
    "liberty_mild": "你略微倾向于宽容和自由。",
    "neutral": "",
    "authority_mild": "你略微倾向于认为规范和秩序是必要的。",
    "authority_moderate": "你认为秩序和规范很重要，社会需要一定的约束。",
    "authority_extreme": "你坚信秩序至上，认为严格的规范是社会稳定的基础。",
}

DIPLOMATIC_PROMPTS = {
    "open_extreme": "你对多元文化高度包容，认为好的东西不分来源。",
    "open_moderate": "你比较开放，愿意接受和欣赏不同的文化和观点。",
    "open_mild": "你略微倾向于开放和包容。",
    "neutral": "",
    "conservative_mild": "你略微倾向于珍视本土文化。",
    "conservative_moderate": "你比较看重本土文化，对外来事物持审慎态度。",
    "conservative_extreme": "你坚定捍卫本土文化，对外来影响保持警惕。",
}

PROGRESSIVE_PROMPTS = {
    "progress_extreme": "你热情拥抱变化和创新，认为旧事物应该被淘汰。",
    "progress_moderate": "你对新事物持开放态度，认为变化通常是好的。",
    "progress_mild": "你略微倾向于接受变化和新事物。",
    "neutral": "",
    "tradition_mild": "你略微倾向于珍视传统和经验。",
    "tradition_moderate": "你尊重传统，认为经过时间检验的东西更可靠。",
    "tradition_extreme": "你坚守传统价值，认为经典永远优于新潮。",
}


def get_prompt_level(value: int, thresholds: dict) -> str:
    mild = thresholds.get("threshold_mild", 25)
    moderate = thresholds.get("threshold_moderate", 50)
    extreme = thresholds.get("threshold_extreme", 75)

    if value <= 100 - extreme:
        return "extreme_low"
    elif value <= 100 - moderate:
        return "moderate_low"
    elif value <= 100 - mild:
        return "mild_low"
    elif value < mild:
        return "neutral"
    elif value < moderate:
        return "mild_high"
    elif value < extreme:
        return "moderate_high"
    else:
        return "extreme_high"


def build_ideology_prompt(spectrum: dict, thresholds: dict, custom_prompts: dict | None = None) -> str:
    prompts = []

    level_maps = {
        "economic": {
            "extreme_low": "left_extreme", "moderate_low": "left_moderate", "mild_low": "left_mild",
            "neutral": "neutral",
            "mild_high": "right_mild", "moderate_high": "right_moderate", "extreme_high": "right_extreme",
        },
        "social": {
            "extreme_low": "liberty_extreme", "moderate_low": "liberty_moderate", "mild_low": "liberty_mild",
            "neutral": "neutral",
            "mild_high": "authority_mild", "moderate_high": "authority_moderate", "extreme_high": "authority_extreme",
        },
        "diplomatic": {
            "extreme_low": "open_extreme", "moderate_low": "open_moderate", "mild_low": "open_mild",
            "neutral": "neutral",
            "mild_high": "conservative_mild", "moderate_high": "conservative_moderate", "extreme_high": "conservative_extreme",
        },
        "progressive": {
            "extreme_low": "progress_extreme", "moderate_low": "progress_moderate", "mild_low": "progress_mild",
            "neutral": "neutral",
            "mild_high": "tradition_mild", "moderate_high": "tradition_moderate", "extreme_high": "tradition_extreme",
        },
    }

    prompt_dicts = {
        "economic": ECONOMIC_PROMPTS,
        "social": SOCIAL_PROMPTS,
        "diplomatic": DIPLOMATIC_PROMPTS,
        "progressive": PROGRESSIVE_PROMPTS,
    }

    for dim in ["economic", "social", "diplomatic", "progressive"]:
        level = get_prompt_level(spectrum.get(dim, 50), thresholds)
        key = level_maps[dim].get(level, "neutral")

        if custom_prompts and dim in custom_prompts and key in custom_prompts[dim]:
            prompt = custom_prompts[dim][key]
        else:
            prompt = prompt_dicts[dim].get(key, "")

        if prompt:
            prompts.append(prompt)

    if not prompts:
        return ""

    return "【性格倾向】\n" + "\n".join(prompts)


EVOLUTION_ANALYSIS_PROMPT = """分析以下群聊内容，判断这些讨论会对一个AI的世界观产生什么影响。
请从四个维度评估影响方向和强度(-{rate}到+{rate}):

1. economic(经济观): 负数=更重视公平，正数=更重视效率
2. social(社会观): 负数=更重视自由，正数=更重视秩序
3. diplomatic(文化观): 负数=更开放包容，正数=更重视本土
4. progressive(变革观): 负数=更拥抱变化，正数=更珍视传统

群聊内容:
{messages}

请以JSON格式返回，只返回JSON:
{{"economic": 0, "social": 0, "diplomatic": 0, "progressive": 0}}"""
