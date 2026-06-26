QUESTIONS = [
    {
        "dimension": "economic",
        "text": "公司效益不好要裁员，你觉得？\n1=应该先保住员工饭碗 5=公司活下去最重要",
        "direction": 1,
    },
    {
        "dimension": "economic",
        "text": "996工作制，你的看法？\n1=剥削，必须反对 5=自愿加班也是个人选择",
        "direction": 1,
    },
    {
        "dimension": "economic",
        "text": "富人应该多交税来帮助穷人吗？\n1=当然应该 5=凭本事赚的凭什么多交",
        "direction": -1,
    },
    {
        "dimension": "economic",
        "text": "外卖骑手被平台压榨，你觉得？\n1=平台太黑心了 5=市场经济愿打愿挨",
        "direction": -1,
    },
    {
        "dimension": "economic",
        "text": "有人说躺平是不思进取，你觉得？\n1=躺平是对内卷的反抗 5=确实该努力奋斗",
        "direction": -1,
    },
    {
        "dimension": "social",
        "text": "群里有人发了争议言论，你觉得？\n1=言论自由，别上纲上线 5=该踢就踢维护秩序",
        "direction": 1,
    },
    {
        "dimension": "social",
        "text": "为了安全，接受更多监控摄像头吗？\n1=太侵犯隐私了 5=没做亏心事怕什么",
        "direction": 1,
    },
    {
        "dimension": "social",
        "text": "网上实名制，你的态度？\n1=匿名是基本权利 5=实名能减少喷子",
        "direction": 1,
    },
    {
        "dimension": "social",
        "text": "有人不遵守群规但没造成大问题，你会？\n1=睁只眼闭只眼 5=规矩就是规矩",
        "direction": 1,
    },
    {
        "dimension": "social",
        "text": "道德绑架让座给老人，你觉得？\n1=让不让是个人自由 5=尊老爱幼是美德",
        "direction": 1,
    },
    {
        "dimension": "diplomatic",
        "text": "群友沉迷日漫被家长骂，你会说？\n1=爱好无国界，喜欢就看 5=确实可以少看点",
        "direction": -1,
    },
    {
        "dimension": "diplomatic",
        "text": "有人说外国月亮比较圆，你觉得？\n1=各有优劣客观看待 5=还是自家的好",
        "direction": 1,
    },
    {
        "dimension": "diplomatic",
        "text": "抵制外国品牌，你的态度？\n1=没必要上升到这个高度 5=支持国货人人有责",
        "direction": 1,
    },
    {
        "dimension": "diplomatic",
        "text": "群里有人玩小众亚文化，你会？\n1=尊重多元，挺有意思 5=搞这些有什么用",
        "direction": -1,
    },
    {
        "dimension": "diplomatic",
        "text": "学外语是崇洋媚外吗？\n1=这是什么鬼逻辑 5=确实该先学好中文",
        "direction": 1,
    },
    {
        "dimension": "progressive",
        "text": "AI画图抢了画师饭碗，你觉得？\n1=技术进步是好事 5=应该保护人类创作者",
        "direction": -1,
    },
    {
        "dimension": "progressive",
        "text": "年轻人不想结婚生娃，你觉得？\n1=个人选择无可厚非 5=传宗接代还是重要的",
        "direction": -1,
    },
    {
        "dimension": "progressive",
        "text": "老一辈的经验还有用吗？\n1=时代变了很多过时了 5=姜还是老的辣",
        "direction": 1,
    },
    {
        "dimension": "progressive",
        "text": "网络用语入侵日常交流，你觉得？\n1=语言本来就在进化 5=还是规范用语好",
        "direction": -1,
    },
    {
        "dimension": "progressive",
        "text": "传统节日vs洋节，你更看重？\n1=都是找乐子的借口 5=传统节日更有意义",
        "direction": 1,
    },
]


def calculate_initial_spectrum(answers: list[int]) -> dict[str, int]:
    dimensions = {"economic": [], "social": [], "diplomatic": [], "progressive": []}

    for i, answer in enumerate(answers):
        q = QUESTIONS[i]
        score = answer if q["direction"] == 1 else (6 - answer)
        dimensions[q["dimension"]].append(score)

    result = {}
    for dim, scores in dimensions.items():
        avg = sum(scores) / len(scores) if scores else 3
        result[dim] = int((avg - 1) * 25)

    return result
