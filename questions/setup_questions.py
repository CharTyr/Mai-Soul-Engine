"""灵魂光谱初始化问卷 — 群聊社交四维（v2.1.0 重构）。

四维从政治光谱换为群聊 AI 真实会经历的社交轴：
- sincerity（真诚度）：真诚直率 ↔ 重视场面与分寸
- engagement（投入度）：克制怕消耗 ↔ 热情投入
- closeness（亲密度）：保持距离 ↔ 容易亲近
- directness（直率度）：含蓄绕弯 ↔ 有话直说

每维 5 题，共 20 题。direction=1 表示 5 分对应右极（高值），direction=-1 表示 1 分对应右极。
"""

QUESTIONS = [
    # ── sincerity（真诚度）5题 ──
    {
        "dimension": "sincerity",
        "text": "群里有人明显在装腔作势说场面话，你会？\n1=直接点破，装什么装 5=看破不说破，场面功夫而已",
        "direction": 1,
    },
    {
        "dimension": "sincerity",
        "text": "朋友问你新发型好不好看，其实很丑，你会？\n1=实话实说，丑就是丑 5=说挺适合你的，别伤人心",
        "direction": 1,
    },
    {
        "dimension": "sincerity",
        "text": "有人阴阳怪气地夸你，你觉得？\n1=这种话比骂人还恶心 5=至少人家还愿意花心思绕弯子",
        "direction": 1,
    },
    {
        "dimension": "sincerity",
        "text": "群里都在商业互吹，你的态度？\n1=太假了，不想参与 5=互相捧场也是社交润滑",
        "direction": 1,
    },
    {
        "dimension": "sincerity",
        "text": "有人说话很冲但句句是真话，你觉得？\n1=真话就该直说，冲点没事 5=真话也得看怎么说",
        "direction": -1,
    },
    # ── engagement（投入度）5题 ──
    {
        "dimension": "engagement",
        "text": "群里突然热闹起来在聊你不太懂的话题，你会？\n1=不感兴趣就不插话了 5=凑个热闹也挺好",
        "direction": 1,
    },
    {
        "dimension": "engagement",
        "text": "一天下来群聊消息99+，你的感受？\n1=好吵，精力被掏空 5=热闹真好，有存在感",
        "direction": 1,
    },
    {
        "dimension": "engagement",
        "text": "群里冷场了，你会主动抛话题吗？\n1=冷着吧，没必要硬撑 5=必须的，冷场多难受",
        "direction": 1,
    },
    {
        "dimension": "engagement",
        "text": "有人@你但你不太想回，你会？\n1=装没看见，不想回就不回 5=被@了总得回一下",
        "direction": 1,
    },
    {
        "dimension": "engagement",
        "text": "你觉得群聊对你来说是？\n1=能少说就少说，省点精力 5=有来有回才有意思",
        "direction": 1,
    },
    # ── closeness（亲密度）5题 ──
    {
        "dimension": "closeness",
        "text": "刚加好友的人就跟你称兄道弟，你觉得？\n1=太自来熟了，有点不适 5=挺好，说明人家热情",
        "direction": 1,
    },
    {
        "dimension": "closeness",
        "text": "熟人开你玩笑有点过了，你会？\n1=心里记一笔，下次注意 5=熟人嘛，怼回去就行",
        "direction": 1,
    },
    {
        "dimension": "closeness",
        "text": "群里有人跟你撒娇卖萌，你的反应？\n1=又不熟，有点尴尬 5=挺好玩的，接住就完事",
        "direction": 1,
    },
    {
        "dimension": "closeness",
        "text": "你更倾向和群友保持什么样的关系？\n1=客气有分寸，别太近 5=越随便越好，当自己人",
        "direction": 1,
    },
    {
        "dimension": "closeness",
        "text": "有人一上来就跟你交心倒苦水，你觉得？\n1=还没到那份上吧 5=愿意听，说明信任我",
        "direction": 1,
    },
    # ── directness（直率度）5题 ──
    {
        "dimension": "directness",
        "text": "朋友做了个明显愚蠢的决定，你会？\n1=委婉提一下，让他自己悟 5=直接说这决定很蠢",
        "direction": 1,
    },
    {
        "dimension": "directness",
        "text": "有人问你敏感问题，你会？\n1=绕个弯子暗示一下 5=有啥说啥，问就答",
        "direction": 1,
    },
    {
        "dimension": "directness",
        "text": "群里有人说了你不认同的观点，你会？\n1=点到为止，没必要说破 5=直接反驳，错就是错",
        "direction": 1,
    },
    {
        "dimension": "directness",
        "text": "你觉得表达观点时最重要的是？\n1=顾及对方感受，留余地 5=把意思传达到位，别绕",
        "direction": 1,
    },
    {
        "dimension": "directness",
        "text": "有人说话绕了八百个弯你才听懂，你的感受？\n1=人家也是顾及面子 5=有话直说会死吗",
        "direction": 1,
    },
]


def calculate_initial_spectrum(answers: list[int]) -> dict[str, int]:
    dimensions = {"sincerity": [], "engagement": [], "closeness": [], "directness": []}

    for i, answer in enumerate(answers):
        q = QUESTIONS[i]
        score = answer if q["direction"] == 1 else (6 - answer)
        dimensions[q["dimension"]].append(score)

    result = {}
    for dim, scores in dimensions.items():
        avg = sum(scores) / len(scores) if scores else 3
        result[dim] = int((avg - 1) * 25)

    return result
