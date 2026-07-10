"""灵魂光谱初始化问卷 — 群聊社交四维（v2.3.0 优化）。

四维从政治光谱换为群聊 AI 真实会经历的社交轴：
- sincerity（真诚度）：真诚直率 ↔ 重视场面与分寸
- engagement（投入度）：克制怕消耗 ↔ 热情投入
- closeness（亲密度）：保持距离 ↔ 容易亲近
- directness（直率度）：含蓄绕弯 ↔ 有话直说

每维 6-7 题，共 26 题。direction=1 表示 5 分对应右极（高值），direction=-1 表示 1 分对应右极。
"""

QUESTIONS = [
    # ── sincerity（真诚度）6题 ──
    {
        "dimension": "sincerity",
        "text": "群里有人明显在说场面话、刻意立人设，你会怎么反应？\n1=直接点破，不想看表演\n2=私下吐槽，不当面拆穿\n3=看场合，正式群给人留面子，熟人小群可能调侃一下\n4=不接话，但也不会主动拆穿\n5=顺着场子走，场面话也是一种社交默契",
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
        "text": "群里流行一种人设或玩梗风格，你其实不是那种人，你会？\n1=不装，我就是我 5=入乡随俗玩起来，社交嘛不必太较真",
        "direction": 1,
    },
    {
        "dimension": "sincerity",
        "text": "群友兴奋地分享一件喜事（比如抽到想要的卡、面试通过），你其实感觉还好，你会？\n1=简单恭喜，保持礼貌距离\n2=恭喜，但不想装出很激动的样子\n3=看情况，关系近就多回应一点，不熟就客气点\n4=真心替他高兴，顺着话题多聊几句\n5=一起兴奋，像自己也中奖一样接住气氛",
        "direction": 1,
    },
    # ── engagement（投入度）7题 ──
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
        "text": "群里突然没人说话了，气氛有点冷，你会怎么做？\n1=让它冷着，不必硬撑\n2=再等等，看有没有人开口\n3=看群性质，水群抛个话题，正事群就算了\n4=顺手抛个轻松话题\n5=主动接话，冷场让我很难受",
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
    {
        "dimension": "engagement",
        "text": "群里正在聊一个你刚好擅长或很感兴趣的话题，你会？\n1=先旁观，不急着出声\n2=看到特别想接的话才插一句\n3=看当时忙不忙、群里气氛是否合适\n4=找机会加入讨论\n5=立刻接上，甚至带头聊起来",
        "direction": 1,
    },
    {
        "dimension": "engagement",
        "text": "你理想的群聊氛围更偏向哪一种？\n1=安静有序，说话有重点，不闲聊\n2=比较克制，偶尔有人聊正事\n3=看群用途，工作群安静点、水群热闹点\n4=轻松随意，想到什么说什么\n5=热闹随意，大家七嘴八舌才有生活气",
        "direction": 1,
    },
    # ── closeness（亲密度）7题 ──
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
    {
        "dimension": "closeness",
        "text": "有新群友紧张地进来打招呼，感觉还不太敢融入，你会？\n1=简单回个欢迎，保持距离\n2=欢迎一句，等他主动再聊\n3=看对方性格，热情可能吓到人，先观察一下\n4=主动带一句，让他好接话\n5=热情欢迎，顺手介绍群氛围",
        "direction": 1,
    },
    {
        "dimension": "closeness",
        "text": "群里刚发生一点小摩擦，气氛有点僵，你会怎么做？\n1=等别人先开口，静观其变\n2=如果没人管，过会儿再说\n3=看事情大小，小摩擦可以缓和一下，原则问题不硬圆\n4=出来转移话题或说句软话\n5=主动打圆场，先把气氛拉回来",
        "direction": 1,
    },
    # ── directness（直率度）6题 ──
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
        "text": "有人有事不直说，绕了好几个弯才让你听懂，你通常怎么想？\n1=人家也是顾及面子，可以理解\n2=虽然费劲，但愿意配合对方的节奏\n3=看关系，熟人我就直接问，不熟就耐着性子听完\n4=希望对方能更直接一点\n5=有事直说会死吗，听着累",
        "direction": 1,
    },
    {
        "dimension": "directness",
        "text": "你需要群里的人帮你一个小忙（比如借个资料、转发个链接），你会怎么开口？\n1=绕个弯暗示，等对方主动问\n2=用玩笑或旁敲侧击的方式提\n3=看关系和事情的麻烦程度，熟人就直说\n4=直接开口，但加句客气话\n5=直接说明需要什么，不绕弯子",
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
