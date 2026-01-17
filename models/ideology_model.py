from datetime import datetime
from peewee import Model, CharField, IntegerField, BooleanField, DateTimeField, TextField, AutoField
from src.common.database.database import db


class IdeologySpectrum(Model):
    scope_id = CharField(primary_key=True, default="global")
    economic = IntegerField(default=50)
    social = IntegerField(default=50)
    diplomatic = IntegerField(default=50)
    progressive = IntegerField(default=50)
    last_economic_dir = IntegerField(default=0)
    last_social_dir = IntegerField(default=0)
    last_diplomatic_dir = IntegerField(default=0)
    last_progressive_dir = IntegerField(default=0)
    initialized = BooleanField(default=False)
    last_evolution = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        table_name = "soul_ideology_spectrum"


class GroupEvolutionRecord(Model):
    group_id = CharField(primary_key=True)
    last_analyzed = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        table_name = "soul_group_evolution"


class EvolutionHistory(Model):
    id = AutoField()
    timestamp = DateTimeField(default=datetime.now)
    group_id = CharField()
    economic_delta = IntegerField(default=0)
    social_delta = IntegerField(default=0)
    diplomatic_delta = IntegerField(default=0)
    progressive_delta = IntegerField(default=0)
    reason = TextField(default="")

    class Meta:
        database = db
        table_name = "soul_evolution_history"


class ThoughtSeed(Model):
    """思维种子数据模型 - 存储待审核的思维种子"""

    seed_id = CharField(primary_key=True)
    seed_type = CharField()  # 道德审判、权力质疑等
    event = TextField()  # 触发事件
    intensity = IntegerField()  # 强度 (0-100)
    reasoning = TextField()  # 检测原因
    potential_impact_json = TextField()  # JSON格式的预期光谱影响
    created_at = DateTimeField(default=datetime.now)
    status = CharField(default="pending")  # pending, approved, rejected

    class Meta:
        database = db
        table_name = "soul_thought_seeds"


def init_tables():
    db.create_tables([IdeologySpectrum, GroupEvolutionRecord, EvolutionHistory, ThoughtSeed], safe=True)


def get_or_create_spectrum(scope_id: str = "global") -> IdeologySpectrum:
    spectrum, _ = IdeologySpectrum.get_or_create(scope_id=scope_id)
    return spectrum
