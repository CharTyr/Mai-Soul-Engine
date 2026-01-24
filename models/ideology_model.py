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
    stream_id = CharField(default="")
    seed_type = CharField()  # 道德审判、权力质疑等
    event = TextField()  # 触发事件
    intensity = IntegerField()  # 强度 (0-100)
    confidence = IntegerField(default=0)  # 置信度 (0-100)
    evidence_json = TextField(default="[]")  # JSON 格式的来源片段（最多若干条）
    reasoning = TextField()  # 检测原因
    potential_impact_json = TextField()  # JSON格式的预期光谱影响
    created_at = DateTimeField(default=datetime.now)
    status = CharField(default="pending")  # pending, approved, rejected

    class Meta:
        database = db
        table_name = "soul_thought_seeds"


class CrystallizedTrait(Model):
    """固化 trait - 存储已内化的观点（可禁用/删除）。"""

    trait_id = CharField(primary_key=True)
    stream_id = CharField(default="")
    seed_id = CharField(default="")
    name = CharField()
    question = TextField(default="")
    thought = TextField()
    tags_json = TextField(default="[]")
    confidence = IntegerField(default=0)  # 置信度 (0-100)
    evidence_json = TextField(default="[]")  # JSON 格式的来源片段/形成证据（可累计）
    spectrum_impact_json = TextField(default="{}")
    created_at = DateTimeField(default=datetime.now)
    enabled = BooleanField(default=True)
    deleted = BooleanField(default=False)

    class Meta:
        database = db
        table_name = "soul_crystallized_traits"


def _sqlite_has_column(table_name: str, column_name: str) -> bool:
    try:
        rows = db.execute_sql(f"PRAGMA table_info('{table_name}')").fetchall()
    except Exception:
        return True
    return any((row[1] == column_name) for row in rows)


def _sqlite_add_column(table_name: str, column_name: str, ddl: str) -> None:
    db.execute_sql(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")


def init_tables():
    db.create_tables([IdeologySpectrum, GroupEvolutionRecord, EvolutionHistory, ThoughtSeed, CrystallizedTrait], safe=True)

    if not _sqlite_has_column(ThoughtSeed._meta.table_name, "stream_id"):
        _sqlite_add_column(ThoughtSeed._meta.table_name, "stream_id", "TEXT DEFAULT ''")
    if not _sqlite_has_column(ThoughtSeed._meta.table_name, "confidence"):
        _sqlite_add_column(ThoughtSeed._meta.table_name, "confidence", "INTEGER DEFAULT 0")
    if not _sqlite_has_column(ThoughtSeed._meta.table_name, "evidence_json"):
        _sqlite_add_column(ThoughtSeed._meta.table_name, "evidence_json", "TEXT DEFAULT '[]'")
    if not _sqlite_has_column(CrystallizedTrait._meta.table_name, "question"):
        _sqlite_add_column(CrystallizedTrait._meta.table_name, "question", "TEXT DEFAULT ''")
    if not _sqlite_has_column(CrystallizedTrait._meta.table_name, "tags_json"):
        _sqlite_add_column(CrystallizedTrait._meta.table_name, "tags_json", "TEXT DEFAULT '[]'")
    if not _sqlite_has_column(CrystallizedTrait._meta.table_name, "confidence"):
        _sqlite_add_column(CrystallizedTrait._meta.table_name, "confidence", "INTEGER DEFAULT 0")
    if not _sqlite_has_column(CrystallizedTrait._meta.table_name, "evidence_json"):
        _sqlite_add_column(CrystallizedTrait._meta.table_name, "evidence_json", "TEXT DEFAULT '[]'")


def get_or_create_spectrum(scope_id: str = "global") -> IdeologySpectrum:
    spectrum, _ = IdeologySpectrum.get_or_create(scope_id=scope_id)
    return spectrum
