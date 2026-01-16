from datetime import datetime
from peewee import Model, CharField, IntegerField, BooleanField, DateTimeField, TextField
from src.common.database.database import db


class IdeologySpectrum(Model):
    scope_id = CharField(primary_key=True, default="global")
    economic = IntegerField(default=50)
    social = IntegerField(default=50)
    diplomatic = IntegerField(default=50)
    progressive = IntegerField(default=50)
    initialized = BooleanField(default=False)
    last_evolution = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        table_name = "worldview_ideology_spectrum"


class GroupEvolutionRecord(Model):
    group_id = CharField(primary_key=True)
    last_analyzed = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        table_name = "worldview_group_evolution"


class EvolutionHistory(Model):
    id = IntegerField(primary_key=True)
    timestamp = DateTimeField(default=datetime.now)
    group_id = CharField()
    economic_delta = IntegerField(default=0)
    social_delta = IntegerField(default=0)
    diplomatic_delta = IntegerField(default=0)
    progressive_delta = IntegerField(default=0)
    reason = TextField(default="")

    class Meta:
        database = db
        table_name = "worldview_evolution_history"


def init_tables():
    db.create_tables([IdeologySpectrum, GroupEvolutionRecord, EvolutionHistory], safe=True)


def get_or_create_spectrum(scope_id: str = "global") -> IdeologySpectrum:
    spectrum, _ = IdeologySpectrum.get_or_create(scope_id=scope_id)
    return spectrum
