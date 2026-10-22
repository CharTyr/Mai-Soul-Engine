"""灵魂引擎后台的输入模型与 JSON 返回契约。"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class TaskHandle(Protocol):
    def done(self) -> bool: ...


class ReviewRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    seed_id: str = Field(min_length=1, max_length=128, pattern=r"^[\w-]+$")
    decision: Literal["approve", "reject"]


class ReviewResult(TypedDict):
    message: str
    operation_id: str


class TaskRow(TypedDict):
    name: str
    status: str
    restart_count: int
    error: str
    last_success: str
    heartbeat: str


class OperationRow(TypedDict):
    operation_id: str
    seed_id: str
    status: str
    error: str


class Overview(TypedDict):
    mode: str
    can_review: bool
    notice: str
    pending: int
    fermenting: int
    traits: int
    tasks: list[TaskRow]
    operations: list[OperationRow]
    spectra: list[SpectrumRow]


class SpectrumRow(TypedDict):
    scope: str
    sincerity: int
    engagement: int
    closeness: int
    directness: int
    summary: str


class SeedRow(TypedDict):
    position: int
    seed_id: str
    stream_id: str
    seed_type: str
    event: str
    reasoning: str
    evidence: str
    context: str
    impact: str
    intensity: int
    confidence: int
    created_at: str
    status: str
    reviewable: bool
    status_label: str


class SeedList(TypedDict):
    rows: list[SeedRow]
