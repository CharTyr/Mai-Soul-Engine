"""主人后台的灵魂引擎数据与种子审核接口（无聊天消息副作用）。"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Protocol

from maibot_sdk import API
from typing_extensions import assert_never

from ..models._conn import _get_conn
from ..models.operations import claim_seed_operation, list_recent_operations
from ..models.seeds import mark_seed_fermenting, update_seed_status
from ..plugin_ui_schema import MaiSoulEngineConfig
from ..utils.runtime_mode import ensure_mutation_allowed, resolve_runtime_mode
from ..utils.task_supervisor import TaskSupervisor
from .webui_types import (
    OperationRow,
    Overview,
    ReviewRequest,
    ReviewResult,
    SeedList,
    SeedRow,
    SpectrumRow,
    TaskHandle,
    TaskRow,
)

STATUS_LABELS: Final = {
    "pending": "待审核",
    "queued": "内化排队中",
    "fermenting": "发酵中",
    "running": "运行中",
    "waiting": "等待下一轮",
    "backoff": "退避重试",
    "failed": "失败",
    "stopped": "已停止",
    "done": "已完成",
}


class SoulWebUI(Protocol):
    """通过私有 API 接入宿主已认证的主人后台；仍遵守 api.enabled。"""

    config: MaiSoulEngineConfig
    _task_supervisor: TaskSupervisor
    _evolution_task: TaskHandle | None
    _notion_sync_task: TaskHandle | None
    _self_reflection_task: TaskHandle | None
    _fermentation_task: TaskHandle | None
    _internalization_task: TaskHandle | None

    def _require_webui(self) -> None:
        if not self.config.api.enabled:
            raise PermissionError("请先在插件配置中开启 api.enabled，才能使用灵魂引擎管理页")

    @API("soul.webui_overview", description="灵魂引擎后台状态", version="1", public=False)
    async def webui_overview(self) -> Overview:
        """只读快照，不创建光谱、不运行模型、不暴露配置密钥。"""
        self._require_webui()
        conn = _get_conn()
        counts = dict(conn.execute("SELECT status, COUNT(*) FROM soul_thought_seeds GROUP BY status"))
        mode = resolve_runtime_mode(self.config)
        can_review = mode.mutation_allowed and self.config.thought_cabinet.enabled
        tasks: list[TaskRow] = []
        for name, label, task in (
            ("evolution", "光谱演化", self._evolution_task),
            ("notion", "Notion 同步", self._notion_sync_task),
            ("reflection", "自我评价", self._self_reflection_task),
            ("fermentation", "种子发酵", self._fermentation_task),
            ("internalization", "后台内化", self._internalization_task),
        ):
            state = self._task_supervisor.state_of(name)
            status = state.status
            if status in ("running", "waiting") and (task is None or task.done()):
                status = "stopped" if task is None else "failed"
            tasks.append(
                TaskRow(
                    name=label,
                    status=STATUS_LABELS.get(status, status),
                    restart_count=state.restart_count,
                    error=state.last_death_reason,
                    last_success=datetime.fromtimestamp(state.last_success_at).isoformat(timespec="seconds")
                    if state.last_success_at
                    else "—",
                    heartbeat=datetime.fromtimestamp(state.heartbeat_at).isoformat(timespec="seconds")
                    if state.heartbeat_at
                    else "—",
                )
            )
        return Overview(
            mode=mode.mode,
            can_review=can_review,
            notice=(
                "选中种子可查看证据并审批；批准后进入发酵或后台内化队列。"
                if can_review
                else "当前只读：审批需要 apply 模式，并开启思维阁。"
            ),
            pending=conn.execute(
                "SELECT COUNT(*) FROM soul_thought_seeds s WHERE s.status = 'pending' "
                "AND NOT EXISTS(SELECT 1 FROM soul_seed_operations o WHERE o.seed_id=s.seed_id AND o.status='running')"
            ).fetchone()[0],
            fermenting=counts.get("fermenting", 0),
            traits=conn.execute("SELECT COUNT(*) FROM soul_crystallized_traits WHERE deleted = 0").fetchone()[0],
            tasks=tasks,
            operations=[
                OperationRow(
                    operation_id=op.operation_id,
                    seed_id=op.seed_id,
                    status="排队／执行中" if op.status == "running" else STATUS_LABELS.get(op.status, op.status),
                    error=op.error,
                )
                for op in list_recent_operations(limit=30)
            ],
            spectra=[
                SpectrumRow(
                    scope=row["scope_id"],
                    sincerity=row["sincerity"],
                    engagement=row["engagement"],
                    closeness=row["closeness"],
                    directness=row["directness"],
                    summary=f"真诚 {row['sincerity']} / 投入 {row['engagement']} / "
                    f"亲近 {row['closeness']} / 直率 {row['directness']}",
                )
                for row in conn.execute("SELECT * FROM soul_ideology_spectrum ORDER BY scope_id LIMIT 20")
            ],
        )

    @API("soul.webui_seeds", description="待审种子与发酵状态", version="1", public=False)
    async def webui_seeds(self, status: str = "all") -> SeedList:
        """返回待审及发酵种子；证据作为纯文本交给宿主渲染。"""
        self._require_webui()
        if status not in {"all", "pending", "fermenting", "queued"}:
            raise ValueError("无效的种子状态筛选")
        rows = (
            _get_conn()
            .execute(
                "SELECT * FROM (SELECT s.*, EXISTS(SELECT 1 FROM soul_seed_operations o "
                "WHERE o.seed_id = s.seed_id AND o.status = 'running') AS queued "
                "FROM soul_thought_seeds s WHERE s.status IN ('pending', 'fermenting') "
                ") WHERE (? = 'all' OR (? = 'queued' AND queued) "
                "OR (? = status AND NOT queued)) ORDER BY created_at DESC LIMIT 200",
                (status, status, status),
            )
            .fetchall()
        )
        return SeedList(
            rows=[
                SeedRow(
                    position=index,
                    seed_id=row["seed_id"],
                    stream_id=row["stream_id"],
                    seed_type=row["seed_type"],
                    event=row["event"],
                    reasoning=row["reasoning"],
                    evidence=row["evidence_json"],
                    context=row["context_json"],
                    impact=row["potential_impact_json"],
                    intensity=row["intensity"],
                    confidence=row["confidence"],
                    created_at=row["created_at"],
                    status="queued" if row["queued"] else row["status"],
                    status_label=STATUS_LABELS["queued"]
                    if row["queued"]
                    else STATUS_LABELS.get(str(row["status"]), str(row["status"])),
                    reviewable=(
                        row["status"] == "pending"
                        and not row["queued"]
                        and resolve_runtime_mode(self.config).mutation_allowed
                        and self.config.thought_cabinet.enabled
                    ),
                )
                for index, row in enumerate(rows, 1)
            ]
        )

    @API("soul.webui_approve", description="主人后台批准种子", version="1", public=False)
    async def webui_approve(self, seed_id: str) -> ReviewResult:
        return self._review_seed(ReviewRequest(seed_id=seed_id, decision="approve"))

    @API("soul.webui_reject", description="主人后台拒绝种子", version="1", public=False)
    async def webui_reject(self, seed_id: str) -> ReviewResult:
        return self._review_seed(ReviewRequest(seed_id=seed_id, decision="reject"))

    def _review_seed(self, request: ReviewRequest) -> ReviewResult:
        """快速提交审批，长时间内化复用现有持久队列。"""
        self._require_webui()
        ensure_mutation_allowed(self, action="网页审核思维种子")
        if not self.config.thought_cabinet.enabled:
            raise PermissionError("思维阁系统未启用")
        conn = _get_conn()
        # 在同一写事务中检查并提交，避免其它连接在审批预检后改变种子状态。
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            seed = conn.execute(
                "SELECT status FROM soul_thought_seeds WHERE seed_id = ?", (request.seed_id,)
            ).fetchone()
            if seed is None or seed["status"] != "pending":
                raise PermissionError("种子不存在或已不在待审核状态，请刷新列表")
            running = conn.execute(
                "SELECT 1 FROM soul_seed_operations WHERE seed_id = ? AND status = 'running' LIMIT 1",
                (request.seed_id,),
            ).fetchone()
            if running:
                raise PermissionError("种子已进入内化队列，请在最近操作中查看进度")
            operation_id = ""
            match request.decision:
                case "approve":
                    if self.config.thought_cabinet.fermentation_enabled:
                        if not mark_seed_fermenting(request.seed_id):
                            raise PermissionError("种子状态已变化，请刷新列表")
                        message = "已批准，种子进入发酵阶段"
                    else:
                        operation_id = claim_seed_operation(request.seed_id) or ""
                        if not operation_id:
                            raise PermissionError("种子已有内化操作，请刷新列表")
                        message = f"已进入后台内化队列：{operation_id}"
                case "reject":
                    if not update_seed_status(request.seed_id, "rejected", expected_status="pending"):
                        raise PermissionError("种子状态已变化，请刷新列表")
                    message = "种子已拒绝，记录保留供追溯"
                case unreachable:
                    assert_never(unreachable)
            return ReviewResult(message=message, operation_id=operation_id)
