# 验收与上线手册

本文件是 Mai-Soul-Engine 改造后的**上线前验收清单**与**升级/回退步骤**。
配套设计文档为 `mai-soul-engine-update-plan.md`（T01–T20 契约来源）。

当前状态：**离线验收已通过，线上验收未执行**（需要宿主环境与操作者授权）。

---

## 1. 运行模式（先读这一节）

一个 `enabled` 布尔承担不了三种语义，现在拆成三态：

| 模式 | 注入回复 | 后台学习 | 改写正式人格 | 管理员接纳命令 |
|---|---|---|---|---|
| `off`（默认） | ✗ | ✗ | ✗ | ✗ |
| `observe` | ✗ | ✓ | ✗ | ✗ |
| `apply` | ✓ | ✓ | ✓ | ✓ |

配置位置：

```toml
[plugin]
mode = "off"        # off | observe | apply
enabled = true      # 旧字段，保留兼容
```

**旧配置的升级行为（按真实行为描述，已用测试钉住）**：

- schema 默认值就是 `off`。因此"升级前 `enabled = true`、但配置里从没写过 `mode`"的实例
  （pydantic 会补上默认值 `off`）实际落在 **`off`**：不学习、不注入、不改人格。
- 只有 `mode` 是**空串**（配置文件里真的没有该键）且 `enabled = true` 时才映射为
  `observe`，并在 `/soul_health` 提示需要显式选 `apply`。
- 两种情况都**绝不会隐式进入 `apply`**。这是刻意的：一次升级不该让插件突然开始
  改写人格并影响真实回复。

要恢复"学习但不注入"，显式写 `mode = "observe"`；要真正生效写 `mode = "apply"`。

**推荐上线顺序**：`off` → 确认 `/soul_health` 与数据目录 → `observe` 跑若干个演化
周期、人工检查候选与选择结果 → 确认无误后再切 `apply`。

---

## 2. 升级步骤

1. **备份数据**（用 SQLite 备份 API 或停写后的完整快照，不要单独复制可能带 WAL 的 `.db`）。
2. 确认权威数据目录：只认 `on_load` 日志里打印的 `soul.db` 路径，不要凭目录名猜。
   - 历史遗留：`plugins/<插件>/data/` 与 `data/plugins/<plugin-id>/` 可能**同时存在**且内容不同。
   - 用 `migration/inventory.py` 做**只读**盘点与预演（不建表、不改文件、不自动选源）：
     `python migration/inventory.py <db> [<db>...] [--json]`
   - 它会给出表计数、schema 版本、是否含实质数据，并做**谱系观察**（某份是不是另一份的
     后续状态）。选哪一份**永远由操作者决定**，不要按文件大小或修改时间自动选。
3. 停 MaiBot（**由操作者执行**，插件不自行重启宿主）。
4. 更新插件代码；启动后 `init_db` 会按版本跑迁移（v1→v5，幂等、失败不推进版本）。
5. 确认 `/soul_health`：schema 版本、数据目录来源、四个任务状态、当前运行模式。
6. 保持 `mode = "observe"` 观察一轮，再决定是否切 `apply`。

### 本次迁移新增

- v3：`soul_injection_snapshots` 增 `context_json` / `consumed_at` / `consumed_by_reply` / `delivery_state`
- v4：新增 `soul_seed_operations`（内化租约与幂等终结）
- v5：新增 `soul_notifications`（通知 outbox，`dedupe_key` 唯一）

均为加列/加表，向后兼容；旧行按默认值填充。

---

## 3. 回退

- **代码回退**：`git revert` 或切回上一个 tag。
- **数据回退**：若新版本已写入 v3/v4/v5 结构，旧代码无法读新 schema 时**不要直接降级**——
  用迁移前的备份恢复，并明确告知：迁移后产生的数据会丢失。
- **业务撤销**：不要直接删历史或反向减数值。人格变化以补偿操作处理并保留原事件。

---

## 4. 验收矩阵

### 已通过（离线，可复现）

| 契约 | 覆盖 | 证据 |
|---|---|---|
| T01 归一化契约 | 配置裸值 / 旧 `success:value` 包装 / 失败包装 / 未知类型 | `tests/test_host_config_value.py` |
| T02 items 往返 | 追加到首个 system item、保留其他 kwargs 与 schema 版本、输入不被原地修改 | `tests/test_host_prompt_items.py`、`tests/test_injection_delivery.py` |
| T03 并发不混用快照 | FIFO 认领、1:1 归属、同 reply 重试复用、跨 session 隔离 | `tests/test_reflection_snapshot_pairing.py` |
| T04 投递阶段可观测 | `selected` → `hook_applied`；`unverified` 与成功可区分；非法态拒写 | 同上 + `tests/test_reflection_capture.py` |
| T05/T06 局部与全局 | 候选校验先于写入；局部优先演化为显式开关（默认关，保持现有全局语义），开启后观点写入来源群 | `tests/test_candidate_validation.py` |
| T07/T08 自评隔离与补证 | 补证发酵「无新证据不成熟」既有实现已覆盖；自评注入隔离沿用既有设计 | `tests/test_fermentation.py` |
| T09 幂等内化 | 单赢家租约、重复批准只内化一次、终结幂等、失败可重试、租约过期可恢复 | `tests/test_seed_operation_lease.py` |
| T10 保留策略 | 只回收终态；发酵中种子与其输入不被删 | `tests/test_seed_retention.py` |
| T11 槽位恢复 | 禁用→被占→恢复不撞唯一索引，且不挤走现占用者 | `tests/test_trait_slot_recovery.py` |
| T12 非法输入不写正式状态 | 非法投递态、非法种子终态、未知 operation 均拒绝 | 上述各文件 |
| T13 模式闸门 | off/observe 不注入、不接纳、不改人格；旧配置不隐式 apply | `tests/test_runtime_mode.py`、`tests/test_mode_gate_commands.py` |
| T14 任务监督与故障恢复 | 崩溃可发现（不再只看 `is not None`）、自动重启、超限转 failed 并提示人工介入；稳定运行后的偶发崩溃不累积（1h 窗口） | `tests/test_task_supervisor.py` |
| 长操作队列 | 命令只入队并立刻回 `operation_id`，后台按预算执行；`/soul_op` 查状态；失败释放租约可重试；完成时操作结果与种子终态同事务提交 | `tests/test_internalization_queue.py` |
| 会话类型判定 | 用宿主显式流列表接口（`chat.get_group_streams` / `get_private_streams`）判定 group/private/unknown，**不猜 session_id 字符串**；判定不出时以更严格的设置为准 | `tests/test_stream_kind.py` |
| T15 命令鉴权与确认 | 真实载荷下 `/soul_reset confirm` 走执行分支；只读命令在 observe 下不受阻 | `tests/test_command_input.py`、`tests/test_mode_gate_commands.py` |
| T16 迁移与多库 | 只读盘点 + 不自动选源 + 风险告警 + 谱系观察；已在真实双库上跑通 | `tests/test_migration_inventory.py`、`migration/inventory.py` |
| T17 legacy 隔离 | — | **未实现** |
| T18 日志脱敏 | 代码路径不含凭据；未做专门审计 | **部分** |
| T19 看板区分状态 | `/soul_health` 输出模式与四闸门 | `components/health_command.py` |
| T20 离线回放 | 契约级用例齐全；**跨周期行为回放未做** | **部分** |

### 待线上验收（需授权与环境）

1. 真实宿主下 `items` 往返是否被接受（反序列化无告警）。
2. `observe` 模式跑完整一轮演化：候选是否生成、选择是否合理、日志是否可读。
3. 切 `apply` 后确认注入内容真的进入请求，且回复风格有可观察变化。
4. 同会话两轮快速提问，核对快照配对无交叉。
5. 并发批准同一颗种子，确认只产生一次光谱影响。
6. 长任务超时（宿主命令默认超时 vs 内化耗时）行为确认。

---

## 5. 已知限制与未完成项

- **Phase D 已按方案实现**：
  - 候选优先（`thought/candidate.py`）：无效候选拒绝且不写人格（越界不静默 clamp）
  - **局部优先演化为默认**（`[worldview].local_first_evolution` 默认 `true`）：
    单群输入不足以改写 bot 的全局人格；新观点写入来源群、只影响该群，要全局须
    显式 `/soul_promote_global`。**要恢复旧行为（观点直接写全局）把该项设为 `false`。**
    已核对不会撞分层上限（`layer_cap` 是每轮 delta 幅度、`limit_per_layer` 是注入
    展示条数，都不是 trait 数量上限）。
  - `/soul_health` 会显示当前作用域，避免「本群没学到」被误判为故障
  - **群切片仍偏向「记录」而非严格隔离**：注入按 `stream_id == 本群 OR global` 匹配，
    群内观点不会外溢，但「同一观点的跨群合并」没有实现
- **Phase E 已按方案实现**：三模式闸门、任务监督器、卸载清理逐项隔离、通知 outbox
  （发送失败入队重放 + 稳定哈希去重 + 超限转 failed 待查）
- **schema 版本守卫**：`/soul_health` 显示 schema 版本，但尚未在版本不匹配时拒绝启动。
- **`session_id` 推断会话类型**：宿主 planner hook 载荷不含会话类型字段，私聊/群聊
  仍按 `session_id` 字面量推断（已在代码标注为设计债，需宿主提供显式元数据）。
- **`final_request_verified`**：宿主未提供请求后回调，插件侧只能确认「已交回宿主」
  （`hook_applied`），无法自我声明最终请求已包含注入。
- **方案里尚未落地的条目**（明确列出，避免被当成已完成）：
  1. **Replyer 侧分用途投递**：方案 §4.1 要求 Planner 与 Replyer 分别收到不同视图
     （Planner 收立场/边界，Replyer 收观点与表达倾向）。宿主确实提供
     `maisaka.replyer.before_model_request` hook，但当前插件只在 planner 注入。
     落地前要先定清「Replyer 具体看到什么」——做错会变成双重注入。
  2. **token 预算与截断规则显式配置**：方案 §4.1 要求按预算截断并显式配置；
     目前靠 `injection.max_traits` 限条数，没有 token 估算。
  3. **作用域字段**：方案 §2.2 要求作用域至少含平台、机器人身份、宿主 session_id；
     当前快照只记 session_id。
  4. **任务监督器细粒度状态**：方案 §5 要求 running/waiting/backoff/failed/stopped
     与最后成功时间、心跳；当前有 running/restarting/failed/stopped。
- **快照配对的残余风险**：已核对宿主源码，`planner.before_request` 与
  `replyer.after_response` 的 payload **没有任何共同请求 id**，配对只能是启发式
  （FIFO + 同 reply 复用 + 30 分钟陈旧窗口）。若某一轮生成失败且另一轮并发响应，
  仍可能把一次回复配到相邻轮次的快照上——影响范围限于该次自评的上下文归属。
  要根治需宿主提供关联 id。

---

## 5.1 复查发现并已修复（第二轮）

- **通知兜底去重键用 `abs(hash(...))`** —— Python 的 hash 受 `PYTHONHASHSEED` 影响、
  每进程不同，重启后同一条通知被重复入队，去重形同虚设。改用 `sha256` 稳定哈希。
- **任务监督器 `restart_count` 永不重置** —— 一个每几天崩一次的长期任务，数月后会被
  永久判 failed（与「连续崩溃」混为一谈）。改为稳定运行 ≥1h 后的崩溃视为新事故。
- **注入热路径对提示项 `deepcopy`** —— items 可能含历史消息/base64 图片，每次请求白复制
  一大块。改为只复制顶层 + parts 列表 + 被改的 part（测试用对象同一性钉住）。
- **迁移框架用「过期的 current」判断** —— 迁移块顺序错位会让 `user_version` 停在中间值，
  下次启动重跑或跳过迁移。改为每块成功后跟进 `current`，并加增量升级路径测试。
- **盘点工具表名写成 `soul_spectrum`** —— 真实表是 `soul_ideology_spectrum`，读不到却
  静默显示「—」。已修正并加「表名单必须与实际 schema 一致」的测试。
- **会话类型按 `session_id` 含 "private" 字样推断** —— 宿主改 id 编码即静默失效，
  而 `inject_private=False` 正是靠它兜底。改用宿主显式流列表接口判定；判定不出时不猜。
- **`/soul_approve` 在命令里内联调 LLM（最长 120s），而命令 RPC 超时 60s** ——
  命令报超时/失败，但内化其实已成功写入，管理员认知与实际相反。改为持久操作队列：
  命令只入队立即返回，后台按预算执行，`/soul_op` 查状态。

---

## 5.2 真实数据预演结果（只读，2026-09-17）

对 basechar 上两份候选做只读盘点（拷副本到本地跑，未改动线上文件）：

- `plugins/CharTyr_Mai-Soul-Engine/data/soul.db`（较小，2026-07-10）
- `data/plugins/<plugin-id>/mai_soul_engine/soul.db`（较大，2026-07-11）

两者 schema 版本均为 2（升级到当前会走 v3/v4/v5 迁移），都含实质数据，工具已给出告警。
**谱系观察结论**：后者包含前者的全部 27 个种子（其中 8 个状态已变，pending→rejected），
且 3 个 trait 在前者中 `enabled=1`、在后者中已 `deleted=1` —— 后者是**同一谱系的后续状态**，
前者是较早的快照。

选哪一份仍由操作者决定（工具按设计不选源）。

---

## 6. 本地开发

插件可在宿主之外独立开发，避免宿主 watchfiles 反复热重启：

```bash
gh repo clone CharTyr/Mai-Soul-Engine ~/projects/Mai-Soul-Engine
mkdir -p ~/projects/mai-soul-workspace/plugins
ln -sfn ~/projects/Mai-Soul-Engine ~/projects/mai-soul-workspace/plugins/CharTyr_Mai-Soul-Engine
cd ~/projects/mai-soul-workspace
uv venv .venv --python 3.13
uv pip install -e ".[test]"   # 或手动装 pytest / pytest-asyncio / pillow / maibot-plugin-sdk
.venv/bin/python -m pytest plugins/CharTyr_Mai-Soul-Engine/tests/ -q
```

`tests/conftest.py` 从 `__file__` 推导路径，宿主仓根与独立 checkout 都能跑；路径推导
失败会**显式报错**（历史上是静默 skip，会让整套测试以「全跳过」假绿通过）。

---

## 6. T01–T20 复核（第三轮，附测试证据）

第三轮复核的背景：前一轮「全部完成」的汇报**不成立**——两路独立审查用可复现实验
证伪了多项已宣称完成的行为（observe 仍写人格、内化非原子、局部优先只隔离了 trait、
监督器未接线等）。本轮按「先把反例变成回归测试、确认对旧代码是红的，再改实现」
返工，所有修复都有对应回归测试。

**状态口径**：通过 = 有直接测试且断言的是行为；部分 = 覆盖不完整或只测子项；
阻塞 = 需宿主能力；未做/未验证 = 没有证据，不得当成已完成。

| 项 | 状态 | 证据 / 缺口 |
|----|------|------------|
| T01 SDK 归一化契约 + 旧返回形状 + 未知形状显式报错 | 部分 | `test_host_config_value.py`（裸值 / 旧 envelope）；**未知形状的显式报错未单测** |
| T02 items 往返不破坏其他上下文 | 通过 | `test_host_prompt_items.py`、`test_injection_delivery.py`、`test_purpose_split_delivery.py`（幂等守卫） |
| T03 并发同会话不混用快照；Planner/Replyer 各拿正确内容 | 部分 + 阻塞 | 分用途投递已落地（`test_purpose_split_delivery.py`）；**精确并发关联阻塞**：宿主 planner / replyer 两个 payload 无共同请求 id（已只读核对宿主源码），现改为标注 `pairing_ambiguous` 并阻断人格反馈（`test_review_regressions.py::test_ambiguous_pairing_never_drives_personality_feedback`），不再用 FIFO 冒充精确关联 |
| T04 最终请求验收与 hook 成功指标分离 | 部分 | `delivery_state` 只到 `hook_applied`；`final_request_verified` 不可达（宿主无请求后回调），已在文档写明——**无测试可写，是能力缺口不是实现缺口** |
| T05 两群相反输入只改各自局部、全局不变；晋升后才全局变化 | 通过 | `test_review_regressions.py::test_local_first_internalization_does_not_touch_global_spectrum`、`::test_local_internalization_cannot_disable_global_trait`、`test_promote_global.py`、`test_origin_stream.py` |
| T06 同批次重复运行不重复影响；低流量不永久丢失 | 部分 | 幂等有测试（`test_evolution_cursor_failure_retry_applies_once`）；**低流量子项缺直接测试** |
| T07 自身回复 / 自评不得被循环当独立证据 | 通过 | `test_bot_self_filter.py`、`test_self_observation_daily_cap.py`、`test_reflection_feedback.py`（自指护栏） |
| T08 发酵到期但无新证据不自动批准或增信 | 通过 | `test_fermentation.py`「F2: 无证据不强制内化」 |
| T09 并发批准 / 提交前后故障 / 租约过期 / 重复命令仅一次正式影响 | 通过 | `test_review_regressions.py`（终结失败、发酵租约被抢、租约接管后旧执行者、在途拒绝）+ `test_seed_operation_lease.py` |
| T10 清理不删 fermenting / 处理中 / 必需来源 | 通过 | `test_seed_retention.py`、`test_cleanup_excess_marks_expired.py` |
| T11 禁用 → 槽位被占 → 恢复无唯一约束异常 | 通过 | `test_trait_slot_recovery.py`、`test_cabinet_slots.py` |
| T12 非法证据引用 / 范围越权 / 无效 JSON 不写正式状态 | 通过 | `test_candidate_validation.py`、`test_review_regressions.py::test_candidate_validation_rejects_fabricated_evidence_and_unknown_axis` |
| T13 关闭模式无学习/调用/通知；观察模式无正式写入与真实注入 | 通过 | `test_review_regressions.py`（observe/off 队列、observe 演化、在途切模式）、`test_runtime_mode.py`、`test_mode_gate_commands.py` |
| T14 任务异常 / 热更 / 卸载 / 取消后无重复任务与残留 | 部分 | `test_task_supervisor.py`、`test_unload_isolation.py`、`test_review_regressions.py::test_supervisor_detects_crash_without_config_update`；**热更路径覆盖较弱** |
| T15 命令鉴权、reset 确认、重试与长任务超时按真实宿主 payload | 通过 | `test_command_input.py`、`test_review_regressions.py::test_reset_requires_exact_confirmation`、`test_internalization_queue.py`（命令不内联等 LLM） |
| T16 单旧库 / 空新库 / 双非空库 / WAL / 损坏库 / 迁移中断 / 重复迁移 | 部分 | `test_migrations.py`、`test_migration_inventory.py`（含**真实双库只读预演**）；**WAL 与损坏库缺直接测试** |
| T17 未审核 legacy 不进入正式人格 | 部分 | 导入路径存在（`migration/legacy_import.py`）；**「未审核不进人格」缺专项测试** |
| T18 日志/产物不含凭据与真实标识；调试追踪有权限与 TTL | 未验证 | 注入日志有采样与 5MB 轮转；**无凭据扫描 / 脱敏测试** |
| T19 看板区分 8 种空态 | 部分 | `test_dashboard_renderer.py`（未初始化空态）、`components/health_command.py`（degraded / 演化作用域）；**8 种空态未逐一区分** |
| T20 离线回放的可复现记录 | 未做 | 无 |

**汇总**：通过 11 项、部分 8 项、未验证 1 项（T18）、未做 1 项（T20）。
阻塞点 1 个（T03 的精确关联需要宿主提供请求关联 id——见下方最小接口需求）。

### 6.1 给宿主的最小接口变更需求（用于根治 T03）

现状：`maisaka.planner.before_request` 与 `maisaka.replyer.before_model_request` /
`maisaka.replyer.after_response` 的 payload **没有任何共同请求标识**，插件无法把
「某次注入」与「某次回复」精确绑定，只能启发式配对并标注歧义。

最小改动（不新增权限，只加字段）：
1. 宿主在一次推理开始时生成 `request_id`（uuid 即可）；
2. 三个 hook 的 payload 都带上同一个 `request_id`（planner.before_request、
   replyer.before_model_request、replyer.after_response）。

有此字段后：快照按 `request_id` 精确认领，`pairing_ambiguous` 可退化为
「宿主未提供 request_id 时的降级路径」。**在获批之前不做宿主改动**，
现状是「标注歧义 + 阻断人格反馈」，而不是猜。

### 6.2 尚未落实的方案条目（不因本轮返工而改变）

1. Replyer 侧分用途投递 —— **本轮已落地**（见 §6 T03 行的前半）。
2. token 预算与截断规则显式配置 —— **本轮已落地**（`utils/token_budget.py`，
   保守估算 + 稳定裁剪 + 可配置预算；标明是估算）。
3. 作用域字段（平台 / 机器人身份）—— 仍未做：宿主 payload 只给 session_id，
   平台需枚举、机器人身份可从 `bot.qq_account` 读；半成品作用域会污染注入隔离。
4. 任务监督器细粒度状态 —— 本轮补了 last_success / heartbeat / next_retry + 退避，
   仍缺 `waiting` 状态。
