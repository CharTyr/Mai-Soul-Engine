# Mai-Soul-Engine 插件（Agent 指引）

独立仓库：https://github.com/CharTyr/Mai-Soul-Engine。分支：
- `main` = v2.0.0 稳定（SDK 2.x 基线）
- `dev` = **v2.5.0**（v2.4.0 发酵基线 + Phase 0 正确性 + 12 槽接入 + 插件侧 H1/H2；本文档对应此分支）
- 旧版归档：`archive/legacy-sdk1-v1`

在 Maibot 宿主中路径：`plugins/CharTyr_Mai-Soul-Engine/`，**自带 `.git`**，勿把 `config.toml` / `data/` / `config_back/` 提交进插件仓。

## 架构要点（易踩坑）

| 项 | 约定 |
|----|------|
| 运行时 | **maibot-plugin-sdk 2.x**，独立 Runner；入口 `plugin.py` + `create_plugin()` |
| 禁止 | `import src.*`、写宿主 `data/MaiBot.db`、恢复 POST_LLM 注入 |
| 注入 | **两个注入点**，都走 `components/ideology_injector.py`：**planner**（`maisaka.planner.before_request`，主视图——立场/分层/情绪/图谱/自评，**落快照、打冷却**）+ **replyer**（`maisaka.replyer.before_model_request`，只给「与本轮相关的观点 + 表达倾向」，**不落快照、不打冷却**，避免双重注入与同会话多快照歧义）；自评捕获另有 `@HookHandler(mode=OBSERVE)`（`replyer.after_response`，见"自我评价反馈回路"） |
| 注入载荷 | **宿主传入/回读的是 `items`（Context Item 列表），不是 `messages`**。形状差异统一由 `utils/host_prompt_items.py` 处理。回写键与传入形状不一致时宿主**整份忽略且不报错**——历史上注入全程空转、日志一切正常，就是踩的这个 |
| 注入合并 | **追加到首个 system item 的最后一个 text part**，保留 `item_schema_version`；无 system item → fail-open 跳过注入，不得凭空 prepend |
| 宿主返回解包 | `config.get` 被 SDK 解包成**裸值**（用 `utils/host_config.py` 归一）；`llm.generate` / `send.*` / `chat.open_session` **不解包**（读 `success` / `response`）；`render.html2png` 解包成 payload |
| 命令入参 | 命令文本取**顶层 `text`**（= `message.processed_plain_text`）；`message` 字典里**没有** `text` 键（用 `spectrum_utils.extract_command_text`） |
| 数据 | **`soul.db`**（`models/`，stdlib sqlite3）。**优先**宿主 `ctx.paths.data_dir/mai_soul_engine`（`utils/data_dir.resolve_and_prepare_data_dir`，SQLite backup 迁移，失败回退 `plugin_dir/data`）；`plugin._data_dir` / `_data_dir_info` 记录实际路径与 source。`models/` 按实体拆分，`ideology_model.py` 为重导出 shim |
| 旧数据 | `on_load` → `migration/legacy_import.py` 只读宿主 `data/MaiBot.db` 的 `soul_*` 表，一次性导入；**注意旧政治轴数值无法映射到社交轴，会丢失**（详见下方"迁移注意"） |
| 配置模型 | `plugin_ui_schema.py`（`MaiSoulEngineConfig`）；`plugin.py` 只引用该类 |
| Runner 必填 | `config.toml` 须有 **`[plugin]`** + **`config_version`**；dev 版本号为 `2.5.0`；`normalize_plugin_config` 会补齐旧配置 |
| WebUI 说明 | Dashboard 只显示 `json_schema_extra` 的 **`label` / `hint`**，不是 `Field(description)` |
| 运行模式 | **`plugin.mode` = `off` / `observe` / `apply`**（schema 默认 `off`），唯一模式判定入口 `utils/runtime_mode.py`，拆成四个闸门：注入回复 / 后台学习 / 改写人格 / 管理员接纳。`off` 全关；`observe` 只学习、不改人格、不注入；`apply` 全开。**升级绝不隐式进入 `apply`**：`mode` 未显式设置时，仅当旧配置 `enabled=true` 且 `mode` 为空字符串才映射为 `observe`，否则按 `off`（schema 默认值，pydantic 会补齐）——升级不得让插件突然开始改写人格并影响真实回复。未知 `mode` 值保守回落并提示。**模式判定必须发生在真正写库那一刻**：`utils/runtime_mode.ensure_mutation_allowed(source)` 读**调用时**的当前配置，被拦则抛 `MutationBlocked`——入口检查一次挡不住「LLM 在途时被切到 `observe`」，而那正是假成功的高发路径。模式拒绝要与业务失败**区分开**（前者是设计拦截，不是故障） |
| 任务生命周期 | `on_load`/`on_config_update` 共用 `_reconcile_background_tasks`；`_task_action` 是纯函数（start/restart/stop/keep），**必须用 `done()` 判断任务死活**——只看 `is not None` 会让崩溃的任务静默停摆且看板假绿；状态记入 `utils/task_supervisor.py`（**五态** `running`/`waiting`/`backoff`/`failed`/`stopped`）。崩溃检测 → **退避重启**（5s→10s→…→300s 上限）→ 超限转 `failed` 等人工介入；另有 `last_success`/`heartbeat`/`next_retry_at`。`plugin._supervisor_loop` 每 **30s** 巡检（循环内异常吞掉，观测不得拖垮业务）；`_unloading` 置位后**不得再拉起任务**。循环在等待间隔打 `waiting`——没人调的状态就是装饰 |
| 管理员鉴权 | `plugin.enabled` 仍存在但只作兼容位；`admin_user_id` 仅标识管理员 QQ，**不是**安全边界 |
| Manifest 版本 | 须为**严格三段式 semver**（如 `2.1.0`），**不能带 `-dev` 后缀**，否则 Runner 校验拒绝 |
| 群 Stream 解析 | `utils/runtime_resolution.resolve_monitored_group_stream` 先试 `get_stream_by_group_id`（内存快），回退 `open_session`（持久创建/恢复）；无 platform 时保留 MD5 兼容 |
| 宿主人设读取 | `utils/host_persona.fetch_host_persona` 读 `personality.personality` / `reply_style`；**`config.get` 的返回值已被 SDK 解包成裸值**，必须经 `utils/host_config.fetch_config_value` 归一——旧写法要求 `{success, value}` 包装，会把人设基底静默读成空。`format_persona_for_prompt` 格式化后注入内化 prompt（`thought/internalization_engine.py` 中追加） |

## 光谱轴（v2.1.0 重构，关键）

**v2.0 用政治轴**（economic/social/diplomatic/progressive），在群聊场景下几乎触发不到、演化空转。**v2.1.0 改为群聊社交轴**：

| v2.0 政治轴 | v2.1.0 社交轴 | 三观归属层 | 层上限 |
|-------------|--------------|-----------|--------|
| economic | **sincerity**（真诚） | values（价值观，最慢） | 2 |
| social | **engagement**（投入） | worldview（世界观，中速） | 4 |
| diplomatic | **closeness**（亲近） | conduct（行为观，较快） | 6 |
| progressive | **directness**（直率） | conduct（行为观，较快） | 6 |

- 层映射常量：`worldview/constants.py` → `SPECTRUM_DIM_TO_LAYER`、`IDEOLOGY_LAYERS`、`LIFECYCLE_STATES`
- `api_set_spectrum` 的参数名 `economic/social/diplomatic/progressive` 作为**向后兼容别名**保留，内部映射到 sincerity/engagement/closeness/directness
- DB 迁移用 `ALTER TABLE RENAME COLUMN`（就地重命名，幂等，需 SQLite ≥3.25）
- 问卷与提示词：`questions/setup_questions.py`（20 题重写）、`prompts/ideology_prompts.py`（4 轴 ×9 级 +2 极端）、`prompts/thought_prompts.py`（`ENHANCED_EVOLUTION_PROMPT`）

## P1 三观生长（v2.1.0 新增）

P1 在 `[worldview].p1_enabled` 开关后（默认 off 时：分层/切片/情绪**关闭**，但**社交轴仍生效**，不回滚到政治轴）。

### 分层与限速

三层三观：**values（价值观，慢）→ worldview（世界观，中）→ conduct（行为观，快）**，各有 trait 数量上限。演化 delta 会被 `apply_layer_caps_to_deltas` 按层限额拦截，防止浅层意见过快堆积成"三观"。

### 群切片 `soul_context_slices`

记录"在哪个群偏移了多少"，仅存偏移量，不存对话原文。`WorldviewService.record_local_slice` 写入，`ideology_injector` 注入时会取分群摘要。

### 情绪 `soul_mood_state`

辅助层，随演化 delta 衰减（`nudge_mood_from_deltas`），注入时附情绪行。非核心三观，仅作语态微调参考。

### 思想图谱 `soul_thought_edges`

trait 之间的轻量边，`internalization_engine` 在 create 时写边。`WorldviewService` 提供 graph hint 供注入摘要。

### 生命周期状态

trait 有 `lifecycle_state`，6 个状态现全部有写入路径：
- `active`：新建初始状态。
- `strengthened`：merge 时被重复证据强化（TTL 过期豁免）。
- `expired`：长期未强化、超 `trait_ttl_days` 自动过期（同时 `enabled=0`）。
- `contradicted`：内省矛盾检测判定"与新 trait 在同话题持相反立场"→ `set_trait_lifecycle_state(.., enabled=False)` 禁用 + 写 `contradicted_by` 边。
- `weakened`：新证据部分削弱旧观点 → 标 weakened（`enabled` 不变，降权但可见）+ 写 `weakened_by` 边。
- `revised`：新知是旧知的更精细版本 → 标 revised（`enabled` 不变）+ 写 `revised_by` 边。

矛盾/弱化/修正由内化时 `_classify_trait_relation`（原 `_find_dedup_target`，是独立的 LLM 调用，internalization_engine.py 中第 2 次 LLM）判定。内化每个种子实际 2 次 LLM 调用：第 1 次形成观点，第 2 次判定与已有 trait 的关系。**防误报三重**：(1) 置信度阈值（contradicted≥0.70、weakened/revised≥0.60，低于降级 none）；(2) `strengthened` trait 豁免（仅可判 duplicate，不可判矛盾/弱化/修正）；(3) 可回滚（管理员 `/soul_trait_enable` 重新启用误判 trait，`/soul_trait <id>` 详情展示关系边可追溯）。注入侧 `_trait_quality_score` 对 weakened -0.3 / revised -0.1 / contradicted -1.0 降权；contradicted 因 `enabled=0` 自动排除出注入池。

### 关键文件

| 文件 | P1 职责 |
|------|---------|
| `worldview/constants.py` | 层定义、轴→层映射、归一化、`GLOBAL_STREAM="global"` 全局作用域常量 |
| `worldview/service.py` | `WorldviewService`：层上限、切片、情绪衰减、层摘要、图谱 hint（批量查边）、状态扩展、API payload |
| `models/` | 按实体拆分：`_conn.py`（连接/建表/迁移，含 `""`→`global` 迁移）、`spectrum.py`/`history.py`/`seeds.py`/`traits.py`/`p1.py`（各实体 CRUD，含 `set_trait_lifecycle_state`/`list_thought_edges_for_traits`）；`ideology_model.py` 为重导出 shim，40+ 处历史导入零破坏。P1 相关：+3 表（slices/mood/edges）+2 列（`ideology_layer`/`lifecycle_state`）+ 列重命名迁移 |
| `components/evolution_task.py` | `apply_layer_caps_to_deltas` + `record_local_slice` + `nudge_mood_from_deltas` |
| `components/ideology_injector.py` | P1 块（层摘要 / 情绪行 / 图谱 hint）追加到注入；`_trait_quality_score` 含 6 态生命周期降权 |
| `thought/internalization_engine.py` | 层推断、生命周期（merge→`strengthened`、矛盾→`contradicted`/`weakened`/`revised`）、`_classify_trait_relation` 关系判定、create 时写图谱边 |
| `components/status_command.py` | P1 扩展（切片偏移、情绪、层计数） |
| `plugin_ui_schema.py` | `WorldviewConfig` 段；`CONFIG_VERSION = "2.5.0"` |
| `plugin.py` | 生命周期 `_compute_desired_tasks` + `_reconcile_background_tasks`；`soul.get_worldview` API；`soul.get_traits` 返回 layer/lifecycle；@API 双层访问控制（`public=False` + `api.enabled`）+ `api_set_spectrum` 审计 |
| `tests/` | 插件内测试见下方「开发与验证」（约 220 项） |

## 思维阁（v2.4.0 发酵重构，关键）

v2.4.0 对思维阁做了**根本性架构变更**：从"单点冲刺"（批准→立即内化）到"持续发酵"（批准→~12h 发酵→明确结论→更大光谱影响）。受 `[thought_cabinet].fermentation_enabled` 控制（默认关，关闭时保持旧行为）。

### 种子生成稀有化

v2.3.0 及之前：每轮演化（~1h）每群最多 2 个种子，频率由 LLM 自控，理论每天每群 48 个。v2.4.0 三重约束使其稀有：

1. **Prompt 强化**（`prompts/thought_prompts.py`，v2.4.0）：明确要求"只有在对话中出现了**完整的思想碰撞**（双方/多方就同一话题表达不同立场、进行有意义讨论）才提取种子"，排除闲聊/附和/问答/吐槽，"大多数对话不会产生种子"。
2. **每轮上限 2→1**：`_process_thought_seeds` 中 `seeds[:1]`。
3. **硬性日上限**：`seed_daily_cap_per_group`（默认 1），在 `_process_thought_seeds` 中用 `count_seeds_created_today(stream_id)` 检查。

### 种子生命周期（v2.4.0 状态机扩展）

v2.3.0：`pending → approved/rejected/expired`
v2.4.0（fermentation_enabled）：`pending → fermenting → internalized`（或 `rejected`/`expired`）

- `pending`：待管理员审核。
- `fermenting`：管理员批准后进入发酵，发酵循环持续收集群聊输入。
- `internalized`：发酵完成，trait 已形成，光谱影响已写入（终态）。
- `rejected`/`expired`：不变。

**状态转换函数**（`models/seeds.py`，均带原子守卫）：
- `mark_seed_fermenting(seed_id)`：仅 `pending` → `fermenting`，同时写 `fermentation_started_at`/`fermentation_checked_at`。
- `mark_seed_internalized(seed_id)`：仅 `fermenting` → `internalized`。
- `extend_fermentation_window(seed_id)`：递增 `fermentation_extension_count`，重置 `fermentation_started_at`。
- `update_fermentation_checked(seed_id, checked_at)`：更新发酵检查时间戳。

**审核后不删除记录**：`/soul_approve` → `mark_seed_fermenting`（发酵模式），或**入内化操作队列**（即时模式：种子保持 `pending`，后台 `_internalization_loop` 消费成功后才 `mark_seed_status(.., "approved")`；命令立刻返回，不内联调 LLM）；`/soul_reject` → `"rejected"`。**不要改回 `delete_seed`**。

### 发酵引擎（v2.4.0 新增，核心）

**文件**：`thought/fermentation_engine.py`

独立异步协程 `run_fermentation_loop(plugin)`（`plugin._fermentation_task`，`on_load` 启动、`on_unload` cancel，**不复用演化循环**）。受 `thought_cabinet.enabled` + `fermentation_enabled` 双重控制。每 `fermentation_check_interval_minutes`（默认 30min）执行一轮：

1. 扫描 `status='fermenting'` 的种子
2. 对每个种子取所在群自 `fermentation_checked_at` 以来的新消息（单次窗口上限 6h 防积压）
3. **L1 关键词子串预筛**：从种子 `type+event+reasoning` 提取关键词，用子串匹配过滤 80% 无关消息（零 LLM 成本）
4. **L3 LLM 批量关联度判断**：`FERMENTATION_RELEVANCE_PROMPT` 一次处理多条候选消息，输出每条 0-1 关联度
5. 关联度 ≥ `fermentation_relevance_threshold`（默认 0.5）的消息存入 `soul_fermentation_inputs` 表
6. 更新 `fermentation_checked_at`（**仅 L3 成功时**；LLM 失败返回空列表时**不推进**）
7. 检查 `fermentation_started_at + fermentation_window_hours` 是否到期
8. 到期 + 输入 ≥ `fermentation_min_inputs` → 触发最终内化
9. 到期但输入不足 → `extend_fermentation_window`（最多 `fermentation_max_extensions` 次）
10. 达最大延长次数仍不足 → **不强制内化**，保持 `fermenting` + warning + 尽量通知管理员（无证据不得成熟）

**失败恢复**：
- 关联度判断 LLM 失败 → 返回 `[]`，**不更新 `fermentation_checked_at`**，下轮重试同窗口（与代码一致；勿再写成「返回全 0 分仍推进游标」）
- 最终内化 LLM 失败 → 保持 `fermenting` 状态，下轮重试
- 插件重启 → 从 DB 的 `fermentation_checked_at` 继续，不丢状态
- 单种子异常 → try/except per-seed，不阻断其他种子
- 配置热更 → `on_config_update` 经 `_reconcile_background_tasks` **可启停发酵任务**（不再要求整插件重载）
### 发酵后内化（v2.4.0 改进）

`InternalizationEngine.internalize_seed(seed, dedup, fermentation_inputs)` 新增 `fermentation_inputs` 参数：

- **无 `fermentation_inputs`**（旧模式）：用 `INTERNALIZATION_PROMPT`，`max_internalize_delta`（默认 ±10）
- **有 `fermentation_inputs`**（发酵后）：用 `FERMENTED_INTERNALIZATION_PROMPT`（`prompts/fermentation_prompts.py`），`fermented_max_internalize_delta`（默认 ±15，更大）。prompt 要求 LLM 基于种子原始内容 + 发酵期间累积讨论形成**明确的、带结论的思想**，标注哪些发酵输入对结论有实质影响。

### 注入中的发酵提示（v2.4.0）

发酵中的种子在 `ideology_injector` 注入时追加一行"近期正在思考的问题（尚未形成结论，仅作背景参考）"，让 bot 行为有连续性。仅 `fermentation_enabled` 时触发，取与当前群相关的 fermenting 种子（最多 2 个）。

### 种子上下文窗口

- `soul_thought_seeds.context_json`：存触发种子的**原始群聊片段**（非 LLM 二次总结）。
- `seed_manager._match_evidence_to_context`：用 difflib 把 LLM 的 evidence 模糊匹配回 `msg_lines`，取 ±2 条窗口。**`_process_thought_seeds` 必须把 `msg_lines` 透传给 `create_seed`**，否则上下文为空。
- 通知走 `_notify_admin_seed` → 从 DB 取种子（含 context）→ `format_seed_notification`，不要用原始 LLM `seed_data`（无 context）。

### 内化 prompt

- `INTERNALIZATION_PROMPT`（即时内化，`thought/internalization_engine.py`）：包含 evidence/context/intensity/confidence/potential_impact。内化 LLM 基于真实片段形成观点。
- `FERMENTED_INTERNALIZATION_PROMPT`（发酵后内化，`prompts/fermentation_prompts.py`）：在种子原始内容基础上增加发酵期间累积讨论，要求形成明确结论。
- `FERMENTATION_RELEVANCE_PROMPT`（`prompts/fermentation_prompts.py`）：关联度判断 prompt，批量处理多条消息。

### 注入选择（`ideology_injector`）

- 选择顺序（冷却硬过滤后）：**tag 命中** → **关键词相关补位**（name/question/thought/tags 术语子串）→ 无 tag 按影响分补位 → 仍空才 `fallback_recent_impact`。
- 二级排序用 `_trait_quality_score`（confidence + 生命周期加权：strengthened +0.3 / weakened -0.3）。
- `selection_mode`：`tag_hit` / `tag_hit+keyword` / `keyword_fill` / `tag_hit+tagless` / `keyword+tagless` / `tagless_fill` / `fallback_recent_impact` / `spectrum_only`（及组合）。
- `picked[].activation_reason`：`tag_hit:…` / `keyword:…` / `tagless_impact` / `fallback_recent_impact`；`/soul_inspect` 与 dashboard 可展示。
- 层摘要 `build_layer_trait_summary(.., exclude_trait_ids=selected_ids, traits=已查列表)` 排除已在详细块的 trait，避免重复。
- **合并策略**：`utils/host_prompt_items.append_block_to_first_system` 把动态层追加到**首个 system item 的最后一个 text part**（标记 `[Mai-Soul 动态层 | …]`）；**禁止**再 prepend 新 system。回写键必须与宿主传入形状一致（宿主传 `items`）——写错会被整份忽略且不报错。
- **分用途投递（第四轮）**：planner 与 replyer 是**两个不同视图**——planner（默认）给立场/分层/情绪/图谱/固化观点 + 自评，**落快照、打冷却**；replyer 只给「与本轮相关的观点 + 表达倾向」，**不落快照、不打冷却**（快照锚点只能 planner 落，否则同会话「多快照」歧义永远为真；冷却按**轮**计，否则 replyer 同轮永远选不中）。**不把完整动态层塞两遍**。
- **注入必须幂等**：宿主对同一请求**每次重试都会再调一次 hook**，`append_block_to_first_system` 靠标记守卫避免重复追加；动这块前先验证幂等（历史文档曾把不幂等写成幂等）。
- **token 预算**：`utils/token_budget.py` 对注入块做保守估算（CJK 按 1 token/字）与稳定裁剪，顺序即优先级。**这是估算，不是精确用量**，别当真实 token 数汇报。
- **热路径优化**：`WorldviewConfigView` + `WorldviewService` 缓存在 `plugin._wv_config_view`/`plugin._wv_service`；锁用 `asyncio.Lock`；注入日志采样 + 5MB 轮转 + `asyncio.to_thread` 异步写。

### 种子去重

- `_is_duplicate_pending_seed`：本地 difflib 对**同群 pending** 种子按 `type+event+reasoning` 签名去重（阈值 `seed_dedup_threshold` 默认0.82，**不调 LLM**）。trait 级关系判定由 `_classify_trait_relation`（调 LLM）处理，输出 5 种关系（none/duplicate/contradicted/weakened/revised）。

### 配置项（`ThoughtCabinetConfig`）

v2.4.0 新增发酵配置（9 项）：`fermentation_enabled` / `fermentation_window_hours` / `fermentation_check_interval_minutes` / `fermentation_relevance_threshold` / `fermentation_max_inputs` / `fermentation_min_inputs` / `fermentation_max_extensions` / `fermented_max_internalize_delta` / `seed_daily_cap_per_group`。

已有：`max_seeds` / `min_trigger_intensity` / `auto_dedup_enabled` / `auto_dedup_threshold` / `seed_ttl_hours` / `reviewed_keep_count` / `trait_ttl_days` / `seed_dedup_threshold` / `admin_notification_cooldown_minutes` / `max_internalize_delta`。

**注意**：读取配置值**不要用 `or 默认`**——`0`/`0.0` 是合法的"关闭"值，`or` 会把它误替换成默认值。pydantic 字段实例化后必然存在，直接属性访问即可。此规范适用于**所有配置段**。
**构造 `ThoughtSeedManager`**：统一用 `ThoughtSeedManager.from_plugin_config(plugin)` 工厂。

### 命令

- `/soul_approve <id>`：行为分叉——`fermentation_enabled=true` → 标 `fermenting`；`false` → **入内化操作队列**（命令立刻返回 `operation_id`，**不内联调 LLM**）。完成/失败均出队通知。
- `/soul_op [operation_id]`：查内化操作状态（运行中 / 已完成 / 失败原因），不带参数列最近记录。
- `/soul_reject <id>`：可拒绝 `fermenting`（清理发酵输入）。
- `/soul_seed <id>`：详情 + 发酵进度。
- 只读：`/soul_trait <id>`、`/soul_inspect <文本>`（含 `activation_reason`）、`/soul_dashboard`（含 **12 格槽位**）。
- 批量：`/soul_reject_all`（**仅批量拒绝，无批量批准**）。
- Trait 管理：`/soul_trait_enable|disable|delete|set_tags|merge`；误判回滚用 enable。
- **12 槽**：`/soul_slot <trait_id> <1-12|clear>`（`set_trait_slot` 原子换槽）；注入各阶段 **有槽优先**。
- **群锁→全局**：`/soul_promote_global <id>`（`stream_id=global`，空则写入 `origin_stream_id`）；列表对非 global 标 `[仅群]`。
- 健康：`/soul_health`（任务、data_dir source、schema version、槽占用、迁移失败则 **degraded**）。

### 全局作用域标记（`GLOBAL_STREAM`）与来源溯源

`worldview/constants.py` 定义 `GLOBAL_STREAM = "global"`。trait/光谱以此值表示"不绑定特定群、对所有聊天流生效"的全局作用域。**历史上 trait 曾用空串 `""` 表全局**（与"未设置/异常"无法区分，误写空 stream_id 的群 trait 会泄漏到所有群注入），现统一用显式 `"global"`：`""` = 未设置/异常（不应匹配任何注入），`"global"` = 有意的全局作用域。`init_db` 迁移自动把存量 `""` trait 归一为 `"global"`（幂等）。

**Phase 0B.1（v2.5.0 修订）**：内化新建 trait 的 `stream_id` 由 **`worldview.local_first_evolution`（默认 `true`）** 决定——**默认写来源群**（单群对话不足以改写 bot 的全局人格），要全局须显式 `/soul_promote_global`；把开关设为 `false` 则回到旧行为（固定写 `GLOBAL_STREAM`，思想视为 Bot 全局身份）。来源群始终写入 **`origin_stream_id`**（仅溯源/展示，不参与注入 scope）。`create_crystallized_trait` 接受 `origin_stream_id`；空 `stream_id` 仍归一 global。注入查询 `query_active_traits_for_injection` 按 `stream_id == ? OR stream_id == GLOBAL_STREAM` 匹配——全局 trait 在所有群可召回，群作用域 trait 只在本群召回。存量 trait 不会被自动改写。

### @API 访问控制

7 个 `@API` 均为 SDK 级组件（`public=False`），**无网络暴露面**（插件无 HTTP server/路由/监听）。双层访问控制：`public=False`（SDK 层，仅 Runner 内可信组件可调）+ `api.enabled` 配置守卫（**schema 默认 `False`**，7 个 API 入口全检查）。唯一写接口 `api_set_spectrum` 有审计日志（`data/audit.jsonl`，`type=api_set_spectrum`，记录社交轴 before/after）。**插件层不自行实现网络级认证**（无网络面，加 token 不适用且需改 SDK/宿主）。`token`/`public_mode` 仍为配置位，当前无网络面时勿当作安全边界。

### 状态卡片可视化（`/soul_dashboard` + `/soul_trait` + `/soul_inspect`，v2.2.0）

三个命令把 Soul 引擎状态渲染成图片卡片发到聊天（非 Web 页面——SDK 无插件前端注册机制，WebUI 只能生成配置表单；此处用 `ctx.render.html2png` 把内联 HTML/CSS 经宿主无头浏览器渲染成 PNG，再 `ctx.send.image` 发图，零宿主改动）。三个卡片**共用 `dashboard_renderer.py` 的 `_wrap_html` CSS 底座**（Raycast 暗色开发工具风格：四级表面梯 + hairline 边框 + 无 drop-shadow + Inter ss03 + 生命周期语义色 chip + 顶部红色 hero stripe），各自根容器 id：`#soul-dashboard`/`#soul-trait`/`#soul-inspect`。视觉规范见 `DESIGN-raycast.md`。

- **`/soul_dashboard` 全状态总览**：`components/dashboard_data.py` 的 `collect_dashboard_data(plugin, stream_id)` 聚合光谱四轴/P1 三层 trait 计数/六态生命周期分布/情绪/本群切片/待审种子/最近演化/图谱边去重计数/功能开关。
- **`/soul_trait <id>` 详情卡片**：`handle_trait_detail` 聚合单 trait 全信息（分层/生命周期/置信度/光谱影响/证据/图谱边）→ `render_trait` 出图。**边展示覆盖全部 5 种关系**（`derived_from`/`supports`/`contradicted_by`/`weakened_by`/`revised_by`，此前文本版只展示前 2 种，已修）。
- **`/soul_inspect <文本>` 命中预览**：`components/inspect_command.py` 干跑 `_select_traits`/`_in_cooldown`/`_trait_quality_score`（**不实际注入**），展示"这段文本会命中哪些 trait、按什么优先级选中、哪些被跳过及原因"。管理员诊断"bot 看到这句话会调用哪些人格"。
- **渲染**：`components/dashboard_renderer.py` 的 `DashboardRenderer.render/render_trait/render_inspect` → `ctx.render.html2png` → base64；`build_dashboard_text`/`build_trait_text`/`build_inspect_text` 是纯文本降级版。
- **降级**：`card_enabled=False` 或渲染失败/超时 → 自动降级同内容纯文本（失败时前缀"卡片渲染失败"）；`send.image` 异常捕获具体类型（`OSError`/`RuntimeError`）非裸 except。
- **配置**：`[render]` 段（`card_enabled`/`viewport_width`/`device_scale_factor`/`render_timeout_ms`），`CONFIG_VERSION=2.2.0`。
- **约束**：html2png 走宿主无头浏览器有渲染开销，**只用于管理员主动触发的命令**，不进热路径；CSS 全内联不引外部资源；渲染失败必须降级文本而非崩。

## 自我评价反馈回路（v2.3.0 新增）

补上插件此前最大缺口：**单向注入**（只告诉 planner"你是谁"，从不检查"你表现得像不像自己"）。v2.3.0 加闭环：注入→输出→自评→校准人设+下次提醒。受 `[self_reflection].enabled` 控制（默认关）。

### 接线（唯一新增 hook 点）

- **捕获**：`@HookHandler("maisaka.replyer.after_response")`，`mode=HookMode.OBSERVE` + `error_policy=ErrorPolicy.SKIP`（**零干扰不改写输出**，失败不影响 bot 回复）。宿主触发点（**只读引用，不改**）：`src/chat/replyer/maisaka_generator_base.py:1182`，payload 含 `response`/`session_id`/`reply_message_id`/token 统计。planner 决策不进入自评——planner 的策略选择最终体现在 replyer 输出中，由 replyer 自评覆盖。
- **配对**：`before_request`（现有 `ideology_injector`）注入时落 `soul_injection_snapshots`（session_id + 命中 trait_ids + 光谱 + mood + selection_mode + **触发上下文**）；`after_response` 用 `claim_snapshot_for_response` **FIFO 认领最旧的未认领快照**（同一 reply 重试复用同一快照）。旧行为是"取最近一条"，同会话两轮并发时会把 A 轮的回复配到 B 轮的快照上——trait 归属与触发上文一起串味。时序安全：`inject_ideology` 是 BLOCKING，宿主在 before 完成后才调 LLM 再触发 after。
- **配对歧义（v6）与宿主零改动结论**：陈旧窗口内 ≥2 条未认领快照 → 标 `pairing_ambiguous=1`，**下游阻断会改人格的自评反馈**（不确定就不改人格）。`reply_message_id` 只让「同一回复重试」那条腿**精确**；planner ↔ replyer 之间**没有共同请求 id**（已核实宿主 payload），在「不得修改宿主任何代码」约束下精确绑定**不可达——已知限制，不是待办**。作用域字段：快照记 `bot_identity`（v7）+ `platform`（v8，按配置声明的平台列表探测宿主流列表得出；探不到留空）。
- **context 来源**：`after_response` payload **不含触发消息**，故 before_request 把触发上文随快照一起落库（`context_json`）；`reflection_capture.take_cached_context` 的 session 键缓存只作旧数据兜底。**上下文缺失 = context_json 空 = 合法降级**（评估只基于 response 文本判语气）。
- **投递阶段**：`delivery_state` 记 `selected` → `hook_applied`。宿主未提供请求后回调，插件**无法自我声明"最终请求已包含注入"**（`final_request_verified` 不可达）——不要把它当成已验证。

### 数据（3 表，`models/self_reflection.py`）

- `soul_injection_snapshots`：注入快照（仅 enabled 时写，防膨胀）。
- `soul_pending_reflections`：待评队列，**TTL + 上限 + `expired` 状态**防堆积（`cleanup_expired_pending` 每轮清超龄 + 超量删最旧）。
- `soul_self_reflections`：评价结果（reply_type/evaluated/consistency_score/deviating_axis/deviating_direction/reason/seed_id）。
  - **Phase 0A**：另有 `raw_consistency_score` / `normalized_consistency_score` / `correction_consumed_at`。
  - **主列 `consistency_score` = raw**（兼容旧读路径）；归一化分单独存。

### 评价层（`components/reflection_evaluator.py`）

独立异步协程（`plugin._self_reflection_task`，`on_unload` cancel，**不复用演化循环**）。每周期：清理 pending → 取队列 → 批量送 LLM → 落 self_reflections + 更新 pending 状态 → 显著偏离生成 `self_observation` 种子。

- **prompt（`prompts/self_reflection_prompts.py`）**：**不给完整光谱+trait"标准答案"**（评估 LLM 与注入 LLM 同模型，共享判断框架→系统性高分），只给**抽象倾向** + **对立视角**（挑剔外部观察者，倾向于找不一致）+ 相关性门槛三档**具体判例**。
- **相关性门槛三档**（`relevance_gate_enabled` 默认开，**运行时已接线**）：`false` 时 prompt 去掉三档判例；`true` 时 `social_glue` 跳过 → `reactive` 只评语气 → `substantive` 完整评。
- **批次归一化**（`normalize_across_batch` 可选）：自评分减本批均值，对冲系统性高估；**种子门槛始终用 raw**，禁止用归一化后分数判定。
- **self_observation 种子**：仅 substantive + **raw** 一致性分<70 + LLM 给了 trait 时生成，走 `/soul_approve` 人工审批。
- **独立日上限**：`self_observation_daily_cap`（默认 2，`0`=不限制），`count_self_observation_seeds_created_today()` 跨群统计；**不与**群聊 `seed_daily_cap_per_group` 共用配额。

### 双路反馈（`components/reflection_feedback.py`）

- **演化路**：`apply_self_reflection_spectrum_correction` 在演化循环末尾调用（仅 enabled）。用 `list_unconsumed_reflections_for_correction`（**跨 session**，不按 `GLOBAL_STREAM` 过滤丢群记录）。自评偏离 ×`self_reflection_weight`(0.5) 折算光谱 delta，**dead zone**（净偏离≥3 才修正）+ weight<1 防自指闭环。参与聚合的记录写 `correction_consumed_at`（含 dead zone 未改光谱时，避免永久重试）。**直接应用原始 delta（±1 经 EMA smooth_delta 会被归零）**。
- **planner 反馈路**：`build_recent_reflection_summary` 聚合近期自评为一行，`ideology_injector._build_injection_block` 按 selection_mode 分场景注入：**有 trait** → trait 块下方"低优先级自查"；**无 trait** → 光谱后"补充参考"。

### 自指风险护栏（关键）

OBSERVE 不改写 / 评价异步批量有 dead zone / weight<1 / strengthened trait 豁免 / self_observation 默认全人工审批 / raw 门槛 + 日 cap / 评估 prompt 不给标准答案+对立视角 / 批次归一化可选 / 修正一次性消费。
### P0 前置修复（v2.3.0 同步，单独提交 f58b41b）

**bot 自消息泄漏**：`get_by_time_in_chat` 会返回 bot 自己消息。插件必须通过宿主 `config.get("bot.qq_account")` 自动识别并短路排除，禁止在插件配置里重复维护 bot 身份。

### 关键文件

| 文件 | 职责 |
|------|------|
| `models/self_reflection.py` | 3 表 dataclass + CRUD（含 `cleanup_expired_pending` TTL/上限、`claim_snapshot_for_response` FIFO 认领 + 陈旧窗口、`mark_snapshot_delivery_state` 投递阶段） |
| `components/reflection_capture.py` | 两个 OBSERVE hook 委托 + context 缓存 + snapshot 守卫。**懒导入 models 避开预存循环导入** |
| `components/reflection_evaluator.py` | 评价协程 + 批量 LLM + 相关性门槛 + self_observation 种子 + 批次归一化 |
| `components/reflection_feedback.py` | 双路反馈：光谱修正（dead zone）+ planner 摘要聚合 |
| `components/reflection_command.py` | `/soul_reflect [N]` 管理员查看 |
| `prompts/self_reflection_prompts.py` | 评价 prompt（抽象倾向+对立视角+门槛判例） |
| `plugin_ui_schema.py` | `SelfReflectionConfig` 段；`CONFIG_VERSION=2.5.0` |
| `plugin.py` | 一个 after_response HookHandler（replyer）+ `_self_reflection_task` 生命周期 + `/soul_reflect` 命令 |

## 迁移注意（重要）

### 从 v1.x（旧 SDK1）→ dev

旧版用政治轴存光谱，dev 用社交轴。`legacy_import` 导入时旧列名与新列名不匹配，**旧光谱数值会丢失**——这是有意为之，政治轴数值在社交轴下语义无意义（economic=60 不代表 sincerity=60）。思维种子和 traits 可正常导入（不涉及轴名）。

迁移后建议 `/soul_reset` → `/soul_setup` 重新做问卷初始化。

### 从 v2.0（main）→ dev

DB 列就地重命名，数值保留但**语义已变**（原 economic=60 现被读作 sincerity=60）。**强烈建议**切换后 `/soul_reset` → `/soul_setup` 重新初始化。演化历史旧 delta 列名同步重命名，语义同样变了但不影响后续演化。

## 监控配置语义

- **`monitored_groups`**：群**白名单**；空 = 不做群演化。
- **`excluded_groups`**：从白名单里再减掉（可选）。
- **`monitored_users` / `excluded_users`**：只过滤**监控群内谁的发言**计入演化，**与私聊无关**；用户列表留空 = 该群全员计入。

## 目录职责

- `components/` — 命令、演化循环、注入、Notion（可选）、状态命令、dashboard 数据聚合/渲染/命令、自我评价捕获/评价/反馈/命令（`reflection_*.py`，v2.3.0）
- `thought/` — 思维阁种子与内化（`thought_cabinet.enabled`）；`seed_manager.py` 含上下文窗口/TTL/去重，`internalization_engine.py` 含 P1 层推断/生命周期/图谱边 + 内化 prompt 上下文 + v2.4.0 发酵后内化（`fermentation_inputs` 参数），`fermentation_engine.py`（**v2.4.0**：发酵循环 + L1 关键词过滤 + L3 LLM 关联度判断 + 到期检测/延长/最终内化触发）
- `worldview/` — **P1 新增**：`constants.py`（层/轴映射）、`service.py`（`WorldviewService`）
- `prompts/`、`questions/` — 问卷与 LLM 提示词（v2.1.0 社交轴版本；v2.3.0 +`self_reflection_prompts.py`；v2.4.0 +`fermentation_prompts.py`）
- `models/` — 按实体拆分：`_conn.py`（连接/建表/迁移 ledger，v2–**v8**：`cabinet_slot_no` / 快照配对 / 种子操作租约 / 通知 outbox / 快照配对**歧义**标记 / `bot_identity` / `platform`；`CURRENT_SCHEMA_VERSION = 8`）、`spectrum.py`、`history.py`、`seeds.py`（含日上限计数与终态保留）、`traits.py`（`origin_stream_id` / `set_trait_slot` / `promote_trait_to_global` / 槽位让位）、`p1.py`、`self_reflection.py`、`operations.py`（内化单赢家租约）、`notifications.py`（通知 outbox）；`ideology_model.py` 为重导出 shim
- `migration/` — `legacy_import.py`（旧库只读导入）、`inventory.py`（双数据目录**只读**盘点与迁移预演，含谱系观察，**不自动选源**）
- `config_template.toml` — 脱敏模板（示例 ID 用 `12345678`）；真实配置在本地 `config.toml`
- `utils/` — `data_dir.py`（宿主 data_dir 解析 + backup 迁移）、`host_persona.py`（人设快照）、`host_config.py`（`config.get` 裸值归一）、`host_prompt_items.py`（`items` 契约适配/合并）、`runtime_resolution.py`（群 stream：get_stream → open_session）、`runtime_mode.py`（三模式闸门）、`task_supervisor.py`（任务存活监督）、`stream_kind.py`（会话类型显式判定）、`notify.py`（通知发送 + 失败入队）、`spectrum_utils.py`（命令文本/模式闸门）、`card_render.py`、`token_budget.py`（token 预算：保守估算 + 稳定裁剪，顺序即优先级）
- `tools/` — `replay.py`（**离线回放**：候选/接纳/注入/配对四阶段全走**真实代码路径**，输出可复现记录并逐条标注「LLM 是固定 fixture」；**不得把 fixture 输出当真实模型表现**）
- `tests/` — 约 **569** 项（宿主根 `uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/ -q`）；覆盖宿主契约、快照配对、种子保留、槽位恢复、操作租约与队列、卸载隔离、通知 outbox、运行模式与命令闸门、候选校验、任务监督、会话类型、迁移盘点、**迁移鲁棒性**（WAL / 损坏库 / 中断 / 重复）、**平台探测**、**隐私与保留期**、**离线回放** 等
  - **加新迁移必须登记**到 `tests/test_migration_robustness.py` 的 `_MIGRATION_ARTIFACTS`（表 + 列）：不登记测试会**明确报错**，而不是悄悄测不着（本轮栽过两次）

## 开发与验证

在 **Maibot 仓库根**（非仅插件目录），dev 分支需额外跑 P1 模型测试：

```bash
uv run pytest pytests/test_mai_soul_legacy_import.py pytests/test_mai_soul_engine_manifest.py pytests/test_mai_soul_p1_model.py -q
```

插件内测试（同样从宿主根运行）：

```bash
uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/ -q
```

> 注：`pytests/test_mai_soul_p1_model.py` 在宿主仓维护，当前可能未提交到宿主 `seren` 分支。

插件内自检（需宿主 PYTHONPATH）：

```bash
cd /path/to/Maibot
.venv/bin/python -c "import importlib; p=importlib.import_module('plugins.CharTyr_Mai-Soul-Engine.plugin'); i=p.create_plugin(); print(len(i.get_components()))"
```

重载后联调建议：`/soul_health` → `/soul_setup`/`/soul_status` → 有种子时 `/soul_approve`（即时模式**入队**，用 `/soul_op` 看进展；发酵模式进 fermenting）→ `/soul_slot` → `/soul_inspect` / `/soul_dashboard`（12 格）。自评开时 `/soul_reflect`。完整清单见本地 `.slim/deepwork/acceptance-2.5.0.md`（不入库）。

## 修改约束

- **不要改 Maibot 主程序**（`src/`）除非维护者明确许可。
- 配置示例与文档中的 QQ/群号用占位符，勿提交真实 ID。
- 可选能力默认关：**Notion**、**思维阁**、**@API**（`api.enabled` 默认 False）、**自我评价反馈回路**（`[self_reflection].enabled`）、**发酵**（`[thought_cabinet].fermentation_enabled`）；**P1 三观生长**受 `[worldview].p1_enabled` 控制。
- **`plugin.mode = "off"`（schema 默认）**：不学习、不注入、不改人格；`observe` 只学习并生成候选（不注入、不改人格、接纳类命令被拒）；`apply` 才真正注入并允许改写人格。**`mode` 未显式设置时不隐式放行**：仅当旧配置 `enabled=true` 且 `mode` 为空字符串才映射为 `observe`，否则按 schema 默认 `off`（pydantic 会补齐默认值，所以「旧配置没写 mode」实际落在 `off`）。
- **候选优先**：内化 LLM 的输出先过 `thought/candidate.py` 结构化校验，无效候选（空观点 / 数值无法解析 / 越界 / 类型错误）**拒绝且不写任何人格状态**，返回结构化原因；越界不静默 clamp。`spectrum_impact` 是 `spectrum_deltas` 的历史别名，仍须支持。**未知光谱轴 = 拒绝**（早期实现是「告警后丢弃」，已改）；**证据引用必须来自本次输入**（白名单精确匹配或输入全文子串），并校验来源范围与身份边界；**模型不得自选 `global` 作用域**——全局变化只能走显式 `/soul_promote_global`。
- **内化幂等**：LLM 调用前先 `claim_seed_operation` 拿租约（`models/operations.py`），同一颗种子并发批准/崩后重试只施加一次光谱影响；终结时操作结果与种子终态在同一事务提交。**终结失败不得直接标 `failed`——走有界重试**（管理员的批准意图不能丢）。LLM 调用必须移出事务；提交时重验所有权 / 种子状态 / 运行模式，校验不过整笔回滚。
- **演化批次必须同一事务**：光谱 + 群切片 + 情绪 + **游标推进**包成**同一个 COMMIT**，否则重放批次会重复施加影响。`models/p1.py` 的 `get_or_create_mood` / `save_mood` / `upsert_context_slice` 带 `commit=False`——**它们内部的 `conn.commit()` 会提前结束外层事务，把回滚变成空操作**（本轮查了一轮的根因），**别再往事务里放不带 `commit=False` 的写函数**。
- **`/soul_reset` 必须严格确认**：确认串**整串精确匹配**（`disconfirm` 之类不算确认）、确认状态绑定**操作者 + 会话**（同群他人不得替他补确认）、有效期 300s、执行前**重新鉴权 + 重新查运行模式**；提示文案必须写明重置范围（全局）。**含糊请求不得默认解释为全局重置**。
- **种子保留只碰终态**：`approved`/`rejected`/`expired`/`internalized` 才可回收，`pending`/`fermenting` 不得删（发酵中是在途工作，删了会连发酵输入一起丢）。
- **槽位恢复**：重新启用 trait 时若其槽已被别的启用 trait 占用，**让出自己的槽号**（不挤走现占用者），避免撞 `cabinet_slot_no` 部分唯一索引。
- **`local_first_evolution`（默认 true）**：开=内化观点写来源群、只影响该群，要全局须显式 `/soul_promote_global`（单群输入不足以改写 bot 的全局人格）；关=观点直接写全局（旧行为，可切回）。`/soul_health` 显示当前作用域。
- **`p1_enabled=false` 只关分层/切片/情绪，社交轴仍然生效**，不会回滚到政治轴。
- **`fermentation_enabled=false` 只关发酵**，批准后的内化走操作队列（见下一条）；**`fermentation_min_inputs=0` 会弱化「最少证据」门槛（到期易直接内化），不推荐**。
- 光谱边界：硬 clamp 0–100，禁止越界反弹。
- 演化：`_analyze_group` → `"success"|"skipped"|"failed"`，禁止无条件 True。
- **长操作一律走队列**：宿主命令 RPC 超时 60s，而插件给 LLM 的超时最长 120s——**禁止在命令里内联调 LLM**，否则会出现「命令报超时/失败，但副作用其实已写入」。`/soul_approve` 只入队并立刻回 `operation_id`，后台 `_internalization_loop` 消费，`/soul_op` 查状态。`@Command` 不吃 `timeout_ms`（那是 `@HookHandler` 的参数）。
- **会话类型不猜字符串**：用 `utils/stream_kind.py`（宿主 `chat.get_group_streams` / `get_private_streams`）判定；判定不出时以更严格的设置为准。**禁止**按 `session_id` 含 "private" 字样推断。
- 新内化 trait 默认写**来源群**（`local_first_evolution=true`），要全局用 `/soul_promote_global`；槽位**不自动占用**，需 `/soul_slot`（或内化文案提示）。
- 发版：插件仓 `git push`；宿主 `plugins/*` 常在 gitignore。
- Manifest / `CONFIG_VERSION`：**2.5.0** 严格三段式 semver，**禁止 `-dev` 后缀**。

## v2.5.0 相对 v2.4.0（正确性 + 最小 12 槽 + 插件侧 Capability）

线上假闭环与产品缺口的修复汇总（**不建全量 v3 表**；优先用宿主**已有** Capability，勿默认「必须改宿主」）：

| 切片 | 要点 | 关键路径 |
|------|------|----------|
| 0A 演化 | 三态 success/skipped/failed；光谱硬 clamp | `evolution_task.py`、`spectrum_utils.py` |
| 0A 发酵 | LLM 失败 `[]` 不推进 checked_at；无证据不强制 finalize | `fermentation_engine.py` |
| 0A 自评 | raw/normalized；种子用 raw；gate 接线；跨 session + consumed | `self_reflection.py`、`reflection_*` |
| 0A 生命周期 | `_compute_desired_tasks` + `_reconcile_background_tasks`；API 默认关 | `plugin.py`、`plugin_ui_schema.py` |
| 0B.1 | 内化 `stream_id` **默认写来源群**（`worldview.local_first_evolution`，默认 true）+ `origin_stream_id` | `traits.py`、`internalization_engine.py`、`plugin_ui_schema.py` |
| 0B.2 | 关键词补位 + `activation_reason` | `ideology_injector.py` |
| 0B.3 | 追加到宿主首个 system item 的**最后一个 text part**（传/回都是 `items`）；无 system fail-open | `ideology_injector.py`、`utils/host_prompt_items.py` |
| 0B.4 | `self_observation_daily_cap`（默认 2） | `seeds.py`、`reflection_evaluator.py` |
| 0C | 内化光谱+trait+边 `commit=False` + BEGIN/COMMIT | `internalization_engine.py`、`spectrum.py`、`traits.py`、`p1.py` |
| 1.mix | `soul_schema_migrations` + `user_version`；`cabinet_slot_no` UNIQUE partial | `_conn.py`、`traits.py` |
| 1.slot | `set_trait_slot`；注入 has_slot 优先；dashboard 12 格；`/soul_slot` | `traits.py`、`ideology_injector.py`、dashboard_*、`thought_commands.py` |
| 产品 | 内化/发酵完成劝槽；`/soul_promote_global`；health degraded | `thought_commands.py`、`fermentation_engine.py`、`health_command.py` |
| data.dir | 宿主 `data_dir/mai_soul_engine` + backup 迁移 | `utils/data_dir.py`、`plugin.py` |
| H1 插件侧 | get_stream → **open_session** 回退（**非**宿主新 PR） | `utils/runtime_resolution.py` |
| H2 插件侧 | **config.get** 读 personality/reply_style → 内化基底 | `utils/host_persona.py` |
| 0D 队列 | 内化走持久操作队列：命令只入队回 `operation_id`，后台 `_internalization_loop` 消费（**禁止在命令里内联调 LLM**——命令 RPC 60s < LLM 120s，会出现「报超时但已写入」） | `thought/internalization_queue.py`、`components/thought_commands.py`、`plugin.py` |
| 0D 通知 | 发送失败入 outbox 重放；去重键必须稳定（`hash()` 受 PYTHONHASHSEED 影响，重启即失效） | `models/notifications.py`、`utils/notify.py` |
| 0D 运维 | 卸载清理**逐项隔离**（一步失败不阻断其余）；任务监督用 `done()` 判定存活 + 稳定运行后重置重启计数 | `plugin.py`、`utils/task_supervisor.py` |
| 0D 判定 | 会话类型走宿主显式流列表（不猜 `session_id` 字符串）；判定不出时以更严格设置为准 | `utils/stream_kind.py`、`ideology_injector.py` |
| 0D 盘点 | 双数据目录**只读**盘点 + 迁移预演 + 谱系观察（**不自动选源**） | `migration/inventory.py` |

**方案条目落地状态（第四轮返工后更新；本节旧表述「尚未落地」已全部作废）**：
1. **Replyer 侧分用途投递**——**已落地**。`maisaka.replyer.before_model_request` 已接线；replyer 视图只给「与本轮相关的观点 + 表达倾向」，**不落快照、不打冷却**（快照锚点只能由 planner 落，否则同会话「多快照」歧义永远为真）；冷却按**轮**计，否则 planner 注入后 replyer 同轮永远选不中刚选中的观点。
2. **token 预算与截断**——**已落地**：`utils/token_budget.py`。CJK 按 1 token/字**保守估算**，顺序即优先级，预算可配置。**没有真实分词器，这是估算不是精确值，不许当精确用量汇报**。
3. **作用域字段**——**已落地（宿主零改动）**。快照记 `bot_identity`（宿主 `bot.qq_account`）+ `platform`。平台 = **配置声明**平台列表（`plugin.platforms`，默认 `["qq"]`）→ 按平台探测宿主流列表（SDK 的 `chat.get_*_streams` 吃 platform **入参**，不返回平台）→ 命中即归属，探不到**留空**。**禁止按 `session_id` 字符串猜平台**。
4. **任务监督器五态**——**已落地**：`running` / `waiting` / `backoff` / `failed` / `stopped`；退避 5s→…→300s，另有 `last_success` / `heartbeat` / `next_retry_at`；五个真实循环在等待间隔打 `waiting`（否则该状态只是装饰）。

**宿主零改动（用户硬约束）**：**不得修改宿主任何代码**。因此 T03 的精确并发绑定（planner ↔ replyer 的共同请求 id）**不可达，属已知限制而非待办**——已核实宿主 payload：planner hook 只有 `items / item_schema_version / tool_definitions / selected_history_count / built_message_count / selection_reason / session_id` 七项，两侧没有共同 id。插件侧穷尽为：replyer 重试按 `reply_message_id` **精确**复用 + 陈旧窗口内最旧未认领 + **窗口内 ≥2 条未认领即标 `pairing_ambiguous` 并阻断会改人格的自评反馈**。宁可不动，不许猜。

**尚未做（YAGNI / 可选）**：完整 v3 candidates/runs/versions；宿主 H3 结构化 persona extension；冷却改分惩罚；自动静默占槽。

**已知债务**：存量群锁 trait 不自动改 global（用 promote）；空池时注入「无 trait 跳过」正常；关键词 2-gram 精度有限。

本地（不入库）：`.slim/deepwork/production-polish.md`、`acceptance-2.5.0.md`、`next-plan.md`、评估/蓝图。

## 参考

- 用户向：`README.md`（含 v2.5.0 要点）
- 变更：`CHANGELOG.md`（**[2.5.0]** 汇总；更早 2.4/2.3/2.1 条目保留）
- 升级计划：`MIGRATION_PLAN.md`
- SDK：https://github.com/Mai-with-u/maibot-plugin-sdk/blob/main/docs/guide.md
