# Mai-Soul-Engine 插件（Agent 指引）

独立仓库：https://github.com/CharTyr/Mai-Soul-Engine。分支：
- `main` = v2.0.0 稳定（SDK 2.x 基线）
- `dev` = v2.4.0 基线 + **Phase 0A/0B 正确性打磨**（思想全局归属、可追踪召回、宿主 system 追加注入、自评/发酵假闭环修复；本文档对应此分支）
- 旧版归档：`archive/legacy-sdk1-v1`

在 Maibot 宿主中路径：`plugins/CharTyr_Mai-Soul-Engine/`，**自带 `.git`**，勿把 `config.toml` / `data/` / `config_back/` 提交进插件仓。

## 架构要点（易踩坑）

| 项 | 约定 |
|----|------|
| 运行时 | **maibot-plugin-sdk 2.x**，独立 Runner；入口 `plugin.py` + `create_plugin()` |
| 禁止 | `import src.*`、写宿主 `data/MaiBot.db`、恢复 POST_LLM 注入 |
| 注入 | 主接线：`@HookHandler("maisaka.planner.before_request")` → `components/ideology_injector.py`；自评捕获另有一个 `@HookHandler(mode=OBSERVE)`（`replyer.after_response`，见"自我评价反馈回路"） |
| 注入合并 | **追加到宿主首条 system**（`_apply_soul_injection_to_messages`），不得 prepend 竞争性 system；找不到 system → fail-open 跳过注入 |
| 数据 | 插件自有 **`data/soul.db`**（`models/`，stdlib sqlite3）；持久化目录绑在 **插件目录 `data/`**（`plugin._data_dir`），不是 `ctx.paths.data_dir`。`models/` 已按实体拆为 7 子模块，`ideology_model.py` 保留为重导出 shim（见"目录职责"） |
| 旧数据 | `on_load` → `migration/legacy_import.py` 只读宿主 `data/MaiBot.db` 的 `soul_*` 表，一次性导入；**注意旧政治轴数值无法映射到社交轴，会丢失**（详见下方"迁移注意"） |
| 配置模型 | `plugin_ui_schema.py`（`MaiSoulEngineConfig`）；`plugin.py` 只引用该类 |
| Runner 必填 | `config.toml` 须有 **`[plugin]`** + **`config_version`**；dev 版本号为 `2.4.0`；`normalize_plugin_config` 会补齐旧配置 |
| WebUI 说明 | Dashboard 只显示 `json_schema_extra` 的 **`label` / `hint`**，不是 `Field(description)` |
| 功能总开关 | `plugin.enabled` 管**注入 + 四后台任务**（演化/Notion/自评/发酵）；`on_load`/`on_config_update` 共用 `_reconcile_background_tasks`；`admin_user_id` 仅标识管理员 QQ |
| Manifest 版本 | 须为**严格三段式 semver**（如 `2.1.0`），**不能带 `-dev` 后缀**，否则 Runner 校验拒绝 |

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
| `plugin_ui_schema.py` | `WorldviewConfig` 段；`CONFIG_VERSION = "2.4.0"` |
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

**审核后不删除记录**：`/soul_approve` → `mark_seed_fermenting`（发酵模式）或 `mark_seed_status(.., "approved")` + 立即内化（旧模式），`/soul_reject` → `"rejected"`。**不要改回 `delete_seed`**。

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
- **合并策略**：`_apply_soul_injection_to_messages` 把动态层追加到宿主首条 system 末尾（标记 `[Mai-Soul 动态层 | …]`）；**禁止**再 prepend 新 system。
- **热路径优化**：`WorldviewConfigView` + `WorldviewService` 缓存在 `plugin._wv_config_view`/`plugin._wv_service`；锁用 `asyncio.Lock`；注入日志采样 + 5MB 轮转 + `asyncio.to_thread` 异步写。

### 种子去重

- `_is_duplicate_pending_seed`：本地 difflib 对**同群 pending** 种子按 `type+event+reasoning` 签名去重（阈值 `seed_dedup_threshold` 默认0.82，**不调 LLM**）。trait 级关系判定由 `_classify_trait_relation`（调 LLM）处理，输出 5 种关系（none/duplicate/contradicted/weakened/revised）。

### 配置项（`ThoughtCabinetConfig`）

v2.4.0 新增发酵配置（9 项）：`fermentation_enabled` / `fermentation_window_hours` / `fermentation_check_interval_minutes` / `fermentation_relevance_threshold` / `fermentation_max_inputs` / `fermentation_min_inputs` / `fermentation_max_extensions` / `fermented_max_internalize_delta` / `seed_daily_cap_per_group`。

已有：`max_seeds` / `min_trigger_intensity` / `auto_dedup_enabled` / `auto_dedup_threshold` / `seed_ttl_hours` / `reviewed_keep_count` / `trait_ttl_days` / `seed_dedup_threshold` / `admin_notification_cooldown_minutes` / `max_internalize_delta`。

**注意**：读取配置值**不要用 `or 默认`**——`0`/`0.0` 是合法的"关闭"值，`or` 会把它误替换成默认值。pydantic 字段实例化后必然存在，直接属性访问即可。此规范适用于**所有配置段**。
**构造 `ThoughtSeedManager`**：统一用 `ThoughtSeedManager.from_plugin_config(plugin)` 工厂。

### 命令

- `/soul_approve <id>`：v2.4.0 行为分叉——`fermentation_enabled=true` → 标 `fermenting`（进入发酵）；`false` → 立即内化（旧行为）。
- `/soul_reject <id>`：v2.4.0 允许拒绝 `fermenting` 状态种子（清理其发酵输入）。
- `/soul_seed <id>`：v2.4.0 展示发酵进度（已收集输入数/预览最近3条/关联度分数）。
- 只读详情：`/soul_trait <id>`（trait 详情卡片）、`/soul_inspect <文本>`（注入命中预览）、`/soul_dashboard`（全状态总览）。
- 批量：`/soul_reject_all`（**只有批量拒绝，无批量批准**）。
- 误判 trait 回滚：`/soul_trait_enable <id>`。

### 全局作用域标记（`GLOBAL_STREAM`）与来源溯源

`worldview/constants.py` 定义 `GLOBAL_STREAM = "global"`。trait/光谱以此值表示"不绑定特定群、对所有聊天流生效"的全局作用域。**历史上 trait 曾用空串 `""` 表全局**（与"未设置/异常"无法区分，误写空 stream_id 的群 trait 会泄漏到所有群注入），现统一用显式 `"global"`：`""` = 未设置/异常（不应匹配任何注入），`"global"` = 有意的全局作用域。`init_db` 迁移自动把存量 `""` trait 归一为 `"global"`（幂等）。

**Phase 0B.1（关键）**：内化新建 trait 时 **`stream_id` 固定写 `GLOBAL_STREAM`**（思想属于 Bot 全局身份）；来源群写入 **`origin_stream_id`**（仅溯源/展示，不参与注入 scope）。`create_crystallized_trait` 接受 `origin_stream_id`；空 `stream_id` 仍归一 global。注入查询 `query_active_traits_for_injection` 按 `stream_id == ? OR stream_id == GLOBAL_STREAM` 匹配——因此 A 群形成的思想可在 B 群相关话题被召回。存量群锁 trait 不会被自动改写，仅新内化走全局。

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
- **配对**：`before_request`（现有 `ideology_injector`）注入时落 `soul_injection_snapshots`（session_id + 命中 trait_ids + 光谱 + mood + selection_mode）；`after_response` 用 session_id 取最近 snapshot 配对（**1:N**，一个 snapshot 可配多条 pending：replyer 多次重试）。时序安全：`inject_ideology` 是 BLOCKING，宿主在 before 完成后才调 LLM 再触发 after。
- **context 来源**：`after_response` payload **不含触发消息**，故 `reflection_capture.cache_session_context` 在 before_request 内存缓存触发上文（session_id 作 key，TTL 10min），after 取回（一次性）。**缓存缺失/超龄 = context_json 空 = 合法降级**（评估只基于 response 文本判语气）。

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
| `models/self_reflection.py` | 3 表 dataclass + CRUD（含 `cleanup_expired_pending` TTL/上限、`get_injection_snapshot` 按 id 配对） |
| `components/reflection_capture.py` | 两个 OBSERVE hook 委托 + context 缓存 + snapshot 守卫。**懒导入 models 避开预存循环导入** |
| `components/reflection_evaluator.py` | 评价协程 + 批量 LLM + 相关性门槛 + self_observation 种子 + 批次归一化 |
| `components/reflection_feedback.py` | 双路反馈：光谱修正（dead zone）+ planner 摘要聚合 |
| `components/reflection_command.py` | `/soul_reflect [N]` 管理员查看 |
| `prompts/self_reflection_prompts.py` | 评价 prompt（抽象倾向+对立视角+门槛判例） |
| `plugin_ui_schema.py` | `SelfReflectionConfig` 段；`CONFIG_VERSION=2.4.0` |
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
- `models/` — 按实体拆分：`_conn.py`（全局连接/建表/迁移/时间工具，含发酵表+列迁移 + origin/raw/consumed 列）、`spectrum.py`（光谱+群演化记录）、`history.py`（演化历史）、`seeds.py`（思维种子 CRUD + 发酵 CRUD + `count_seeds_created_today` / `count_self_observation_seeds_created_today`）、`traits.py`（trait CRUD + `origin_stream_id`）、`p1.py`（切片/情绪/图谱边）、`self_reflection.py`（注入快照/待评/自评 + unconsumed 修正查询）；`ideology_model.py` 保留为**重导出 shim**。**新代码直接从子模块 import**
- `config_template.toml` — 脱敏模板（示例 ID 用 `12345678`）；真实配置在本地 `config.toml`
- `tests/` — 插件内测试（约 **220** 项，从宿主根 `uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/ -q`）；覆盖原子守卫/生命周期/全局标记/origin/光谱 clamp/演化三态/发酵 cursor 与无证据不强制内化/自评 raw·consumed·日 cap/关键词召回/system 合并/FeatureSupervisor/dashboard+inspect 等

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

重载插件后联调：`/soul_setup` → `/soul_answer` → `/soul_status`（dev 下 status 会显示层计数、切片偏移、情绪、发酵中种子数）；开启 `[self_reflection].enabled` 后 `/soul_reflect` 看自评记录；开启 `[thought_cabinet].fermentation_enabled` 后 `/soul_approve` 进入发酵、`/soul_seed <id>` 看发酵进度；看 Runner 日志与 `data/migration_state.json`。

## 修改约束

- **不要改 Maibot 主程序**（`src/`）除非维护者明确许可。
- 配置示例与文档中的 QQ/群号用占位符，勿提交真实 ID。
- 可选能力默认关：**Notion**、**思维阁**、**@API**（`api.enabled` 默认 False）、**自我评价反馈回路**（`[self_reflection].enabled`）、**发酵**（`[thought_cabinet].fermentation_enabled`）；**P1 三观生长**受 `[worldview].p1_enabled` 控制。
- **`plugin.enabled=false`**：注入跳过 + 四后台任务 desired 全 false（经 `_reconcile_background_tasks`）。
- **`p1_enabled=false` 只关分层/切片/情绪，社交轴仍然生效**，不会回滚到政治轴。
- **`fermentation_enabled=false` 只关发酵，种子仍会生成（经稀有化过滤），批准后立即内化（旧行为）**。
- 光谱边界：`update_spectrum_value` 为**硬 clamp**（0–100），禁止越界反弹。
- 演化群分析：`_analyze_group` 返回 `"success"|"skipped"|"failed"`；`_analyze_with_sem` 必须透传，禁止无条件 True。
- 发版：插件仓自行 `git push`；宿主侧 `plugins/*` 多在 `.gitignore`，pytest 文件在宿主仓维护。
- Manifest 版本须严格三段式 semver（`2.4.0`），**禁止 `-dev` 后缀**。

## Phase 0A/0B 正确性打磨（dev 上 v2.4.0 之后，关键）

线上曾出现：演化 skip 假成功、发酵 LLM 失败丢窗口、无证据强制内化、自评修正查不到群 session、trait 锁在来源群、注入全 spectrum_only、Soul 抢宿主 system。Phase 0 **不建 v3 表、不改宿主**，在现模型上修闭环：

| 切片 | 要点 | 关键路径 |
|------|------|----------|
| 0A 演化 | AnalysisResult 三态；光谱硬 clamp | `evolution_task.py`、`spectrum_utils.py` |
| 0A 发酵 | LLM 失败 `[]` 不推进 checked_at；max_extensions 无证据不 finalize | `fermentation_engine.py` |
| 0A 自评 | raw/normalized 列；种子用 raw；gate 接线；跨 session + `correction_consumed_at` | `self_reflection.py`、`reflection_*` |
| 0A 生命周期 | `_compute_desired_tasks` + `_reconcile_background_tasks`；API 默认关 | `plugin.py`、`plugin_ui_schema.py` |
| 0B.1 | 内化 `stream_id=global` + `origin_stream_id` | `traits.py`、`internalization_engine.py` |
| 0B.2 | 关键词补位 + `activation_reason` | `ideology_injector.py` |
| 0B.3 | 追加宿主首条 system；无 system fail-open | `ideology_injector.py` |
| 0B.4 | `self_observation_daily_cap`（默认 2） | `seeds.py`、`reflection_evaluator.py` |

| 0C | 内化 `commit=False` + BEGIN/COMMIT 包裹光谱/trait/边 | `internalization_engine.py`、`spectrum.py`、`traits.py`、`p1.py` |

| 1.mix | `soul_schema_migrations` + `user_version`；`cabinet_slot_no` UNIQUE partial index | `_conn.py`、`traits.py` |

| 1.slot | `set_trait_slot`；注入 has_slot 优先；dashboard/trait 展示 | `traits.py`、`ideology_injector.py`、dashboard_* |

**尚未做（后续）**：data_dir 迁移；完整 v3 Thought 表；宿主 H1–H3；`/soul_slot` 管理命令（可选）。

本地会话进度（不入库）：`.slim/deepwork/production-polish.md` 与评估/蓝图文档。

## 参考

- 升级计划与 P0 验收、P1 分阶段计划：`MIGRATION_PLAN.md`
- 用户向：`README.md`（dev 含完整 P1 架构/轴表/层映射/隐私/API/差异表）
- 变更：`CHANGELOG.md`（v2.4.0 含思维阁发酵重构；v2.3.0 含自我评价反馈回路；v2.1.0 含轴重构 + P1）
- SDK 指南：https://github.com/Mai-with-u/maibot-plugin-sdk/blob/main/docs/guide.md
