# Mai-Soul-Engine 开发文档（维护者指南）

本文面向后续接手的开发者（人类/AI），目标是让你在 **不修改 MaiBot 主程序** 的前提下，能快速理解本插件的结构、数据流、API 合同与演进规则，并安全地做迭代。

---

## 0. 快速结论（你应该先记住的 5 件事）

1) **两套版本号**：插件版本在 `plugins/MaiBot_Soul_Engine/_manifest.json`，API 合同版本在 `plugins/MaiBot_Soul_Engine/webui/http_api.py` 的 `SCHEMA_VERSION`。  
2) **所有数据都在 MaiBot 的 DB**：表定义在 `plugins/MaiBot_Soul_Engine/models/ideology_model.py`，迁移只允许 `ADD COLUMN`（幂等）。  
3) **思维阁是核心体验点**：演化任务会产出 `ThoughtSeed`，管理员审核后由 `InternalizationEngine` 固化为 `CrystallizedTrait`（带 tags/evidence/confidence）。  
4) **注入策略以 tags 为主，impact 为 fallback**：注入逻辑在 `plugins/MaiBot_Soul_Engine/components/ideology_injector.py`。  
5) **对外公共展示要“前后端都开公共模式”**：前端默认 `VITE_SOUL_UI_MODE=public`，后端建议 `api.public_mode=true` 做脱敏/降能力，避免仅靠 UI 隐藏但数据仍可被直接抓取。
6) **公共展示前端二选一**：要么部署 `mai-soul-archive`，要么启用 `notion.enabled=true` 同步到 Notion 数据库（Notion 侧自建页面/视图）。

---

## 1. 目录结构（只列关键入口）

- `plugin.py`：插件注册与配置 schema，注册 HTTP Router 到 Core Server
- `components/`
  - `setup_command.py`：问卷初始化 `/soul_setup` 与答题 `/soul_answer`
  - `status_command.py`：`/soul_status`
  - `reset_command.py`：`/soul_reset`
  - `evolution_task.py`：`ON_START` 启动周期性演化 +（可选）思维阁种子产出
  - `ideology_injector.py`：`POST_LLM` 注入 prompt（光谱 + traits）
  - `notion_frontend_sync.py`：`ON_START` 启动 Notion 数据库同步（公共展示，可选）
  - `thought_commands.py`：思维阁管理命令（seeds/traits/merge/tags/enable/disable/delete）
- `models/ideology_model.py`：Peewee 模型 + `init_tables()` 迁移逻辑
- `thought/`
  - `seed_manager.py`：种子写入/读取适配（含 evidence/confidence 规范化）
  - `internalization_engine.py`：LLM 内化，生成 trait、去重合并、写 evidence/tags/confidence
- `utils/`
  - `spectrum_utils.py`：stream_id 解析、脱敏、EMA/阻力等
  - `audit_log.py`：`data/audit.jsonl` 写入（演化审计/初始化/重置）
  - `trait_tags.py`：tags 解析/规范化（JSON 存储）
  - `trait_evidence.py`：evidence 结构化 parse/dumps/append
  - `notion_frontend.py`：Notion API（urllib）+ traits 同步（数据库模式，仅公共展示）
- `webui/http_api.py`：FastAPI Router（`/api/v1/soul/*`），`SCHEMA_VERSION`、注入历史、治理建议、公共模式脱敏

---

## 2. 关键概念（术语表）

- `stream_id`：MaiBot 内部的会话标识（群/私聊/频道）。插件支持按 `stream_id` 过滤 seeds/traits/注入历史等。  
  - “全局”使用空字符串或 `"global"`（注意不同层的表示：DB 多用 `""`，API 允许 `"global"`）。
- `ThoughtSeed`（思维种子）：待审核的候选思想，来自演化阶段的 LLM 分析输出。字段含 `event/reasoning/intensity/confidence/evidence_json`。
- `CrystallizedTrait`（已固化 trait）：已内化为人格的一部分，可用于注入。字段含 `question/thought/tags_json/confidence/evidence_json/spectrum_impact_json`。
- `tags`：关键词（用于注入命中）。存储在 `tags_json`，注入时按大小写不敏感的子串匹配（`tag in message_text`）。
- `evidence`：形成证据（可累计）。为结构化 JSON（见 `utils/trait_evidence.py`），前端可展开查看。

---

## 3. 核心流程（数据从哪里来、走到哪里去）

### 3.1 初始化（问卷）

- 入口命令：`/soul_setup`、`/soul_answer`（`components/setup_command.py`）
- 结果写入：`IdeologySpectrum.initialized=True`（`models/ideology_model.py`）
- 产物：`data/audit.jsonl` 写入一条 `init` 记录（`utils/audit_log.py`）

### 3.2 周期性演化（ON_START + loop）

- 启动：`components/evolution_task.py`（`EventType.ON_START`）
- 数据源：MaiBot 历史消息（通过 `get_messages_by_time_in_chat`）
- 前处理：用户过滤 + 文本脱敏/截断（`utils/spectrum_utils.py`）
- LLM 输出：
  - 默认：仅返回四轴 `deltas`
  - 启用思维阁（`thought_cabinet.enabled=true`）后：返回 `spectrum_deltas` + `thought_seeds`
- 应用到光谱：EMA 平滑 + 方向阻力 + 边界回弹（同文件 `spectrum_utils.py`）
- 审计：写入 `EvolutionHistory` + `data/audit.jsonl`（用于 WebUI 的 Introspection/Fragments）

### 3.3 思维阁（种子 -> 审核 -> 内化 -> trait）

- 产种子：`EvolutionTaskHandler._process_thought_seeds(...)`（在 `components/evolution_task.py`）
- 管理员审核：`components/thought_commands.py`（`/soul_seeds`、`/soul_approve`、`/soul_reject`）
- 内化固化：`thought/internalization_engine.py`
  - LLM 生成：`thought`（正文）、`spectrum_impact`、`reasoning`、`confidence`、`tags`
  - 去重合并（可选）：`thought_cabinet.auto_dedup_enabled` + `auto_dedup_threshold`
  - 写入：`CrystallizedTrait`（并写入 `tags_json`、`evidence_json`、`confidence`）
- trait 管理：
  - tags：`/soul_trait_set_tags`（写入 `tags_json`）
  - 合并：`/soul_trait_merge source target`（source 软删除，evidence 合并，confidence 取 max）
  - 禁用/启用/删除：`/soul_trait_disable`、`/soul_trait_enable`、`/soul_trait_delete`

### 3.4 注入（POST_LLM）

- 入口：`components/ideology_injector.py`（`EventType.POST_LLM`）
- 组成：`ideology_prompt`（光谱提示词）+（可选）`trait_lines`（固化观点）
- trait 选择策略：
  1) **tag 命中**：按命中文本次数排序，取 `injection.max_traits`
  2) **fallback**：当无 tag 命中且开启 `injection.fallback_recent_impact=true`，按 `spectrum_impact_json` 的绝对值综合排序挑选
  3) **冷却/去重**：`injection.trait_cooldown_seconds`，避免同一 trait 连续刷屏
- 可观测：每次注入会记录到 `webui/http_api.py:record_last_injection(...)`
  - 内存态：最近一次注入 + ring buffer 历史
  - 文件：`data/injections.jsonl`（会做截断，避免无限增长）

### 3.5 Notion 前端同步（数据库模式，可选）

> 目标：把 **traits + 意识形态光谱** 写入 Notion 数据库，你可以在 Notion 里基于数据库自建公共展示页面；该前端仅用于展示，不做管理后台。

- 入口：`components/notion_frontend_sync.py`（`EventType.ON_START` 启动 loop）
- 同步实现：`utils/notion_frontend.py`（标准库 `urllib`，不引入额外依赖）
- 配置：
  - `notion.enabled=true`
  - `notion.database_id`：traits 数据库 ID（32 位）
  - `notion.sync_spectrum=true` + `notion.spectrum_database_id`：光谱数据库 ID（32 位）
  - `notion.token` 或环境变量 `MAIBOT_SOUL_NOTION_TOKEN`
  - 字段映射：
    - traits：`notion.property_*`（默认 `Name/TraitId/Tags/...`）
    - spectrum：`notion.spectrum_property_*`（默认 `Name/ScopeId/Economic/...`）
- 去重/关联：以 `TraitId` 字段为主键（`rich_text equals trait_id`），并在 `data/notion_frontend_state.json` 缓存 `trait_id -> page_id` 映射，减少 query 成本。
- 光谱 upsert：以 `ScopeId` 字段为主键（默认 `global`），维护 `Economic/Social/Diplomatic/Progressive/Initialized/LastEvolution/UpdatedAt`。
- “永不覆盖可编辑字段”：
  - `notion.never_overwrite_user_fields=true` 时，插件仅在 **新建页面** 时写入 `Name/Question/Thought/Visibility`
  - 后续更新只维护 `Tags/Confidence/ImpactScore/Status/UpdatedAt`，方便你在 Notion 里润色公开文案。
- 隐私：不会把 `evidence_json`、注入命中细节、聊天原文写入 Notion（默认按“公网展示”标准对齐）。

---

## 4. Web API（/api/v1/soul/*）

### 4.1 Router 注册

- 注册位置：`plugins/MaiBot_Soul_Engine/plugin.py:register_plugin()`
- Router 实现：`plugins/MaiBot_Soul_Engine/webui/http_api.py:create_soul_api_router()`
- 默认挂载到 Core Server：`/api/v1/soul/*`

### 4.2 鉴权

- `api.token` 或环境变量 `SOUL_API_TOKEN`
- Header：`X-Soul-Token: <token>`

### 4.3 SCHEMA_VERSION

- 位置：`plugins/MaiBot_Soul_Engine/webui/http_api.py`
- 原则：只要 API 合同（字段/端点）变化就递增
- 前端可以通过 `GET /api/v1/soul/health` 的 `schema_version` 做兼容判断

### 4.4 公共展示模式（api.public_mode）

对外公共展示时建议开启 `api.public_mode=true`，后端会对敏感信息进行脱敏/降能力：

- `/targets`：返回空（隐藏群/会话）
- `/trait_merge_suggestions`：返回空（隐藏治理）
- `/injection`、`/injections`：去除命中详情（`picked` 清空等）
- `/cabinet`：隐藏 slot 的 `introspection/debug_params`，trait 的 `evidence`
- `/introspection`：统一返回 `[REDACTED]`

> 这一步很关键：**仅靠前端隐藏仍可能被直接抓取 API 数据**。

### 4.5 与前端工程的合同

前端工程独立仓库：`https://github.com/CharTyr/mai-soul-archive`  
接口合同文档：`../dev/mai-soul-archive/docs/SOUL_API.md`（在本地工作区）

---

## 5. tags / evidence / confidence 的数据规范

### 5.1 tags（关键词）

- 存储：`CrystallizedTrait.tags_json`（JSON list string）
- 规范化：`utils/trait_tags.py`
- 注入命中：当前实现为子串匹配（`tag.casefold() in text.casefold()`）
  - 优点：简单、零依赖、成本低
  - 缺点：容易误命中（短 tag），对分词/同义词无能为力
  - 若要提升精度：建议先做 tag 规范（最小长度/停用词），再考虑分词或 embedding（注意成本与隐私）

### 5.2 evidence（形成证据）

- 存储：`evidence_json`（结构化 JSON list）
- parse/dumps：`utils/trait_evidence.py`
- 合并策略：merge 或 dedup 合并时 append/merge evidence（不覆盖）

### 5.3 confidence（置信度）

- 存储：整数 0-100（DB），API/前端通常映射为 0-1
- 来源：
  - seed 阶段：LLM/规则产出（`ThoughtSeed.confidence`）
  - 内化阶段：LLM `confidence` + seed_confidence + intensity 合并取最大
  - 去重阶段：会把 `dedup_similarity` 也计入（避免高相似合并后置信度过低）

---

## 6. 版本与迁移（强约束）

### 6.1 插件版本（SemVer）

文件：`plugins/MaiBot_Soul_Engine/_manifest.json`

- `MAJOR`：破坏性变更（配置键语义变化、API break、数据库无法自动兼容）
- `MINOR`：新增功能但保持兼容（新增可选配置/新增只读 API/新增字段）
- `PATCH`：修 bug/文档，不改变对外行为

### 6.2 数据库迁移（init_tables）

文件：`plugins/MaiBot_Soul_Engine/models/ideology_model.py`

- 只允许 **幂等** 的 `ALTER TABLE ADD COLUMN ... DEFAULT ...`
- 不做列删除（SQLite 不安全/成本高）
- 新字段必须有默认值，旧数据可直接兼容

---

## 7. 开发/调试（最小可行流程）

### 7.1 后端（插件）本地验证

建议最小检查：

```bash
cd "plugins/MaiBot_Soul_Engine"
python3 -m compileall -q .
```

运行 MaiBot（项目根目录）：

```bash
uv sync
uv run python "bot.py"
```

然后：
- 私聊管理员跑 `/soul_setup` 完成初始化
- 群里发几条消息触发演化/注入
- 用浏览器访问 `GET /api/v1/soul/health` 确认 `schema_version`
- 若启用 Notion 前端：检查 `data/notion_frontend_state.json` 是否生成，并在 Notion 数据库中看到 traits 记录

### 7.2 前端（可选）

默认是“公共展示模式”，如需内部监控/管理展示：

```bash
VITE_SOUL_UI_MODE=internal npm run dev
```

生产对外建议：
- 前端保持默认 `public`
- 后端设置 `api.public_mode=true`
- 最好同域反代 `/api`，避免 token 暴露在公网

---

## 8. 常见改动点速查（避免迷路）

- 新增配置项：`plugin.py:config_schema` + `config_template.toml` + README（必要时）
- 新增 DB 字段：`models/ideology_model.py` + `init_tables()` 迁移 +（可选）API 输出
- 新增 HTTP 端点：`webui/http_api.py` + 递增 `SCHEMA_VERSION` + 更新前端 `docs/SOUL_API.md`
- 新增管理命令：`components/thought_commands.py`（遵循现有命令 pattern）
- 改注入策略：`components/ideology_injector.py`（注意冷却与记录注入历史）

---

## 9. 安全与隐私（默认按“公网对外”标准思考）

- 强烈建议：公网展示时开启 `api.public_mode=true`（后端脱敏）
- token 不要硬编码在前端（`VITE_` 变量会被打包进浏览器）
- `data/*.jsonl` 可能包含敏感上下文（尤其是 evidence），不要提交到仓库
