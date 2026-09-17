# Mai-Soul-Engine 升级与迁移计划

> **插件仓本文**：记录 SDK 2.x 落地实施与验收状态。  
> **产品/架构远景**（三观分层、图谱、P1/P2）：见 Maibot 宿主 `docs/mai-soul-engine-migration-plan.md`（讨论稿，非逐步改代码清单）。

## 1. 目标

| 层级 | 目标 |
|------|------|
| **P0（本次）** | 旧 Soul 安全迁到 rdev-Maibot + maibot-plugin-sdk 2.x：独立 Runner、自有 `soul.db`、Hook 注入、旧数据迁移、公开独立仓库 |
| **P1/P2（后续）** | 三观分层、思想图谱、时间线/周报等 — 按宿主讨论稿迭代，**不阻塞** P0 发版 |

一句话（P0）：**聊天 → 脱敏缓存 → 内省/演化 → 光谱变化 → `before_request` 注入 → 回复**，闭环在新架构下继续成立。

## 2. SDK 1.x → 2.x 对照

| 项 | 旧版（归档 `archive/legacy-sdk1-v1`） | 新版 `main`（2.0.0） |
|----|--------------------------------------|----------------------|
| 加载 | 同进程 `src.plugin_system` | Host + **Runner 子进程** |
| 依赖宿主 | `import src.*`、Peewee 写 **MaiBot.db** | 禁止 `src.*`；**sqlite3** + 插件 **`data/soul.db`** |
| 注入 | POST_LLM 等旧 Hook | **`maisaka.planner.before_request`**（`ideology_injector`） |
| 对外能力 | 偷读内部对象 | `@API` + manifest；可选 HTTP 能力 |
| 配置 | 散落 / `admin.enabled` 等 | **`plugin_ui_schema.py`** + `[plugin].config_version` |
| 仓库 | 嵌在旧 MaiBot | **https://github.com/CharTyr/Mai-Soul-Engine** 独立公开仓 |

## 3. P0 实施清单（SDK 2.0）

### 3.1 代码与结构

- [x] 插件树：`plugin.py`、`components/`、`models/`、`migration/`、`thought/`、`prompts/`、`questions/`
- [x] manifest v2，`id`: `char-tyr.mai-soul-engine`
- [x] `create_plugin()` + 生命周期：`on_load` 迁移、`on_enable` 演化任务、`on_disable` 清理
- [x] `normalize_plugin_config`：补 `[plugin]`、`config_version`、旧 `admin.enabled` → `plugin.enabled`
- [x] 功能开关：`plugin.config.plugin.enabled`（注入等）
- [x] 监控语义：`monitored_groups` 白名单；`monitored_users` **仅群内发言人**；空用户 = 群内全员

### 3.2 数据与安全

- [x] `migration/legacy_import.py`：只读宿主 `data/MaiBot.db` 的 `soul_*` → `soul.db`（幂等，`migration_state.json`）
- [x] 插件侧脱敏与缓存上限（沿用旧工程底线；**不**要求改宿主 `bot_config.toml` 才能发插件）
- [x] 持久化目录：当前实现绑 **插件目录 `data/`**（`plugin._data_dir`）

### 3.3 配置与 WebUI

- [x] `config_template.toml` 脱敏占位（如 `12345678`）
- [x] `config.toml` **gitignore**，真实 QQ/群仅本地
- [x] WebUI：`json_schema_extra` 的 **`label` / `hint`**（Dashboard 不读 `Field(description)`）
- [x] `[plugin].enabled` + `config_version = "2.0.0"`

### 3.4 仓库与文档

- [x] 独立 git；旧 `main` → `archive/legacy-sdk1-v1` + 标签 `archive-sdk1-last`
- [x] `README.md`、`CHANGELOG.md`、`_locales/zh-CN.json`、`AGENTS.md`
- [x] 标签 `v2.0.0`（`main` 上可有后续文档提交，如 `AGENTS.md`）

### 3.5 宿主侧门禁（Maibot 仓，非插件仓提交物）

- [x] `pytests/test_mai_soul_legacy_import.py`
- [x] `pytests/test_mai_soul_engine_manifest.py`

### 3.6 可选能力（默认关，不挡 P0）

- [x] Notion 同步（`notion.enabled`）
- [x] 思维阁（`thought_cabinet.enabled`）
- [x] Soul @API（`api.enabled` + token）

## 4. P0 与宿主讨论稿的差异（有意收敛）

讨论稿 P0 曾列多条 Hook（`on_plan` / `post_llm` / `after_llm` / `after_send`）。**当前 2.0 实现**为：

- **注入**：仅 `maisaka.planner.before_request`（规划前立场/光谱提示）。
- **观察**：群消息脱敏缓存 + 周期性演化任务（`evolution_task`），未复刻全部旧 Hook 名。

后续若 Host/SDK 暴露等价能力，可按讨论稿 **增量** 接回，不推翻 `soul.db` 与 Runner 架构。

## 5. 验收步骤（运维）

1. 插件放入 `plugins/CharTyr_Mai-Soul-Engine/`，本地 `config.toml` 自 `config_template.toml` 复制并填写 `admin_user_id`、`monitored_groups`。
2. 确认 `[plugin]` 与 `config_version` 存在（否则 Runner 报错）。
3. 重载/启动 Runner，检查 `data/soul.db`、`data/migration_state.json`。
4. 管理员：`/soul_setup` → `/soul_answer` → `/soul_status`。
5. 在监控群发言，等待演化周期后再次 `/soul_status` 或查日志。
6. 宿主：`uv run pytest pytests/test_mai_soul_*.py -q`

## 6. 待办（发版与运营）

- [ ] 视 `main` 最新提交打标签 **v2.0.1**（若仅文档/配置说明变更）
- [ ] GitHub Release 说明 + 插件市场 listing（按 Mai-with-u 贡献流程）
- [ ] 用户本地填真实 ID 后长期观察演化与注入效果

## 7. P1 / P2 路线图（引用）

不在本插件仓展开细节，以宿主 **`docs/mai-soul-engine-migration-plan.md`** 为准：

- **P1**：三观分层、思想图谱、内省分层与自我修正、群/用户切片、情绪辅助层。
- **P2**：三观时间线、周报、证据链、A/B 验证、群际偏移对比。
- **明确不做（当前主线）**：Dream 联动、情绪主角化、复杂关系类型、内省随便群发等（见讨论稿 §7）。

---

## 8. P1 分阶段实施（`dev` 分支）

> **分支策略**：`main` = 稳定 2.x；**`dev` = P1 开发**（从当前 `main` 拉出，勿与旧 `origin/dev` SDK1 历史混用）。  
> **运行实例**：在 `config.toml` 关功能即可，**不影响 Maibot 主程序**（独立 Runner；关注入后不影响其他聊天逻辑）。

### 8.1 本地关闭 Soul 行为（推荐）

在 `plugins/CharTyr_Mai-Soul-Engine/config.toml`：

```toml
[plugin]
enabled = false          # 关闭光谱注入（ideology_injector 直接 return）

[evolution]
evolution_enabled = false  # 关闭群聊演化后台任务
```

可选一并关闭：`[thought_cabinet] enabled = false`、`[notion] enabled = false`、`[api] enabled = false`。

重载插件或重启 Runner 后：主 Bot 照常；Soul Runner 仍可加载，但**不注入、不演化**（若 `plugin.enabled=false` 仅关注入，演化仍跑则需同时关 `evolution_enabled`）。

若希望 Runner **完全不加载**本插件：在宿主插件管理/WebUI 禁用该插件（视 MaiBot 版本而定）；仅改 `plugin.enabled` 不会卸载插件进程。

### 8.2 P1 阶段划分（建议顺序）

| 阶段 | 内容 | 主要改动 | 验收 |
|------|------|----------|------|
| **P1-a** | 数据模型：三观层 + 思想状态 | `soul.db` schema 迁移；trait/思想增加 `layer`（价值观/世界观/处事观）、`lifecycle`（强化/弱化/修正/过期/矛盾） | 迁移幂等；旧 trait 默认映射到一层 |
| **P1-b** | 轻量思想图谱 | 边表或 JSON：`supports` / `contradicts` / `derived_from` / `revises`；内化/演化写入关系 | 单条思想可查到来源边 |
| **P1-c** | 内省刹车 + 变化速度 | 演化/内化 prompt + 规则：重复性、群局部 vs 全局、层别 delta 上限（价值观最慢） | 短期刷屏不大幅改价值观 |
| **P1-d** | 群/用户切片（Mai 偏移） | `stream_id` / hashed uid 维度偏移向量，**不写用户画像**；注入时可叠加局部偏移 | 全局光谱稳定，单群偏移可观测 |
| **P1-e** | 情绪辅助层 | 短期三维状态（开心/兴奋/疲惫等），**只调语气 prompt**，不写长期三观表 | 关 `plugin.enabled` 时情绪也不注入 |
| **P1-f** | 注入与 API 暴露 | `ideology_injector` 分层摘要；`@API` 返回层别摘要（脱敏） | `/soul_status` 或 API 可见分层 |

P2（时间线、周报、A/B）在 P1-f 合并回 `main` 后再开 **`dev-p2`** 或继续在 `dev` 分批。

### 8.3 P1 开发约束

- 不 `import src.*`；schema 变更走插件内 migration 脚本 + `config_version`  bump（如 `2.1.0`）。
- 默认行为与 `main` 兼容：新字段有默认值，未开 P1 配置时表现与 2.0 一致。
- 每阶段在宿主仓补/跑 `pytests/test_mai_soul_*`（若新增迁移逻辑）。

### 8.4 P1 当前进度（dev）

- [x] P1-a 数据模型（层 / lifecycle / 新表）
- [x] P1-b 思想图谱（derived_from / supports）
- [x] P1-c 内省刹车与分层速度（演化限速）
- [x] P1-d 群/用户切片（仅群偏移向量，不写用户画像）
- [x] P1-e 情绪辅助层（三维 + 自动衰减）
- [x] P1-f 注入与 API（分层摘要 + soul.get_worldview）

> 内省「自我修正」判断逻辑（重复性/局部性）当前以**层限速 + lifecycle 状态**形式实现；  
> 完整的 LLM 侧自我修正（接受/暂存/弱化/修正旧观点）留待后续迭代，不阻塞 dev 体验。

## 9. 相关文件

| 文件 | 用途 |
|------|------|
| `AGENTS.md` | Agent/维护者速查 |
| `CHANGELOG.md` | 版本变更 |
| `plugin_ui_schema.py` | 配置模型与 WebUI 文案 |
| `migration/legacy_import.py` | 宿主 DB 一次性导入 |
| `components/ideology_injector.py` | Hook 注入 |
| `components/evolution_task.py` | 群聊演化 |