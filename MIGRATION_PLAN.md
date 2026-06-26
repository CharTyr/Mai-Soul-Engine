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

## 8. 相关文件

| 文件 | 用途 |
|------|------|
| `AGENTS.md` | Agent/维护者速查 |
| `CHANGELOG.md` | 版本变更 |
| `plugin_ui_schema.py` | 配置模型与 WebUI 文案 |
| `migration/legacy_import.py` | 宿主 DB 一次性导入 |
| `components/ideology_injector.py` | Hook 注入 |
| `components/evolution_task.py` | 群聊演化 |