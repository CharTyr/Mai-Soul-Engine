# Changelog

## [2.1.0] — dev 分支（P1 三观生长 + 光谱重构）

### 用户可感知

- **光谱四维重构**：从政治光谱（经济/社会/文化/变革）换为群聊社交轴——真诚度、投入度、亲密度、直率度。这些是 MaiBot 在群聊里真实会经历、会被塑造的倾向，而非宏观政治立场。
- `/soul_setup` 问卷 20 题全部重写，围绕群聊社交场景（装腔作势、冷场、熟人玩笑、有话直说等）。
- `/soul_status` 显示改为社交轴，并额外显示：本群局部偏移、短期情绪、固化观点分层统计。
- 光谱演化速度因三观分层而变化：价值观层（真诚度）最慢，处事观层（亲密度/直率度）较快，避免被短期聊天带歪。
- 注入回复时带「三观分层摘要」与「短期情绪语气」，回复风格更贴合长期人格。
- 新增 API `soul.get_worldview`：返回分层统计、情绪状态、本群切片（脱敏）。

### 开发侧

- **光谱重构**：`IdeologySpectrum` 字段 economic/social/diplomatic/progressive → sincerity/engagement/closeness/directness；DB 列就地 `RENAME COLUMN` 迁移（幂等，SQLite ≥3.25）；演化历史与切片表同步重命名。
- 提示词：`prompts/ideology_prompts.py` 四组档位文案全部重写为社交轴；`EVOLUTION_ANALYSIS_PROMPT` 与 `thought_prompts.py` 同步。
- 问卷：`questions/setup_questions.py` 20 题重写为群聊社交场景。
- 数据层：新增表 `soul_context_slices` / `soul_mood_state` / `soul_thought_edges`；`soul_crystallized_traits` 新增 `ideology_layer` / `lifecycle_state` 列（就地迁移，幂等）。
- 新模块 `worldview/`：`constants.py`（层/lifecycle 枚举与归一化、维→层映射）、`service.py`（`WorldviewService`：分层限速、群切片、情绪衰减、分层注入摘要、思想图谱注册）。
- 配置：`plugin_ui_schema.py` 新增 `[worldview]` 节；`config_version` 升至 `2.1.0`，`normalize_plugin_config` 自动补齐。
- 演化：`evolution_task` 接入 `apply_layer_caps_to_deltas` / `record_local_slice` / `nudge_mood_from_deltas`。
- 内化：`internalization_engine` 归层、写 lifecycle（合并升级为 `strengthened`）、注册 `derived_from` / `supports` 图谱边。
- 注入：`ideology_injector` 叠加分层摘要、情绪语气、图谱来源提示。
- API：`soul.get_traits` 返回 `ideology_layer` / `lifecycle_state`；新增 `soul.get_worldview`。
- Notion：光谱属性默认名改为 Sincerity/Engagement/Closeness/Directness。
- 兼容：`[worldview].p1_enabled = false` 时分层/切片/情绪关闭，行为与 v2.0 一致（四维已是社交轴）。
- 测试：宿主仓新增 `pytests/test_mai_soul_p1_model.py`（4 项），manifest 门禁改为跟随分支版本，legacy_import 测试同步新列名。

## [2.0.0] — 2026-06-26

### 用户可感知

- 适配新版 MaiBot（maibot-plugin-sdk 2.x），插件在独立 Runner 子进程中运行。
- 意识形态光谱：问卷初始化、群聊自动演化、回复前注入（`maisaka.planner.before_request`）。
- 思维阁（可选）：种子审核、内化、trait 管理命令。
- Soul @API（可选）：光谱 / 历史 / traits / 种子查询与健康检查。
- Notion 前端同步（可选，默认关闭）。
- 从旧版宿主 `data/MaiBot.db` 自动迁移 `soul_*` 数据到插件 `data/soul.db`（幂等）。

### 开发侧

- manifest v2，`id`: `char-tyr.mai-soul-engine`。
- 数据层：peewee → stdlib `sqlite3`，插件自有 `soul.db`。
- 旧版 `src.plugin_system` / POST_LLM 注入已移除；仅 SDK 2.x。
- 旧版代码归档分支：`archive/legacy-sdk1-v1`，标签 `archive-sdk1-last`。
