# Mai-Soul-Engine 插件（Agent 指引）

独立仓库：https://github.com/CharTyr/Mai-Soul-Engine。分支：
- `main` = v2.0.0 稳定（SDK 2.x 基线）
- `dev` = v2.1.0（**P1 三观生长 + 光谱轴重构**，本文档对应此分支）
- 旧版归档：`archive/legacy-sdk1-v1`

在 Maibot 宿主中路径：`plugins/CharTyr_Mai-Soul-Engine/`，**自带 `.git`**，勿把 `config.toml` / `data/` / `config_back/` 提交进插件仓。

## 架构要点（易踩坑）

| 项 | 约定 |
|----|------|
| 运行时 | **maibot-plugin-sdk 2.x**，独立 Runner；入口 `plugin.py` + `create_plugin()` |
| 禁止 | `import src.*`、写宿主 `data/MaiBot.db`、恢复 POST_LLM 注入 |
| 注入 | 唯一接线：`@HookHandler("maisaka.planner.before_request")` → `components/ideology_injector.py` |
| 数据 | 插件自有 **`data/soul.db`**（`models/ideology_model.py`，stdlib sqlite3）；持久化目录绑在 **插件目录 `data/`**（`plugin._data_dir`），不是 `ctx.paths.data_dir` |
| 旧数据 | `on_load` → `migration/legacy_import.py` 只读宿主 `data/MaiBot.db` 的 `soul_*` 表，一次性导入；**注意旧政治轴数值无法映射到社交轴，会丢失**（详见下方"迁移注意"） |
| 配置模型 | `plugin_ui_schema.py`（`MaiSoulEngineConfig`）；`plugin.py` 只引用该类 |
| Runner 必填 | `config.toml` 须有 **`[plugin]`** + **`config_version`**；dev 版本号为 `2.1.0`；`normalize_plugin_config` 会补齐旧配置 |
| WebUI 说明 | Dashboard 只显示 `json_schema_extra` 的 **`label` / `hint`**，不是 `Field(description)` |
| 功能总开关 | `plugin.enabled`（注入等）；`admin_user_id` 仅标识管理员 QQ |
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

trait 有 `lifecycle_state`：新建 `active` → `strengthened`（merge 时强化）→ `expired`（长期未强化自动过期）。已实际使用的状态：`active` / `strengthened` / `expired`；`weakened` / `revised` / `contradicted` 仍是预留枚举（尚无写入路径）。基于 LLM 的内省矛盾检测推迟到后续迭代。

### 关键文件

| 文件 | P1 职责 |
|------|---------|
| `worldview/constants.py` | 层定义、轴→层映射、归一化 |
| `worldview/service.py` | `WorldviewService`：层上限、切片、情绪衰减、层摘要、图谱 hint、状态扩展、API payload |
| `models/ideology_model.py` | +3 表（slices/mood/edges）+2 列（`ideology_layer`/`lifecycle_state`）+ 列重命名迁移 |
| `components/evolution_task.py` | `apply_layer_caps_to_deltas` + `record_local_slice` + `nudge_mood_from_deltas` |
| `components/ideology_injector.py` | P1 块（层摘要 / 情绪行 / 图谱 hint）追加到注入 |
| `thought/internalization_engine.py` | 层推断、生命周期（merge→`strengthened`）、create 时写图谱边 |
| `components/status_command.py` | P1 扩展（切片偏移、情绪、层计数） |
| `plugin_ui_schema.py` | `WorldviewConfig` 段；`CONFIG_VERSION = "2.1.0"` |
| `plugin.py` | `soul.get_worldview` API；`soul.get_traits` 返回 layer/lifecycle |

## 思维阁（强化，dev）

种子 → 内化 → trait 全链路在 v2.1.0 做了一轮强化。关键约定（**易踩坑**）：

### 种子生命周期（不再删除）

- 种子有 `status`：`pending` → `approved` / `rejected` / `expired`。
- **审核后不删除记录**：`/soul_approve` → `update_seed_status(.., "approved")`，`/soul_reject` → `"rejected"`，保留种子→trait 审计链。**不要改回 `delete_seed`**。
- `delete_seed`（物理删除）仅用于 `_cleanup_excess_seeds`（pending 超 `max_seeds` 时挤掉最旧）和 `_cleanup_old_reviewed_seeds`（已审核超 `reviewed_keep_count` 时删最旧）。
- 种子超 `seed_ttl_hours`(默认168h) 自动标 `expired`；trait 超 `trait_ttl_days`(默认90) 且仍 `active` 自动 `expired`+`enabled=0`（`strengthened` 豁免）。两者在演化循环每轮触发。

### 种子上下文窗口

- `soul_thought_seeds.context_json`：存触发种子的**原始群聊片段**（非 LLM 二次总结）。
- `seed_manager._match_evidence_to_context`：用 difflib 把 LLM 的 evidence 模糊匹配回 `msg_lines`，取 ±2 条窗口。**`_process_thought_seeds` 必须把 `msg_lines` 透传给 `create_seed`**，否则上下文为空。
- 通知走 `_notify_admin_seed` → 从 DB 取种子（含 context）→ `format_seed_notification`，不要用原始 LLM `seed_data`（无 context）。

### 内化 prompt

- `INTERNALIZATION_PROMPT` 现包含 evidence/context/intensity/confidence/potential_impact。内化 LLM 基于真实片段形成观点，复用种子的 `potential_impact` 作参考。改 prompt 时保持这些占位符。

### 注入选择（`ideology_injector`）

- 选择顺序：tag 命中 → 无 tag 按影响分**补位**填满 `max_traits` → 仍空才 `fallback_recent_impact`。
- 二级排序用 `_trait_quality_score`（confidence + 生命周期加权：strengthened +0.3 / weakened -0.3）。
- `selection_mode`：`tag_hit` / `tag_hit+tagless` / `tagless_fill` / `fallback_recent_impact` / `spectrum_only`。
- 层摘要 `build_layer_trait_summary(.., exclude_trait_ids=selected_ids)` 排除已在详细块的 trait，避免重复。

### 种子去重

- `_is_duplicate_pending_seed`：本地 difflib 对**同群 pending** 种子按 `type+event+reasoning` 签名去重（阈值 `seed_dedup_threshold` 默认0.82，**不调 LLM**）。trait 级去重仍在内化时由 `_find_dedup_target`（调 LLM）处理。

### 配置项（`ThoughtCabinetConfig`）

新增：`seed_ttl_hours` / `reviewed_keep_count` / `trait_ttl_days` / `seed_dedup_threshold`。
**注意**：读取这些值用 `config.get(key, 默认)`，**不要用 `or 默认`**——`0`/`0.0` 是合法的"关闭"值，`or` 会把它误替换成默认值。

### 命令

只读详情：`/soul_seed <id>`（种子详情含上下文）、`/soul_trait <id>`（trait 详情含分层/生命周期/图谱边）。批量：`/soul_reject_all`（**只有批量拒绝，无批量批准**——批准会触发大量 LLM 内化）。

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

- `components/` — 命令、演化循环、注入、Notion（可选）、状态命令
- `thought/` — 思维阁种子与内化（`thought_cabinet.enabled`）；`seed_manager.py` 含上下文窗口/TTL/去重，`internalization_engine.py` 含 P1 层推断/生命周期/图谱边 + 内化 prompt 上下文
- `worldview/` — **P1 新增**：`constants.py`（层/轴映射）、`service.py`（`WorldviewService`）
- `prompts/`、`questions/` — 问卷与 LLM 提示词（v2.1.0 社交轴版本）
- `models/` — `ideology_model.py`（schema + 迁移 + 列重命名）
- `config_template.toml` — 脱敏模板（示例 ID 用 `12345678`）；真实配置在本地 `config.toml`

## 开发与验证

在 **Maibot 仓库根**（非仅插件目录），dev 分支需额外跑 P1 模型测试：

```bash
uv run pytest pytests/test_mai_soul_legacy_import.py pytests/test_mai_soul_engine_manifest.py pytests/test_mai_soul_p1_model.py -q
```

> 注：`pytests/test_mai_soul_p1_model.py` 在宿主仓维护，当前可能未提交到宿主 `seren` 分支。

插件内自检（需宿主 PYTHONPATH）：

```bash
cd /path/to/Maibot
.venv/bin/python -c "import importlib; p=importlib.import_module('plugins.CharTyr_Mai-Soul-Engine.plugin'); i=p.create_plugin(); print(len(i.get_components()))"
```

重载插件后联调：`/soul_setup` → `/soul_answer` → `/soul_status`（dev 下 status 会显示层计数、切片偏移、情绪）；看 Runner 日志与 `data/migration_state.json`。

## 修改约束

- **不要改 Maibot 主程序**（`src/`）除非维护者明确许可。
- 配置示例与文档中的 QQ/群号用占位符，勿提交真实 ID。
- 可选能力默认关：**Notion**、**思维阁**、**@API**（有 `api.enabled` 守卫）；**P1 三观生长**受 `[worldview].p1_enabled` 控制。
- **`p1_enabled=false` 只关分层/切片/情绪，社交轴仍然生效**，不会回滚到政治轴。
- 发版：插件仓自行 `git push`；宿主侧 `plugins/*` 多在 `.gitignore`，pytest 文件在宿主仓维护。
- Manifest 版本须严格三段式 semver（`2.1.0`），**禁止 `-dev` 后缀**。

## 参考

- 升级计划与 P0 验收、P1 分阶段计划：`MIGRATION_PLAN.md`
- 用户向：`README.md`（dev 含完整 P1 架构/轴表/层映射/隐私/API/差异表）
- 变更：`CHANGELOG.md`（v2.1.0 含轴重构 + P1）
- SDK 指南：https://github.com/Mai-with-u/maibot-plugin-sdk/blob/main/docs/guide.md