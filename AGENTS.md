# Mai-Soul-Engine 插件（Agent 指引）

独立仓库：https://github.com/CharTyr/Mai-Soul-Engine（`main` = SDK 2.x；旧版 `archive/legacy-sdk1-v1`）。  
在 Maibot 宿主中路径：`plugins/CharTyr_Mai-Soul-Engine/`，**自带 `.git`**，勿把 `config.toml` / `data/` 提交进插件仓。

## 架构要点（易踩坑）

| 项 | 约定 |
|----|------|
| 运行时 | **maibot-plugin-sdk 2.x**，独立 Runner；入口 `plugin.py` + `create_plugin()` |
| 禁止 | `import src.*`、写宿主 `data/MaiBot.db`、恢复 POST_LLM 注入 |
| 注入 | 唯一接线：`@HookHandler("maisaka.planner.before_request")` → `components/ideology_injector.py` |
| 数据 | 插件自有 **`data/soul.db`**（`models/ideology_model.py`，stdlib sqlite3）；持久化目录当前绑在 **插件目录 `data/`**（`plugin._data_dir`），不是 `ctx.paths.data_dir` |
| 旧数据 | `on_load` → `migration/legacy_import.py` 只读宿主 `data/MaiBot.db` 的 `soul_*` 表，一次性导入 |
| 配置模型 | `plugin_ui_schema.py`（`MaiSoulEngineConfig`）；`plugin.py` 只引用该类 |
| Runner 必填 | `config.toml` 须有 **`[plugin]`** + **`config_version`**（当前 `2.0.0`）；`normalize_plugin_config` 会补齐旧配置 |
| WebUI 说明 | Dashboard 只显示 `json_schema_extra` 的 **`label` / `hint`**，不是 `Field(description)` |
| 功能总开关 | `plugin.enabled`（注入等）；`admin_user_id` 仅标识管理员 QQ |

## 监控配置语义

- **`monitored_groups`**：群**白名单**；空 = 不做群演化。
- **`excluded_groups`**：从白名单里再减掉（可选）。
- **`monitored_users` / `excluded_users`**：只过滤**监控群内谁的发言**计入演化，**与私聊无关**；用户列表留空 = 该群全员计入。

## 目录职责

- `components/` — 命令、演化循环、注入、Notion（可选）
- `thought/` — 思维阁种子与内化（`thought_cabinet.enabled`）
- `prompts/`、`questions/` — 问卷与 LLM 提示词
- `config_template.toml` — 脱敏模板（示例 ID 用 `12345678`）；真实配置在本地 `config.toml`

## 开发与验证

在 **Maibot 仓库根**（非仅插件目录）：

```bash
uv run pytest pytests/test_mai_soul_legacy_import.py pytests/test_mai_soul_engine_manifest.py -q
```

插件内自检（需宿主 PYTHONPATH）：

```bash
cd /path/to/Maibot
.venv/bin/python -c "import importlib; p=importlib.import_module('plugins.CharTyr_Mai-Soul-Engine.plugin'); i=p.create_plugin(); print(len(i.get_components()))"
```

重载插件后联调：`/soul_setup` → `/soul_answer` → `/soul_status`；看 Runner 日志与 `data/migration_state.json`。

## 修改约束

- **不要改 Maibot 主程序**（`src/`）除非维护者明确许可。
- 配置示例与文档中的 QQ/群号用占位符，勿提交真实 ID。
- 可选能力默认关：**Notion**、**思维阁**；**@API** 有 `api.enabled` 守卫。
- 发版：插件仓自行 `git push`；宿主侧 `plugins/*` 多在 `.gitignore`，pytest 文件在宿主仓维护。

## 参考

- 升级计划与 P0 验收：`MIGRATION_PLAN.md`
- 用户向：`README.md`
- 变更：`CHANGELOG.md`
- SDK 指南：https://github.com/Mai-with-u/maibot-plugin-sdk/blob/main/docs/guide.md