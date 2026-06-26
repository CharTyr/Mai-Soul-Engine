# Mai-Soul-Engine

通过聊天塑造 MaiBot 三观的人格底座 — 意识形态光谱演化 + 思维阁内化 + 回复注入。

> **分支说明**
> - **`main`（v2.0.0，本分支）**：SDK 2.x 稳定版，聊天→演化→注入的完整闭环，生产可用。
> - **`dev`（v2.1.0）**：在 v2.0 之上叠加 **P1 三观生长**（三观分层限速、轻量思想图谱、群聊局部切片、短期情绪辅助、分层注入摘要、`soul.get_worldview` API）。**向下兼容**：`[worldview].p1_enabled=false` 时行为与 v2.0 完全一致。详见 `dev` 分支 README 与 `MIGRATION_PLAN.md` §8。
> - **`archive/legacy-sdk1-v1`**：旧 SDK1 归档，不再维护。
>
> 想体验「让 Mai 的三观随聊天分层生长」请切到 `dev`；要稳定运行请留在 `main`。

## 版本说明

**v2.0.0** — 适配新版 MaiBot（maibot-plugin-sdk 2.x）。不再兼容旧版 MaiBot（0.x）。

主要变化：
- 插件运行在独立 Runner 子进程中，崩溃不影响主程序
- 数据存储改为插件自有 SQLite（`data/soul.db`），不再写入宿主数据库
- 意识形态注入改用 `maisaka.planner.before_request` Hook
- HTTP API 改为 `@API` 组件，通过 Host RPC 供 WebUI / 其他插件调用
- 支持从旧版数据自动迁移

## 快速上手

1. 将本目录放入 MaiBot 的 `plugins/` 目录
2. 复制 `config_template.toml` 为 `config.toml`
3. 最小必填配置：

```toml
[admin]
admin_user_id = "qq:12345678"  # 平台:ID

[monitor]
monitored_groups = ["qq:12345678:group"]  # 监控的群
```

4. 启动 MaiBot
5. 管理员私聊初始化光谱：

```
/soul_setup
/soul_answer <1-5>  # 按提示回答 20 题
```

完成后，插件会：
- 在每次回复前注入"光谱倾向提示词"（影响回复风格）
- 周期性分析监控群的群聊内容，让光谱随群聊演化

## 命令

### 用户命令

- `/soul_status` — 查看当前光谱

### 管理员命令

- `/soul_setup` — 初始化光谱问卷
- `/soul_answer <1-5>` — 问卷答题
- `/soul_reset` — 重置光谱
- `/soul_seeds` — 查看待审核思维种子
- `/soul_approve <seed_id>` — 批准种子内化
- `/soul_reject <seed_id>` — 拒绝种子
- `/soul_traits [stream_id]` — 查看已固化 traits
- `/soul_trait_set_tags <trait_id> <tags>` — 设置 trait tags
- `/soul_trait_merge <source_id> <target_id>` — 合并 traits
- `/soul_trait_disable <trait_id>` — 禁用 trait
- `/soul_trait_enable <trait_id>` — 启用 trait
- `/soul_trait_delete <trait_id>` — 删除 trait（软删除）

## 可选功能

以下功能默认关闭，不影响核心闭环（光谱演化 + 注入 + 思维阁）：

### 思维阁（`[thought_cabinet]`）

启用后，演化分析会额外提取"思维种子"（深层价值观冲突），管理员审核后内化为固化 trait，进一步影响回复。

```toml
[thought_cabinet]
enabled = true
max_seeds = 20
min_trigger_intensity = 0.7
```

### Soul @API 组件（`[api]`）

启用后，其他插件和 WebUI 可通过 Host RPC 查询光谱、traits、种子等数据。

```toml
[api]
enabled = true
token = ""  # 可选访问令牌
```

可用 API：
- `soul.get_spectrum` — 获取当前光谱
- `soul.get_evolution_history` — 获取演化历史
- `soul.get_traits` — 获取 traits 列表
- `soul.get_seeds` — 获取待审核种子
- `soul.set_spectrum` — 手动设置光谱
- `soul.health` — 健康检查

### Notion 前端展示（`[notion]`）

启用后，将 traits 与光谱同步到 Notion 数据库，用于公共展示。

```toml
[notion]
enabled = true
token = ""  # 或环境变量 MAIBOT_SOUL_NOTION_TOKEN
database_id = "..."
```

## 从旧版迁移

如果你之前使用过旧版 Mai-Soul-Engine（v1.x，基于旧版 MaiBot），插件会在首次加载时自动迁移旧数据：

1. 从宿主 `data/MaiBot.db` 读取 `soul_*` 五张表
2. 导入到插件自有 `data/soul.db`
3. 拷贝旧 `data/audit.jsonl`、`injections.jsonl` 等文件
4. 记录迁移状态到 `data/migration_state.json`（幂等，不会重复迁移）

迁移是自动的，无需手动操作。

## 数据存储

- `data/soul.db` — SQLite 数据库（光谱、演化历史、种子、traits）
- `data/audit.jsonl` — 演化审计日志
- `data/injections.jsonl` — 注入记录
- `data/migration_state.json` — 旧版数据迁移状态
- `data/notion_frontend_state.json` — Notion 同步状态（可选）

## 配置项

完整配置见 `config_template.toml`，每个字段都有注释说明。

## dev 分支（v2.1.0，P1 三观生长）新增内容

> 本节为 `main` 用户的预告；要使用请切到 `dev` 分支。详情见 `dev` 分支的 README 与 `MIGRATION_PLAN.md` §8。

P1 的目标是让「聊天塑造三观」更可信、更克制，避免 Mai 被短期聊天带歪：

- **三观分层限速**：经济维→价值观（最慢）、社会/外交维→世界观、变革维→处事观（较快），每层独立单次演化上限。
- **轻量思想图谱**：trait 内化时记录 `derived_from` / `supports` 关系，注入可附来源提示。
- **自我修正状态**：trait 带 `lifecycle_state`（强化/弱化/修正/过期/矛盾），不直接删除。
- **群聊局部切片**：每群记录「Mai 相对全局的偏移」，不写用户画像，防局部污染全局。
- **短期情绪辅助**：三维状态（愉悦/兴奋/精力）仅影响语气，自动衰减，不写三观。
- **分层注入摘要 + `soul.get_worldview` API**。

兼容性：schema 幂等迁移（+3 表 +2 列）；`[worldview].p1_enabled=false` 即退化为 v2.0。

## License

GPL-3.0-or-later
