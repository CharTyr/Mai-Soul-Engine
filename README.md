# Mai-Soul-Engine

通过聊天塑造 MaiBot 三观的人格底座 — 群聊社交光谱演化 + 思维阁内化 + 回复注入。

> **本分支为 `dev`（v2.1.0）**：在 `main`/v2.0 闭环之上叠加 **P1 三观生长**，并将光谱四维从政治轴重构为**群聊社交轴**。  
> 生产稳定版请用 `main` 分支（`archive/legacy-sdk1-v1` 为旧 SDK1 归档）。  
> P1 是**向下兼容**的：关闭 `[worldview].p1_enabled` 后行为与 v2.0 一致（但四维已是社交轴，不回退到政治轴）。

## 版本说明

**v2.1.0（dev 分支）— P1 三观生长 + 光谱重构**

**光谱四维重构（最重要）**：从政治光谱（economic/social/diplomatic/progressive）换为群聊 AI 真实会经历的社交轴：

| 维度 | 低极（<50） | 高极（>50） | 三观层 |
|------|------------|------------|--------|
| **sincerity** 真诚度 | 真诚直率、反感装腔作势 | 重视场面与分寸 | 价值观（最慢） |
| **engagement** 投入度 | 克制、怕被消耗 | 热情投入、爱参与 | 世界观（中） |
| **closeness** 亲密度 | 保持距离、谨慎 | 容易亲近、对熟人放松 | 处事观（较快） |
| **directness** 直率度 | 含蓄绕弯、顾及面子 | 有话直说、不绕 | 处事观（较快） |

原因：政治轴在群聊里几乎演化不出来（没人聊再分配/文化入侵），导致闭环空转。社交轴是 MaiBot 日复一日真实经历的东西，光谱数值的变化才有意义。DB 列就地重命名迁移（幂等）。

在 v2.0「聊天 → 演化 → 注入」闭环之上，新增：

- **三观分层限速（§5.1 / §5.4）**：经济维→价值观（最慢）、社会/外交维→世界观、变革维→处事观（较快），每层有独立单次演化上限，避免短期刷屏改写根本判断。
- **轻量思想图谱（§5.2）**：trait 内化时写入 `derived_from`（源自种子）与 `supports`（去重合并）边，注入时可附带来源提示。
- **自我修正状态（§5.3）**：trait 新增 `ideology_layer` 与 `lifecycle_state`（active/strengthened/weakened/revised/expired/contradicted），观点被强化时升级为 `strengthened`，**不直接删除**。
- **群聊局部切片（§5.5）**：每个监控群记录「Mai 相对全局的局部偏移向量」，**只记偏移、不记用户画像**，防止局部氛围污染全局三观；`/soul_status` 与 `soul.get_worldview` 可见。
- **短期情绪辅助（§5.6）**：三维短期状态（愉悦/兴奋/精力），仅影响注入语气提示，按 `mood_decay_hours` 自动归零，**不写入长期三观**。
- **分层注入摘要（P1-f）**：注入时按价值观/世界观/处事观各取要点，并叠加本群偏移与情绪语气。
- **新增 API `soul.get_worldview`**：返回分层统计、情绪状态与本群切片（脱敏）。
- **思维阁强化**：种子审核通知附「原始对话上下文」（真实群聊片段，非二次总结）；审核后保留种子记录（可追溯 trait 来源）；种子/trait 自动过期清理；注入按 confidence+生命周期加权、无 tag 观点也能补位、层摘要与详细块去重；同话题种子去重。详见思维阁配置项 `seed_ttl_hours` / `reviewed_keep_count` / `trait_ttl_days` / `seed_dedup_threshold`。

**v2.0.0（main 分支）— SDK 2.x 迁移**

- 插件运行在独立 Runner 子进程中，崩溃不影响主程序
- 数据存储改为插件自有 SQLite（`data/soul.db`），不再写入宿主数据库
- 意识形态注入改用 `maisaka.planner.before_request` Hook
- HTTP API 改为 `@API` 组件，通过 Host RPC 供 WebUI / 其他插件调用
- 支持从旧版数据自动迁移

## P1 架构（开发者必读）

```
聊天消息
  ↓
演化任务 evolution_task
  ├─ WorldviewService.apply_layer_caps_to_deltas   # P1-c 分层限速
  ├─ record_local_slice                            # P1-d 群切片偏移
  └─ nudge_mood_from_deltas                        # P1-e 短期情绪
  ↓
trait 内化 internalization_engine
  ├─ normalize_ideology_layer                      # P1-a 归层
  ├─ create_crystallized_trait(... ideology_layer) # 落库
  └─ register_trait_graph                          # P1-b 图谱边
  ↓
回复注入 ideology_injector (before_request)
  ├─ build_layer_trait_summary                     # P1-f 分层摘要
  ├─ mood_prompt_lines                             # P1-e 语气
  └─ build_graph_hint                              # P1-b 关联提示
  ↓
/soul_status + soul.get_worldview                  # 可观察
```

| 模块 | 文件 | P1 职责 |
|------|------|---------|
| 常量 | `worldview/constants.py` | 三层定义、维→层映射、lifecycle 枚举、归一化 |
| 服务 | `worldview/service.py` | 限速、切片、情绪衰减、分层摘要、图谱注册 |
| 数据 | `models/ideology_model.py` | 新表 `soul_context_slices` / `soul_mood_state` / `soul_thought_edges`，trait 新增 `ideology_layer` / `lifecycle_state` |
| 配置 | `plugin_ui_schema.py` `[worldview]` | P1 总开关与各层上限、切片比例、情绪衰减 |
| 注入 | `components/ideology_injector.py` | 叠加分层摘要/情绪/图谱提示 |
| 演化 | `components/evolution_task.py` | 应用分层限速、写切片、推情绪 |
| 内化 | `thought/internalization_engine.py` | 归层、写 lifecycle、注册图谱 |

### 三观层 ↔ 光谱维映射

| 层 | 速度 | 映射的维 | 单次上限默认 |
|----|------|----------|--------------|
| 价值观 values | 最慢 | sincerity（真诚度） | 2 |
| 世界观 worldview | 中 | engagement（投入度） | 4 |
| 处事观 conduct | 较快 | closeness、directness（亲密度、直率度） | 6 |

### 隐私边界（P1-d 群切片）

- 只记录 **Mai 自己相对全局的偏移**，不存用户画像、真名、原句、QQ。
- 切片仅用于观察「某群氛围如何局部塑造 Mai」，不能反推群友个人。

## 快速上手

1. 切到 `dev` 分支后，将插件目录放入 MaiBot 的 `plugins/` 目录。
2. 复制 `config_template.toml` 为 `config.toml`，最小必填：

```toml
[admin]
admin_user_id = "qq:12345678"

[monitor]
monitored_groups = ["qq:12345678:group"]

[worldview]
p1_enabled = true   # 关闭后行为与 main/v2.0 一致
```

3. 启动 MaiBot，管理员私聊：

```
/soul_setup
/soul_answer <1-5>
/soul_status        # P1 会额外显示本群偏移、情绪、分层统计
```

## 命令

### 用户命令

- `/soul_status` — 查看当前光谱（P1：附本群偏移、情绪、分层统计）
- `/soul_dashboard` — Soul 引擎全状态可视化卡片（光谱四轴/三观分层/生命周期分布/情绪/切片/种子/演化/图谱/开关一览，渲染成图片；渲染不可用时降级纯文本）

### 管理员命令

- `/soul_setup` — 初始化光谱问卷
- `/soul_answer <1-5>` — 问卷答题
- `/soul_reset` — 重置光谱
- `/soul_seeds` — 查看待审核思维种子
- `/soul_seed <seed_id>` — 查看单个种子完整详情（事件/证据/原始对话上下文）
- `/soul_approve <seed_id>` / `/soul_reject <seed_id>` — 审核
- `/soul_reject_all` — 批量拒绝所有待审核种子
- `/soul_traits [stream_id]` — 查看 traits
- `/soul_trait <trait_id>` — 查看 trait 详情卡片（分层/生命周期/置信度/光谱影响/证据/图谱边含矛盾关系；渲染不可用降级文本）
- `/soul_inspect <文本>` — 注入命中预览卡片（模拟"这段文本会命中哪些 trait、优先级排序"，不实际注入；渲染不可用降级文本）
- `/soul_trait_set_tags` / `_merge` / `_disable` / `_enable` / `_delete` — trait 管理

## 可选功能

默认关闭或可调，不影响核心闭环：

### 状态卡片渲染（`[render]`，v2.2.0 新增）

`/soul_dashboard` 默认把 Soul 引擎全状态渲染成图片卡片。渲染经宿主无头浏览器完成，关闭或不可用时自动降级纯文本。

```toml
[render]
card_enabled = true          # 关闭后 /soul_dashboard 输出纯文本
viewport_width = 1100        # 渲染视口宽度（像素）
device_scale_factor = 2.0    # 图片清晰度倍率
render_timeout_ms = 60000    # 渲染超时（毫秒），超时降级文本
```

### P1 三观生长（`[worldview]`，dev 新增）

```toml
[worldview]
p1_enabled = true
values_max_delta = 2        # 价值观层
worldview_max_delta = 4     # 世界观层
conduct_max_delta = 6       # 处事观层
local_influence_ratio = 0.35
mood_enabled = true
mood_decay_hours = 8.0
mood_inject = true
graph_inject = true
```

影响：
- 演化速度因分层限速而整体变慢（价值观几乎不动，处事观可较快调整）。
- 注入文本会变长（分层摘要 + 情绪 + 图谱提示），LLM token 成本略增。
- `soul.db` schema 自动迁移新增三表与两个列（幂等）。

### 思维阁（`[thought_cabinet]`）、Soul @API（`[api]`）、Notion（`[notion]`）

见 v2.0 说明；P1 下内化出的 trait 会带 `ideology_layer` 与 `lifecycle_state`，`soul.get_traits` 已返回这两个字段。

## API（@API 组件）

启用 `[api]` 后：

- `soul.get_spectrum` / `soul.set_spectrum` — 光谱读写
- `soul.get_evolution_history` — 演化历史
- `soul.get_traits` — traits（**P1**：含 `ideology_layer` / `lifecycle_state`）
- `soul.get_seeds` — 待审核种子
- `soul.get_worldview` — **P1 新增**：分层统计 + 情绪 + 本群切片（脱敏）
- `soul.health` — 健康检查

## 数据存储

`data/soul.db`（SQLite，幂等迁移）：

- v2.0 表：`soul_ideology_spectrum`、`soul_group_evolution`、`soul_evolution_history`、`soul_thought_seeds`、`soul_crystallized_traits`
- **P1 新增表**：`soul_context_slices`（群切片）、`soul_mood_state`（情绪）、`soul_thought_edges`（思想图谱）
- **P1 新增列**：`soul_crystallized_traits.ideology_layer` / `lifecycle_state`

其他：`data/audit.jsonl`、`data/injections.jsonl`、`data/migration_state.json`。

## 从旧版迁移（重要）

### 从 v1.x（旧 SDK1）迁移到 dev（v2.1.0）

旧版插件用**政治轴**（economic/social/diplomatic/progressive）存光谱，dev 用**群聊社交轴**（sincerity/engagement/closeness/directness）。`legacy_import` 从宿主 `MaiBot.db` 导入时，旧列名与新列名不匹配，**旧光谱数值会丢失**。这是有意为之——政治轴数值在社交轴下语义上无意义（economic=60 不代表 sincerity=60）。

迁移后请用 `/soul_reset` → `/soul_setup` 重新做问卷初始化。思维种子和 traits 可正常导入（不涉及轴名）。

### 从 v2.0（main）切换到 dev（v2.1.0）

如果已有 v2.0 的 `soul.db`，切到 dev 后 DB 列会**就地重命名**（`ALTER TABLE RENAME COLUMN`，幂等，SQLite ≥3.25）：economic→sincerity、social→engagement 等。数值会保留但**语义已变**——原来的 economic=60 现在被读作 sincerity=60，含义完全不同。

**强烈建议**切换后执行 `/soul_reset` → `/soul_setup` 重新初始化光谱，让数值在新社交轴下有正确含义。演化历史和切片表的旧列名也会同步重命名，旧 delta 记录的语义同样变了，但不会影响后续演化。

## 验证

宿主仓根：

```bash
uv run pytest pytests/test_mai_soul_legacy_import.py pytests/test_mai_soul_engine_manifest.py pytests/test_mai_soul_p1_model.py -q
```

## 与 main/v2.0 的差异速查

| 维度 | main (v2.0) | dev (v2.1.0) |
|------|-------------|--------------|
| 光谱四维 | 政治轴（经济/社会/文化/变革） | **群聊社交轴（真诚/投入/亲近/直率）** |
| 三观分层 | 无 | 有（限速） |
| 群切片 | 无 | 有（仅偏移） |
| 情绪 | 无 | 辅助层（衰减） |
| 思想图谱 | 扁平 | 轻量边 |
| 注入摘要 | 扁平 trait | 分层 + 情绪 + 图谱 |
| schema | 5 表 | 8 表 + 2 列 |
| 关闭方式 | — | `[worldview].p1_enabled=false` 即退化为 v2.0 |

## License

GPL-3.0-or-later
