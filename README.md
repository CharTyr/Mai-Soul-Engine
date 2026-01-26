# Mai-Soul-Engine

可被塑造的人格演化系统 - 让MaiBot通过问卷初始化灵魂光谱，并根据群聊内容动态演化，影响回复风格。

## 快速上手（5 分钟）

> 目标：先跑起来、先开始注入与演化；思维阁与对外展示稍后再加。

1) 安装插件：把 `MaiBot_Soul_Engine` 文件夹放入 MaiBot 的 `plugins/`  
2) 生成配置：在插件目录复制 `config_template.toml` 为 `config.toml`  
   - 若你在 MaiBot 根目录操作，对应路径是 `plugins/MaiBot_Soul_Engine/config_template.toml` → `plugins/MaiBot_Soul_Engine/config.toml`  
3) 最小必填配置（`config.toml`）：

```toml
[admin]
admin_user_id = "qq:123456" # 平台:ID，如 qq:768295235

[monitor]
monitored_groups = ["qq:123456:group"] # 空=不演化、不产思维种子

[thought_cabinet]
enabled = false # 先关闭，跑通后再开
```

4) 启动 MaiBot  
5) 管理员私聊初始化光谱：

```
/soul_setup
/soul_answer <1-5>  # 按提示回答 20 题
```

完成后，插件会：
- 在每次回复前注入“光谱倾向提示词”（影响回复风格）
- 周期性分析 `monitor.monitored_groups` 的群聊内容，让光谱随群聊演化

## 最常用命令

用户可用：
- `/soul_status`：查看当前光谱

管理员可用：
- `/soul_setup`、`/soul_answer`：初始化问卷
- `/soul_reset`：重置光谱

思维阁开启后（见下文）新增：
- `/soul_seeds`、`/soul_approve <seed_id>`、`/soul_reject <seed_id>`
- `/soul_traits [stream_id]`
- `/soul_trait_set_tags <trait_id> <tag1 tag2 / tag1,tag2>`
- `/soul_trait_merge <source_trait_id> <target_trait_id>`
- `/soul_trait_disable <trait_id>`、`/soul_trait_enable <trait_id>`、`/soul_trait_delete <trait_id>`

## 配置指南（只讲你最可能会改的）

> 完整配置请看 `config_template.toml`（带中文注释）。在 MaiBot 根目录安装时路径是 `plugins/MaiBot_Soul_Engine/config_template.toml`。

### ID 格式

- 用户：`平台:ID`（如 `qq:768295235`）
- 群：`平台:群号:group`（如 `qq:123456:group`）

### 影响体验的关键项

- `monitor.monitored_groups`：不填则**不会演化**（也不会产思维种子）
- `injection.inject_private`：是否允许私聊注入（默认开启）
- `injection.max_traits`：每次注入最多携带的 trait 数量（默认 3）
- `injection.fallback_recent_impact`：无 tags 命中时是否 fallback（默认开启）
- `injection.trait_cooldown_seconds`：trait 冷却，避免刷屏

## 思维阁（主体验点，可选开启）

思维阁会在演化阶段产出“思维种子”，管理员审核后固化为 trait，用于后续注入。

开启：
```toml
[thought_cabinet]
enabled = true
```

### tags/关键词（非常重要）

trait 的注入选择优先依赖 tags：消息文本命中 trait 的 tags 才会被选入注入（最多 `injection.max_traits` 条）。

tags 来源：
- `/soul_approve` 内化时，LLM 生成初始 tags（不一定准）
- 你可以用 `/soul_trait_set_tags` 手工调整，使命中更稳、更贴题

## 对外展示（可选部署：二选一）

不部署任何展示前端也不影响插件运行（注入/演化/思维阁都照常）。  
如需对外公共展示：

- 方案 A：**Mai‑Soul‑Archive**（独立静态站点）
- 方案 B：**Notion**（同步到 Notion 数据库，用 Notion 自建页面/视图）

### 方案 A：Mai‑Soul‑Archive（独立部署）

仓库：https://github.com/CharTyr/mai-soul-archive  

生产建议：
- 前端默认公共展示模式（避免展示管理信息）
- 后端同时开启 `api.public_mode=true`（对外展示建议开启，避免敏感内容被直接请求抓取）

### 方案 B：Notion（公共展示）

插件会把 **traits（思维阁固化观点）** 与 **光谱图表数据** 写入 Notion 数据库，你在 Notion 里做筛选/排序/分组/图表即可。

#### Notion 侧准备（一次性）

1) 创建 Integration：`https://www.notion.so/my-integrations` → `New integration`  
2) 新建 **traits（思维阁 / Traits）数据库**（表）  
   - 必需字段：`Name`(Title)、`TraitId`(Rich text)、`Tags`(Multi-select)、`Thought`(Rich text)、`Visibility`(Select)  
   - 其余字段（Question/Confidence/ImpactScore/Status/UpdatedAt）可按模板增补（见 `config_template.toml`）
3) 新建 **光谱数据库**（表，推荐“4 行 + Value”结构）：  
   - 必需字段：`Dimension`(Title)、`ScopeId`(Rich text 或 Select)、`Value`(Number)
4) 将这两个数据库所在页面 **Share 给 Integration（Can edit）**  
5) 获取数据库 ID：
   - `database_id`：traits 数据库 ID（复制链接取 32 位 ID）
   - `spectrum_database_id`：光谱数据库 ID

#### 插件侧配置（推荐用环境变量放 token）

```toml
[notion]
enabled = true
database_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
spectrum_database_id = "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"
spectrum_mode = "dimension_rows"
token = "" # 推荐留空，用环境变量 MAIBOT_SOUL_NOTION_TOKEN
```

环境变量示例：`MAIBOT_SOUL_NOTION_TOKEN="secret_xxx"`

#### 同步策略（对外安全默认）

- 不写入聊天原文、evidence、注入命中细节等敏感信息
- 新建 trait 默认 `Visibility=Public`
- 创建后**永不覆盖** `Name/Question/Thought/Visibility`（你可以在 Notion 里润色公开文案）

## 其它（可选阅读）

### 示例截图

![Mai-Soul-Engine 示例](./image.png)

### 意识形态维度

| 维度 | 左端（0） | 右端（100） |
|------|-----------|-------------|
| 经济观 | 重视公平 | 重视效率 |
| 社会观 | 重视自由 | 重视秩序 |
| 文化观 | 开放包容 | 本土优先 |
| 变革观 | 拥抱变化 | 珍视传统 |

### 开发者文档

维护者/二次开发请阅读：`./DEVELOPMENT.md`。

## 许可证

GPL-3.0-or-later
