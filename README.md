# Mai-Soul-Engine（MaiBot 插件）

为 MaiBot 提供一个“可被群体塑造、带生活感”的动态意识形态系统：  
- **System A：思维维度光谱（Spectrum）**：四轴坐标随群聊语义变化而位移（Dream 窗口批处理）。  
- **System B：思维阁（Thought Cabinet）**：热议话题 → 思维种子 → 多轮品鉴 → 固化为 Trait → 反哺坐标与提示词。  
- **System C：黑盒模式（Black Box）**：Dream 期间的思考日志 + 潜意识片段（不发群，供后续对话风格参考）。  

并暴露 `/api/v1/soul/*` 供外部前端展示（前端需求见 `前端需求.md`）。

开发文档（架构/落地功能/实现细节）在：`docs/开发文档.md`。  
AI 交接指南（新 AI 直接接手）在：`docs/交接指南_AI.md`。

---

## 你能看到的“实际效果”（一句话版本）

1) 群里正常聊天 → **到 Dream 启动时**，插件会把“上一次 Dream 到这一次 Dream”的消息窗口交给 LLM，得到四轴位移并更新坐标。  
2) 群里出现持续热议 → 思维阁生成一个槽位并显示“内化中”，Dream 静默时刻后台多轮品鉴，成熟后“固化”为 Trait。  
3) 固化后的 Trait 会：  
   - 对光谱造成一次更明显的加/减偏移  
   - 在命中相关话题时，把“思想定义 + 内化结果”注入到回复提示词里（效果更可感知）  
4) 黑盒页面会看到 Dream 期间每轮的“思考过程/判决摘要”。  
5) 灵魂雕刻师会展示影响力 Top5 用户（影响力分数 + 权重乘子）。  

---

## 安装

把整个文件夹放到：`MaiBot/plugins/MaiBot_Soul_Engine/`，然后启动 MaiBot 即可加载。

插件目录应包含：
- `plugin.py`
- `_manifest.json`
- `config.toml`

---

## 最小配置（新手只改 3 项就能跑）

打开 `config.toml`：

### 1) 启用插件
```toml
[plugin]
enabled = true
```

### 2) 选择默认作用域（建议先 `group`）
```toml
[scope]
default_scope = "group"
```

### 3) 设置 API Token（强烈建议）
```toml
[api]
token = "改成你自己的token"
```

调用 API 时带上任意一种：
- Header `Authorization: Bearer <token>`
- Header `X-Soul-Token: <token>`

---

## Dream 完全绑定（强烈推荐的运行方式）

如果你希望“做梦 = 内化开始 / 潜意识活跃”，打开 Dream 联动：

```toml
[dream]
integrate_with_dream = true
full_bind = true
bind_cabinet_to_dream = true
bind_blackbox_to_dream = true
```

验证绑定是否成功：
- `GET /api/v1/soul/pulse?scope=global`
- `dream.hook_ok == true`
- `dream.full_bind_effective == true`

---

## 调试命令（快速验收，可选）

1) 在 `config.toml` 打开：
```toml
[debug]
enabled = true
```

2) 重启 MaiBot 后，在群里发送：
- `/soul status`：查看当前群状态
- `/soul simulate 你的一句话`：立刻跑一轮分析并更新
- `/soul dream`：模拟一次 Dream 周期（验证 Dream 链路）
- `/soul masticate`：推进一次思维阁品鉴
- `/soul fragment`：生成一条黑盒片段

---

## API（给前端展示）

前端需求：`前端需求.md`  
前端仓库（示例实现）：`https://github.com/CharTyr/mai-soul-archive`

关键约定：
- `scope` 推荐用 `local` / `global`（默认 `local`）；兼容 `group` 作为 `local` 的别名。
- 所有 GET 接口默认返回**前端友好结构与单位**（同 `前端需求.md`）；如需原始结构可加 `format=raw`。

只读接口：
- `GET /api/v1/soul/spectrum`
- `GET /api/v1/soul/cabinet`
- `GET /api/v1/soul/sculptors`
- `GET /api/v1/soul/social`
- `GET /api/v1/soul/sociolect`
- `GET /api/v1/soul/fragments`
- `GET /api/v1/soul/pulse`
- `GET /api/v1/soul/targets`

写接口（默认禁用，生产环境谨慎开）：
- `POST /api/v1/soul/reset`
- `PUT /api/v1/soul/base_tone`
- `POST /api/v1/soul/cabinet/decision`

---

## 生产环境建议（很重要）

- **务必设置 `api.token`**，并保持 `api.allow_mutations=false`（除非你明确需要远程写操作）。
- **备份人格硬盘**：`plugins/MaiBot_Soul_Engine/data/state.json`（长期运行建议定期备份）。
- **global 模式一定要上护栏**：至少配置 `cabinet.global_whitelist_targets` 或 `cabinet.global_daily_shift_cap_abs`。

---

## 常见问题（排错）

### 1) Dream 完全绑定没生效
看 `GET /api/v1/soul/pulse?scope=global`：
- `dream.hook_ok=false`：hook 失败（看 `dream.reason`）
- `dream.full_bind_requested=true` 但 `dream.full_bind_effective=false`：你想绑定，但 hook 失败；插件会回退到“非 full_bind”避免停摆

### 2) 思维阁一直不出槽位 / 槽位不推进
优先看：
- `GET /api/v1/soul/cabinet`
- `GET /api/v1/soul/fragments`（黑盒里会看到 Dream 日志，能判断是否在跑）

更深入的实现细节与调参思路请看：`docs/开发文档.md`。

---

## 全局内化安全护栏（global 模式强烈建议）

当 `scope.default_scope="global"` 时，一个群的话题可能影响全局人格，所以建议至少打开其中一些限制：

- `cabinet.global_daily_shift_cap_abs`：每天每轴最大累计偏移（防止一天被带飞）
- `cabinet.global_whitelist_targets`：只有白名单群允许影响全局人格（最安全）
- `cabinet.global_min_slot_energy`：话题热度太低不进入全局
- `cabinet.global_min_avg_confidence`：品鉴置信度太低不进入全局

---

## 可观测性：怎么知道“它在干什么/为什么没触发”

推荐用这两个接口看状态：

1) `GET /api/v1/soul/cabinet`  
每个 slot 会带 `debug`，里面会告诉你：
- 片段数/还差多少片段
- 已品鉴轮数/还差多少轮
- 平均置信度
- 需要等待静默多久/还要等多久才能下一轮品鉴
- global 模式下是否被白名单/日限额等挡住（会给出原因）

2) `GET /api/v1/soul/pulse`  
能看到 Dream hook 是否成功，以及 full_bind 是否真的生效。

---

## API 一览（用于前端展示）

前端兼容提示：
- `scope` 推荐用 `local` / `global`（默认 `local`）；同时兼容 `group` 作为 `local` 的别名。
- 所有 GET 接口默认返回与 `前端需求.md` 更贴近的结构与单位（`format=frontend`）；如需调试原始结构可加 `format=raw`。

只读接口（常用）：
- `GET /api/v1/soul/spectrum`
- `GET /api/v1/soul/cabinet`
- `GET /api/v1/soul/sculptors`
- `GET /api/v1/soul/fragments`
- `GET /api/v1/soul/pulse`
- `GET /api/v1/soul/targets`
- `GET /api/v1/soul/social`
- `GET /api/v1/soul/sociolect`

需要 `api.allow_mutations=true` 的接口（谨慎开）：
- `POST /api/v1/soul/reset`
- `PUT /api/v1/soul/base_tone`
- `POST /api/v1/soul/cabinet/decision`（手动批准/拒绝；可用于“种子批准”或“最终内化批准”）

---

## 常见问题（排错）

### 1) Dream 完全绑定没生效
看 `GET /api/v1/soul/pulse?scope=global`：
- `dream.hook_ok=false`：hook 失败（看 `dream.reason`）
- `dream.full_bind_requested=true` 但 `dream.full_bind_effective=false`：你想完全绑定，但 hook 没成功，所以为了不让系统停摆，插件会回退到“非 full_bind”的行为

### 2) 思维阁一直不出槽位 / 槽位不推进
看 `GET /api/v1/soul/cabinet` 的 `slot.debug.reasons_blocking_progress`：
常见原因：
- `not_enough_fragments`：片段不够（群里话题太分散/太短）
- `not_quiet_yet`：还没到静默时刻
- `mastication_interval_not_reached`：两轮品鉴间隔还没到
- `awaiting_manual_approval`：需要手动批准

### 3) 想要“重新反思”
把 `cabinet.rethink_enabled=true`，然后在 Dream 周期（或非 full_bind 时的后台周期）插件会对已固化思维重新生成“思想定义/内化结果”，可能会得到不一样的版本（并保留少量历史）。

---

## 隐私与安全说明（重要）

- 思维阁会保存一定量的“上下文切片 fragments”（做了脱敏/截断），请你根据自己群的隐私要求调整保留量与 API 展示尾部条数。
- 群体语言画像（sociolect）明确要求“不复读语录/不模仿个体口癖”，并对注入长度做了限制。

---

## 生产部署建议（想要“能长期跑”）

- **务必设置 `api.token`**：并保持 `api.allow_mutations=false`（除非你明确需要远程重置/批准）。
- **如果你一定要开 mutations**：现在插件要求必须设置 token，否则会直接拒绝写操作（避免误开“无鉴权写接口”）。
- **global 模式一定要加护栏**：推荐至少设置 `cabinet.global_daily_shift_cap_abs`，更安全的做法是配置 `cabinet.global_whitelist_targets`。
- **state.json 记得备份**：`plugins/MaiBot_Soul_Engine/data/state.json` 是人格演化的“硬盘”，建议放在持久化磁盘并定期备份。
