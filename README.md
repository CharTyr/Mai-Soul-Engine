# Mai‑Soul‑Engine（MaiBot 插件）

Mai‑Soul‑Engine 为 MaiBot 提供一个**可被群聊塑造**的后端人格演化系统，核心由三个部分组成：

1) **思想光谱（Spectrum）**：四轴量化立场，范围 `-100 ~ +100`，并按档位影响回复风格  
2) **思维阁（Thought Cabinet）**：把热议/争议话题沉淀为“思想种子”，在后续内省中内化固化为思想  
3) **思维内省（Introspection）**：定期回想最近时间窗的群聊记录，多轮思考后输出独白、偏移量与新种子

插件会在回复前（`POST_LLM`）把“当前光谱倾向 + 相关固化思想（定义/内化结论）”注入到提示词，从而对 replyer 的表达产生可感知影响。

同时也会在 `ON_PLAN` 阶段把“光谱倾向”注入到 planner prompt，让 planner 的动作选择/回复策略与 replyer 的立场表现形成联动。

同时暴露 `/api/v1/soul/*` 给前端可视化使用。

此外，插件会在启动时**注册主程序的记忆检索工具（Memory Retrieval tools）**，用于在需要时检索“固化思想”（不依赖主程序数据库，不受“最近 50 条”限制）。

开发文档：`docs/开发文档.md`

---

## 安装

将插件目录放入：

`MaiBot/plugins/MaiBot_Soul_Engine/`

确保目录包含：
- `plugin.py`
- `_manifest.json`
- `config.toml`

启动 MaiBot 后插件会自动加载。

---

## 快速配置（建议先改这几项）

打开 `config.toml`：

1) 启用插件
```toml
[plugin]
enabled = true
```

2) 为 API 设置 Token（生产必设）
```toml
[api]
token = "改成你自己的token"
```

3) 调整内省频率与窗口
```toml
[introspection]
interval_minutes = 20.0
window_minutes = 30.0
rounds = 4
min_messages_to_run = 30
```

4) “麦麦自己”的账号识别（自动）

插件会自动读取主程序 `config/bot_config.toml` 中的：
- `[bot].qq_account`
- `[bot].platforms = ["platform:account", ...]`

用于区分“自己发言 vs 他人发言”，无需在插件里额外配置。

可选：是否把主程序人格设定用于内省/内化：
```toml
[introspection]
use_main_personality = true
```

---

## 它是怎么运作的（面向用户）

1) 群里正常聊天 → 插件只做脱敏后缓存（不逐条调用 LLM）  
2) 满足“静默窗口 + 到达内省间隔 + 有新消息/有待内化种子”后，触发一次**思维内省**  
3) 内省会：
   - 优先内化队列里的思想种子（固化成思想，并产生一次明显光谱影响）
   - 再对时间窗聊天进行多轮回想（每轮输出独白与要点，最终输出偏移与新种子）
4) 麦麦准备回复时：将光谱档位倾向与相关固化思想注入提示词（不显式提及系统名/数值）

光谱档位（温和曲线）：
- `0~5`：几乎中立（基本不影响）
- `5~25`：第 1 档（轻微影响）
- `25~50`：第 2 档（偏向）
- `50~75`：第 3 档（明显）
- `75~100`：第 4 档（强烈）

### 光谱稳定性参数（高级调参）

在 `GET /api/v1/soul/spectrum` 的每个维度里，你会看到三类数：
- `values.current`：当前立场（-100~100）
- `values.ema`：平滑后的“长期底色”（-100~100）
- `volatility`：最近一段时间的波动率（0~1，仅用于展示）

对应 `config.toml` 里的两个关键参数：

1) `spectrum.ema_alpha`（EMA 平滑系数）  
EMA（指数移动平均）会把“每次思维内省后的 current”平滑成更稳定的 `ema`，用于展示底色、以及推导 `base_tone / dominant_trait`。  
公式直观理解：`ema = (1-α)*ema_prev + α*current`  
- `α` 越大：越“跟手”，立场/特质变化更快，但更容易抖动  
- `α` 越小：越“稳定”，变化更慢、更像长期塑造  

2) `spectrum.fluctuation_window`（波动窗口）  
插件会记录每次内省产生的有效偏移，并只保留“最近 N 次内省”的偏移作为采样窗口，用来计算 `/spectrum` 的 `volatility`。  
- `N` 越大：波动率更平滑（但更滞后）  
- `N` 越小：波动率更敏感（但更跳）  
注意：这里的 N 指“最近 N 次内省”，不是分钟/小时。

调参建议：
- 觉得界面上的 `dominant_trait/base_tone` 变化太快：把 `ema_alpha` 调小（例如 `0.10`）；变化太慢：调大（例如 `0.20`）
- 觉得 `volatility` 太跳：把 `fluctuation_window` 调大（例如 `50`）；太钝：调小（例如 `20`）

---

## API（给前端）

Base：`/api/v1/soul`

鉴权（如果设置了 `api.token`）：
- Header `Authorization: Bearer <token>`
- 或 Header `X-Soul-Token: <token>`

常用只读接口：
- `GET /api/v1/soul/spectrum`
- `GET /api/v1/soul/cabinet`
- `GET /api/v1/soul/introspection`（思维内省日志）
- `GET /api/v1/soul/pulse`
- `GET /api/v1/soul/targets`
- `GET /api/v1/soul/injection`（最近一次注入命中信息）
- `GET /api/v1/soul/health`（健康检查/统计）
- `GET /api/v1/soul/export`（脱敏导出，用于迁移/备份）

兼容接口：
- `GET /api/v1/soul/fragments`：等价于 `/introspection`（用于兼容旧前端）

---

## 可选：前端 WebUI（推荐）

本插件只提供后端能力与 API。如果你希望可视化查看“光谱 / 思维阁 / 内省日志”，可以选装配套前端：

前端仓库：`https://github.com/CharTyr/mai-soul-archive`

### 方式 A：本地开发预览（最简单）

1) 克隆并安装依赖：
```bash
git clone https://github.com/CharTyr/mai-soul-archive.git
cd mai-soul-archive
npm install
```

2) 配置前端环境变量（创建 `.env.local`）：
```bash
VITE_SOUL_API_URL=http://127.0.0.1:<MaiBot端口>/api/v1/soul
VITE_SOUL_API_TOKEN=<填你在 config.toml 里设置的 api.token>
VITE_USE_MOCK_DATA=false
```

3) 启动前端：
```bash
npm run dev
```

说明：
- 本地开发推荐使用前端的 Vite 代理（默认 `/api -> http://127.0.0.1:8000`），通常不需要处理跨域/CORS。
- 如果你选择“不走代理、直接跨域请求后端”，请在后端设置 `api.token`，插件会自动允许浏览器 Origin（无需额外 CORS 配置）。

### 方式 B：生产部署（静态站点）

1) 构建静态文件：
```bash
npm run build
```

2) 用任意静态服务器部署 `dist/`（Nginx / Caddy / CDN 均可）。

3) 后端与前端不在同域时：
   - 后端必须设置 `api.token`
   - 插件会自动处理 CORS（反射 Origin）；生产环境仍建议把前端与后端放在同域或通过反代统一域名

---

## 聊天命令（可选）

在 `config.toml` 打开：
```toml
[debug]
enabled = true
admin_only = true
# admin_user_ids 允许两种格式：
# - 仅 user_id：["123456"]
# - platform:user_id：["qq:123456"]（推荐：多平台/防冲突）
admin_user_ids = ["qq:123456"]
```

注意：如果你不在白名单中，机器人会直接忽略 `/soul ...`（不会回复）。

群里发送：
- `/soul status`：查看状态（类似 WebUI 的 `/pulse`）
- `/soul spectrum`：查看思想光谱
- `/soul cabinet`：查看思维阁（固化思想 / 待内化种子队列）
- `/soul logs 12`：查看最近 12 条内省日志
- `/soul injection`：查看最近一次注入命中信息
- `/soul targets`：列出已记录的群/会话 `sid`（便于私聊查看群状态）
- `/soul introspect`：强制执行一次思维内省（需要 `debug.allow_force_introspect=true`）

私聊里使用（避免刷屏）：
1) 先 `/soul targets` 复制要查看的 `sid`
2) 再用：`/soul spectrum sid=<sid>`（其它命令同理）

---

## 生产环境注意事项

- 不要提交密钥到仓库；务必设置 `api.token`。
- 插件会持久化人格状态到 `plugins/MaiBot_Soul_Engine/data/state.json`（原子写入），并保留 `state.json.bak` 作为上一次可用备份。
- `pending_messages` 不落盘真实用户名/账号：用户与群的标识会被哈希化；聊天内容会进行更强脱敏（链接/邮箱/手机号/身份证/长数字等）。
- 聊天窗口过大时会自动“分段总结/降级”以避免超上下文：可用 `performance.prompt_budget_chars` / `performance.summary_chunk_chars` 调整。
- 内省日志/思想 fragments 已做脱敏与截断，但仍建议结合你的隐私要求调整 `introspection.pending_max_messages` 与 `introspection.max_log_items`。
- 迁移/备份：用 `GET /api/v1/soul/export` 导出（不含 pending_messages 全文）；导入仅支持本地文件（见 `config.toml` 的 `[persistence]` 注释）。
