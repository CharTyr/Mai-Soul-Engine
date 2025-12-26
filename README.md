# Mai‑Soul‑Engine（MaiBot 插件）

Mai‑Soul‑Engine 为 MaiBot 提供一个**可被群聊塑造**的后端人格演化系统，核心由三个部分组成：

1) **思想光谱（Spectrum）**：四轴量化立场，范围 `-100 ~ +100`，并按档位影响回复风格  
2) **思维阁（Thought Cabinet）**：把热议/争议话题沉淀为“思想种子”，在后续内省中内化固化为思想  
3) **思维内省（Introspection）**：定期回想最近时间窗的群聊记录，多轮思考后输出独白、偏移量与新种子

插件会在回复前（`POST_LLM`）把“当前光谱倾向 + 相关固化思想（定义/内化结论）”注入到提示词，从而对 replyer 的表达产生可感知影响。

同时暴露 `/api/v1/soul/*` 给前端可视化使用。

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
```

4) 设置麦麦自己的 user_id（用于区分“自己发言 vs 他人发言”）
```toml
[introspection]
self_user_ids = ["123456"]
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

同时在后端 `config.toml` 中把你的前端地址加入 CORS 白名单（示例）：
```toml
[api]
cors_allow_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
```

### 方式 B：生产部署（静态站点）

1) 构建静态文件：
```bash
npm run build
```

2) 用任意静态服务器部署 `dist/`（Nginx / Caddy / CDN 均可）。

3) 后端与前端不在同域时：
   - 后端必须设置 `api.token`
   - 后端 `api.cors_allow_origins` 只放你的前端域名（不要 `*`）

---

## 调试命令（可选）

在 `config.toml` 打开：
```toml
[debug]
enabled = true
admin_only = true
admin_user_ids = ["qq:123456"] # 或只写 ["123456"]
```

群里发送：
- `/soul status`：查看 pulse 状态（含内省下一次时间）
- `/soul introspect`：强制执行一次思维内省

---

## 生产环境注意事项

- 不要提交密钥到仓库；务必设置 `api.token`。
- 插件会持久化人格状态到 `plugins/MaiBot_Soul_Engine/data/state.json`，建议放到持久化磁盘并定期备份。
- 内省日志/思想 fragments 已做脱敏与截断，但仍建议结合你的隐私要求调整 `introspection.pending_max_messages` 与 `introspection.max_log_items`。
