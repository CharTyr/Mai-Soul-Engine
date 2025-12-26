# 交接指南（面向 AI）— Mai‑Soul‑Engine

目标：让一个“全新接手的 AI”在**不改 MaiBot 主程序**的前提下，能快速定位代码、复现链路、做安全改动并完成验收。

---

## 0. 你接手时必须先接受的硬约束（不可违反）

1) **禁止修改 MaiBot 主程序代码**（`MaiBot/src/**`）。只能改本插件目录文件。  
2) **Dream 不能被阻塞**：任何耗时操作必须“快进快出”，失败要兜底，不得让 Dream 卡死。  
3) **隐私/脱敏**：不要在持久化/日志/API 中输出原句复刻、链接、号码、邮箱、可识别个人信息。  
4) **生产安全**：默认禁止写接口；一旦开启写接口必须要求 token；global 内化必须有护栏。  
5) **前端契约优先**：`前端需求.md` 里的字段名/单位是“对接契约”，变更必须同步前端与文档。  

---

## 1. 你要先读哪些文件（建议顺序）

1) `README.md`：面向用户的运行方式、配置入口、API 入口  
2) `前端需求.md`：API 字段/单位/枚举的硬约束  
3) `docs/开发文档.md`：模块解释（Spectrum/Cabinet/Blackbox）与调参项  
4) `plugin.py`：唯一实现文件（核心逻辑全部在这里）  

---

## 2. “成功标准”：你怎么证明自己接手成功

在一个真实运行的 MaiBot 环境里（插件已加载），你能完成以下验证：

1) Dream hook 生效：
   - `GET /api/v1/soul/pulse?scope=global`
   - 看到 `dream.hook_ok=true` 且（如果配置了）`dream.full_bind_effective=true`

2) 光谱会动（Dream window 批处理）：
   - 群里聊几分钟 → 等一次 Dream → `GET /api/v1/soul/spectrum?scope=local`
   - 四轴 `dimensions[*].values.current` 发生变化（不是永远 0）

3) 思维阁会出槽位并推进：
   - `GET /api/v1/soul/cabinet?scope=local`
   - `slots` 出现 `status=internalizing`，进度会随 Dream 周期增长，最终出现 `status=crystallized`

4) 黑盒有“思考过程”：
   - `GET /api/v1/soul/fragments?scope=local&limit=50`
   - 看到 source/tag 包含 `dream/rational/memory/subconscious` 等日志片段

5) 雕刻师有 Top5：
   - `GET /api/v1/soul/sculptors?scope=local&limit=5`
   - 返回 `users[0..4]`，并能看到 `influence` 与 `weight`（权重乘子）

---

## 3. 快速复现链路（开发者最常用）

### 3.1 只验证 API（无需前端）

建议按这个顺序：
- `GET /api/v1/soul/pulse?scope=global`
- `GET /api/v1/soul/targets`
- `GET /api/v1/soul/spectrum?scope=local`
- `GET /api/v1/soul/cabinet?scope=local`
- `GET /api/v1/soul/fragments?scope=local&limit=50`
- `GET /api/v1/soul/sculptors?scope=local&limit=5`

如果配置了 token，记得带 Header：
- `X-Soul-Token: <token>`

### 3.2 快速触发（调试命令）

在 `config.toml` 打开：
```toml
[debug]
enabled = true
```

然后群里发：
- `/soul status`：看当前群数值/traits/slots/Dream 状态
- `/soul dream`：模拟 Dream 周期（验收 Dream 链路）
- `/soul simulate ...`：立即跑一次分析并更新（验收 LLM 解析链路）

---

## 4. 代码地图（你要改什么就去哪里）

所有实现都在 `plugin.py`。下面给“按任务找入口”的地图（用函数名可直接 `rg` 搜索）：

### 4.1 Dream 绑定与周期
- Dream hook：`try_hook_dream()`
- Dream 周期入口：`_on_dream_cycle(phase="start"|"end")`
- Dream 主题/计数：`_dream_cycle_theme` / `_dream_cycle_fragments_generated`

### 4.2 光谱（Dream window 批处理）
- 收消息（dream_window）：`on_message()` → `_append_pending_message_locked()`
- 批处理 sweep：`_dream_spectrum_window_sweep()`
- 单群窗口处理：`_process_spectrum_window_for_group()`
- LLM 窗口分析 prompt：`_semantic_shift_window()`
- 应用分析结果：`_apply_window_analysis()`

### 4.3 思维阁（Slot/品鉴/固化）
- 从 topics 生成种子槽：在 `_apply_window_analysis()` 的 “topics -> slots” 段
- 推进品鉴：`_mastication_sweep_once()` / `_mastication_once()`
- evaluator prompt：`_evaluate_slot_with_llm()`（或同名 evaluator 函数，按 `rg \"evaluator\"` 查）
- 内化/固化：`_assimilate_slot_locked()` / `_finalize_slot_locked()`（按 `rg \"assimil\"` 查）
- Rethink：`_rethink_sweep_once()` / `_rethink_next_global_trait()`

### 4.4 黑盒（潜意识片段 + Dream 思考日志）
- 生成片段：`_blackbox_sweep()` / `_generate_and_store_fragment()`
- 写 Dream trace：`_append_trace_locked()`
- 前端输出合并：`_snapshot_fragments_frontend()`

### 4.5 提示词注入（让“效果可感知”）
- 注入块构建：`build_injection_block()`
- 6 档力度曲线：`_axis_tier()` / `_axis_instruction()` / `_spectrum_to_instructions()`
- 相关 trait 选择：`_select_relevant_traits()`

### 4.6 API（对接前端）
- 路由注册：`setup_router()` / `register_api_routes()`
- scope 解析：`_resolve_stream_id()`
- 前端格式快照：`_snapshot_*_frontend()`（spectrum/cabinet/sculptors/social/sociolect/fragments/pulse/targets）
- 鉴权：`_require_api_token()` / `_require_mutation_safety()`

### 4.7 持久化（人格硬盘）
- 写：`_persist_unlocked()`（输出到 `data/state.json`）
- 读：`load_persisted()`
- 关键：所有 list 都必须做 cap（避免 state.json 无限膨胀）

---

## 5. 你改动时必须维持的不变量（Checklist）

每次提交前自检：

1) **不改主程序**：diff 里不能出现 `MaiBot/src/**`  
2) **不泄露原句**：
   - fragments/quotes/pending_messages 必须走脱敏/截断（`_sanitize_*`）
3) **失败可兜底**：
   - LLM 返回非 JSON：必须能跳过/回退（不能让 Dream 报错退出）
4) **状态不膨胀**：
   - dream_trace/quote_bank/pending_messages/fragments/slots 的 cap 必须生效
5) **API 契约不破坏**：
   - 对照 `前端需求.md`，字段名/单位/枚举不随意改
6) **写接口安全**：
   - `api.allow_mutations` 默认 false
   - 开启 mutations 必须要求 token
7) **global 安全护栏**：
   - 新增任何“影响全局人格”的路径，都要走 whitelist + daily cap

---

## 6. 常见问题定位（AI 常见误踩坑）

### 6.1 Dream full_bind 没生效
现象：`full_bind_requested=true` 但 `full_bind_effective=false`  
原因：hook 失败（函数名变了/导入失败）  
定位：看 `GET /api/v1/soul/pulse?scope=global` 的 `dream.reason`

### 6.2 为什么光谱一直不动
常见原因：
- 群没被识别为 session（`targets` 里没出现）
- Dream 没跑（或 hook 失败）
- `pending_messages` 被清空但分析失败（LLM/JSON 解析失败）
定位：
- `pulse` 看 Dream
- `fragments` 看 dream_trace 是否记录 `spectrum_window`

### 6.3 为什么思维阁没出槽位
常见原因：
- LLM window 没返回 topics
- `cabinet.seed_energy_threshold` 太高
- `cabinet.max_slots` 太小被占满
定位：先看 `fragments` 里的 `seed_created` / `spectrum_window` trace

---

## 7. 建议的提交策略（让协作更稳定）

1) 任何改动先保证：
   - `python -m py_compile plugin.py` 通过
   - API 输出不崩（至少能返回空结构）
2) 变更 API 时：
   - 先改 `前端需求.md`（或至少在 PR 说明里标出差异）
   - 再改 `README.md` + `docs/开发文档.md`
3) 每个 PR 只做一件大事：例如“改阈值策略”、“改 API 字段”、“改持久化结构”

