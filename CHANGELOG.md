# Changelog

## [2.1.0] — dev 分支（P1 三观生长 + 光谱重构）

### 用户可感知

- **光谱四维重构**：从政治光谱（经济/社会/文化/变革）换为群聊社交轴——真诚度、投入度、亲密度、直率度。这些是 MaiBot 在群聊里真实会经历、会被塑造的倾向，而非宏观政治立场。
- `/soul_setup` 问卷 20 题全部重写，围绕群聊社交场景（装腔作势、冷场、熟人玩笑、有话直说等）。
- `/soul_status` 显示改为社交轴，并额外显示：本群局部偏移、短期情绪、固化观点分层统计。
- 光谱演化速度因三观分层而变化：价值观层（真诚度）最慢，处事观层（亲密度/直率度）较快，避免被短期聊天带歪。
- 注入回复时带「三观分层摘要」与「短期情绪语气」，回复风格更贴合长期人格。
- 新增 API `soul.get_worldview`：返回分层统计、情绪状态、本群切片（脱敏）。

### 思维阁强化（用户可感知）

- 种子审核通知新增「原始对话上下文」：除 LLM 摘录的证据片段外，附带触发该种子的真实群聊片段（evidence 前后各 2 条），管理员审核时能看到真实对话而非二次总结。
- 种子审核后不再删除：批准/拒绝改为标记状态，保留种子→trait 的可追溯审计链。
- 自动卫生清理：超期未审的种子自动过期；长期未被强化的固化观点自动停用，避免陈旧观点持续影响人格。
- 注入更聪明：高置信度、已强化的观点优先注入；无标签的观点也能按影响力被注入（此前几乎不可见）；分层摘要不再与详细观点重复。
- 同话题种子去重：同一群聊反复讨论同一话题不再重复产生待审种子。

### 思维阁强化（开发侧）

- 种子上下文窗口：`soul_thought_seeds` 新增 `context_json` 列（就地迁移）；`seed_manager._match_evidence_to_context` 用 difflib 把 LLM evidence 模糊匹配回原始消息行，取 ±2 条窗口（每行截断 200 字，最多 10 行）。
- 种子审计链（P0）：`update_seed_status` 替代删除；approve→`approved`、reject→`rejected`；`_cleanup_old_reviewed_seeds` 保留最近 `reviewed_keep_count`(默认200) 条已审种子。
- 种子 TTL（P0）：`expire_old_pending_seeds` 将超过 `seed_ttl_hours`(默认168h) 的 pending 种子标记 `expired`，演化循环每轮自动执行。
- trait 生命周期（P0）：`expire_old_traits` 将超过 `trait_ttl_days`(默认90) 且未被强化的 `active` trait 标记 `expired` 并 `enabled=0`；`strengthened` 不受影响。
- 内化上下文（P1）：`INTERNALIZATION_PROMPT` 新增 evidence/context/intensity/confidence/potential_impact 字段，内化 LLM 基于真实片段形成观点，复用种子的预期光谱影响。
- 注入选择（P1+P2）：无 tag trait 按影响分补位填满 `max_traits`（新增 `tag_hit+tagless`/`tagless_fill` 模式）；二级排序加质量分（confidence + 生命周期加权）；`build_layer_trait_summary` 支持 `exclude_trait_ids` 与详细注入块去重。
- 种子去重（P2）：`_is_duplicate_pending_seed` 用本地 difflib 对同群 pending 种子按 type+event+reasoning 签名去重（阈值 `seed_dedup_threshold` 默认0.82，不调 LLM）。
- 新增配置：`seed_ttl_hours` / `reviewed_keep_count` / `trait_ttl_days` / `seed_dedup_threshold`。
- 种子类型字典 `THOUGHT_TYPES` 对齐社交轴；approve/reject 路径配置改为读取而非硬编码。
- 修复 `__init__` 中 `or 默认值` 误把合法 `0.0` 替换为默认值的 bug。

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

### 重构与加固（开发侧）

- **数据层拆分**：`models/ideology_model.py`（1006 行单体）按实体拆为 `_conn.py`/`spectrum.py`/`history.py`/`seeds.py`/`traits.py`/`p1.py` 六个子模块；`ideology_model.py` 保留为重导出 shim（`import *` + `__getattr__` 委托动态变量），40+ 处历史导入零破坏。
- **`update_seed_status` 原子守卫**：新增 `expected_status="pending"` 参数，默认仅当当前状态为 `pending` 才更新，避免竞态下复活已过期/已审核种子；所有现有 2 参调用自动获得守卫。
- **`_cleanup_excess_seeds` 改标 expired**：pending 超 `max_seeds` 时不再物理删除最旧种子，改为 `update_seed_status(.., "expired")`，保留被挤掉种子的审计记录。
- **消除 `or` fallback 反模式**：全量清除配置字段侧 `int/float(... or 默认)`（会吞掉合法 `0`/`0.0`，如 `max_traits=0` 被改成 3、`trait_ttl_days=0` 被改成 90）；改为直接属性访问（pydantic 字段必然存在）。残留的 `dict.get(k,0) or 0`/`dbfield or 0` 是 None 守卫且默认值=0（0 被保留），非反模式。
- **`ThoughtSeedManager.from_plugin_config` 工厂**：收敛 4 处重复的配置 dict 构造（evolution_task + thought_commands 三处）。
- **注入热路径优化**（`before_request` 每条消息触发）：`WorldviewConfigView`/`WorldviewService` 缓存到 `plugin._wv_config_view`/`_wv_service`（`on_load` 构造、`on_config_update` 重建）；`build_layer_trait_summary` 复用已查 traits 避免二次 SQL；冷却筛选从 per-trait 改为一次批量过滤；`inject_ideology`（290 行）拆为 `_should_inject`/`_select_traits`/`_build_injection_block` 等子函数；`threading.Lock` → `asyncio.Lock`；注入日志采样（每 10 条）+ 5MB 轮转 + `asyncio.to_thread` 异步写。
- **`getattr` → 直接属性访问**：`ideology_injector`/`worldview/service` 中对 dataclass/pydantic 字段的冗余 `getattr(.., default) or default` 全部改为直接访问。
- **`except` 收窄**：`ideology_injector`/`evolution_task`/`thought_commands`/`internalization_engine`/`seed_manager` 中裸 `except Exception` 改为具体异常类型（顶层循环恢复保留并加注释）。
- **杂项**：`spectrum_utils.py` 的 `import re` 上移到文件顶部；`worldview/constants.py` 的 `LIFECYCLE_STATES` 补文档说明预留枚举（`weakened`/`revised`/`contradicted` 尚无写入路径，injector 已预置降权分，保留而非删除）。

### 内省矛盾检测与剩余加固（开发侧）

- **trait 生命周期矛盾检测**：`_find_dedup_target` 重命名为 `_classify_trait_relation`，复用同一次 LLM 调用（零额外 token）扩展输出 5 种关系（none/duplicate/contradicted/weakened/revised）。`_upsert_crystallized_trait` 按 relation 分支处理：duplicate→merge 强化（现有）；contradicted→旧 trait 标 `contradicted`+`enabled=0`+写 `contradicted_by` 边+建新 active；weakened→旧 trait 标 `weakened`+写 `weakened_by` 边+建新；revised→旧 trait 标 `revised`+写 `revised_by` 边+建新。**防误报三重**：置信度阈值（contradicted≥0.70、weakened/revised≥0.60，低于降级 none）；`strengthened` trait 豁免（prompt 注明仅可判 duplicate）；可回滚（`/soul_trait_enable` 重新启用，`/soul_trait <id>` 展示关系边可追溯）。`_trait_quality_score` 补 `contradicted: -1.0` 降权。至此 `LIFECYCLE_STATES` 6 个状态全部有写入路径。
- **全局 trait 标记显式化**：新增 `GLOBAL_STREAM = "global"` 常量（`worldview/constants.py`）。trait 表从用空串 `""` 表全局改为显式 `"global"`，消除"未设置/异常空值"与"有意的全局作用域"的歧义。`init_db` 迁移自动把存量 `""` trait 归一为 `"global"`（幂等）；`create_crystallized_trait`/`query_crystallized_traits` 传入空串自动归一；`query_active_traits_for_injection` 按 `stream_id == ? OR stream_id == GLOBAL_STREAM` 匹配。
- **图谱边批量查询**：`models/p1.py` 新增 `list_thought_edges_for_traits(trait_ids)`，一次 SQL 查所有边按 trait_id 分组；`worldview/service.py` 的 `build_graph_hint` 从 per-trait N+1 改为批量调用。保留单数版 `list_thought_edges_for_trait` 供 `/soul_trait <id>` 详情使用。
- **`set_trait_lifecycle_state` setter**：`models/traits.py` 新增，用于矛盾检测标记旧 trait 状态（可选同时改 `enabled`），不改 DB schema。
- **@API 访问控制加固**：`api_health` 补上缺失的 `api.enabled` 守卫（7 个 API 现全部一致）；唯一写接口 `api_set_spectrum` 接入 `log_api_set_spectrum` 审计（记录社交轴 before/after）；文档化安全模型（无网络暴露面，`public=False` + `api.enabled` 双层控制，插件层不自行实现网络认证）。
- **插件内测试**：新建 `tests/` 目录（28 项），覆盖种子状态原子守卫、`set_trait_lifecycle_state`、全局标记归一+迁移、批量边查询、`_trait_quality_score` 6 态降权、`_cleanup_excess_seeds` 标 expired、矛盾 trait 注入排除。从宿主根 `uv run pytest plugins/CharTyr_Mai-Soul-Engine/tests/ -q` 运行。

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
