# 验收与上线手册

本文件是 Mai-Soul-Engine 改造后的**上线前验收清单**与**升级/回退步骤**。
配套设计文档为 `mai-soul-engine-update-plan.md`（T01–T20 契约来源）。

当前状态：**离线验收已通过，线上验收未执行**（需要宿主环境与操作者授权）。

---

## 1. 运行模式（先读这一节）

一个 `enabled` 布尔承担不了三种语义，现在拆成三态：

| 模式 | 注入回复 | 后台学习 | 改写正式人格 | 管理员接纳命令 |
|---|---|---|---|---|
| `off`（默认） | ✗ | ✗ | ✗ | ✗ |
| `observe` | ✗ | ✓ | ✗ | ✗ |
| `apply` | ✓ | ✓ | ✓ | ✓ |

配置位置：

```toml
[plugin]
mode = "off"        # off | observe | apply
enabled = true      # 旧字段，保留兼容
```

**旧配置的升级行为（重要）**：若没有显式写 `mode`，`enabled = true` 只会映射为
`observe` —— 学习继续、但**不注入、不改人格**，`/soul_health` 会提示需要显式选
`apply`。这是刻意的：一次升级不该让插件突然开始改写人格并影响真实回复。

**推荐上线顺序**：`off` → 确认 `/soul_health` 与数据目录 → `observe` 跑若干个演化
周期、人工检查候选与选择结果 → 确认无误后再切 `apply`。

---

## 2. 升级步骤

1. **备份数据**（用 SQLite 备份 API 或停写后的完整快照，不要单独复制可能带 WAL 的 `.db`）。
2. 确认权威数据目录：只认 `on_load` 日志里打印的 `soul.db` 路径，不要凭目录名猜。
   - 历史遗留：`plugins/<插件>/data/` 与 `data/plugins/<plugin-id>/` 可能**同时存在**且内容不同。
   - 迁移前先只读比对两者的表计数与初始化状态，**由操作者决定用哪一份**，不要按文件大小或修改时间自动选。
3. 停 MaiBot（**由操作者执行**，插件不自行重启宿主）。
4. 更新插件代码；启动后 `init_db` 会按版本跑迁移（v1→v4，幂等、失败不推进版本）。
5. 确认 `/soul_health`：schema 版本、数据目录来源、四个任务状态、当前运行模式。
6. 保持 `mode = "observe"` 观察一轮，再决定是否切 `apply`。

### 本次迁移新增

- v3：`soul_injection_snapshots` 增 `context_json` / `consumed_at` / `consumed_by_reply` / `delivery_state`
- v4：新增 `soul_seed_operations`（内化租约与幂等终结）

均为加列/加表，向后兼容；旧行按默认值填充。

---

## 3. 回退

- **代码回退**：`git revert` 或切回上一个 tag。
- **数据回退**：若新版本已写入 v3/v4 结构，旧代码无法读新 schema 时**不要直接降级**——
  用迁移前的备份恢复，并明确告知：迁移后产生的数据会丢失。
- **业务撤销**：不要直接删历史或反向减数值。人格变化以补偿操作处理并保留原事件。

---

## 4. 验收矩阵

### 已通过（离线，可复现）

| 契约 | 覆盖 | 证据 |
|---|---|---|
| T01 归一化契约 | 配置裸值 / 旧 `success:value` 包装 / 失败包装 / 未知类型 | `tests/test_host_config_value.py` |
| T02 items 往返 | 追加到首个 system item、保留其他 kwargs 与 schema 版本、输入不被原地修改 | `tests/test_host_prompt_items.py`、`tests/test_injection_delivery.py` |
| T03 并发不混用快照 | FIFO 认领、1:1 归属、同 reply 重试复用、跨 session 隔离 | `tests/test_reflection_snapshot_pairing.py` |
| T04 投递阶段可观测 | `selected` → `hook_applied`；`unverified` 与成功可区分；非法态拒写 | 同上 + `tests/test_reflection_capture.py` |
| T05/T06 局部与全局 | 候选校验先于写入；局部优先演化为显式开关（默认关，保持现有全局语义），开启后观点写入来源群 | `tests/test_candidate_validation.py` |
| T07/T08 自评隔离与补证 | 补证发酵「无新证据不成熟」既有实现已覆盖；自评注入隔离沿用既有设计 | `tests/test_fermentation.py` |
| T09 幂等内化 | 单赢家租约、重复批准只内化一次、终结幂等、失败可重试、租约过期可恢复 | `tests/test_seed_operation_lease.py` |
| T10 保留策略 | 只回收终态；发酵中种子与其输入不被删 | `tests/test_seed_retention.py` |
| T11 槽位恢复 | 禁用→被占→恢复不撞唯一索引，且不挤走现占用者 | `tests/test_trait_slot_recovery.py` |
| T12 非法输入不写正式状态 | 非法投递态、非法种子终态、未知 operation 均拒绝 | 上述各文件 |
| T13 模式闸门 | off/observe 不注入、不接纳、不改人格；旧配置不隐式 apply | `tests/test_runtime_mode.py`、`tests/test_mode_gate_commands.py` |
| T14 任务监督与故障恢复 | — | **未实现**（Phase E 剩余） |
| T15 命令鉴权与确认 | 真实载荷下 `/soul_reset confirm` 走执行分支；只读命令在 observe 下不受阻 | `tests/test_command_input.py`、`tests/test_mode_gate_commands.py` |
| T16 迁移与多库 | — | **未实现**（需连真实数据库，待授权） |
| T17 legacy 隔离 | — | **未实现** |
| T18 日志脱敏 | 代码路径不含凭据；未做专门审计 | **部分** |
| T19 看板区分状态 | `/soul_health` 输出模式与四闸门 | `components/health_command.py` |
| T20 离线回放 | 契约级用例齐全；**跨周期行为回放未做** | **部分** |

### 待线上验收（需授权与环境）

1. 真实宿主下 `items` 往返是否被接受（反序列化无告警）。
2. `observe` 模式跑完整一轮演化：候选是否生成、选择是否合理、日志是否可读。
3. 切 `apply` 后确认注入内容真的进入请求，且回复风格有可观察变化。
4. 同会话两轮快速提问，核对快照配对无交叉。
5. 并发批准同一颗种子，确认只产生一次光谱影响。
6. 长任务超时（宿主命令默认超时 vs 内化耗时）行为确认。

---

## 5. 已知限制与未完成项

- **Phase D 部分实现**：
  - 已做：候选优先（`thought/candidate.py`，无效候选拒绝且不写人格）、局部优先演化的
    开关（`[worldview].local_first_evolution`，**默认关**）、显式晋升（`/soul_promote_global` 已有）
  - 未做：把「局部优先」设为默认。当前默认仍是全局写入——这是刻意的，因为反转它会
    改变产品语义，需要操作者决策。
  - **群切片目前仍是「记录」而非「隔离」**，文档与看板措辞不应把它宣传成隔离机制。
- **Phase E 剩余**：任务监督器（running/waiting/backoff/failed/stopped 与故障恢复）、
  卸载清理的逐项隔离、通知 outbox。
- **Phase C 剩余**：迁移工具（只读盘点 + 显式选源 + 预演报告）——需要操作者授权后
  才能连真实数据库。
- **schema 版本守卫**：`/soul_health` 显示 schema 版本，但尚未在版本不匹配时拒绝启动。
- **`session_id` 推断会话类型**：宿主 planner hook 载荷不含会话类型字段，私聊/群聊
  仍按 `session_id` 字面量推断（已在代码标注为设计债，需宿主提供显式元数据）。
- **`final_request_verified`**：宿主未提供请求后回调，插件侧只能确认「已交回宿主」
  （`hook_applied`），无法自我声明最终请求已包含注入。

---

## 6. 本地开发

插件可在宿主之外独立开发，避免宿主 watchfiles 反复热重启：

```bash
gh repo clone CharTyr/Mai-Soul-Engine ~/projects/Mai-Soul-Engine
mkdir -p ~/projects/mai-soul-workspace/plugins
ln -sfn ~/projects/Mai-Soul-Engine ~/projects/mai-soul-workspace/plugins/CharTyr_Mai-Soul-Engine
cd ~/projects/mai-soul-workspace
uv venv .venv --python 3.13
uv pip install -e ".[test]"   # 或手动装 pytest / pytest-asyncio / pillow / maibot-plugin-sdk
.venv/bin/python -m pytest plugins/CharTyr_Mai-Soul-Engine/tests/ -q
```

`tests/conftest.py` 从 `__file__` 推导路径，宿主仓根与独立 checkout 都能跑；路径推导
失败会**显式报错**（历史上是静默 skip，会让整套测试以「全跳过」假绿通过）。
