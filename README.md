# MaiBot Mai-Soul-Engine

让MaiBot通过问卷初始化意识形态光谱，并根据群聊内容动态演化，影响回复风格。

## 功能特性

- **问卷初始化**：20道问题设定初始灵魂光谱
- **四维度光谱**：经济观、社会观、文化观、变革观
- **动态演化**：周期性分析群聊内容，自动调整光谱
- **提示词注入**：根据当前光谱向回复注入性格倾向
- **回弹机制**：光谱值超出边界时可反向变化
- **EMA平滑**：防止光谱剧烈波动，变化更平稳
- **隐私脱敏**：分析前自动过滤敏感信息
- **审计日志**：记录所有演化事件到日志文件
- **WebUI接口**：预留API供前端调用查看状态

## 安装

将 `Mai-Soul-Engine` 文件夹放入 `plugins/` 目录即可。

## 配置

在 `config/plugins/Mai-Soul-Engine/config.toml` 中配置：

```toml
[admin]
admin_user_id = "qq:768295235"  # 管理员用户ID（必填，格式：平台:ID）
enabled = true
initialized = false

[evolution]
evolution_enabled = true
evolution_interval_hours = 1.0  # 演化周期（小时）
evolution_rate = 5  # 每次最大变化值
ema_alpha = 0.3  # EMA平滑系数（0-1，越大变化越快）

[monitor]
monitored_groups = ["qq:123456:group", "telegram:789012:group"]  # 监控的群ID（格式：平台:ID:group）
excluded_groups = []  # 排除的群ID（格式：平台:ID:group）
monitored_users = []  # 监控的用户ID（格式：平台:ID）
excluded_users = []  # 排除的用户ID（格式：平台:ID）

[threshold]
threshold_mild = 25  # 轻微倾向阈值
threshold_moderate = 50  # 明显倾向阈值
threshold_extreme = 75  # 极端倾向阈值
custom_prompts = {}  # 自定义提示词（可选）
```

### ID 格式说明

- **用户ID**：`平台:ID`，如 `qq:768295235`
- **群ID**：`平台:ID:group`，如 `qq:123456:group`

## 使用方法

### 1. 初始化灵魂光谱

管理员私聊发送：
```
/worldview_setup
```

依次回答20道问题（1-5分），完成初始化。

### 2. 查看当前状态

任意用户发送：
```
/worldview_status
```

显示当前四维度光谱值。

### 3. 重置灵魂光谱

管理员发送：
```
/worldview_reset
```

重置为中立状态。

## 意识形态维度

| 维度 | 左端（0） | 右端（100） | 影响范围 |
|------|-----------|-------------|----------|
| **经济观** | 重视公平 | 重视效率 | 职场、劳资、内卷等话题 |
| **社会观** | 重视自由 | 重视秩序 | 群规、隐私、道德等话题 |
| **文化观** | 开放包容 | 本土优先 | 外来文化、亚文化等话题 |
| **变革观** | 拥抱变化 | 珍视传统 | 新技术、新观念、婚育等话题 |

## WebUI 接口

插件预留了以下API供WebUI调用：

```python
from plugins.Mai-Soul-Engine.webui.api import (
    get_current_spectrum,
    get_evolution_history,
    get_spectrum_chart_data,
    manual_evolution,
    set_spectrum,
)

# 获取当前光谱
spectrum = await get_current_spectrum()

# 获取演化历史
history = await get_evolution_history(limit=100)

# 获取图表数据
chart_data = await get_spectrum_chart_data(days=30)

# 手动触发演化
result = await manual_evolution(group_id="123456")

# 手动设置光谱
result = await set_spectrum(economic=60, social=40, diplomatic=70, progressive=50)
```

## 工作原理

1. **初始化**：问卷回答通过计分算法转换为四个维度的0-100数值
2. **提示词注入**：每次回复前，根据当前光谱值选择对应档位的提示词注入
3. **周期演化**：每周期分析监控群的聊天内容，用LLM评估对灵魂光谱的影响，调整光谱
4. **回弹机制**：当光谱值接近0或100时，反向变化会推动回弹

## 高级功能

### EMA平滑

使用指数移动平均防止光谱剧烈波动。通过 `ema_alpha` 配置系数：
- 值越大（接近1）：变化越快，响应越灵敏
- 值越小（接近0）：变化越慢，越稳定
- 默认值 0.3 适合大多数场景

### 隐私脱敏

分析群聊内容前自动过滤敏感信息：
- URL → `<url>`
- 邮箱 → `<email>`
- 手机号 → `<phone>`
- QQ号 → `<qq>`
- 身份证号 → `<id>`
- 长数字串 → `<num>`

### 审计日志

所有演化事件记录到 `data/audit.jsonl`：

```json
{"ts": "2026-01-16T01:45:00", "type": "evolution", "group_id": "qq:123:group", "before": {"economic": 50}, "after": {"economic": 52}, "deltas": {"economic": 2}, "message_count": 42}
{"ts": "2026-01-16T01:30:00", "type": "init", "admin_id": "qq:768295235", "spectrum": {"economic": 45, "social": 55, "diplomatic": 50, "progressive": 60}}
{"ts": "2026-01-16T02:00:00", "type": "reset", "admin_id": "qq:768295235"}
```

## 注意事项

- 必须先配置 `admin_user_id` 才能使用
- 只有管理员可以执行 `/worldview_setup` 和 `/worldview_reset`
- 演化任务需要配置 `monitored_groups` 才会生效
- 自定义提示词可覆盖默认的价值观描述

## 许可证

MIT
