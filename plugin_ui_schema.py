"""插件配置模型（含 WebUI label / hint）。

maibot_sdk 生成的 Schema 中，Dashboard 只展示 ``hint`` 与 ``label``，
不会渲染 ``description``。请通过 ``json_schema_extra`` 填写说明。
"""

from __future__ import annotations

from typing import Any

from maibot_sdk import Field, PluginConfigBase

CONFIG_VERSION = "2.1.0"


def _ui(
    label: str,
    hint: str,
    *,
    placeholder: str = "",
    input_type: str | None = None,
    x_widget: str | None = None,
    rows: int | None = None,
    ge: float | None = None,
    le: float | None = None,
    step: float | None = None,
    advanced: bool = False,
) -> dict[str, Any]:
    extra: dict[str, Any] = {"label": label, "hint": hint}
    if placeholder:
        extra["placeholder"] = placeholder
    if input_type:
        extra["input_type"] = input_type
    if x_widget:
        extra["x-widget"] = x_widget
    if rows is not None:
        extra["rows"] = rows
    if step is not None:
        extra["step"] = step
    if advanced:
        extra["advanced"] = True
    return extra


class PluginSectionConfig(PluginConfigBase):
    """Runner 要求的 [plugin] 节：版本号与功能总开关。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(
        default=True,
        description="启用 Soul 功能",
        json_schema_extra=_ui(
            "启用 Soul 功能",
            "关闭后跳过注入等核心逻辑；与 WebUI 插件总开关配合使用。",
        ),
    )
    config_version: str = Field(
        default=CONFIG_VERSION,
        description="配置版本",
        json_schema_extra=_ui(
            "配置版本",
            "须与插件默认一致；升级插件后由 Runner 按版本合并配置，请勿随意修改。",
        ),
    )


class AdminConfig(PluginConfigBase):
    """可执行 /soul_setup、重置与思维阁审核的账号。"""

    __ui_label__ = "管理员"
    __ui_icon__ = "shield"
    __ui_order__ = 1

    admin_user_id: str = Field(
        default="",
        description="管理员用户ID",
        json_schema_extra=_ui(
            "管理员用户 ID",
            "私聊执行 /soul_setup、/soul_reset 等管理命令的账号。格式：平台:QQ号，例如 qq:12345678。",
            placeholder="qq:12345678",
        ),
    )


class EvolutionConfig(PluginConfigBase):
    """按周期分析监控群聊天记录，自动微调四维意识形态光谱。"""

    __ui_label__ = "演化"
    __ui_icon__ = "trending-up"
    __ui_order__ = 2

    evolution_enabled: bool = Field(
        default=True,
        description="启用自动演化",
        json_schema_extra=_ui("启用自动演化", "关闭后光谱不会随群聊更新，仅保留问卷初始化与手动/API 修改。"),
    )
    evolution_interval_hours: float = Field(
        default=1.0,
        ge=0.1,
        description="演化周期",
        json_schema_extra=_ui("演化周期（小时）", "两次群聊分析之间的最短间隔。", step=0.1),
    )
    evolution_rate: int = Field(
        default=5,
        ge=1,
        le=20,
        description="单次最大变化",
        json_schema_extra=_ui("单次最大变化值", "每一维光谱单次演化允许变化的上限（0–100 刻度）。"),
    )
    ema_alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="EMA 系数",
        json_schema_extra=_ui("EMA 平滑系数", "0–1，越大对新分析结果越敏感、变化越快。", step=0.05),
    )
    direction_resistance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="反向阻力",
        json_schema_extra=_ui("反向变动阻力", "0–1，越大越难朝与近期趋势相反的方向移动。", step=0.05),
    )
    max_messages_per_analysis: int = Field(
        default=200,
        ge=10,
        description="最大消息数",
        json_schema_extra=_ui("每次分析消息数上限", "单次演化从群内拉取并送入模型的消息条数上限。"),
    )
    max_chars_per_message: int = Field(
        default=200,
        ge=50,
        description="单条消息截断",
        json_schema_extra=_ui("单条消息最大字符", "过长消息截断，控制 LLM 成本与噪声。"),
    )


class MonitorConfig(PluginConfigBase):
    """决定哪些群/用户参与演化分析；列表为空则不做群演化。"""

    __ui_label__ = "监控"
    __ui_icon__ = "eye"
    __ui_order__ = 3

    monitored_groups: list[str] = Field(
        default_factory=list,
        description="监控群",
        json_schema_extra=_ui(
            "监控的群",
            "仅这些群会参与光谱演化。格式：qq:群号:group，每行一项。留空表示不分析任何群。",
            placeholder="qq:12345678:group",
        ),
    )
    excluded_groups: list[str] = Field(
        default_factory=list,
        description="排除群",
        json_schema_extra=_ui("排除的群", "从监控列表中再排除的群，格式同监控群。"),
    )
    monitored_users: list[str] = Field(
        default_factory=list,
        description="群内发言人",
        json_schema_extra=_ui(
            "群内发言人（可选）",
            "仅对「监控群」里谁的发言计入演化，与私聊无关。留空=该群所有成员都算；非空=只统计列表内用户（高级）。",
            placeholder="qq:12345678",
            advanced=True,
        ),
    )
    excluded_users: list[str] = Field(
        default_factory=list,
        description="排除发言人",
        json_schema_extra=_ui(
            "群内排除的发言人（可选）",
            "这些 QQ 在监控群里的发言不参与演化（如机器人、小号）。与私聊无关。",
            placeholder="qq:12345678",
            advanced=True,
        ),
    )


class ThresholdConfig(PluginConfigBase):
    """光谱档位与注入用提示词模板的进阶选项。"""

    __ui_label__ = "阈值"
    __ui_icon__ = "gauge"
    __ui_order__ = 4

    enable_extreme: bool = Field(
        default=False,
        description="极端档位",
        json_schema_extra=_ui("启用极端档位", "开启后 98–100 分档会使用更强烈的倾向提示词。"),
    )
    custom_prompts: dict[str, Any] = Field(
        default_factory=dict,
        description="自定义提示词",
        json_schema_extra=_ui(
            "自定义提示词模板",
            "覆盖内置档位文案的 JSON 对象；不熟悉请保持 {}。",
            x_widget="json",
        ),
    )


class InjectionConfig(PluginConfigBase):
    """在 Maisaka 规划请求前向模型注入光谱倾向与相关 trait。"""

    __ui_label__ = "注入"
    __ui_icon__ = "syringe"
    __ui_order__ = 5

    scope: str = Field(
        default="global",
        description="注入范围",
        json_schema_extra=_ui(
            "群聊注入范围",
            "global：所有群聊回复前都可能注入；monitored_only：仅监控列表中的群。",
            placeholder="global 或 monitored_only",
        ),
    )
    inject_private: bool = Field(
        default=True,
        description="私聊注入",
        json_schema_extra=_ui("私聊注入", "是否在私聊回复前也注入光谱提示。"),
    )
    max_traits: int = Field(
        default=3,
        ge=0,
        le=10,
        description="trait 数量",
        json_schema_extra=_ui("每次最多 trait 数", "单次注入附带多少个已固化观点（trait）。"),
    )
    fallback_recent_impact: bool = Field(
        default=True,
        description="fallback",
        json_schema_extra=_ui("无标签命中时 fallback", "没有 tag 命中当前话题时，是否注入最近影响最大的 traits。"),
    )
    trait_cooldown_seconds: int = Field(
        default=180,
        ge=0,
        description="冷却",
        json_schema_extra=_ui("trait 冷却（秒）", "同一 trait 重复注入的最小间隔，避免刷屏。"),
    )


class ThoughtCabinetConfig(PluginConfigBase):
    """从群聊提取思维种子，管理员审核后内化为 trait（默认关闭）。"""

    __ui_label__ = "思维阁"
    __ui_icon__ = "brain"
    __ui_order__ = 6

    enabled: bool = Field(
        default=False,
        description="启用思维阁",
        json_schema_extra=_ui("启用思维阁", "关闭时不产生新种子，不影响已有 trait 与光谱演化。"),
    )
    max_seeds: int = Field(
        default=20,
        ge=1,
        description="种子上限",
        json_schema_extra=_ui("待审核种子上限", "超过后需先批准或拒绝旧种子。"),
    )
    min_trigger_intensity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="触发强度",
        json_schema_extra=_ui("最小触发强度", "0–1，只有强度不低于此值的冲突才会记为种子。", step=0.05),
    )
    admin_notification_enabled: bool = Field(
        default=True,
        description="审核通知",
        json_schema_extra=_ui("管理员审核通知", "有新种子时是否私聊通知管理员。"),
    )
    auto_dedup_enabled: bool = Field(
        default=True,
        description="自动去重",
        json_schema_extra=_ui("自动合并相似 trait", "内化时尝试合并语义相近的 trait。"),
    )
    auto_dedup_threshold: float = Field(
        default=0.78,
        ge=0.0,
        le=1.0,
        description="去重阈值",
        json_schema_extra=_ui("自动去重相似度阈值", "0–1，越高越保守、越少自动合并。", step=0.02),
    )


class ApiConfig(PluginConfigBase):
    """供 WebUI 或其他插件通过 Host RPC 查询 Soul 数据（可选）。"""

    __ui_label__ = "API"
    __ui_icon__ = "key"
    __ui_order__ = 7

    enabled: bool = Field(
        default=True,
        description="启用 API",
        json_schema_extra=_ui("启用 Soul @API", "关闭后 soul.* 接口返回未启用；不影响命令与注入。"),
    )
    token: str = Field(
        default="",
        description="访问令牌",
        json_schema_extra=_ui(
            "访问令牌（可选）",
            "若 Host 侧校验令牌，请填写；也可设置环境变量 SOUL_API_TOKEN。",
            input_type="password",
        ),
    )
    public_mode: bool = Field(
        default=False,
        description="公共模式",
        json_schema_extra=_ui("公共展示模式", "开启后对 API 结果脱敏，适合对外只读展示。"),
    )


class NotionConfig(PluginConfigBase):
    """将 traits / 光谱同步到 Notion（可选，默认关闭）。"""

    __ui_label__ = "Notion"
    __ui_icon__ = "book-open"
    __ui_order__ = 8

    enabled: bool = Field(
        default=False,
        description="启用 Notion",
        json_schema_extra=_ui("启用 Notion 同步", "关闭时不访问 Notion API。"),
    )
    token: str = Field(
        default="",
        description="Notion Token",
        json_schema_extra=_ui(
            "Notion Integration Token",
            "也可使用环境变量 MAIBOT_SOUL_NOTION_TOKEN。",
            input_type="password",
        ),
    )
    database_id: str = Field(
        default="",
        description="traits 库 ID",
        json_schema_extra=_ui("Traits 数据库 ID", "Notion 数据库 32 位 ID（traits 表）。"),
    )
    sync_spectrum: bool = Field(
        default=True,
        description="同步光谱",
        json_schema_extra=_ui("同步光谱到 Notion", "需同时配置光谱数据库 ID。"),
    )
    spectrum_database_id: str = Field(
        default="",
        description="光谱库 ID",
        json_schema_extra=_ui("光谱数据库 ID", "留空则不同步光谱。"),
    )
    spectrum_scope_id: str = Field(
        default="global",
        description="scope_id",
        json_schema_extra=_ui("光谱 scope_id", "写入 Notion 时标识哪条光谱记录，一般用 global。"),
    )
    spectrum_mode: str = Field(
        default="dimension_rows",
        description="同步模式",
        json_schema_extra=_ui("光谱同步模式", "dimension_rows：每维一行；single_row：单行四列。"),
    )
    sync_interval_seconds: int = Field(
        default=600,
        ge=60,
        description="同步间隔",
        json_schema_extra=_ui("同步间隔（秒）", "后台定时同步周期，最小 60。"),
    )
    first_delay_seconds: int = Field(
        default=5,
        ge=0,
        description="首次延迟",
        json_schema_extra=_ui("启动后首次同步延迟（秒）", "避免与 on_load 其它任务抢资源。"),
    )
    max_traits: int = Field(
        default=200,
        ge=1,
        description="同步 trait 上限",
        json_schema_extra=_ui("单次同步 trait 上限", "一次 Notion 同步最多处理的 trait 条数。"),
    )
    visibility_default: str = Field(
        default="Public",
        description="默认可见性",
        json_schema_extra=_ui("新建 trait 默认 Visibility", "与 Notion 库中选项一致。"),
    )
    never_overwrite_user_fields: bool = Field(
        default=True,
        description="不覆盖用户字段",
        json_schema_extra=_ui("不覆盖用户在 Notion 中改过的字段", "建议保持开启。"),
    )
    max_rich_text_chars: int = Field(
        default=1800,
        ge=100,
        description="富文本长度",
        json_schema_extra=_ui("写入 Notion 的文本最大长度", "防止超长块导致 API 失败。"),
    )
    property_title: str = Field(default="Name", json_schema_extra=_ui("Traits：标题字段名", "Notion 属性名映射，与库结构一致即可。", advanced=True))
    property_trait_id: str = Field(default="TraitId", json_schema_extra=_ui("Traits：TraitId 字段名", "高级：属性名映射。", advanced=True))
    property_tags: str = Field(default="Tags", json_schema_extra=_ui("Traits：Tags 字段名", "高级：属性名映射。", advanced=True))
    property_question: str = Field(default="Question", json_schema_extra=_ui("Traits：Question 字段名", "高级：属性名映射。", advanced=True))
    property_thought: str = Field(default="Thought", json_schema_extra=_ui("Traits：Thought 字段名", "高级：属性名映射。", advanced=True))
    property_confidence: str = Field(default="Confidence", json_schema_extra=_ui("Traits：Confidence 字段名", "高级：属性名映射。", advanced=True))
    property_impact_score: str = Field(default="ImpactScore", json_schema_extra=_ui("Traits：ImpactScore 字段名", "高级：属性名映射。", advanced=True))
    property_status: str = Field(default="Status", json_schema_extra=_ui("Traits：Status 字段名", "高级：属性名映射。", advanced=True))
    property_visibility: str = Field(default="Visibility", json_schema_extra=_ui("Traits：Visibility 字段名", "高级：属性名映射。", advanced=True))
    property_updated_at: str = Field(default="UpdatedAt", json_schema_extra=_ui("Traits：UpdatedAt 字段名", "高级：属性名映射。", advanced=True))
    spectrum_property_title: str = Field(default="Name", json_schema_extra=_ui("光谱：标题字段名", "高级：属性名映射。", advanced=True))
    spectrum_property_scope_id: str = Field(default="ScopeId", json_schema_extra=_ui("光谱：ScopeId 字段名", "高级：属性名映射。", advanced=True))
    spectrum_property_economic: str = Field(default="Sincerity", json_schema_extra=_ui("光谱：Sincerity 字段名", "高级：Notion 属性名映射（真诚度）。", advanced=True))
    spectrum_property_social: str = Field(default="Engagement", json_schema_extra=_ui("光谱：Engagement 字段名", "高级：Notion 属性名映射（投入度）。", advanced=True))
    spectrum_property_diplomatic: str = Field(default="Closeness", json_schema_extra=_ui("光谱：Closeness 字段名", "高级：Notion 属性名映射（亲密度）。", advanced=True))
    spectrum_property_progressive: str = Field(default="Directness", json_schema_extra=_ui("光谱：Directness 字段名", "高级：Notion 属性名映射（直率度）。", advanced=True))
    spectrum_property_value: str = Field(default="Value", json_schema_extra=_ui("光谱：Value 字段名", "dimension_rows 模式用。", advanced=True))
    spectrum_property_initialized: str = Field(default="Initialized", json_schema_extra=_ui("光谱：Initialized 字段名", "高级：属性名映射。", advanced=True))
    spectrum_property_last_evolution: str = Field(default="LastEvolution", json_schema_extra=_ui("光谱：LastEvolution 字段名", "高级：属性名映射。", advanced=True))
    spectrum_property_updated_at: str = Field(default="UpdatedAt", json_schema_extra=_ui("光谱：UpdatedAt 字段名", "高级：属性名映射。", advanced=True))


class WorldviewConfig(PluginConfigBase):
    """P1 三观生长：分层限速、群切片、情绪辅助（dev 分支）。"""

    __ui_label__ = "三观生长 (P1)"
    __ui_icon__ = "layers"
    __ui_order__ = 25

    p1_enabled: bool = Field(
        default=True,
        description="启用 P1",
        json_schema_extra=_ui(
            "启用 P1 三观生长",
            "关闭后行为与 main/v2.0 一致：不做分层限速、切片、情绪与分层注入摘要。",
        ),
    )
    values_max_delta: int = Field(
        default=2,
        ge=0,
        le=20,
        description="价值观层单次上限",
        json_schema_extra=_ui("价值观层单次演化上限", "经济维等映射到价值观层，变化最慢。"),
    )
    worldview_max_delta: int = Field(
        default=4,
        ge=0,
        le=20,
        description="世界观层单次上限",
        json_schema_extra=_ui("世界观层单次演化上限", "社会/外交维映射到世界观层。"),
    )
    conduct_max_delta: int = Field(
        default=6,
        ge=0,
        le=20,
        description="处事观层单次上限",
        json_schema_extra=_ui("处事观层单次演化上限", "变革维等映射到处事观层，变化较快。"),
    )
    local_influence_ratio: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="群局部偏移比例",
        json_schema_extra=_ui(
            "群聊局部偏移累积比例",
            "演化 delta 的一部分记入本群切片，用于观察局部氛围，不替代全局光谱。",
            step=0.05,
        ),
    )
    mood_enabled: bool = Field(
        default=True,
        description="情绪辅助",
        json_schema_extra=_ui("短期情绪辅助", "仅影响注入中的语气提示，不写入长期三观。"),
    )
    mood_decay_hours: float = Field(
        default=8.0,
        ge=0.5,
        le=72.0,
        description="情绪衰减",
        json_schema_extra=_ui("情绪归零时间（小时）", "超过该时间未演化则短期情绪自动归零。", step=0.5),
    )
    mood_inject: bool = Field(
        default=True,
        description="注入情绪",
        json_schema_extra=_ui("在注入中加入情绪语气", "关闭则情绪仅可在 /soul_status 查看。"),
    )
    graph_inject: bool = Field(
        default=True,
        description="注入图谱提示",
        json_schema_extra=_ui("在注入中加入思想关联摘要", "展示 derived_from / supports 等轻量关系。"),
    )


class MaiSoulEngineConfig(PluginConfigBase):
    """Mai-Soul-Engine 插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    worldview: WorldviewConfig = Field(default_factory=WorldviewConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    thought_cabinet: ThoughtCabinetConfig = Field(default_factory=ThoughtCabinetConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    notion: NotionConfig = Field(default_factory=NotionConfig)