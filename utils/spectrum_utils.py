def update_spectrum_value(current: int, delta: int) -> int:
    new_value = current + delta
    if new_value > 100:
        new_value = 100 - (new_value - 100)
    elif new_value < 0:
        new_value = 0 - new_value
    return max(0, min(100, new_value))


def format_spectrum_display(spectrum: dict) -> str:
    def bar(value: int) -> str:
        pos = int(value / 10)
        return f"{'━' * pos}●{'━' * (10 - pos)}"

    lines = [
        f"经济观: 公平 {bar(spectrum.get('economic', 50))} 效率 ({spectrum.get('economic', 50)})",
        f"社会观: 自由 {bar(spectrum.get('social', 50))} 秩序 ({spectrum.get('social', 50)})",
        f"文化观: 开放 {bar(spectrum.get('diplomatic', 50))} 本土 ({spectrum.get('diplomatic', 50)})",
        f"变革观: 变化 {bar(spectrum.get('progressive', 50))} 传统 ({spectrum.get('progressive', 50)})",
    ]
    return "\n".join(lines)


def parse_user_id(config_id: str) -> tuple[str, str]:
    """解析 平台:ID 格式，返回 (platform, user_id)"""
    parts = config_id.split(":")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", config_id


def parse_chat_id(config_id: str) -> tuple[str, str, str]:
    """解析 平台:ID:private/group 格式，返回 (platform, chat_id, chat_type)"""
    parts = config_id.split(":")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return parts[0], parts[1], "group"
    return "", config_id, "group"


def chat_config_to_stream_id(config_id: str) -> str:
    """
    将配置中的聊天标识转换为 MaiBot 使用的 stream_id。

    配置支持两种形式：
    - 平台:ID:private/group（推荐，便于人工配置）
    - 直接填写 stream_id（例如 md5 值）
    """
    import hashlib

    platform, chat_id, chat_type = parse_chat_id(config_id)
    if not platform:
        return chat_id

    is_private = chat_type == "private"
    components = [platform, chat_id, "private"] if is_private else [platform, chat_id]
    key = "_".join(components)
    return hashlib.md5(key.encode()).hexdigest()


def match_user(platform: str, user_id: str, config_id: str) -> bool:
    """检查用户是否匹配配置的ID"""
    cfg_platform, cfg_user_id = parse_user_id(config_id)
    if cfg_platform and cfg_platform != platform:
        return False
    return cfg_user_id == user_id


def match_chat(platform: str, chat_id: str, chat_type: str, config_id: str) -> bool:
    """检查聊天是否匹配配置的ID"""
    cfg_platform, cfg_chat_id, cfg_chat_type = parse_chat_id(config_id)
    if cfg_platform and cfg_platform != platform:
        return False
    if cfg_chat_type and cfg_chat_type != chat_type:
        return False
    return cfg_chat_id == chat_id


def is_group_monitored(platform: str, chat_id: str, chat_type: str, config: dict) -> bool:
    monitored = config.get("monitored_groups", [])
    excluded = config.get("excluded_groups", [])

    for exc in excluded:
        if match_chat(platform, chat_id, chat_type, exc):
            return False

    if not monitored:
        return True

    for mon in monitored:
        if match_chat(platform, chat_id, chat_type, mon):
            return True

    return False


def is_user_monitored(platform: str, user_id: str, config: dict) -> bool:
    monitored = config.get("monitored_users", [])
    excluded = config.get("excluded_users", [])

    for exc in excluded:
        if match_user(platform, user_id, exc):
            return False

    if not monitored:
        return True

    for mon in monitored:
        if match_user(platform, user_id, mon):
            return True

    return False


# EMA平滑
def ema_update(current: float, new_value: float, alpha: float = 0.3) -> float:
    """指数移动平均，alpha越大新值权重越高"""
    return current * (1 - alpha) + new_value * alpha


def smooth_delta(current: int, delta: int, alpha: float = 0.3) -> int:
    """平滑后的delta值"""
    if delta == 0:
        return 0
    target = current + delta
    smoothed = ema_update(float(current), float(target), alpha)
    return int(round(smoothed)) - current


def apply_resistance(delta: int, last_dir: int, resistance: float = 0.5) -> tuple[int, int]:
    """应用反向变动阻力，返回(调整后delta, 新方向)"""
    if delta == 0:
        return 0, last_dir

    current_dir = 1 if delta > 0 else -1

    if last_dir != 0 and current_dir != last_dir:
        adjusted_delta = int(delta * (1 - resistance))
    else:
        adjusted_delta = delta

    return adjusted_delta, current_dir


# 隐私脱敏
import re


def sanitize_text(text: str, max_chars: int = 500) -> str:
    """过滤敏感信息"""
    s = (text or "").replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"https?://\S+", "<url>", s, flags=re.IGNORECASE)
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<email>", s)
    s = re.sub(r"@\S+", "@某人", s)
    s = re.sub(r"(?:\+?86[-\s]?)?1[3-9]\d{9}", "<phone>", s)
    s = re.sub(r"\b\d{17}[\dXx]\b", "<id>", s)
    s = re.sub(r"(?:QQ群|群号|群|QQ|qq)\s*[:：]?\s*([1-9]\d{4,11})", "<qq>", s)
    s = re.sub(r"\b\d{7,}\b", "<num>", s)
    return s[:max_chars]
