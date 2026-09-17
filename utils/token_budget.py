"""提示词预算：保守 token 估算 + 稳定裁剪。

**为什么要估算**：插件侧拿不到宿主模型的 tokenizer（那是宿主进程内的东西，
SDK 不暴露），但「拿不到精确分词器」不是「不做预算」的理由——超长注入会挤掉
宿主自己的关键上下文，而且是静默的。所以这里做**保守估算**：

- CJK 字符按 **1 token / 字**（真实多数在 1~1.5 字/token，这里高估 = 保守）
- 其余字符按 **3.5 字符 / token**（偏低估计 = 同样保守）

估算值一律**标明是估算**（``ESTIMATE_LABEL``），不要在任何输出里写成"精确"。

**裁剪是稳定的**：调用方给出的顺序**就是**优先级顺序，从尾部丢；
丢弃数量会报出来（不静默截断）。
"""

from __future__ import annotations

from typing import Iterable

__all__ = [
    "ESTIMATE_LABEL",
    "estimate_tokens",
    "fit_to_budget",
    "is_cjk",
]

ESTIMATE_LABEL = "估算（无宿主分词器）"

_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),   # CJK 标点
    (0x3400, 0x4DBF),   # 扩展 A
    (0x4E00, 0x9FFF),   # 基本区
    (0xF900, 0xFAFF),   # 兼容表意
    (0xFF00, 0xFFEF),   # 全角
    (0x3040, 0x30FF),   # 假名
    (0xAC00, 0xD7AF),   # 谚文
    (0x20000, 0x2FA1F),  # 扩展 B+
)


def is_cjk(ch: str) -> bool:
    """该字符是否按「1 token / 字」计价。"""
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """保守估算 token 数（向上取整，宁可高估）。

    输入为空 → 0。返回的是**估算值**，调用方展示时要带 ``ESTIMATE_LABEL``。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if is_cjk(ch))
    other = len(text) - cjk
    return cjk + int(other / 3.5 + 0.9999)


def fit_to_budget(
    lines: Iterable[str],
    budget_tokens: int,
) -> tuple[list[str], int]:
    """按**给定顺序**（即优先级，高→低）保留条目直到预算用尽。

    规则：
    - 第一条**总是保留**（哪怕它自己就超预算）：把「有内容」裁成「没内容」
      比略超预算更难排查；空注入应当来自明确的空选择，而不是预算误算。
    - 之后逐条累加，超出即停（后续全部丢弃）。
    - 顺序即优先级，因此裁剪是**确定、可复现**的——同样的输入永远留下同样的条目。

    Args:
        lines: 条目，顺序即优先级。
        budget_tokens: 预算（估算 token，<=0 视为「只保留第一条」）。

    Returns:
        ``(保留的条目, 被丢弃的条数)``。丢弃数 >0 时调用方必须记日志/上报，
        不要静默截断。
    """
    items = list(lines)
    if not items:
        return [], 0

    budget = max(0, int(budget_tokens))
    kept: list[str] = [items[0]]
    used = estimate_tokens(items[0])
    for line in items[1:]:
        cost = estimate_tokens(line)
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    return kept, len(items) - len(kept)
