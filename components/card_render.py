"""共享卡片渲染内核：html2png 多 attempt + crop + 高度校验。

三个 P0 卡片插件（Soul / Plugin-Ops / chat_summary）各内置一份同源副本。
MaiBot 插件互相独立、不能跨插件 import，故采用「同源复制」而非共享包。
任何修改需同步三处：
  - plugins/CharTyr_Mai-Soul-Engine/components/card_render.py
  - plugins/CharTyr_Mai-Plugin-Ops/plugin_ops/card_render.py
  - plugins/chat_summary/card_render.py

稳定性要点（2026-07-18 重构）：
- 统一参数名 ``render_timeout_ms``（宿主兼容旧名 ``timeout_ms``，但新代码不再使用）
- 多 attempt：body 视口 → #root 元素 → body full_page，指数退避
- PIL 内容裁剪：去掉 body 截图的大片空白边
- 高度合法性校验：异常尺寸（空白高图/截断矮图）拒绝并进入下一 attempt
- 可选 CSS 泄漏顶栏检测（Soul 启用）：live Chromium 偶发把 <style> 渲成可见文本
"""

from __future__ import annotations

import asyncio
import base64
import logging
from io import BytesIO
from typing import Any, Callable, Optional

_logger = logging.getLogger(__name__)


def png_size(blob: bytes) -> tuple[int | None, int | None]:
    """从 PNG 字节流读 (width, height)；非 PNG 返回 (None, None)。"""
    if blob[:8] == b"\x89PNG\r\n\x1a\n" and len(blob) >= 24:
        return int.from_bytes(blob[16:20], "big"), int.from_bytes(blob[20:24], "big")
    return None, None


def crop_content_png(blob: bytes, *, pad: int = 12, min_size: int = 120) -> bytes:
    """裁掉四周近白边，保留内容区 + pad。失败/无内容/太小则返回原图。"""
    try:
        from PIL import Image

        with Image.open(BytesIO(blob)) as im:
            im = im.convert("RGB")
            pixels = im.load()
            w, h = im.size
            left, top, right, bottom = w, h, 0, 0
            found = False
            step = 2
            for y in range(0, h, step):
                for x in range(0, w, step):
                    r, g, b = pixels[x, y]
                    if r + g + b < 735:
                        found = True
                        left = min(left, x)
                        top = min(top, y)
                        right = max(right, x)
                        bottom = max(bottom, y)
            if not found:
                return blob
            box = (
                max(0, left - pad),
                max(0, top - pad),
                min(w, right + 1 + pad),
                min(h, bottom + 1 + pad),
            )
            if box[2] - box[0] < min_size or box[3] - box[1] < min_size:
                return blob
            cropped = im.crop(box)
            out = BytesIO()
            cropped.save(out, format="PNG", optimize=True)
            return out.getvalue()
    except Exception:
        return blob


def top_looks_like_css_leak(blob: bytes) -> bool:
    """启发式：顶部 28px 横带暗像素占比 >8% → 疑似 CSS 文本泄漏到页面顶部。"""
    try:
        from PIL import Image

        with Image.open(BytesIO(blob)) as im:
            im = im.convert("RGB")
            w, h = im.size
            band_h = min(28, h)
            pixels = im.load()
            dark = total = 0
            for y in range(band_h):
                for x in range(0, w, 3):
                    r, g, b = pixels[x, y]
                    total += 1
                    if r + g + b < 420:
                        dark += 1
            return total > 0 and (dark / total) > 0.08
    except Exception:
        return False


async def render_card_png(
    ctx: Any,
    html: str,
    *,
    root_id: str,
    viewport_width: int,
    viewport_height: int,
    device_scale_factor: float,
    render_timeout_ms: int,
    min_height: int | None = None,
    max_height: int | None = None,
    detect_css_leak: bool = False,
    logger: Any = None,
) -> tuple[str, str | None]:
    """渲染 HTML 卡片为 PNG base64。

    返回 ``(image_base64, error_reason)``；成功时 error_reason 为 None。
    高度窗口校验：``min_height``/``max_height`` 任一提供才启用；超窗拒绝并尝试下一 attempt。
    """
    log = logger if logger is not None else _logger
    if ctx is None:
        return "", "无渲染上下文"

    attempts = [
        {"name": "a1-body", "selector": "body", "full_page": False, "wait_ms": 300, "crop": True},
        {"name": "a2-element", "selector": f"#{root_id}", "full_page": False, "wait_ms": 300, "crop": False},
        {"name": "a3-full-crop", "selector": "body", "full_page": True, "wait_ms": 250, "crop": True},
    ]

    last_err = "卡片渲染失败"
    for idx, attempt in enumerate(attempts):
        try:
            result = await ctx.render.html2png(
                html,
                selector=attempt["selector"],
                viewport={"width": viewport_width, "height": viewport_height},
                device_scale_factor=device_scale_factor,
                full_page=attempt["full_page"],
                omit_background=False,
                wait_until="load",
                wait_for_selector=f"#{root_id}",
                wait_for_timeout_ms=attempt["wait_ms"],
                render_timeout_ms=render_timeout_ms,
                allow_network=False,
            )
        except Exception as exc:
            last_err = f"html2png 异常: {exc}"
            if hasattr(log, "exception"):
                log.exception("html2png failed (%s)", attempt["name"])
            await asyncio.sleep(0.2 * (2 ** idx))
            continue

        if not isinstance(result, dict):
            last_err = "html2png 返回异常"
            await asyncio.sleep(0.2 * (2 ** idx))
            continue
        image_base64 = result.get("image_base64")
        if not isinstance(image_base64, str) or not image_base64:
            last_err = "html2png 未返回图片"
            await asyncio.sleep(0.2 * (2 ** idx))
            continue

        try:
            blob = base64.b64decode(image_base64)
        except Exception:
            # 无法解码的 base64 直接透传（保持旧 Ops 行为）
            return image_base64, None

        if attempt.get("crop"):
            cropped = crop_content_png(blob)
            if cropped is not blob:
                blob = cropped
                image_base64 = base64.b64encode(blob).decode("ascii")

        width, height = png_size(blob)

        if detect_css_leak and top_looks_like_css_leak(blob):
            last_err = f"疑似 CSS 泄漏帧 {width}x{height}"
            if hasattr(log, "warning"):
                log.warning("rejected CSS-leak-looking frame %s: %sx%s", attempt["name"], width, height)
            await asyncio.sleep(0.2 * (2 ** idx))
            continue

        if min_height is not None and height is not None and height < min_height:
            last_err = f"卡片高度异常 {width}x{height}（<{min_height}）"
            await asyncio.sleep(0.2 * (2 ** idx))
            continue
        if max_height is not None and height is not None and height > max_height:
            last_err = f"卡片高度异常 {width}x{height}（>{max_height}）"
            await asyncio.sleep(0.2 * (2 ** idx))
            continue

        return image_base64, None

    return "", last_err
