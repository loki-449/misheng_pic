"""
为 PIL 解析可显示中文的 TrueType 字体，避免默认点阵字体导致标题「乱码」方框。
"""
import os
from pathlib import Path
from typing import Optional

from PIL import ImageFont


def _candidate_font_paths() -> list[Path]:
    env = os.getenv("CJK_FONT_PATH", "").strip()
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    # Windows 常见中文字体
    windir = os.environ.get("WINDIR", r"C:\Windows")
    paths.extend(
        [
            Path(windir) / "Fonts" / "msyh.ttc",
            Path(windir) / "Fonts" / "msyhbd.ttc",
            Path(windir) / "Fonts" / "simhei.ttf",
            Path(windir) / "Fonts" / "simsun.ttc",
        ]
    )
    # Linux
    paths.extend(
        [
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        ]
    )
    # macOS
    paths.extend(
        [
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/System/Library/Fonts/STHeiti Light.ttc"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
        ]
    )
    return paths


def resolve_cjk_font_path() -> Optional[Path]:
    for p in _candidate_font_paths():
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def load_title_fonts(canvas_size: int) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    """
    主标题 + 副标题字体。canvas 越大字号略增。
    """
    path = resolve_cjk_font_path()
    title_px = max(22, min(86, int(canvas_size / 22)))
    sub_px = max(16, min(56, int(title_px * 0.72)))
    if path and path.suffix.lower() in {".ttc", ".ttf", ".otf"}:
        try:
            # .ttc 为字体集合，index=0 通常为常规体
            title = ImageFont.truetype(str(path), size=title_px, index=0)
            sub = ImageFont.truetype(str(path), size=sub_px, index=0)
            return title, sub
        except OSError:
            pass
    return ImageFont.load_default(), ImageFont.load_default()
