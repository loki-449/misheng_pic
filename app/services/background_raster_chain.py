"""
按环境变量 **MVP_BACKGROUND_RASTER_ORDER** 依次尝试背景位图生成（nanobanana | stable_diffusion | gemini），
任一成功即返回；全部失败则返回 (None, "none", attempts) 供前端与日志排查。
"""
from __future__ import annotations

import os
from typing import Any

from loguru import logger

from app.services.nanobanana_service import NanobananaBackgroundClient
from app.services.stable_diffusion_service import StableDiffusionClient
from app.services.gemini_image_service import GeminiImageClient


def _parse_order() -> list[str]:
    raw = (os.getenv("MVP_BACKGROUND_RASTER_ORDER") or "nanobanana,stable_diffusion,gemini").strip().lower()
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    allowed = {"nanobanana", "stable_diffusion", "gemini", "sd"}
    out: list[str] = []
    for p in parts:
        if p == "sd":
            p = "stable_diffusion"
        if p in allowed and p not in out:
            out.append(p)
    return out or ["nanobanana", "stable_diffusion", "gemini"]


def try_generate_background_raster(
    *,
    prompt_for_api: str,
    negative_en: str,
    canvas_size: int,
    variant_index: int,
    nanobanana: NanobananaBackgroundClient,
    stable_diffusion: StableDiffusionClient,
    gemini_image: GeminiImageClient,
) -> tuple[bytes | None, str, list[dict[str, Any]]]:
    """
    返回 (image_bytes | None, 实际使用的来源标识, 尝试记录列表)。
    """
    if not prompt_for_api.strip():
        return None, "none", [{"source": "skip", "ok": False, "reason": "empty_prompt"}]

    order = _parse_order()
    attempts: list[dict[str, Any]] = []
    cs = max(512, min(4096, int(canvas_size)))

    def _append(source: str, ok: bool, reason: str = "") -> None:
        attempts.append({"source": source, "ok": ok, "reason": reason})

    for name in order:
        if name == "nanobanana":
            if not nanobanana.is_configured():
                _append("nanobanana", False, "not_configured")
                continue
            b = nanobanana.generate(prompt_for_api, cs, cs, variant_index)
            if b:
                _append("nanobanana", True, "")
                logger.info("background_raster_ok source=nanobanana variant={}", variant_index)
                return b, "nanobanana", attempts
            _append("nanobanana", False, "empty_or_request_failed")
            continue

        if name == "stable_diffusion":
            if not stable_diffusion.is_configured():
                _append("stable_diffusion", False, "not_configured")
                continue
            b = stable_diffusion.generate(
                prompt_en=prompt_for_api,
                negative_en=negative_en,
                width=cs,
                height=cs,
                variant_index=variant_index,
            )
            if b:
                _append("stable_diffusion", True, "")
                logger.info("background_raster_ok source=stable_diffusion variant={}", variant_index)
                return b, "stable_diffusion", attempts
            _append("stable_diffusion", False, "empty_or_request_failed")
            continue

        if name == "gemini":
            if not gemini_image.is_configured():
                _append("gemini", False, "not_configured")
                continue
            b = gemini_image.generate_image_bytes(prompt_for_api, variant_index)
            if b:
                _append("gemini", True, "")
                logger.info("background_raster_ok source=gemini variant={}", variant_index)
                return b, "gemini", attempts
            _append("gemini", False, "empty_or_request_failed")
            continue

        _append(name, False, "unknown_backend")

    return None, "none", attempts


def try_generate_promo_raster(
    *,
    prompt_en: str,
    canvas_size: int,
    pixel_width: int,
    pixel_height: int,
    variant_index: int,
    nanobanana: NanobananaBackgroundClient,
    stable_diffusion: StableDiffusionClient,
) -> tuple[bytes | None, str]:
    """宣传条：与背景相同顺序，但仅尝试 nanobanana 与 stable_diffusion（Gemini 文案条兼容性差）。"""
    order = [x for x in _parse_order() if x in ("nanobanana", "stable_diffusion")]
    if not order:
        order = ["nanobanana", "stable_diffusion"]
    cs = max(512, min(4096, int(canvas_size)))
    for name in order:
        if name == "nanobanana" and nanobanana.is_configured():
            b = nanobanana.generate(
                prompt_en, cs, cs, variant_index, pixel_width=pixel_width, pixel_height=pixel_height
            )
            if b:
                return b, "nanobanana"
        if name == "stable_diffusion" and stable_diffusion.is_configured():
            b = stable_diffusion.generate(
                prompt_en=prompt_en,
                negative_en="extra text, watermark, low quality",
                width=pixel_width,
                height=pixel_height,
                variant_index=variant_index,
            )
            if b:
                return b, "stable_diffusion"
    return None, "none"
