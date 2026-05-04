"""
Stable Diffusion（或兼容 Automatic1111 WebUI）HTTP 文生图，用于 MVP 背景位图。

在 .env 中设置 **完整 URL**，例如：
  STABLE_DIFFUSION_API_URL=http://127.0.0.1:7860/sdapi/v1/txt2img

模式：
  a1111（默认）：请求体为 Automatic1111 txt2img JSON，响应 images[] 为 base64 PNG。
  openai_images：与 Nanobanana 相同，POST JSON { prompt, size, n }，解析 data[0].b64_json/url。
"""
from __future__ import annotations

import base64
import os
from typing import Any

import httpx
from loguru import logger

from app.services.nanobanana_service import _extract_bytes_from_json


class StableDiffusionClient:
    def __init__(self) -> None:
        self.api_url = (os.getenv("STABLE_DIFFUSION_API_URL") or "").strip()
        self.mode = (os.getenv("STABLE_DIFFUSION_MODE") or "a1111").strip().lower()
        self.api_key = (os.getenv("STABLE_DIFFUSION_API_KEY") or "").strip()
        self.timeout = float(os.getenv("STABLE_DIFFUSION_TIMEOUT_SEC", "180"))
        self.steps = int(os.getenv("SD_STEPS", "28"))
        self.cfg = float(os.getenv("SD_CFG_SCALE", "7.0"))
        self.sampler = (os.getenv("SD_SAMPLER_NAME") or "DPM++ 2M Karras").strip()

    def is_configured(self) -> bool:
        return bool(self.api_url)

    @staticmethod
    def _align_sd_dim(n: int) -> int:
        n = max(256, min(2048, int(n)))
        return n - (n % 8)

    def generate(
        self,
        prompt_en: str,
        negative_en: str,
        width: int,
        height: int,
        variant_index: int,
    ) -> bytes | None:
        if not self.api_url or not prompt_en.strip():
            return None

        w = self._align_sd_dim(width)
        h = self._align_sd_dim(height)
        vhint = ["", " subtle lighting variation.", " gentle color shift."][variant_index % 3]
        prompt = (prompt_en + vhint).strip()
        neg = (negative_en or "").strip() or "low quality, worst quality, blurry, text, watermark, logo, ugly"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                if self.mode == "openai_images":
                    body: dict[str, Any] = {
                        "prompt": f"{prompt}\nNegative: {neg}",
                        "n": 1,
                        "size": f"{w}x{h}",
                    }
                    m = (os.getenv("STABLE_DIFFUSION_MODEL") or "").strip()
                    if m:
                        body["model"] = m
                    r = client.post(self.api_url, headers=headers, json=body)
                    r.raise_for_status()
                    ct = (r.headers.get("content-type") or "").lower()
                    if "image/" in ct and "json" not in ct:
                        return r.content
                    return _extract_bytes_from_json(r.json(), "openai_images")

                # a1111 txt2img
                body = {
                    "prompt": prompt,
                    "negative_prompt": neg,
                    "width": w,
                    "height": h,
                    "steps": self.steps,
                    "cfg_scale": self.cfg,
                    "sampler_name": self.sampler,
                    "seed": 20000 + variant_index * 7919,
                    "restore_faces": False,
                    "batch_size": 1,
                    "n_iter": 1,
                }
                r = client.post(self.api_url, headers=headers, json=body)
                if r.status_code >= 400:
                    snippet = (r.text or "")[:500]
                    logger.warning(
                        "stable_diffusion_http_error status={} url={} body_snip={}",
                        r.status_code,
                        self.api_url,
                        snippet,
                    )
                    return None
                data = r.json()
                imgs = data.get("images")
                if isinstance(imgs, list) and imgs and isinstance(imgs[0], str):
                    raw = base64.b64decode(imgs[0])
                    if raw:
                        return raw
                logger.warning("stable_diffusion_no_images keys={}", list(data.keys()) if isinstance(data, dict) else type(data))
                return None
        except Exception as e:
            logger.warning("stable_diffusion_generate_failed variant={} err={}", variant_index, str(e))
            return None
