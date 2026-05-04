"""
可选：Nanobanana 或其它图生图 HTTP 接口，用于生成背景位图。

请在 .env 中填写 **完整请求地址** NANOBANANA_API_URL（例如 …/v1/images/generations），
留空则仅用本地渐变 / 素材库背景。
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any

import httpx
from loguru import logger


class NanobananaBackgroundClient:
    """
    NANOBANANA_RESPONSE_MODE（默认 openai_images）:
      - openai_images: JSON 含 data[0].url 或 data[0].b64_json
      - json_url:      JSON 含 url / image_url / output_url
      - json_base64:   JSON 含 image / image_base64 / b64
    若响应 Content-Type 为 image/*，则正文即图片字节。
    """

    def __init__(self) -> None:
        self.api_url = (os.getenv("NANOBANANA_API_URL") or "").strip()
        self.api_key = (os.getenv("NANOBANANA_API_KEY") or "").strip()
        self.model = (os.getenv("NANOBANANA_MODEL") or "").strip()
        self.mode = (os.getenv("NANOBANANA_RESPONSE_MODE") or "openai_images").strip().lower()
        self.timeout = float(os.getenv("NANOBANANA_TIMEOUT_SEC", "120"))

    def is_configured(self) -> bool:
        return bool(self.api_url)

    def generate(
        self,
        prompt_en: str,
        width: int,
        height: int,
        variant_index: int,
        pixel_width: int | None = None,
        pixel_height: int | None = None,
    ) -> bytes | None:
        if not self.api_url or not prompt_en.strip():
            return None

        api_w = int(pixel_width or 1024)
        api_h = int(pixel_height or 1024)
        api_w = max(256, min(2048, api_w))
        api_h = max(256, min(2048, api_h))
        vhint = ["", " slight color variation.", " alternate soft lighting."][variant_index % 3]
        full_prompt = (prompt_en + vhint).strip()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.mode == "openai_images":
            body: dict[str, Any] = {
                "prompt": full_prompt,
                "n": 1,
                "size": f"{api_w}x{api_h}",
            }
            if self.model:
                body["model"] = self.model
        else:
            body = {
                "prompt": full_prompt,
                "width": api_w,
                "height": api_h,
                "seed": 1000 + variant_index * 17,
            }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(self.api_url, headers=headers, json=body)
                if r.status_code >= 400:
                    snippet = (r.text or "")[:600]
                    logger.warning(
                        "nanobanana_http_error status={} variant={} url={} snippet={}",
                        r.status_code,
                        variant_index,
                        self.api_url,
                        snippet,
                    )
                    return None
                ct = (r.headers.get("content-type") or "").lower()
                if "image/" in ct and "json" not in ct:
                    return r.content
                data = r.json()
            out = _extract_bytes_from_json(data, self.mode)
            if not out:
                logger.warning(
                    "nanobanana_empty_image variant={} mode={} json_keys={}",
                    variant_index,
                    self.mode,
                    list(data.keys()) if isinstance(data, dict) else type(data),
                )
            return out
        except Exception as e:
            logger.warning("nanobanana_generate_failed variant={} err={}", variant_index, str(e))
            return None


def _extract_bytes_from_json(data: Any, mode: str) -> bytes | None:
    if not isinstance(data, dict):
        return None
    if mode == "openai_images":
        arr = data.get("data")
        if isinstance(arr, list) and arr and isinstance(arr[0], dict):
            item = arr[0]
            b64 = item.get("b64_json")
            if b64:
                return base64.b64decode(str(b64))
            url = item.get("url")
            if isinstance(url, str) and url.startswith("http"):
                with httpx.Client(timeout=90.0) as client:
                    ir = client.get(url)
                    ir.raise_for_status()
                    return ir.content
    for key in ("image_base64", "b64", "image", "base64", "output"):
        v = data.get(key)
        if isinstance(v, str) and len(v) > 80:
            try:
                return base64.b64decode(v)
            except Exception:
                continue
    for key in ("url", "image_url", "output_url"):
        u = data.get(key)
        if isinstance(u, str) and u.startswith("http"):
            with httpx.Client(timeout=90.0) as client:
                ir = client.get(u)
                ir.raise_for_status()
                return ir.content
    return None
