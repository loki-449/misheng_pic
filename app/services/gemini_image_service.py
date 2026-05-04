"""
使用 Google Gemini「原生出图」模型生成背景位图（Nano Banana 系列）。

文档：https://ai.google.dev/gemini-api/docs/image-generation
需在 .env 配置 GEMINI_API_KEY 与 GEMINI_IMAGE_MODEL。
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any

import httpx
from loguru import logger


class GeminiImageClient:
    def __init__(self) -> None:
        self.api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        # 绘图专用模型，与纯文本 GEMINI_MODEL 分开配置
        self.image_model = (os.getenv("GEMINI_IMAGE_MODEL") or "gemini-2.5-flash-image").strip()
        self.base_url = (
            os.getenv("GEMINI_API_BASE_URL") or "https://generativelanguage.googleapis.com"
        ).strip().rstrip("/")
        self.timeout = float(os.getenv("GEMINI_IMAGE_TIMEOUT_SEC", "120"))

    def is_configured(self) -> bool:
        return bool(self.api_key) and bool(self.image_model)

    def generate_image_bytes(self, prompt_en: str, variant_index: int = 0) -> bytes | None:
        if not self.is_configured() or not prompt_en.strip():
            return None

        hint = ["", " Slight color variation.", " Alternate soft studio lighting."][variant_index % 3]
        text = (prompt_en + hint).strip()

        url = f"{self.base_url}/v1beta/models/{self.image_model}:generateContent"
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": text}],
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
            },
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            logger.warning("gemini_image_request_failed model={} err={}", self.image_model, str(e))
            return None

        return _extract_first_image_bytes(data)


def _part_inline_b64(part: dict[str, Any]) -> bytes | None:
    inline = part.get("inlineData") or part.get("inline_data")
    if not isinstance(inline, dict):
        return None
    raw = inline.get("data")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return base64.b64decode(raw)
    except Exception:
        return None


def _extract_first_image_bytes(data: dict[str, Any]) -> bytes | None:
    cands = data.get("candidates")
    if not isinstance(cands, list) or not cands:
        return None
    content = cands[0].get("content") if isinstance(cands[0], dict) else None
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    for part in parts:
        if not isinstance(part, dict):
            continue
        img = _part_inline_b64(part)
        if img:
            return img
    return None
