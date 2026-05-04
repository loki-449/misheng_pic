"""
Google Gemini 纯文本 JSON 调用（与 GEMINI_IMAGE_MODEL 分离，使用 GEMINI_MODEL）。

用于 MVP 受控风格词、拼接方案等短输出；失败时抛出带 HTTP 状态与正文的异常，便于返回给前端修改配置。
"""
from __future__ import annotations

import os
from typing import Any

import httpx


class GeminiTextError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class GeminiTextClient:
    def __init__(self) -> None:
        self.api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        self.model = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip()
        self.base_url = (
            os.getenv("GEMINI_API_BASE_URL") or "https://generativelanguage.googleapis.com"
        ).strip().rstrip("/")
        self.timeout = float(os.getenv("GEMINI_TEXT_TIMEOUT_SEC", "90"))

    def is_configured(self) -> bool:
        return bool(self.api_key) and bool(self.model)

    def generate_text(self, *, system_instruction: str, user_text: str) -> str:
        if not self.is_configured():
            raise GeminiTextError("GEMINI_API_KEY 或 GEMINI_MODEL 未配置", status_code=None)

        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {"temperature": 0.15},
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(url, headers=headers, json=body)
                if r.status_code >= 400:
                    try:
                        payload = r.json()
                    except Exception:
                        payload = {"raw": (r.text or "")[:800]}
                    raise GeminiTextError(
                        f"Gemini HTTP {r.status_code}: {str(payload)[:500]}",
                        status_code=r.status_code,
                        payload=payload,
                    )
                data = r.json()
        except GeminiTextError:
            raise
        except Exception as e:
            raise GeminiTextError(f"Gemini 请求异常: {e}", status_code=None) from e

        text = _extract_text_from_generate_content(data)
        if not text.strip():
            raise GeminiTextError("Gemini 返回空文本", status_code=200, payload=data)
        return text


def _extract_text_from_generate_content(data: dict[str, Any]) -> str:
    cands = data.get("candidates")
    if not isinstance(cands, list) or not cands:
        return ""
    c0 = cands[0] if isinstance(cands[0], dict) else {}
    content = c0.get("content") if isinstance(c0.get("content"), dict) else {}
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            chunks.append(p["text"])
    return "".join(chunks).strip()


def gemini_model_error_hints(status_code: int | None, payload: Any) -> list[str]:
    """面向运维/用户的修改说明（中文）。"""
    fixes: list[str] = []
    err_msg = ""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            err_msg = str(err.get("message", "") or "")
    low = err_msg.lower()
    if status_code == 404 or "not found" in low or "is not found" in low:
        fixes.append(
            "请在 .env 中检查 GEMINI_MODEL：须为当前 Google AI Studio / Generative Language API 支持的「文本」模型 ID。"
        )
        fixes.append(
            "常见可尝试值：gemini-2.0-flash、gemini-1.5-flash、gemini-1.5-pro（以官方文档为准，会随时间更新）。"
        )
        fixes.append("若您把绘图模型 GEMINI_IMAGE_MODEL 填到了 GEMINI_MODEL，也会 404；文本分析与绘图模型需分开配置。")
    elif status_code == 400:
        fixes.append("请检查 GEMINI_API_KEY 是否有效、是否未过期，以及请求体是否与该模型版本兼容。")
        if err_msg:
            fixes.append(f"接口返回说明：{err_msg[:300]}")
    elif status_code in (401, 403):
        fixes.append("请检查 GEMINI_API_KEY 是否正确、是否对该 API 已启用计费/权限。")
    else:
        fixes.append("请查看服务器日志中的完整 Gemini 响应；确认 GEMINI_API_BASE_URL（如有自定义）与网络连通性。")
        if err_msg:
            fixes.append(f"接口返回说明：{err_msg[:300]}")
    return fixes
