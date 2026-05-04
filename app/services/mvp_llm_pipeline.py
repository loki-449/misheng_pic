"""
MVP：受控中文风格词（仅从词表选择）+ 拼接方案（二选一 + 预设 distribution），
通过 DeepSeek / Gemini 两个通道按顺序尝试；失败时抛出结构化异常供 API 返回。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI

from app.services.gemini_text_client import GeminiTextClient, GeminiTextError, gemini_model_error_hints


class MvpLlmPipelineError(Exception):
    """携带给前端的 code / message_zh / fixes。"""

    def __init__(
        self,
        *,
        code: str,
        message_zh: str,
        fixes: list[str],
        provider: str = "",
        raw: str = "",
    ) -> None:
        super().__init__(message_zh)
        self.code = code
        self.message_zh = message_zh
        self.fixes = fixes
        self.provider = provider
        self.raw = raw

    def to_detail(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "message_zh": self.message_zh,
            "fixes": self.fixes,
        }
        if self.provider:
            out["provider"] = self.provider
        if self.raw:
            out["raw_error"] = self.raw[:1200]
        return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def snap_keywords_to_vocab(heuristic: list[str], vocab: list[str]) -> list[str]:
    """将启发式中文词对齐到词表（无法对齐时用词表首项或互相兜底）。"""
    if not vocab:
        pool = [str(x).strip() for x in heuristic if str(x).strip()][:3]
        while len(pool) < 3:
            pool.append(pool[-1] if pool else "简约")
        return pool[:3]
    vocab_set = set(vocab)
    out: list[str] = []
    for h in heuristic:
        h = str(h).strip()
        if h in vocab_set:
            out.append(h)
            continue
        hit = next((v for v in vocab if (h in v or v in h)), None)
        out.append(hit or vocab[0])
    while len(out) < 3:
        out.append(out[-1])
    return out[:3]


def any_text_llm_configured() -> bool:
    if (os.getenv("GEMINI_API_KEY") or "").strip() and (os.getenv("GEMINI_MODEL") or "").strip():
        return True
    return bool((os.getenv("DEEPSEEK_API_KEY") or "").strip())


def load_mvp_style_vocab() -> list[str]:
    path = Path(os.getenv("MVP_STYLE_VOCAB_PATH") or (_repo_root() / "prompts" / "mvp_style_vocab.txt"))
    if not path.is_file():
        return []
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        words.append(s)
    return words


def load_mvp_stitch_system_prompt() -> str:
    path = Path(os.getenv("MVP_STITCH_PROMPT_PATH") or (_repo_root() / "prompts" / "mvp_stitch_prompt.md"))
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return (
        '只输出 JSON：{"stitch_mode":"single_main_aux|multi_equal_grid",'
        '"distribution":"grid_auto_square"}。不要其它文字。'
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1]
            if t.lstrip().startswith("json"):
                t = t.lstrip()[4:].strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object in model output")
    return json.loads(t[start : end + 1])


def _provider_order() -> list[str]:
    raw = (os.getenv("MVP_STYLE_LLM_ORDER") or "gemini,deepseek").strip().lower()
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    order: list[str] = []
    for p in parts:
        if p in ("gemini", "deepseek") and p not in order:
            order.append(p)
    if not order:
        order = ["gemini", "deepseek"]
    return order


def _deepseek_client() -> tuple[OpenAI | None, str, str]:
    key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    base = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip()
    model = (os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip()
    if not key:
        return None, base, model
    return OpenAI(api_key=key, base_url=base), base, model


def _call_deepseek_json(system: str, user: str, request_id: str) -> str:
    client, _base, model = _deepseek_client()
    if not client:
        raise MvpLlmPipelineError(
            code="deepseek_not_configured",
            message_zh="未配置 DEEPSEEK_API_KEY，无法调用 DeepSeek。",
            fixes=["在 .env 中填写 DEEPSEEK_API_KEY，或将 MVP_STYLE_LLM_ORDER 调整为仅使用 gemini（若已配置 GEMINI_API_KEY）。"],
            provider="deepseek",
        )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.15,
            stream=False,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("empty_content")
        return content
    except MvpLlmPipelineError:
        raise
    except Exception as e:
        msg = str(e)
        logger.warning("mvp_deepseek_failed request_id={} err={}", request_id, msg)
        fixes = [
            "请确认 DEEPSEEK_MODEL（例如 deepseek-chat）与 DEEPSEEK_BASE_URL 是否与当前账号一致。",
            "若接口返回 model not found，请在 .env 中更换为控制台列出的可用模型名。",
        ]
        if "model" in msg.lower() and ("not found" in msg.lower() or "does not exist" in msg.lower()):
            fixes.insert(0, "当前 DEEPSEEK_MODEL 可能无效：请在 DeepSeek 控制台核对可用模型 ID 并修改 .env。")
        raise MvpLlmPipelineError(
            code="deepseek_request_failed",
            message_zh=f"DeepSeek 调用失败：{msg[:400]}",
            fixes=fixes,
            provider="deepseek",
            raw=msg,
        ) from e


def _call_gemini_json(system: str, user: str, request_id: str) -> str:
    gem = GeminiTextClient()
    if not gem.is_configured():
        raise MvpLlmPipelineError(
            code="gemini_not_configured",
            message_zh="未配置 Gemini 文本模型（GEMINI_API_KEY / GEMINI_MODEL）。",
            fixes=["在 .env 中填写 GEMINI_API_KEY 与 GEMINI_MODEL；文本模型与 GEMINI_IMAGE_MODEL 不同。"],
            provider="gemini",
        )
    try:
        return gem.generate_text(system_instruction=system, user_text=user)
    except GeminiTextError as e:
        hints = gemini_model_error_hints(e.status_code, e.payload)
        raise MvpLlmPipelineError(
            code="gemini_request_failed",
            message_zh=f"Gemini 调用失败：{str(e)[:500]}",
            fixes=hints or ["请检查 GEMINI_MODEL 与 GEMINI_API_KEY。"],
            provider="gemini",
            raw=str(e),
        ) from e


def _call_json_llm(system: str, user: str, request_id: str, provider: str) -> str:
    if provider == "gemini":
        return _call_gemini_json(system, user, request_id)
    if provider == "deepseek":
        return _call_deepseek_json(system, user, request_id)
    raise MvpLlmPipelineError(
        code="unknown_provider",
        message_zh=f"不支持的 LLM 通道：{provider}",
        fixes=["请将 MVP_STYLE_LLM_ORDER 设为 gemini,deepseek 的组合。"],
        provider=provider,
    )


ALLOWED_DISTRIBUTIONS = frozenset(
    {
        "single_main_bottom_aux_top_strip",
        "single_main_center_aux_right_stack",
        "grid_1x2",
        "grid_1x3",
        "grid_2x2",
        "grid_2x3",
        "grid_auto_square",
    }
)


def run_style_constrained_llm(
    *,
    vocab: list[str],
    stats_summary: str,
    product_name: str,
    product_desc: str,
    image_count: int,
    request_id: str,
) -> tuple[list[str], str]:
    """
    返回 (三个中文风格词, 实际使用的 provider 名)。
    若词表为空则回退为不受控的旧逻辑（由调用方处理）。
    """
    if not vocab:
        raise MvpLlmPipelineError(
            code="style_vocab_missing",
            message_zh="风格词表文件为空或不存在。",
            fixes=["请确认 prompts/mvp_style_vocab.txt 存在且含有词语；或设置环境变量 MVP_STYLE_VOCAB_PATH 指向有效词表。"],
            provider="",
        )

    vocab_inline = "、".join(vocab[:80])
    if len(vocab) > 80:
        vocab_inline += "（词表过长已截断展示；实际校验使用完整词表）"

    system = (
        "你是电商视觉助理。只输出一个 JSON 对象，不要 Markdown、不要解释。"
        '{"keywords":["词1","词2","词3"]} '
        "keywords：恰好 3 个字符串，每一个必须**完全等于**下面词表中的某一个词（可复制粘贴，不得自造词、不得写英文）：\n"
        f"{vocab_inline}"
    )
    user = (
        f"request_id={request_id}\n"
        f"商品图张数：{image_count}\n"
        f"商品名称：{product_name}\n"
        f"商品描述：{product_desc}\n"
        f"图像统计摘要：{stats_summary}\n"
        "请严格从词表中选 3 个词填入 keywords。"
    )

    last_err: MvpLlmPipelineError | None = None
    vocab_set = set(vocab)
    for provider in _provider_order():
        try:
            raw = _call_json_llm(system, user, request_id, provider)
            parsed = _extract_json_object(raw)
            kws = parsed.get("keywords") or parsed.get("style_keywords")
            if not isinstance(kws, list) or len(kws) < 3:
                raise ValueError("keywords invalid")
            out = [str(x).strip() for x in kws[:3]]
            for w in out:
                if w not in vocab_set:
                    raise ValueError(f"out_of_vocab:{w}")
            return out, provider
        except MvpLlmPipelineError as e:
            last_err = e
            logger.warning("mvp_style_provider_failed request_id={} provider={} err={}", request_id, provider, e.message_zh)
        except Exception as e:
            last_err = MvpLlmPipelineError(
                code="style_parse_failed",
                message_zh=f"风格 JSON 解析失败：{e}",
                fixes=["请略调高温度无效；一般是模型未按要求输出 JSON。可尝试更换 MVP_STYLE_LLM_ORDER 中的优先模型。"],
                provider=provider,
                raw=str(e),
            )
            logger.warning("mvp_style_parse_failed request_id={} err={}", request_id, str(e))

    if last_err:
        merged_fixes = list(
            dict.fromkeys(
                last_err.fixes
                + [
                    "已按 MVP_STYLE_LLM_ORDER 顺序尝试所有已配置的通道；请至少保证 Gemini 或 DeepSeek 其一可用。",
                ]
            )
        )
        raise MvpLlmPipelineError(
            code="style_all_providers_failed",
            message_zh="所有已配置的风格分析模型均失败或输出不符合词表约束。",
            fixes=merged_fixes,
            provider="",
            raw=last_err.raw,
        )
    raise MvpLlmPipelineError(
        code="style_no_provider",
        message_zh="没有可用的风格分析模型。",
        fixes=["配置 GEMINI_API_KEY+GEMINI_MODEL 或 DEEPSEEK_API_KEY。"],
        provider="",
    )


def run_stitch_decision_llm(
    *,
    stats_summary: str,
    product_name: str,
    product_desc: str,
    image_count: int,
    request_id: str,
) -> tuple[str, str, str]:
    """
    返回 (stitch_mode, distribution, provider)
    """
    system = load_mvp_stitch_system_prompt()
    user = (
        f"request_id={request_id}\n"
        f"商品图张数：{image_count}\n"
        f"商品名称：{product_name}\n"
        f"商品描述：{product_desc}\n"
        f"图像统计摘要：{stats_summary}\n"
        "请只输出 JSON。"
    )

    last_err: MvpLlmPipelineError | None = None
    for provider in _provider_order():
        try:
            raw = _call_json_llm(system, user, request_id, provider)
            parsed = _extract_json_object(raw)
            mode = str(parsed.get("stitch_mode", "")).strip()
            dist = str(parsed.get("distribution", "")).strip()
            if image_count <= 1:
                mode = "single_main_aux"
                dist = "single_main_bottom_aux_top_strip"
            if mode not in ("single_main_aux", "multi_equal_grid"):
                raise ValueError("bad stitch_mode")
            if dist not in ALLOWED_DISTRIBUTIONS:
                raise ValueError("bad distribution")
            return mode, dist, provider
        except MvpLlmPipelineError as e:
            last_err = e
            logger.warning("mvp_stitch_provider_failed request_id={} p={} err={}", request_id, provider, e.message_zh)
        except Exception as e:
            last_err = MvpLlmPipelineError(
                code="stitch_parse_failed",
                message_zh=f"拼接方案 JSON 无效：{e}",
                fixes=["可尝试更换 MVP_STYLE_LLM_ORDER 优先顺序，或检查 prompts/mvp_stitch_prompt.md 是否被误改。"],
                provider=provider,
                raw=str(e),
            )

    # 确定性回退（离线可用）
    if image_count <= 1:
        return "single_main_aux", "single_main_bottom_aux_top_strip", "fallback"
    if image_count == 2:
        return "multi_equal_grid", "grid_1x2", "fallback"
    if image_count == 3:
        return "multi_equal_grid", "grid_1x3", "fallback"
    if image_count == 4:
        return "multi_equal_grid", "grid_2x2", "fallback"
    if image_count <= 6:
        return "multi_equal_grid", "grid_2x3", "fallback"
    return "multi_equal_grid", "grid_auto_square", "fallback"


def build_promo_image_prompt_en(promo_copy: str, style_keywords_cn: list[str]) -> str:
    """供 Nanobanana 生成宣传条漫/字块（英文 prompt 更稳）。"""
    tags = ", ".join(style_keywords_cn[:3])
    safe = re.sub(r"[\r\n]+", " ", promo_copy.strip())[:200]
    return (
        f"Horizontal promotional typography strip for e-commerce poster, mood keywords: {tags}. "
        f"Display the following Chinese marketing copy prominently with clean modern layout, "
        f"high legibility, subtle decorative elements matching the mood, flat vector or print-ready graphic style, "
        f"no extra Chinese characters beyond the quoted text, no watermark, no QR code. "
        f"Copy to render exactly: 「{safe}」"
    )
