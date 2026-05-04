import json
import os
import uuid
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from app.services.style_service import StyleClassifier
from app.services.background_service import BackgroundRetriever
from app.services.image_analysis_service import (
    analyze_product_image,
    analyze_product_images,
    heuristic_style_keywords_cn,
)
from app.services.mvp_llm_pipeline import (
    MvpLlmPipelineError,
    any_text_llm_configured,
    load_mvp_style_vocab,
    run_stitch_decision_llm,
    run_style_constrained_llm,
    snap_keywords_to_vocab,
)
from app.services.layout_templates import get_layout_template, template_to_layout_dict

# 自动加载根目录下的 .env 文件
load_dotenv()

class YayoiBrain:
    def __init__(self):
        # 1. 这里的配置全部来自你刚刚写的 .env
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.review_rule_weight = float(os.getenv("REVIEW_RULE_WEIGHT", "0.65"))
        self.review_llm_weight = float(os.getenv("REVIEW_LLM_WEIGHT", "0.35"))
        self.style_classifier = StyleClassifier()
        self.background_retriever = BackgroundRetriever()

        # 2. 初始化 OpenAI 兼容客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _default_plan(self, canvas_size: int) -> dict[str, Any]:
        return {
            "style": {
                "tags": ["clean"],
                "confidence": 0.3,
                "reason": "model_fallback",
                "source": "fallback",
            },
            "layout": {
                "canvas_size": canvas_size,
                "composition": "product_center_bottom",
                "text_area": "top_left",
                "safe_margin_percent": 8
            },
            "background": {
                "mode": "retrieve_or_generate",
                "keywords": ["soft light", "low saturation"],
                "negative_keywords": ["busy pattern", "strong noise"],
                "note": "prioritize readability",
                "candidates": [],
            }
        }

    def _extract_json(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            parts = text.split("```")
            # 支持 ```json ... ``` 格式
            if len(parts) >= 3:
                text = parts[1]
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        return json.loads(text[start:end + 1])

    def get_visual_plan(
        self,
        product_name: str,
        product_desc: str,
        canvas_size: int = 2048,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """
        根据产品信息生成严格 JSON 的视觉方案。
        带解析失败自动重试，确保 API 稳定返回结构化数据。
        """
        system_prompt = (
            "你是一位资深电商视觉总监。"
            "你必须只输出一个 JSON 对象，不允许输出任何解释、Markdown、代码块。"
            "JSON schema 如下："
            "{"
            "\"style\":{\"tags\":[\"string\"],\"confidence\":0.0,\"reason\":\"string\"},"
            "\"layout\":{\"canvas_size\":2048,\"composition\":\"string\",\"text_area\":\"string\",\"safe_margin_percent\":8},"
            "\"background\":{\"mode\":\"retrieve_or_generate\",\"keywords\":[\"string\"],\"negative_keywords\":[\"string\"],\"note\":\"string\"}"
            "}"
        )

        user_content = (
            f"产品名称：{product_name}\n"
            f"产品描述：{product_desc}\n"
            f"目标画布：{canvas_size}\n"
            "请返回可直接用于后续渲染的结构化视觉方案。"
        )

        retries = 2
        last_error = None
        rid = request_id or str(uuid.uuid4())
        for attempt in range(retries + 1):
            try:
                logger.info(
                    "visual_plan_request request_id={} attempt={} product={} canvas={}",
                    rid,
                    attempt + 1,
                    product_name,
                    canvas_size,
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2,
                    stream=False
                )
                content = response.choices[0].message.content or ""
                parsed = self._extract_json(content)
                # 最小字段校验，缺失则抛错进入重试
                _ = parsed["style"]["tags"]
                _ = parsed["layout"]["composition"]
                _ = parsed["background"]["keywords"]
                # M1: 风格分类器主判，LLM 作为解释补充
                style_result = self.style_classifier.classify(
                    product_name=product_name,
                    product_desc=product_desc,
                )
                parsed["style"]["tags"] = style_result.tags
                parsed["style"]["confidence"] = style_result.confidence
                parsed["style"]["reason"] = f"{style_result.reason}; llm_hint:{parsed['style'].get('reason', '')}"
                parsed["style"]["source"] = style_result.source
                # M2: 背景检索（优先素材库）
                candidates = self.background_retriever.retrieve(
                    style_tags=style_result.tags,
                    product_name=product_name,
                    product_desc=product_desc,
                    top_k=3,
                )
                parsed["background"]["candidates"] = [
                    {
                        "background_id": c.background_id,
                        "path": c.path,
                        "tags": c.tags,
                        "score": c.score,
                    }
                    for c in candidates
                ]
                logger.info(
                    "visual_plan_success request_id={} attempt={} product={} style_tags={}",
                    rid,
                    attempt + 1,
                    product_name,
                    parsed.get("style", {}).get("tags", []),
                )
                return parsed
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "visual_plan_retry request_id={} attempt={} product={} error={}",
                    rid,
                    attempt + 1,
                    product_name,
                    last_error,
                )

        fallback = self._default_plan(canvas_size=canvas_size)
        fallback["style"]["reason"] = f"fallback_after_retry: {last_error}"
        try:
            style_result = self.style_classifier.classify(product_name=product_name, product_desc=product_desc)
            fallback["style"]["tags"] = style_result.tags
            fallback["style"]["confidence"] = style_result.confidence
            fallback["style"]["source"] = style_result.source
            fallback["background"]["candidates"] = [
                {
                    "background_id": c.background_id,
                    "path": c.path,
                    "tags": c.tags,
                    "score": c.score,
                }
                for c in self.background_retriever.retrieve(
                    style_tags=style_result.tags,
                    product_name=product_name,
                    product_desc=product_desc,
                    top_k=3,
                )
            ]
        except Exception:
            pass
        logger.error(
            "visual_plan_fallback request_id={} product={} reason={}",
            rid,
            product_name,
            fallback["style"]["reason"],
        )
        return fallback

    def review_visual_plan(
        self,
        plan: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """
        对视觉方案进行规则化评分，给出可执行修正建议。
        先走规则引擎，后续可升级为模型评审。
        """
        rid = request_id or str(uuid.uuid4())
        issues: list[str] = []
        suggestions: list[str] = []
        score = 100

        style = plan.get("style", {})
        layout = plan.get("layout", {})
        background = plan.get("background", {})

        confidence = float(style.get("confidence", 0.0) or 0.0)
        tags = style.get("tags", [])
        text_area = str(layout.get("text_area", ""))
        margin = int(layout.get("safe_margin_percent", 0) or 0)
        bg_keywords = background.get("keywords", [])
        bg_negative = background.get("negative_keywords", [])

        if confidence < 0.6:
            score -= 20
            issues.append("style_confidence_low")
            suggestions.append("增加风格样本或补充产品描述细节，提高风格识别稳定性")
        if not tags:
            score -= 15
            issues.append("style_tags_missing")
            suggestions.append("至少返回一个明确风格标签")
        if margin < 6:
            score -= 20
            issues.append("safe_margin_too_small")
            suggestions.append("将安全边距提高到 6-10%，保证文案不贴边")
        if text_area not in {"top_left", "top_right", "left", "right"}:
            score -= 10
            issues.append("text_area_unclear")
            suggestions.append("将文案区设为 top_left/top_right/left/right 之一")
        if not bg_keywords:
            score -= 15
            issues.append("background_keywords_missing")
            suggestions.append("补充 2-5 个背景关键词，强化检索/生成约束")
        if "busy pattern" not in bg_negative:
            score -= 10
            issues.append("negative_keyword_missing_busy_pattern")
            suggestions.append("建议加入 busy pattern 等负面关键词，避免背景抢主体")

        score = max(0, min(100, score))
        level = "good" if score >= 80 else "medium" if score >= 60 else "poor"
        rule_review = {
            "score": score,
            "level": level,
            "issues": issues,
            "suggestions": suggestions,
            "source": "rule",
            "confidence": 0.9,
        }
        logger.info(
            "visual_plan_review_rule request_id={} score={} level={} issues={}",
            rid,
            rule_review["score"],
            rule_review["level"],
            rule_review["issues"],
        )
        llm_review = self._review_visual_plan_by_llm(plan=plan, request_id=rid)
        if llm_review is None:
            return rule_review
        merged = self._merge_reviews(rule_review=rule_review, llm_review=llm_review, request_id=rid)
        return merged

    def _review_visual_plan_by_llm(self, plan: dict[str, Any], request_id: str) -> dict[str, Any] | None:
        """
        LLM 审美复核通道。失败时返回 None，不影响主流程。
        """
        system_prompt = (
            "你是资深电商视觉审稿专家。"
            "请仅输出一个 JSON 对象，不要输出任何多余内容。"
            "JSON schema: "
            "{"
            "\"score\": 0,"
            "\"level\": \"good|medium|poor\","
            "\"issues\": [\"string\"],"
            "\"suggestions\": [\"string\"],"
            "\"confidence\": 0.0"
            "}"
        )
        user_content = (
            "请评估以下视觉方案的可读性、主体突出度、风格一致性与背景匹配度：\n"
            f"{json.dumps(plan, ensure_ascii=False)}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                stream=False,
            )
            content = response.choices[0].message.content or ""
            parsed = self._extract_json(content)
            score = int(parsed["score"])
            level = str(parsed["level"])
            issues = list(parsed["issues"])
            suggestions = list(parsed["suggestions"])
            confidence = float(parsed.get("confidence", 0.6))
            llm_review = {
                "score": max(0, min(100, score)),
                "level": level if level in {"good", "medium", "poor"} else "medium",
                "issues": [str(i) for i in issues],
                "suggestions": [str(s) for s in suggestions],
                "source": "llm",
                "confidence": max(0.0, min(1.0, confidence)),
            }
            logger.info(
                "visual_plan_review_llm request_id={} score={} level={} confidence={}",
                request_id,
                llm_review["score"],
                llm_review["level"],
                llm_review["confidence"],
            )
            return llm_review
        except Exception as e:
            logger.warning(
                "visual_plan_review_llm_failed request_id={} error={}",
                request_id,
                str(e),
            )
            return None

    def _merge_reviews(
        self,
        rule_review: dict[str, Any],
        llm_review: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """
        融合规则评分与 LLM 评分，提升稳定性与审美判断能力。
        """
        rule_w = self.review_rule_weight
        llm_w = self.review_llm_weight
        weight_sum = rule_w + llm_w
        if weight_sum <= 0:
            rule_w, llm_w = 0.65, 0.35
            weight_sum = 1.0
        # 标准化，防止外部配置和为非 1
        rule_w = rule_w / weight_sum
        llm_w = llm_w / weight_sum
        merged_score = int(round(rule_review["score"] * rule_w + llm_review["score"] * llm_w))
        merged_score = max(0, min(100, merged_score))
        merged_level = "good" if merged_score >= 80 else "medium" if merged_score >= 60 else "poor"

        merged_issues = list(dict.fromkeys(rule_review["issues"] + llm_review["issues"]))
        merged_suggestions = list(dict.fromkeys(rule_review["suggestions"] + llm_review["suggestions"]))
        merged_confidence = round((float(rule_review["confidence"]) * rule_w + float(llm_review["confidence"]) * llm_w), 3)

        merged = {
            "score": merged_score,
            "level": merged_level,
            "issues": merged_issues,
            "suggestions": merged_suggestions,
            "source": "hybrid(rule+llm)",
            "confidence": merged_confidence,
        }
        logger.info(
            "visual_plan_review_merged request_id={} score={} level={} confidence={}",
            request_id,
            merged["score"],
            merged["level"],
            merged["confidence"],
        )
        return merged

    def get_visual_strategy(self, product_name: str, product_desc: str) -> str:
        """
        兼容旧接口：保留原有字符串输出能力。
        """
        plan = self.get_visual_plan(product_name=product_name, product_desc=product_desc, canvas_size=2048)
        return json.dumps(plan, ensure_ascii=False)

    def get_public_config(self) -> dict[str, Any]:
        """
        仅返回可公开的非敏感配置，用于联调与排障。
        """
        gemini_key = bool((os.getenv("GEMINI_API_KEY") or "").strip())
        gemini_img = (os.getenv("GEMINI_IMAGE_MODEL") or "gemini-2.5-flash-image").strip()
        gemini_txt = (os.getenv("GEMINI_MODEL") or "").strip()
        return {
            "model": self.model,
            "base_url": self.base_url,
            "review_rule_weight": self.review_rule_weight,
            "review_llm_weight": self.review_llm_weight,
            "nanobanana_api_configured": bool((os.getenv("NANOBANANA_API_URL") or "").strip()),
            "stable_diffusion_configured": bool((os.getenv("STABLE_DIFFUSION_API_URL") or "").strip()),
            "gemini_image_configured": gemini_key and bool(gemini_img),
            "gemini_image_model": gemini_img if gemini_key else "",
            "gemini_text_configured": gemini_key and bool(gemini_txt),
            "gemini_text_model": gemini_txt if gemini_key else "",
            "mvp_style_llm_order": (os.getenv("MVP_STYLE_LLM_ORDER") or "gemini,deepseek").strip(),
        }

    def _cn_keywords_to_en_tags(self, keywords_cn: list[str]) -> list[str]:
        """中文风格词映射为内部英文标签，供背景检索等模块使用。"""
        mapping: dict[str, str] = {
            "清新": "fresh",
            "明亮": "fresh",
            "治愈": "fresh",
            "柔和": "fresh",
            "自然": "fresh",
            "活力": "fresh",
            "温暖": "fresh",
            "鲜明": "fresh",
            "简约": "clean",
            "极简": "clean",
            "干净": "clean",
            "高级": "clean",
            "平衡": "clean",
            "复古": "retro",
            "怀旧": "retro",
            "质感": "retro",
            "浓郁": "retro",
            "饱满": "retro",
            "吸睛": "craft",
            "沉稳": "tech",
            "商务": "tech",
            "科技": "tech",
            "日系": "craft",
            "手作": "craft",
            "文创": "craft",
        }
        out: list[str] = []
        for k in keywords_cn:
            key = str(k).strip()
            tag = mapping.get(key)
            if tag and tag not in out:
                out.append(tag)
        if not out:
            out = ["clean"]
        return out[:3]

    @staticmethod
    def _pad_three_cn_keywords(words: list[str]) -> list[str]:
        kw = [s.strip() for s in words if s.strip()][:3]
        while len(kw) < 3:
            kw.append(kw[-1] if kw else "简约")
        return kw[:3]

    def _fallback_background_prompt_en(self, cn_keywords: list[str], stats: dict[str, Any]) -> str:
        tags = ", ".join(cn_keywords)
        dom = stats.get("dominant_rgb", [220, 215, 210])
        return (
            f"Professional minimal e-commerce product photography backdrop, mood: {tags}, "
            f"soft studio lighting, clean gradient surface, subtle paper or fabric texture, "
            f"color harmony with accent RGB({dom[0]},{dom[1]},{dom[2]}), "
            "absolutely no text, no letters, no watermark, no logo, no people, empty center area for product"
        )

    def _llm_full_mvp_analysis(
        self,
        stats: dict[str, Any],
        product_name: str,
        product_desc: str,
        request_id: str,
    ) -> dict[str, Any] | None:
        """
        一次 LLM：三个中文风格词 + 英文背景生成 prompt（给 Nanobanana / SD 类接口用）。
        """
        if not self.api_key:
            return None
        system_prompt = (
            "你是电商视觉总监。只输出一个 JSON 对象，不要 Markdown、不要解释。"
            "JSON 字段："
            "{\"keywords\":[\"\",\"\",\"\"],"
            "\"background_prompt_en\":\"string\","
            "\"background_negative_en\":\"string\"}。"
            "keywords：恰好3个词，每个2-4个汉字，描述风格氛围，不要英文。"
            "background_prompt_en：一段英文，描述无主体的商品摄影背景/台面/柔光/材质，"
            "必须与商品气质一致；强调无文字、无水印、无 logo。"
            "background_negative_en：英文负面描述，列出要避免的元素（如文字、杂乱图案等）。"
        )
        user_content = (
            f"商品图统计：主色RGB={stats.get('dominant_rgb')}, "
            f"平均亮度={stats.get('mean_brightness')}, 平均饱和度={stats.get('mean_saturation')}, "
            f"尺寸={stats.get('width')}x{stats.get('height')}。\n"
            f"商品名称：{product_name}\n"
            f"商品描述：{product_desc}\n"
            "请输出 JSON。"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.35,
                stream=False,
            )
            content = response.choices[0].message.content or ""
            parsed = self._extract_json(content)
            raw_kw = parsed.get("keywords") or parsed.get("style_keywords")
            if not isinstance(raw_kw, list):
                return None
            kws = [str(x).strip() for x in raw_kw if str(x).strip()][:3]
            if len(kws) < 3:
                return None
            prompt_en = str(parsed.get("background_prompt_en", "")).strip()
            neg_en = str(parsed.get("background_negative_en", "")).strip()
            logger.info(
                "mvp_llm_full request_id={} keywords={} prompt_len={}",
                request_id,
                kws,
                len(prompt_en),
            )
            return {
                "keywords": kws[:3],
                "background_prompt_en": prompt_en,
                "background_negative_en": neg_en,
            }
        except Exception as e:
            logger.warning("mvp_llm_full_failed request_id={} error={}", request_id, str(e))
            return None

    def _llm_background_prompt_only(
        self,
        stats: dict[str, Any],
        product_name: str,
        product_desc: str,
        cn_keywords: list[str],
        request_id: str,
    ) -> dict[str, str] | None:
        """在用户已指定中文风格词时，仅向 LLM 索取英文背景 prompt。"""
        if not self.api_key:
            return None
        system_prompt = (
            "只输出一个 JSON：{\"background_prompt_en\":\"\",\"background_negative_en\":\"\"}。"
            "background_prompt_en 为英文，描述无主体的商品拍摄背景，与给定中文风格一致；"
            "无文字、无水印。background_negative_en 为英文负面词。"
        )
        user_content = (
            f"商品图统计：主色RGB={stats.get('dominant_rgb')}, 亮度={stats.get('mean_brightness')}, "
            f"饱和度={stats.get('mean_saturation')}。\n"
            f"商品名称：{product_name}\n商品描述：{product_desc}\n"
            f"已定中文风格词（逗号分隔）：{','.join(cn_keywords)}\n"
            "请根据以上生成背景绘图英文 prompt。"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.35,
                stream=False,
            )
            content = response.choices[0].message.content or ""
            parsed = self._extract_json(content)
            pe = str(parsed.get("background_prompt_en", "")).strip()
            ne = str(parsed.get("background_negative_en", "")).strip()
            if not pe:
                return None
            logger.info("mvp_llm_bg_only request_id={} prompt_len={}", request_id, len(pe))
            return {"background_prompt_en": pe, "background_negative_en": ne}
        except Exception as e:
            logger.warning("mvp_llm_bg_only_failed request_id={} error={}", request_id, str(e))
            return None

    def build_mvp_plan(
        self,
        product_name: str,
        product_desc: str,
        image_paths: list[str],
        layout_id: str | None,
        canvas_size: int,
        style_keywords_override_cn: list[str] | None,
        request_id: str | None = None,
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        """
        MVP：多图统计 + 受控中文风格词（词表 + Gemini/DeepSeek）+ 拼接方案（LLM 二选一）+ 背景英文 prompt。
        """
        rid = request_id or str(uuid.uuid4())
        paths = [p for p in (image_paths or []) if p and str(p).strip()]
        if not paths:
            stats = analyze_product_image("")
            image_count = 0
        else:
            stats = analyze_product_images(paths)
            image_count = int(stats.get("image_count") or len(paths))

        stats_summary = (
            f"dominant_rgb={stats.get('dominant_rgb')},brightness={stats.get('mean_brightness')},"
            f"saturation={stats.get('mean_saturation')},size={stats.get('width')}x{stats.get('height')},"
            f"image_count={image_count}"
        )

        stitch_mode, distribution, stitch_provider = run_stitch_decision_llm(
            stats_summary=stats_summary,
            product_name=product_name,
            product_desc=product_desc,
            image_count=max(1, image_count),
            request_id=rid,
        )

        mvp_bg: dict[str, str] = {"prompt_en": "", "negative_en": ""}
        style_source = "heuristic"
        vocab = load_mvp_style_vocab()

        if style_keywords_override_cn:
            cn_keywords = self._pad_three_cn_keywords(list(style_keywords_override_cn))
            style_source = "user"
            bg_only = self._llm_background_prompt_only(
                stats, product_name, product_desc, cn_keywords, rid
            )
            if bg_only:
                mvp_bg["prompt_en"] = bg_only.get("background_prompt_en", "")
                mvp_bg["negative_en"] = bg_only.get("background_negative_en", "")
            if not mvp_bg["prompt_en"]:
                mvp_bg["prompt_en"] = self._fallback_background_prompt_en(cn_keywords, stats)
        elif vocab:
            if any_text_llm_configured():
                try:
                    kws, prov_used = run_style_constrained_llm(
                        vocab=vocab,
                        stats_summary=stats_summary,
                        product_name=product_name,
                        product_desc=product_desc,
                        image_count=max(1, image_count),
                        request_id=rid,
                    )
                except MvpLlmPipelineError:
                    raise
                cn_keywords = self._pad_three_cn_keywords(kws)
                style_source = f"llm_{prov_used}"
            else:
                cn_keywords = snap_keywords_to_vocab(heuristic_style_keywords_cn(stats), vocab)
                style_source = "heuristic_vocab"
            bg_only = self._llm_background_prompt_only(
                stats, product_name, product_desc, cn_keywords, rid
            )
            if bg_only and bg_only.get("background_prompt_en"):
                mvp_bg["prompt_en"] = bg_only["background_prompt_en"]
                mvp_bg["negative_en"] = bg_only.get("background_negative_en", "")
                if style_source == "heuristic_vocab":
                    style_source = "heuristic_vocab+llm_bg"
            if not mvp_bg["prompt_en"]:
                mvp_bg["prompt_en"] = self._fallback_background_prompt_en(cn_keywords, stats)
        else:
            full: dict[str, Any] | None = None
            for _ in range(3):
                full = self._llm_full_mvp_analysis(stats, product_name, product_desc, rid)
                if full and isinstance(full.get("keywords"), list) and len(full["keywords"]) == 3:
                    break
            if full and len(full.get("keywords", [])) == 3:
                cn_keywords = full["keywords"][:3]
                mvp_bg["prompt_en"] = (
                    str(full.get("background_prompt_en", "")).strip()
                    or self._fallback_background_prompt_en(cn_keywords, stats)
                )
                mvp_bg["negative_en"] = str(full.get("background_negative_en", "")).strip()
                style_source = "llm"
            else:
                cn_keywords = heuristic_style_keywords_cn(stats)
                bg_only = self._llm_background_prompt_only(
                    stats, product_name, product_desc, cn_keywords, rid
                )
                if bg_only and bg_only.get("background_prompt_en"):
                    mvp_bg["prompt_en"] = bg_only["background_prompt_en"]
                    mvp_bg["negative_en"] = bg_only.get("background_negative_en", "")
                    style_source = "heuristic+llm_bg"
                else:
                    mvp_bg["prompt_en"] = self._fallback_background_prompt_en(cn_keywords, stats)
                    style_source = "heuristic"

        template = get_layout_template(layout_id)
        en_tags = self._cn_keywords_to_en_tags(cn_keywords)

        plan = self.get_visual_plan(
            product_name=product_name,
            product_desc=product_desc,
            canvas_size=canvas_size,
            request_id=rid,
        )
        plan["layout"] = template_to_layout_dict(template, canvas_size)
        plan["style"]["tags"] = en_tags
        plan["style"]["reason"] = f"mvp_image_stats:{json.dumps(stats, ensure_ascii=False)}"
        conf = 0.65
        if style_keywords_override_cn:
            conf = 0.9
        elif style_source.startswith("llm_"):
            conf = 0.86
        elif style_source.startswith("heuristic_vocab"):
            conf = 0.7
        elif style_source == "llm":
            conf = 0.82
        plan["style"]["confidence"] = conf
        plan["style"]["source"] = f"mvp_{style_source}"
        candidates = self.background_retriever.retrieve(
            style_tags=en_tags,
            product_name=product_name,
            product_desc=product_desc,
            top_k=3,
        )
        plan["background"]["candidates"] = [
            {
                "background_id": c.background_id,
                "path": c.path,
                "tags": c.tags,
                "score": c.score,
            }
            for c in candidates
        ]
        neg_hint = mvp_bg.get("negative_en", "")
        plan["background"]["note"] = (
            f"style_cn:{','.join(cn_keywords)}; bg_prompt_en:{mvp_bg.get('prompt_en', '')[:240]}; "
            f"{plan.get('background', {}).get('note', '')}"
        ).strip("; ")
        if neg_hint:
            plan["background"]["negative_keywords"] = list(
                dict.fromkeys(
                    list(plan.get("background", {}).get("negative_keywords", []) or [])
                    + [neg_hint[:200]]
                )
            )
        plan["_mvp_layout"] = {
            "template_id": template.id,
            "label_zh": template.label_zh,
            "product_box": list(template.product_box),
            "title_anchor": template.title_anchor,
        }
        plan["_mvp_stitch"] = {
            "stitch_mode": stitch_mode,
            "distribution": distribution,
            "provider": stitch_provider,
            "image_count": max(1, image_count),
        }
        plan["_mvp_bg_prompt"] = {
            "prompt_en": mvp_bg.get("prompt_en", ""),
            "negative_en": mvp_bg.get("negative_en", ""),
        }
        plan["_mvp_style_source"] = style_source
        return plan, cn_keywords, stats

    @staticmethod
    def strip_mvp_internals(plan: dict[str, Any]) -> dict[str, Any]:
        """供 API 响应序列化前移除内部渲染字段。"""
        out = {k: v for k, v in plan.items() if not str(k).startswith("_")}
        return out

# 测试代码（仅在直接运行此文件时执行）
if __name__ == "__main__":
    brain = YayoiBrain()
    print(brain.get_visual_plan("樱花和纸胶带", "带有人间四月天意象的粉色半透明胶带", 2048))
