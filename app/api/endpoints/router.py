import os
import uuid
from typing import Annotated, Any, Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from app.core.ai_logic import YayoiBrain
from app.services.layout_templates import LAYOUT_TEMPLATES, list_layout_templates
from app.services.gemini_image_service import GeminiImageClient
from app.services.nanobanana_service import NanobananaBackgroundClient
from app.services.render_service import PosterRenderer
from app.services.stable_diffusion_service import StableDiffusionClient
from app.services.background_raster_chain import try_generate_background_raster, try_generate_promo_raster
from app.services.mvp_llm_pipeline import MvpLlmPipelineError, build_promo_image_prompt_en

router = APIRouter()
brain = YayoiBrain()
renderer = PosterRenderer()
nanobanana = NanobananaBackgroundClient()
stable_sd = StableDiffusionClient()
gemini_image = GeminiImageClient()
UPLOAD_DIR = "./app/static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "10485760"))  # 10MB default
ALLOWED_UPLOAD_TYPES = {"image/png", "image/jpeg", "image/webp"}


class PosterPlanRequest(BaseModel):
    product_name: str = Field(..., description="产品名称")
    product_desc: str = Field(..., description="产品描述")
    canvas_size: Optional[int] = Field(default=2048, description="画布尺寸")


class StylePlan(BaseModel):
    tags: list[str]
    confidence: float
    reason: str
    source: str = "unknown"


class LayoutPlan(BaseModel):
    canvas_size: int
    composition: str
    text_area: str
    safe_margin_percent: int


class BackgroundPlan(BaseModel):
    mode: str
    keywords: list[str]
    negative_keywords: list[str]
    note: str
    candidates: list[dict] = Field(default_factory=list)


class PosterPlanResponse(BaseModel):
    request_id: str
    style: StylePlan
    layout: LayoutPlan
    background: BackgroundPlan


class PosterReviewRequest(BaseModel):
    style: StylePlan
    layout: LayoutPlan
    background: BackgroundPlan


class PosterReviewResponse(BaseModel):
    request_id: str
    score: int
    level: str
    issues: list[str]
    suggestions: list[str]
    source: str
    confidence: float


class PosterRenderRequest(BaseModel):
    product_name: str = Field(..., description="产品名称")
    product_desc: str = Field(..., description="产品描述")
    tagline: str = Field(default="新品推荐", description="副标题文案")
    canvas_size: Optional[int] = Field(default=2048, description="画布尺寸")
    product_image_path: Optional[str] = Field(default=None, description="商品图本地路径（可选）")


class PosterRenderResponse(BaseModel):
    request_id: str
    image_path: str
    thumbnail_path: str
    width: int
    height: int
    background_hit: bool
    product_pasted: bool
    render_ms: int
    plan: PosterPlanResponse


class PosterPipelineResponse(BaseModel):
    request_id: str
    plan: PosterPlanResponse
    review: PosterReviewResponse


class HealthResponse(BaseModel):
    status: str
    service: str


class ConfigResponse(BaseModel):
    request_id: str
    config: dict


class MvpVariantView(BaseModel):
    variant_index: int
    image_path: str
    thumbnail_path: str
    background_hit: bool
    product_pasted: bool
    render_ms: int


class MvpRunResponse(BaseModel):
    request_id: str
    style_keywords_cn: list[str]
    style_source: str = Field(
        default="",
        description="llm | user | heuristic | heuristic+llm_bg",
    )
    stitch_mode: str = Field(default="", description="single_main_aux | multi_equal_grid")
    stitch_distribution: str = Field(default="", description="预设 distribution 标识")
    stitch_decision_provider: str = Field(default="", description="拼接方案 LLM 通道或 fallback")
    background_image_prompt_en: str = Field(
        default="",
        description="英文背景 prompt，供 Nanobanana 或外接图生图使用",
    )
    background_negative_en: str = ""
    image_stats: dict
    layout_id: str
    layout_label_zh: str
    available_layouts: list[dict[str, str]]
    saved_product_image: str = Field(default="", description="兼容：首张上传图 basename")
    saved_product_images: list[str] = Field(
        default_factory=list,
        description="本次参与生成的全部上传图 basename，按上传顺序",
    )
    promo_raster_generated: bool = Field(
        default=False,
        description="是否已用 Nanobanana 生成宣传条漫位图（需配置接口且填写宣传文案）",
    )
    promo_raster_source: str = Field(
        default="",
        description="宣传字块实际使用的绘图通道：nanobanana | stable_diffusion | none",
    )
    background_generation_attempts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="背景位图各通道尝试结果（便于排查 Nanobanana/Gemini/SD 是否真正返回了图）",
    )
    background_api_configured: bool = False
    background_raster_source: str = Field(
        default="none",
        description="背景位图最终来源：none | nanobanana | stable_diffusion | gemini（顺序由 MVP_BACKGROUND_RASTER_ORDER 控制）",
    )
    gemini_image_model: str = Field(default="", description="绘图时使用的 GEMINI_IMAGE_MODEL")
    variants: list[MvpVariantView]
    plan: PosterPlanResponse
    review: PosterReviewResponse


def _resolve_request_id(x_request_id: Optional[str]) -> str:
    return x_request_id or str(uuid.uuid4())


def _to_public_static_url(path: str) -> str:
    normalized = path.replace("\\", "/")
    marker = "app/static/"
    idx = normalized.find(marker)
    if idx >= 0:
        return "/static/" + normalized[idx + len(marker):]
    if normalized.startswith("/static/"):
        return normalized
    return normalized


def _parse_style_keywords_override(raw: Optional[str]) -> Optional[list[str]]:
    if not raw or not str(raw).strip():
        return None
    parts = [p.strip() for p in str(raw).replace("，", ",").split(",")]
    return [p for p in parts if p][:3] or None


def _parse_saved_basenames(saved_product_images: Optional[str], saved_product_image: Optional[str]) -> list[str]:
    if saved_product_images and str(saved_product_images).strip():
        return [p.strip() for p in str(saved_product_images).split(",") if p.strip()]
    if saved_product_image and str(saved_product_image).strip():
        return [str(saved_product_image).strip()]
    raise HTTPException(status_code=400, detail="missing_saved_product_images")


def _validate_saved_upload_basename(basename: str) -> str:
    if not basename or basename.strip() != basename:
        raise HTTPException(status_code=400, detail="invalid_saved_image")
    if any(sep in basename for sep in ("/", "\\")):
        raise HTTPException(status_code=400, detail="invalid_saved_image")
    if basename.startswith("."):
        raise HTTPException(status_code=400, detail="invalid_saved_image")
    upload_root = os.path.abspath(UPLOAD_DIR)
    full = os.path.abspath(os.path.join(UPLOAD_DIR, basename))
    try:
        common = os.path.commonpath([upload_root, full])
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid_saved_image")
    if common != upload_root:
        raise HTTPException(status_code=400, detail="invalid_saved_image")
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="upload_not_found")
    return full


def _mvp_run_core(
    rid: str,
    product_name: str,
    product_desc: str,
    tagline: str,
    canvas_size: int,
    layout_id: Optional[str],
    local_paths: list[str],
    saved_basenames: list[str],
    style_keywords_override: Optional[str],
    promo_copy: str = "",
) -> MvpRunResponse:
    override_list = _parse_style_keywords_override(style_keywords_override)
    try:
        plan_raw, cn_keywords, stats = brain.build_mvp_plan(
            product_name=product_name,
            product_desc=product_desc,
            image_paths=local_paths,
            layout_id=layout_id,
            canvas_size=canvas_size,
            style_keywords_override_cn=override_list,
            request_id=rid,
        )
    except MvpLlmPipelineError as e:
        raise HTTPException(status_code=502, detail=e.to_detail()) from e

    mvp_meta = plan_raw.get("_mvp_layout") or {}
    layout_resolved = str(mvp_meta.get("template_id", layout_id or "classic_center"))
    label_zh = str(mvp_meta.get("label_zh", ""))
    stitch_meta = plan_raw.get("_mvp_stitch") or {}

    plan_for_api = YayoiBrain.strip_mvp_internals(plan_raw)
    plan_response = PosterPlanResponse(
        request_id=rid,
        style=StylePlan(**plan_for_api.get("style", {})),
        layout=LayoutPlan(**plan_for_api.get("layout", {})),
        background=BackgroundPlan(**plan_for_api.get("background", {})),
    )
    review = brain.review_visual_plan(plan=plan_response.model_dump(), request_id=rid)
    review_response = PosterReviewResponse(request_id=rid, **review)

    mvp_bg = plan_raw.get("_mvp_bg_prompt") or {}
    prompt_en = str(mvp_bg.get("prompt_en", "")).strip()
    neg_en = str(mvp_bg.get("negative_en", "")).strip()
    style_src = str(plan_raw.get("_mvp_style_source", ""))
    prompt_for_api = prompt_en
    if neg_en:
        prompt_for_api = f"{prompt_en} Avoid: {neg_en}"

    gemini_model_display = gemini_image.image_model if gemini_image.is_configured() else ""

    promo_strip = (promo_copy or "").strip()
    promo_bytes: bytes | None = None
    promo_raster_generated = False
    promo_raster_source = ""
    if promo_strip:
        pep = build_promo_image_prompt_en(promo_strip, cn_keywords)
        pw = min(2048, max(768, int(canvas_size * 1.45)))
        ph = max(256, min(1024, int(canvas_size * 0.24)))
        promo_bytes, promo_raster_source = try_generate_promo_raster(
            prompt_en=pep,
            canvas_size=canvas_size,
            pixel_width=pw,
            pixel_height=ph,
            variant_index=0,
            nanobanana=nanobanana,
            stable_diffusion=stable_sd,
        )
        promo_raster_generated = bool(promo_bytes)

    variants_out: list[MvpVariantView] = []
    primary_local = local_paths[0] if local_paths else ""
    bg_attempts_first: list[dict[str, Any]] = []
    first_variant_raster_source = "none"
    for v in range(3):
        bg_bytes = None
        if prompt_for_api.strip():
            bg_bytes, used_src, attempts = try_generate_background_raster(
                prompt_for_api=prompt_for_api,
                negative_en=neg_en,
                canvas_size=canvas_size,
                variant_index=v,
                nanobanana=nanobanana,
                stable_diffusion=stable_sd,
                gemini_image=gemini_image,
            )
            if v == 0:
                bg_attempts_first = attempts
                first_variant_raster_source = used_src if bg_bytes else "none"
        render_out = renderer.render_mvp(
            plan=plan_raw,
            product_name=product_name,
            tagline=tagline,
            request_id=rid,
            product_image_path=primary_local,
            variant_index=v,
            image_stats=stats,
            background_raster_bytes=bg_bytes,
            product_image_paths=local_paths,
            promo_image_bytes=promo_bytes,
            promo_copy=promo_strip,
        )
        variants_out.append(
            MvpVariantView(
                variant_index=v,
                image_path=_to_public_static_url(render_out["image_path"]),
                thumbnail_path=_to_public_static_url(render_out["thumbnail_path"]),
                background_hit=bool(render_out.get("background_hit")),
                product_pasted=bool(render_out.get("product_pasted")),
                render_ms=int(render_out.get("render_ms", 0)),
            )
        )

    first_saved = saved_basenames[0] if saved_basenames else ""
    return MvpRunResponse(
        request_id=rid,
        style_keywords_cn=cn_keywords,
        style_source=style_src,
        stitch_mode=str(stitch_meta.get("stitch_mode", "")),
        stitch_distribution=str(stitch_meta.get("distribution", "")),
        stitch_decision_provider=str(stitch_meta.get("provider", "")),
        background_image_prompt_en=prompt_en,
        background_negative_en=neg_en,
        image_stats=stats,
        layout_id=layout_resolved,
        layout_label_zh=label_zh,
        available_layouts=list_layout_templates(),
        saved_product_image=first_saved,
        saved_product_images=saved_basenames,
        promo_raster_generated=promo_raster_generated,
        promo_raster_source=promo_raster_source,
        background_generation_attempts=bg_attempts_first,
        background_api_configured=nanobanana.is_configured()
        or stable_sd.is_configured()
        or gemini_image.is_configured(),
        background_raster_source=first_variant_raster_source,
        gemini_image_model=gemini_model_display,
        variants=variants_out,
        plan=plan_response,
        review=review_response,
    )


@router.get("/test")
async def test():
    return {"msg": "endpoints router is working"}


@router.post("/poster/plan", response_model=PosterPlanResponse)
async def poster_plan(payload: PosterPlanRequest, x_request_id: Optional[str] = Header(default=None)):
    rid = _resolve_request_id(x_request_id)
    plan = brain.get_visual_plan(
        product_name=payload.product_name,
        product_desc=payload.product_desc,
        canvas_size=payload.canvas_size or 2048,
        request_id=rid,
    )

    return PosterPlanResponse(
        request_id=rid,
        style=StylePlan(**plan.get("style", {})),
        layout=LayoutPlan(**plan.get("layout", {})),
        background=BackgroundPlan(**plan.get("background", {})),
    )


@router.post("/poster/review", response_model=PosterReviewResponse)
async def poster_review(payload: PosterReviewRequest, x_request_id: Optional[str] = Header(default=None)):
    rid = _resolve_request_id(x_request_id)
    review = brain.review_visual_plan(
        plan=payload.model_dump(),
        request_id=rid,
    )
    return PosterReviewResponse(request_id=rid, **review)


@router.post("/poster/pipeline", response_model=PosterPipelineResponse)
async def poster_pipeline(payload: PosterPlanRequest, x_request_id: Optional[str] = Header(default=None)):
    rid = _resolve_request_id(x_request_id)
    plan = brain.get_visual_plan(
        product_name=payload.product_name,
        product_desc=payload.product_desc,
        canvas_size=payload.canvas_size or 2048,
        request_id=rid,
    )

    plan_response = PosterPlanResponse(
        request_id=rid,
        style=StylePlan(**plan.get("style", {})),
        layout=LayoutPlan(**plan.get("layout", {})),
        background=BackgroundPlan(**plan.get("background", {})),
    )

    review = brain.review_visual_plan(
        plan=plan_response.model_dump(),
        request_id=rid,
    )
    review_response = PosterReviewResponse(request_id=rid, **review)
    return PosterPipelineResponse(request_id=rid, plan=plan_response, review=review_response)


@router.post("/poster/render", response_model=PosterRenderResponse)
async def poster_render(payload: PosterRenderRequest, x_request_id: Optional[str] = Header(default=None)):
    rid = _resolve_request_id(x_request_id)
    plan = brain.get_visual_plan(
        product_name=payload.product_name,
        product_desc=payload.product_desc,
        canvas_size=payload.canvas_size or 2048,
        request_id=rid,
    )
    plan_response = PosterPlanResponse(
        request_id=rid,
        style=StylePlan(**plan.get("style", {})),
        layout=LayoutPlan(**plan.get("layout", {})),
        background=BackgroundPlan(**plan.get("background", {})),
    )
    render_out = renderer.render(
        plan=plan_response.model_dump(),
        product_name=payload.product_name,
        tagline=payload.tagline,
        request_id=rid,
        product_image_path=payload.product_image_path,
    )
    render_out["image_path"] = _to_public_static_url(render_out["image_path"])
    render_out["thumbnail_path"] = _to_public_static_url(render_out["thumbnail_path"])
    return PosterRenderResponse(request_id=rid, plan=plan_response, **render_out)


@router.post("/poster/render/upload", response_model=PosterRenderResponse)
async def poster_render_upload(
    product_name: str = Form(...),
    product_desc: str = Form(...),
    tagline: str = Form("新品推荐"),
    canvas_size: int = Form(2048),
    product_image: UploadFile = File(...),
    x_request_id: Optional[str] = Header(default=None),
):
    rid = _resolve_request_id(x_request_id)
    if product_image.content_type not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(status_code=400, detail="unsupported_file_type")

    safe_name = product_image.filename or "product.png"
    local_name = f"{rid}_{safe_name}".replace(" ", "_")
    local_path = os.path.join(UPLOAD_DIR, local_name)

    content = await product_image.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="file_too_large")
    with open(local_path, "wb") as f:
        f.write(content)

    plan = brain.get_visual_plan(
        product_name=product_name,
        product_desc=product_desc,
        canvas_size=canvas_size,
        request_id=rid,
    )
    plan_response = PosterPlanResponse(
        request_id=rid,
        style=StylePlan(**plan.get("style", {})),
        layout=LayoutPlan(**plan.get("layout", {})),
        background=BackgroundPlan(**plan.get("background", {})),
    )
    render_out = renderer.render(
        plan=plan_response.model_dump(),
        product_name=product_name,
        tagline=tagline,
        request_id=rid,
        product_image_path=local_path,
    )
    render_out["image_path"] = _to_public_static_url(render_out["image_path"])
    render_out["thumbnail_path"] = _to_public_static_url(render_out["thumbnail_path"])
    return PosterRenderResponse(request_id=rid, plan=plan_response, **render_out)


@router.post("/poster/mvp/run", response_model=MvpRunResponse)
async def poster_mvp_run(
    product_images: Annotated[Optional[list[UploadFile]], File(description="一张或多张商品图")] = None,
    product_image: Annotated[Optional[UploadFile], File(description="单张商品图（兼容旧字段名）")] = None,
    product_name: str = Form(""),
    product_desc: str = Form(""),
    tagline: str = Form("新品推荐"),
    promo_copy: str = Form(""),
    canvas_size: int = Form(1024),
    layout_id: str = Form("classic_center"),
    style_keywords_override: Optional[str] = Form(None),
    x_request_id: Optional[str] = Header(default=None),
):
    """
    MVP 闭环：上传一张或多张商品图 → 受控风格词 + 拼接方案 + 背景（优先 Nanobanana）+ 可选宣传条漫 + 三版预览。
    multipart 字段名可为 **product_images**（可重复）或旧版 **product_image**（单文件），二者择一即可。
    """
    rid = _resolve_request_id(x_request_id)
    upload_parts: list[UploadFile] = []
    if product_images:
        upload_parts.extend(product_images)
    elif product_image is not None:
        upload_parts.append(product_image)
    if not upload_parts:
        raise HTTPException(status_code=400, detail="no_product_images")
    if layout_id and layout_id not in LAYOUT_TEMPLATES:
        raise HTTPException(status_code=400, detail="invalid_layout_id")

    local_paths: list[str] = []
    saved_basenames: list[str] = []
    for idx, product_image in enumerate(upload_parts):
        if product_image.content_type not in ALLOWED_UPLOAD_TYPES:
            raise HTTPException(status_code=400, detail="unsupported_file_type")
        safe_name = product_image.filename or f"product_{idx}.png"
        local_name = f"{rid}_{idx}_{safe_name}".replace(" ", "_")
        local_path = os.path.join(UPLOAD_DIR, local_name)
        content = await product_image.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="file_too_large")
        with open(local_path, "wb") as f:
            f.write(content)
        local_paths.append(local_path)
        saved_basenames.append(local_name)

    return _mvp_run_core(
        rid=rid,
        product_name=product_name or "未命名商品",
        product_desc=product_desc or " ",
        tagline=tagline,
        canvas_size=canvas_size,
        layout_id=layout_id,
        local_paths=local_paths,
        saved_basenames=saved_basenames,
        style_keywords_override=style_keywords_override,
        promo_copy=promo_copy,
    )


@router.post("/poster/mvp/regenerate", response_model=MvpRunResponse)
async def poster_mvp_regenerate(
    saved_product_images: Optional[str] = Form(None),
    saved_product_image: Optional[str] = Form(None),
    product_name: str = Form(""),
    product_desc: str = Form(""),
    tagline: str = Form("新品推荐"),
    promo_copy: str = Form(""),
    canvas_size: int = Form(1024),
    layout_id: str = Form("classic_center"),
    style_keywords_override: Optional[str] = Form(None),
    x_request_id: Optional[str] = Header(default=None),
):
    """
    对已保存的上传图再次出图（微调风格词或排版），无需重新上传整图文件。
    saved_product_images：逗号分隔的多个 basename；兼容旧字段 saved_product_image（单张）。
    """
    rid = _resolve_request_id(x_request_id)
    if layout_id and layout_id not in LAYOUT_TEMPLATES:
        raise HTTPException(status_code=400, detail="invalid_layout_id")
    basenames = _parse_saved_basenames(saved_product_images, saved_product_image)
    local_paths = [_validate_saved_upload_basename(b) for b in basenames]
    saved_basenames = [os.path.basename(p) for p in local_paths]
    return _mvp_run_core(
        rid=rid,
        product_name=product_name or "未命名商品",
        product_desc=product_desc or " ",
        tagline=tagline,
        canvas_size=canvas_size,
        layout_id=layout_id,
        local_paths=local_paths,
        saved_basenames=saved_basenames,
        style_keywords_override=style_keywords_override,
        promo_copy=promo_copy,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", service="misheng-poster-agent")


@router.get("/config", response_model=ConfigResponse)
async def config(x_request_id: Optional[str] = Header(default=None)):
    rid = _resolve_request_id(x_request_id)
    return ConfigResponse(request_id=rid, config=brain.get_public_config())
