from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.api.main import app  # noqa: E402
from app.api.endpoints import router as router_module  # noqa: E402


client = TestClient(app)


@pytest.fixture(autouse=True)
def _stub_external_background_raster(monkeypatch: pytest.MonkeyPatch) -> None:
    """单元测试不请求真实 Nanobanana / Gemini / SD，避免慢与 429。"""

    def _fake_bg(**kwargs: object) -> tuple[None, str, list]:
        return None, "none", [{"source": "stub", "ok": False, "reason": "unit_test_no_network"}]

    def _fake_pr(**kwargs: object) -> tuple[None, str]:
        return None, "none"

    monkeypatch.setattr(router_module, "try_generate_background_raster", _fake_bg)
    monkeypatch.setattr(router_module, "try_generate_promo_raster", _fake_pr)


def _mock_plan(*args, **kwargs):
    canvas_size = kwargs.get("canvas_size", 2048)
    return {
        "style": {"tags": ["clean"], "confidence": 0.9, "reason": "mocked", "source": "mock"},
        "layout": {
            "canvas_size": canvas_size,
            "composition": "product_center_bottom",
            "text_area": "top_left",
            "safe_margin_percent": 8,
        },
        "background": {
            "mode": "retrieve_or_generate",
            "keywords": ["soft light"],
            "negative_keywords": ["busy pattern"],
            "note": "mocked",
            "candidates": [
                {
                    "background_id": "bg-clean-paper-001",
                    "path": "assets/backgrounds/clean-paper-001.jpg",
                    "tags": ["clean"],
                    "score": 0.88,
                }
            ],
        },
    }


def _mock_review(*args, **kwargs):
    return {
        "score": 88,
        "level": "good",
        "issues": [],
        "suggestions": [],
        "source": "rule",
        "confidence": 0.9,
    }


def _mock_render(*args, **kwargs):
    return {
        "image_path": "/static/results/poster_mock.png",
        "thumbnail_path": "/static/results/poster_mock_thumb.png",
        "width": 2048,
        "height": 2048,
        "background_hit": True,
        "product_pasted": True,
        "render_ms": 123,
    }


def test_poster_plan_schema():
    router_module.brain.get_visual_plan = _mock_plan
    resp = client.post(
        "/v1/poster/plan",
        headers={"x-request-id": "test-plan-001"},
        json={
            "product_name": "test",
            "product_desc": "test desc",
            "canvas_size": 2048,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "test-plan-001"
    assert "style" in data and "layout" in data and "background" in data
    assert "source" in data["style"]
    assert "candidates" in data["background"]


def test_poster_review_schema():
    router_module.brain.review_visual_plan = _mock_review
    resp = client.post(
        "/v1/poster/review",
        headers={"x-request-id": "test-review-001"},
        json={
            "style": {"tags": ["clean"], "confidence": 0.9, "reason": "ok"},
            "layout": {
                "canvas_size": 2048,
                "composition": "product_center_bottom",
                "text_area": "top_left",
                "safe_margin_percent": 8,
            },
            "background": {
                "mode": "retrieve_or_generate",
                "keywords": ["soft light"],
                "negative_keywords": ["busy pattern"],
                "note": "ok",
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "test-review-001"
    assert "score" in data and "source" in data and "confidence" in data


def test_poster_pipeline_schema():
    router_module.brain.get_visual_plan = _mock_plan
    router_module.brain.review_visual_plan = _mock_review
    resp = client.post(
        "/v1/poster/pipeline",
        headers={"x-request-id": "test-pipeline-001"},
        json={
            "product_name": "test",
            "product_desc": "test desc",
            "canvas_size": 2048,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "test-pipeline-001"
    assert "plan" in data and "review" in data


def test_poster_render_schema():
    router_module.brain.get_visual_plan = _mock_plan
    router_module.renderer.render = _mock_render
    resp = client.post(
        "/v1/poster/render",
        headers={"x-request-id": "test-render-001"},
        json={
            "product_name": "test",
            "product_desc": "test desc",
            "tagline": "新品推荐",
            "canvas_size": 2048,
            "product_image_path": "app/static/uploads/test.png",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "test-render-001"
    assert data["image_path"].startswith("/static/")
    assert data["thumbnail_path"].startswith("/static/")
    assert "plan" in data
    assert "product_pasted" in data
    assert "render_ms" in data


def test_poster_render_upload_schema():
    router_module.brain.get_visual_plan = _mock_plan
    router_module.renderer.render = _mock_render

    tmp_img = ROOT / "app" / "static" / "uploads" / "tmp_test_upload.png"
    tmp_img.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64), color=(255, 0, 0, 255)).save(tmp_img)

    with open(tmp_img, "rb") as f:
        resp = client.post(
            "/v1/poster/render/upload",
            headers={"x-request-id": "test-render-upload-001"},
            data={
                "product_name": "test",
                "product_desc": "test desc",
                "tagline": "新品推荐",
                "canvas_size": "2048",
            },
            files={"product_image": ("tmp_test_upload.png", f, "image/png")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "test-render-upload-001"
    assert data["image_path"].startswith("/static/")
    assert "product_pasted" in data
    assert data["thumbnail_path"].startswith("/static/")
    assert "render_ms" in data


def _mock_mvp_plan(*args, **kwargs):
    canvas = int(kwargs.get("canvas_size", 1024))
    return (
        {
            "style": {"tags": ["clean"], "confidence": 0.8, "reason": "mvp_mock", "source": "mvp_mock"},
            "layout": {
                "canvas_size": canvas,
                "composition": "product_center_bottom",
                "text_area": "top_left",
                "safe_margin_percent": 8,
            },
            "background": {
                "mode": "retrieve_or_generate",
                "keywords": ["soft light"],
                "negative_keywords": ["busy pattern"],
                "note": "mvp_mock",
                "candidates": [],
            },
            "_mvp_layout": {
                "template_id": "classic_center",
                "label_zh": "经典居中",
                "product_box": [0.18, 0.34, 0.64, 0.52],
                "title_anchor": "top_left",
            },
            "_mvp_stitch": {
                "stitch_mode": "single_main_aux",
                "distribution": "single_main_bottom_aux_top_strip",
                "provider": "mock",
                "image_count": 1,
            },
            "_mvp_bg_prompt": {
                "prompt_en": "mock minimal studio backdrop, no text",
                "negative_en": "letters",
            },
            "_mvp_style_source": "llm",
        },
        ["清新", "明亮", "简约"],
        {
            "dominant_rgb": [200, 200, 200],
            "mean_brightness": 180.0,
            "mean_saturation": 20.0,
            "width": 64,
            "height": 64,
            "source": "mock",
        },
    )


def test_poster_mvp_run_schema():
    router_module.brain.build_mvp_plan = _mock_mvp_plan
    router_module.brain.review_visual_plan = _mock_review

    tmp_img = ROOT / "app" / "static" / "uploads" / "tmp_mvp_test.png"
    tmp_img.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (80, 80), color=(10, 120, 200, 255)).save(tmp_img)

    with open(tmp_img, "rb") as f:
        resp = client.post(
            "/v1/poster/mvp/run",
            headers={"x-request-id": "mvp-run-001"},
            data={
                "product_name": "测试贴纸",
                "product_desc": "治愈系植物贴纸",
                "tagline": "新品推荐",
                "canvas_size": "640",
                "layout_id": "classic_center",
            },
            files=[("product_images", ("tmp_mvp_test.png", f, "image/png"))],
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "mvp-run-001"
    assert len(data["style_keywords_cn"]) == 3
    assert data["layout_id"] == "classic_center"
    assert len(data["variants"]) == 3
    for i, v in enumerate(data["variants"]):
        assert v["variant_index"] == i
        assert v["image_path"].startswith("/static/")
        assert v["thumbnail_path"].startswith("/static/")
    assert "plan" in data and "review" in data
    assert data["saved_product_image"]
    assert data.get("style_source") == "llm"
    assert data.get("stitch_mode") == "single_main_aux"
    assert data.get("saved_product_images")
    assert isinstance(data.get("background_generation_attempts"), list)
    assert "mock minimal" in (data.get("background_image_prompt_en") or "")


def test_poster_mvp_run_accepts_legacy_product_image_field_name():
    """旧客户端只传 product_image 时不应再 422。"""
    router_module.brain.build_mvp_plan = _mock_mvp_plan
    router_module.brain.review_visual_plan = _mock_review

    tmp_img = ROOT / "app" / "static" / "uploads" / "tmp_mvp_legacy_field.png"
    tmp_img.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (80, 80), color=(10, 120, 200, 255)).save(tmp_img)

    with open(tmp_img, "rb") as f:
        resp = client.post(
            "/v1/poster/mvp/run",
            headers={"x-request-id": "mvp-run-legacy-001"},
            data={"product_name": "测试", "product_desc": "描述", "canvas_size": "640", "layout_id": "classic_center"},
            files={"product_image": ("tmp_mvp_legacy_field.png", f, "image/png")},
        )
    assert resp.status_code == 200
    assert resp.json()["request_id"] == "mvp-run-legacy-001"


def test_poster_mvp_regenerate_schema():
    router_module.brain.build_mvp_plan = _mock_mvp_plan
    router_module.brain.review_visual_plan = _mock_review

    tmp_img = ROOT / "app" / "static" / "uploads" / "mvp_regen_saved.png"
    tmp_img.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (32, 32), color=(255, 0, 0, 255)).save(tmp_img)

    resp = client.post(
        "/v1/poster/mvp/regenerate",
        headers={"x-request-id": "mvp-regen-001"},
        data={
            "saved_product_image": tmp_img.name,
            "product_name": "测试",
            "product_desc": "描述",
            "style_keywords_override": "复古,质感,高级",
            "canvas_size": "640",
            "layout_id": "hero_top",
            "promo_copy": "",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "mvp-regen-001"
    assert len(data["variants"]) == 3


def test_poster_render_upload_rejects_invalid_content_type():
    router_module.brain.get_visual_plan = _mock_plan
    router_module.renderer.render = _mock_render
    resp = client.post(
        "/v1/poster/render/upload",
        headers={"x-request-id": "test-render-upload-invalid-001"},
        data={
            "product_name": "test",
            "product_desc": "test desc",
            "tagline": "新品推荐",
            "canvas_size": "2048",
        },
        files={"product_image": ("bad.txt", b"not-image", "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "unsupported_file_type"
