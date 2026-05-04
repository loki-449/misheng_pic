import os
from collections import Counter
from typing import Any

from PIL import Image


def analyze_product_image(image_path: str, sample_max: int = 96) -> dict[str, Any]:
    """
    从商品图提取轻量视觉统计，供 LLM 或启发式生成风格关键词。
    不依赖多模态 API，保证离线可跑通 MVP。
    """
    if not image_path or not os.path.exists(image_path):
        return {
            "dominant_rgb": [244, 241, 236],
            "mean_brightness": 200.0,
            "mean_saturation": 30.0,
            "width": 0,
            "height": 0,
            "source": "empty_path",
        }

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        im_small = im.copy()
        im_small.thumbnail((sample_max, sample_max))

    pixels = list(im_small.getdata())
    if not pixels:
        return {
            "dominant_rgb": [244, 241, 236],
            "mean_brightness": 200.0,
            "mean_saturation": 0.0,
            "width": w,
            "height": h,
            "source": "no_pixels",
        }

    quantized: list[tuple[int, int, int]] = []
    brightness_vals: list[float] = []
    sat_vals: list[float] = []
    for r, g, b in pixels:
        qr, qg, qb = (r // 48) * 48, (g // 48) * 48, (b // 48) * 48
        quantized.append((qr, qg, qb))
        brightness_vals.append(0.299 * r + 0.587 * g + 0.114 * b)
        mx, mn = max(r, g, b), min(r, g, b)
        sat_vals.append(float(mx - mn))

    dom = Counter(quantized).most_common(1)[0][0]
    mean_b = sum(brightness_vals) / len(brightness_vals)
    mean_s = sum(sat_vals) / len(sat_vals)

    return {
        "dominant_rgb": [int(dom[0]), int(dom[1]), int(dom[2])],
        "mean_brightness": round(mean_b, 2),
        "mean_saturation": round(mean_s, 2),
        "width": w,
        "height": h,
        "source": "pil_stats_v1",
    }


def analyze_product_images(image_paths: list[str], sample_max: int = 96) -> dict[str, Any]:
    """多张商品图：合并统计（主色取平均、亮度/饱和度取平均，尺寸取最大边）。"""
    paths = [p for p in image_paths if p and os.path.exists(p)]
    if not paths:
        return analyze_product_image("", sample_max=sample_max)
    if len(paths) == 1:
        s = analyze_product_image(paths[0], sample_max=sample_max)
        s["image_count"] = 1
        return s

    stats_list = [analyze_product_image(p, sample_max=sample_max) for p in paths]
    doms = [tuple(s.get("dominant_rgb", [200, 200, 200])) for s in stats_list]
    avg_d = [int(sum(d[i] for d in doms) / len(doms)) for i in range(3)]
    mb = sum(float(s.get("mean_brightness", 180)) for s in stats_list) / len(stats_list)
    ms = sum(float(s.get("mean_saturation", 40)) for s in stats_list) / len(stats_list)
    w = max(int(s.get("width", 0) or 0) for s in stats_list)
    h = max(int(s.get("height", 0) or 0) for s in stats_list)
    return {
        "dominant_rgb": [avg_d[0], avg_d[1], avg_d[2]],
        "mean_brightness": round(mb, 2),
        "mean_saturation": round(ms, 2),
        "width": w,
        "height": h,
        "source": "pil_stats_multi_v1",
        "image_count": len(paths),
    }


def heuristic_style_keywords_cn(stats: dict[str, Any]) -> list[str]:
    """无 LLM 时根据统计返回 3 个中文风格词。"""
    b = float(stats.get("mean_brightness", 180))
    s = float(stats.get("mean_saturation", 40))
    r, g, b_ = stats.get("dominant_rgb", [200, 200, 200])
    warm_score = (r + g * 0.5) - (b_ * 1.2)

    if b > 200 and s < 45:
        pool = ["清新", "明亮", "简约"]
    elif b < 90:
        pool = ["沉稳", "质感", "高级"]
    elif s > 70 and warm_score > 40:
        pool = ["温暖", "活力", "鲜明"]
    elif s > 55:
        pool = ["浓郁", "饱满", "吸睛"]
    else:
        pool = ["柔和", "自然", "平衡"]

    return pool[:3]
