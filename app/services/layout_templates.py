from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutTemplate:
    id: str
    label_zh: str
    composition: str
    text_area: str
    safe_margin_percent: int
    # 商品区相对画布的比例：x, y, w, h（0~1）
    product_box: tuple[float, float, float, float]
    # 标题锚点：top_left | top_center
    title_anchor: str


DEFAULT_LAYOUT_ID = "classic_center"

LAYOUT_TEMPLATES: dict[str, LayoutTemplate] = {
    "classic_center": LayoutTemplate(
        id="classic_center",
        label_zh="经典居中",
        composition="product_center_bottom",
        text_area="top_left",
        safe_margin_percent=9,
        # 略下移、加宽，给标题与留白更均衡的视觉权重
        product_box=(0.18, 0.34, 0.64, 0.52),
        title_anchor="top_left",
    ),
    "hero_top": LayoutTemplate(
        id="hero_top",
        label_zh="顶部大标题",
        composition="hero_title_upper",
        text_area="top_center",
        safe_margin_percent=9,
        product_box=(0.12, 0.42, 0.76, 0.48),
        title_anchor="top_center",
    ),
    "editorial_split": LayoutTemplate(
        id="editorial_split",
        label_zh="杂志分栏",
        composition="split_editorial",
        text_area="left",
        safe_margin_percent=10,
        product_box=(0.46, 0.20, 0.48, 0.62),
        title_anchor="top_left",
    ),
}


def list_layout_templates() -> list[dict[str, str]]:
    return [{"id": t.id, "label_zh": t.label_zh} for t in LAYOUT_TEMPLATES.values()]


def get_layout_template(layout_id: str | None) -> LayoutTemplate:
    if not layout_id:
        return LAYOUT_TEMPLATES[DEFAULT_LAYOUT_ID]
    return LAYOUT_TEMPLATES.get(layout_id, LAYOUT_TEMPLATES[DEFAULT_LAYOUT_ID])


def template_to_layout_dict(template: LayoutTemplate, canvas_size: int) -> dict[str, int | str]:
    return {
        "canvas_size": canvas_size,
        "composition": template.composition,
        "text_area": template.text_area,
        "safe_margin_percent": template.safe_margin_percent,
    }
