任务：根据风格与文案，输出可渲染版式。

输出 JSON：
{
  "canvas": {"width": 2048, "height": 2048},
  "layers": [
    {"id": "product", "type": "image", "x": 0, "y": 0, "w": 0, "h": 0, "z": 10},
    {"id": "title", "type": "text", "x": 0, "y": 0, "w": 0, "h": 0, "z": 20},
    {"id": "subtitle", "type": "text", "x": 0, "y": 0, "w": 0, "h": 0, "z": 20}
  ],
  "constraints": [
    "text_not_overlap_product",
    "keep_safe_margin_5_percent"
  ]
}

要求：
- 坐标必须是整数像素。
- 文案图层不能越界。
- 约束条件至少包含可读性与留白规则。
