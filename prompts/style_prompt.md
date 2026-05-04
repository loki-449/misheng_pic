任务：根据产品图和描述识别视觉风格。

输出 JSON：
{
  "style_tags": ["minimal", "fresh", "craft"],
  "confidence": 0.0,
  "color_palette": ["#RRGGBB", "#RRGGBB"],
  "reason": "一句话说明判断依据"
}

要求：
- style_tags 最多 3 个。
- confidence 范围 [0,1]。
- 如果判断不确定，必须在 reason 中明确说明不确定来源。
