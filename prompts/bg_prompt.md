任务：根据产品与风格，选择或生成背景方案。

输出 JSON：
{
  "background_mode": "retrieve_or_generate",
  "keywords": ["soft light", "paper texture"],
  "negative_keywords": ["busy pattern", "high contrast noise"],
  "composition_note": "left area for headline, center for product"
}

规则：
- 优先检索已有素材库。
- 素材不命中时再生成。
- 背景必须服务主体，不可喧宾夺主。
