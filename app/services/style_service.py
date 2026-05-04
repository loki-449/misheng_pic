import math
import re
from dataclasses import dataclass


@dataclass
class StyleResult:
    tags: list[str]
    confidence: float
    reason: str
    source: str


class StyleClassifier:
    """
    轻量风格分类器（文本特征检索版）。
    先以规则+相似度实现稳定输出，后续可替换为视觉模型。
    """

    def __init__(self) -> None:
        self.style_keywords = {
            "clean": {"简约", "干净", "纯色", "minimal", "clean"},
            "fresh": {"清新", "春日", "花", "植物", "fresh", "pastel"},
            "craft": {"手作", "文创", "纸", "胶带", "手账", "craft"},
            "retro": {"复古", "怀旧", "胶片", "retro", "vintage"},
            "tech": {"科技", "未来", "金属", "极客", "tech"},
        }

    def _tokenize(self, text: str) -> list[str]:
        # 中文和英文关键词混用，做最小切分
        text = text.lower()
        text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
        return [t for t in text.split() if t]

    def classify(self, product_name: str, product_desc: str) -> StyleResult:
        text = f"{product_name} {product_desc}"
        tokens = self._tokenize(text)
        token_set = set(tokens)
        if not token_set:
            return StyleResult(
                tags=["clean"],
                confidence=0.3,
                reason="empty_text_fallback",
                source="style_classifier_v1",
            )

        scored: list[tuple[str, float]] = []
        for style, words in self.style_keywords.items():
            hit = len(token_set.intersection(words))
            denom = max(1, int(math.sqrt(len(words))))
            score = hit / denom
            scored.append((style, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [s for s, v in scored if v > 0][:2]
        if not top:
            top = ["clean"]

        best_score = scored[0][1]
        confidence = max(0.35, min(0.95, 0.35 + best_score * 0.35))
        reason = f"keyword_match:{','.join(top)}"
        return StyleResult(
            tags=top,
            confidence=round(confidence, 3),
            reason=reason,
            source="style_classifier_v1",
        )
