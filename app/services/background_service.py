import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class BackgroundCandidate:
    background_id: str
    path: str
    tags: list[str]
    score: float


class BackgroundRetriever:
    """
    轻量背景检索器（本地 metadata 检索版）。
    后续可替换为 embedding + FAISS。
    """

    def __init__(self) -> None:
        default_path = "./assets/backgrounds/metadata.json"
        self.metadata_path = os.getenv("BACKGROUND_METADATA_PATH", default_path)

    def _load_metadata(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.metadata_path):
            return []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []

    def retrieve(
        self,
        style_tags: list[str],
        product_name: str,
        product_desc: str,
        top_k: int = 3,
    ) -> list[BackgroundCandidate]:
        entries = self._load_metadata()
        query_tokens = set(style_tags + [product_name.lower(), product_desc.lower()])
        scored: list[BackgroundCandidate] = []
        for item in entries:
            tags = [str(t).lower() for t in item.get("tags", [])]
            if not tags:
                continue
            hit = len(query_tokens.intersection(set(tags)))
            if hit <= 0:
                continue
            score = round(min(0.99, 0.4 + 0.15 * hit), 3)
            scored.append(
                BackgroundCandidate(
                    background_id=str(item.get("id", "unknown")),
                    path=str(item.get("path", "")),
                    tags=tags,
                    score=score,
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
