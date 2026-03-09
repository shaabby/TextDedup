from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SimilarityResult:
    index: int
    score: float
    text: str


class SimilarityEngine:
    """基于 TF-IDF + 余弦相似度的文本相似度引擎。"""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (2, 4),
        min_df: int = 1,
        analyzer: str = "char_wb",
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            analyzer=analyzer,
        )
        self._matrix = None
        self._texts: List[str] = []

    def fit(self, texts: Sequence[str]) -> None:
        if not texts:
            raise ValueError("texts 不能为空")
        self._texts = list(texts)
        self._matrix = self.vectorizer.fit_transform(self._texts)

    def query(self, text: str, top_k: int = 3) -> List[SimilarityResult]:
        if self._matrix is None:
            raise RuntimeError("请先调用 fit() 构建索引")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_vec = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self._matrix).ravel()
        top_indices = np.argsort(-scores)[:top_k]

        return [
            SimilarityResult(index=int(i), score=float(scores[i]), text=self._texts[int(i)])
            for i in top_indices
        ]

    def deduplicate(self, threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """返回相似度超过阈值的文本对: (i, j, score)。"""
        if self._matrix is None:
            raise RuntimeError("请先调用 fit() 构建索引")

        sim = cosine_similarity(self._matrix)
        duplicates: List[Tuple[int, int, float]] = []
        n = sim.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim[i, j])
                if score >= threshold:
                    duplicates.append((i, j, score))

        return duplicates


def pairwise_similarity(text_a: str, text_b: str) -> float:
    """计算两段文本的 TF-IDF 余弦相似度。"""
    engine = SimilarityEngine()
    engine.fit([text_a, text_b])
    return engine.query(text_a, top_k=2)[1].score
