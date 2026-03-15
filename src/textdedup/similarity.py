from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .cache import TfidfVocabularyCache, load_tfidf_vocabulary_cache, save_tfidf_vocabulary_cache
from .config import TfidfConfig, TokenizationConfig
from .preprocess import build_preprocessor


@dataclass
class SimilarityResult:
    index: int
    score: float
    text: str


class SimilarityEngine:
    """基于 TF-IDF + 余弦相似度的文本相似度引擎。"""

    def __init__(
        self,
        config: Optional[TfidfConfig] = None,
        tokenization_config: Optional[TokenizationConfig] = None,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        analyzer: str = "word",
    ) -> None:
        self.preprocessor = build_preprocessor(tokenization_config)
        tfidf_config = config or TfidfConfig(
            ngram_range=ngram_range,
            min_df=min_df,
            analyzer=analyzer,
        )
        self.config = tfidf_config
        vectorizer_kwargs = {
            "ngram_range": tfidf_config.ngram_range,
            "min_df": tfidf_config.min_df,
            "analyzer": tfidf_config.analyzer,
        }
        if tfidf_config.analyzer == "word":
            vectorizer_kwargs.update(
                {
                    "tokenizer": self.preprocessor.tokenize,
                    "token_pattern": None,
                    "lowercase": False,
                }
            )
        else:
            vectorizer_kwargs["preprocessor"] = self.preprocessor.normalize

        self.vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        self._matrix = None
        self._texts: List[str] = []
        self._feature_weights: Optional[np.ndarray] = None

    def _term_weight(self, feature: str) -> float:
        if self.config.analyzer != "word":
            return 1.0
        parts = feature.split(" ")
        if not parts:
            return 1.0
        weights = [self.preprocessor.token_weight(part) for part in parts]
        return float(sum(weights) / len(weights))

    def _build_feature_weights(self) -> np.ndarray:
        feature_weights = np.ones(len(self.vectorizer.vocabulary_), dtype=float)
        idf_values = self.vectorizer.idf_
        for feature, idx in self.vectorizer.vocabulary_.items():
            term_weight = self._term_weight(feature)
            idf_weight = 1.0
            if self.config.idf_cap is not None and idf_values[idx] > 0:
                idf_weight = min(float(idf_values[idx]), self.config.idf_cap) / float(idf_values[idx])
            feature_weights[idx] = term_weight * idf_weight
        return feature_weights

    def fit(self, texts: Sequence[str]) -> None:
        if not texts:
            raise ValueError("texts 不能为空")
        self._texts = list(texts)
        self._matrix = self.vectorizer.fit_transform(self._texts)
        self._feature_weights = self._build_feature_weights()
        self._matrix = self._matrix.multiply(self._feature_weights).tocsr()

    def query(self, text: str, top_k: int = 3) -> List[SimilarityResult]:
        if self._matrix is None:
            raise RuntimeError("请先调用 fit() 构建索引")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_input = text if self.config.analyzer == "word" else self.preprocessor.normalize(text)
        query_vec = self.vectorizer.transform([query_input])
        if self._feature_weights is not None:
            query_vec = query_vec.multiply(self._feature_weights)
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

    def export_vocabulary_cache(self) -> TfidfVocabularyCache:
        if self._matrix is None:
            raise RuntimeError("请先调用 fit() 构建索引")

        return TfidfVocabularyCache(
            document_count=len(self._texts),
            analyzer=self.config.analyzer,
            ngram_range=list(self.config.ngram_range),
            min_df=self.config.min_df,
            vocabulary={str(k): int(v) for k, v in self.vectorizer.vocabulary_.items()},
            idf=[float(v) for v in self.vectorizer.idf_.tolist()],
        )

    def save_vocabulary_cache(self, path: str | Path) -> None:
        save_tfidf_vocabulary_cache(path, self.export_vocabulary_cache())

    @staticmethod
    def load_vocabulary_cache(path: str | Path) -> TfidfVocabularyCache:
        return load_tfidf_vocabulary_cache(path)


def pairwise_similarity(text_a: str, text_b: str) -> float:
    """计算两段文本的 TF-IDF 余弦相似度。"""
    engine = SimilarityEngine()
    engine.fit([text_a, text_b])
    return engine.query(text_a, top_k=2)[1].score
