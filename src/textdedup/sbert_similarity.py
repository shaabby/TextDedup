from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .cache import (
    SbertEmbeddingsCache,
    build_sbert_embeddings_cache,
    load_sbert_embeddings_cache,
    save_sbert_embeddings_cache,
)
from .config import SBERTConfig


@dataclass
class SbertResult:
    index: int
    score: float
    text: str


class SbertSimilarityEngine:
    """基于 sentence-transformers 的语义相似度引擎。"""

    def __init__(self, config: SBERTConfig) -> None:
        self.config = config
        self._model = None
        self._texts: List[str] = []
        self._embeddings: np.ndarray | None = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sentence-transformers，请先安装依赖后再启用 SBERT"
            ) from exc

        model_ref = self.config.local_model_path or self.config.model_name
        if self.config.local_model_path:
            model_ref = str(Path(self.config.local_model_path).expanduser().resolve())

        self._model = SentenceTransformer(model_ref, device=self.config.device)
        return self._model

    def fit(self, texts: Sequence[str]) -> None:
        if not texts:
            raise ValueError("texts 不能为空")
        model = self._ensure_model()
        self._texts = list(texts)
        embeddings = model.encode(
            self._texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self._embeddings = np.asarray(embeddings, dtype=np.float32)

    def query(self, text: str, top_k: int = 10) -> List[SbertResult]:
        if self._embeddings is None:
            raise RuntimeError("请先调用 fit() 构建向量索引")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        model = self._ensure_model()
        q = model.encode(
            [text],
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        if self.config.normalize_embeddings:
            scores = self._embeddings @ q
        else:
            denom = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q)
            denom = np.where(denom == 0.0, 1e-12, denom)
            scores = (self._embeddings @ q) / denom

        top_indices = np.argsort(-scores)[:top_k]
        return [
            SbertResult(index=int(i), score=float(scores[i]), text=self._texts[int(i)])
            for i in top_indices
        ]

    def export_embeddings_cache(self) -> SbertEmbeddingsCache:
        if self._embeddings is None:
            raise RuntimeError("请先调用 fit() 构建向量索引")
        return build_sbert_embeddings_cache(
            model_name=self.config.model_name,
            normalize_embeddings=self.config.normalize_embeddings,
            embeddings=self._embeddings.tolist(),
        )

    def save_embeddings_cache(self, path: str | Path) -> None:
        save_sbert_embeddings_cache(path, self.export_embeddings_cache())

    @staticmethod
    def load_embeddings_cache(path: str | Path) -> SbertEmbeddingsCache:
        return load_sbert_embeddings_cache(path)
