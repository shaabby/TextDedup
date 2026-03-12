from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .similarity import SimilarityEngine
from .simhash import SimHash


@dataclass
class TwoStageResult:
    index: int
    text: str
    simhash_distance: int
    score: float


class TwoStageSearchEngine:
    """两阶段文本检索: SimHash 粗筛 + TF-IDF 精排。"""

    def __init__(self, hash_bits: int = 64) -> None:
        self.simhash = SimHash(hash_bits=hash_bits)
        self.ranker = SimilarityEngine()
        self._texts: List[str] = []
        self._fingerprints: List[int] = []

    def fit(self, texts: Sequence[str]) -> None:
        if not texts:
            raise ValueError("texts 不能为空")
        self._texts = list(texts)
        self._fingerprints = [self.simhash.fingerprint(t) for t in self._texts]
        self.ranker.fit(self._texts)

    def query(self, text: str, top_k: int = 3, candidate_k: int = 20) -> List[TwoStageResult]:
        if not self._texts:
            raise RuntimeError("请先调用 fit() 构建索引")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if candidate_k <= 0:
            raise ValueError("candidate_k 必须大于 0")

        candidate_k = min(candidate_k, len(self._texts))
        top_k = min(top_k, candidate_k)

        q_fp = self.simhash.fingerprint(text)
        distances = [
            (i, self.simhash.hamming_distance(q_fp, fp))
            for i, fp in enumerate(self._fingerprints)
        ]
        distances.sort(key=lambda x: x[1])
        candidate_indices = [i for i, _ in distances[:candidate_k]]

        # 仅在候选集合内做 TF-IDF 余弦精排，控制计算开销。
        candidate_texts = [self._texts[i] for i in candidate_indices]
        local_ranker = SimilarityEngine()
        local_ranker.fit(candidate_texts)
        ranked = local_ranker.query(text, top_k=top_k)

        results: List[TwoStageResult] = []
        for item in ranked:
            global_idx = candidate_indices[item.index]
            simhash_dist = self.simhash.hamming_distance(q_fp, self._fingerprints[global_idx])
            results.append(
                TwoStageResult(
                    index=global_idx,
                    text=self._texts[global_idx],
                    simhash_distance=simhash_dist,
                    score=item.score,
                )
            )

        return results
