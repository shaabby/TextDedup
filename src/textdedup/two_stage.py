from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .config import PipelineConfig
from .sbert_similarity import SbertSimilarityEngine
from .similarity import SimilarityEngine


@dataclass
class TwoStageResult:
    index: int
    text: str
    simhash_distance: Optional[int]
    tfidf_score: Optional[float]
    sbert_score: Optional[float]
    final_score: float
    score: float


class TwoStageSearchEngine:
    """二阶段文本检索: TF-IDF 粗排 + (可选) SBERT 精排。"""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._texts: List[str] = []

    def fit(
        self,
        texts: Sequence[str],
        progress_callback: Optional[Callable[[str], None]] = None,
        progress_interval: int = 10000,
    ) -> None:
        if not texts:
            raise ValueError("texts 不能为空")
        if progress_interval <= 0:
            raise ValueError("progress_interval 必须大于 0")
        self._texts = list(texts)
        if progress_callback is not None:
            progress_callback(f"文本索引就绪，共 {len(self._texts)} 条")

    def load_texts_only(self, texts: Sequence[str]) -> None:
        if not texts:
            raise ValueError("texts 不能为空")
        self._texts = list(texts)

    def query(
        self,
        text: str,
        top_k: Optional[int] = None,
        tfidf_candidate_k: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[TwoStageResult]:
        if not self._texts:
            raise RuntimeError("请先调用 fit() 构建索引")

        effective_top_k = self.config.output_top_n if top_k is None else top_k
        effective_tfidf_candidate_k = len(self._texts) if tfidf_candidate_k is None else tfidf_candidate_k

        if effective_top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if effective_tfidf_candidate_k <= 0:
            raise ValueError("tfidf_candidate_k 必须大于 0")

        if not self.config.tfidf.enabled:
            raise RuntimeError("two-stage 模式必须启用 TF-IDF")

        effective_top_k = min(effective_top_k, len(self._texts))
        tfidf_pool_k = min(effective_tfidf_candidate_k, len(self._texts))

        candidate_texts = self._texts
        if progress_callback is not None:
            progress_callback(f"开始 TF-IDF 拟合，语料 {len(candidate_texts)} 条")
        local_ranker = SimilarityEngine(
            config=self.config.tfidf,
            tokenization_config=self.config.tokenization,
        )
        local_ranker.fit(candidate_texts)

        rerank_top_k = min(tfidf_pool_k, len(self._texts))
        if progress_callback is not None:
            progress_callback(f"TF-IDF 拟合完成，开始重排 Top {rerank_top_k}")
        ranked = local_ranker.query(text, top_k=rerank_top_k)
        if progress_callback is not None:
            progress_callback(f"TF-IDF 重排完成，命中 {len(ranked)} 条")

        tfidf_scores: dict[int, float] = {}
        tfidf_ordered_indices: list[int] = []
        for item in ranked:
            global_idx = item.index
            tfidf_scores[global_idx] = item.score
            tfidf_ordered_indices.append(global_idx)

        if not self.config.sbert.enabled:
            results: List[TwoStageResult] = []
            for item in ranked[:effective_top_k]:
                global_idx = item.index
                results.append(
                    TwoStageResult(
                        index=global_idx,
                        text=self._texts[global_idx],
                        simhash_distance=None,
                        tfidf_score=item.score,
                        sbert_score=None,
                        final_score=item.score,
                        score=item.score,
                    )
                )
            return results

        sbert_candidate_indices = tfidf_ordered_indices[: min(self.config.sbert.top_n, len(tfidf_ordered_indices))]
        if not sbert_candidate_indices:
            return []

        sbert_texts = [self._texts[i] for i in sbert_candidate_indices]
        if progress_callback is not None:
            progress_callback(f"开始 SBERT 编码候选，数量 {len(sbert_texts)}")
        sbert_engine = SbertSimilarityEngine(config=self.config.sbert)
        sbert_engine.fit(sbert_texts)
        if progress_callback is not None:
            progress_callback("SBERT 编码完成，开始语义重排")
        sbert_ranked = sbert_engine.query(text, top_k=min(effective_top_k, len(sbert_texts)))
        if progress_callback is not None:
            progress_callback(f"SBERT 重排完成，命中 {len(sbert_ranked)} 条")

        results: List[TwoStageResult] = []
        for item in sbert_ranked:
            global_idx = sbert_candidate_indices[item.index]
            results.append(
                TwoStageResult(
                    index=global_idx,
                    text=self._texts[global_idx],
                    simhash_distance=None,
                    tfidf_score=tfidf_scores.get(global_idx),
                    sbert_score=item.score,
                    final_score=item.score,
                    score=item.score,
                )
            )

        return results
