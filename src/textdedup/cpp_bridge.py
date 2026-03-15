from __future__ import annotations

import hashlib
import importlib
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


@dataclass(frozen=True)
class WeightedDocument:
    doc_id: str
    tokens: List[str]
    weights: List[float]


@dataclass(frozen=True)
class SimHashFingerprint:
    doc_id: str
    hash_bits: int
    fingerprint: int


@dataclass(frozen=True)
class HammingCandidate:
    doc_id: str
    distance: int
    score: float


@dataclass(frozen=True)
class EngineError:
    error_code: str
    error_message: str
    stage: str


@dataclass(frozen=True)
class EngineMeta:
    engine: str
    version: str
    latency_ms: float
    fallback_used: bool


@dataclass(frozen=True)
class TopKResult:
    query_doc_id: str
    candidates: List[HammingCandidate]
    meta: EngineMeta
    error: Optional[EngineError] = None


class CppBridge:
    """Bridge for future C++ kernels with Python fallback implementations."""

    def __init__(self, prefer_cpp: bool = True) -> None:
        self.prefer_cpp = prefer_cpp
        self._cpp_module = self._load_cpp_module()

    def _load_cpp_module(self) -> Optional[Any]:
        if not self.prefer_cpp:
            return None
        try:
            # Reserved module name for future pybind11 extension.
            return importlib.import_module("textdedup._cpp_core")
        except Exception:
            return None

    def is_available(self) -> bool:
        return self._cpp_module is not None

    @staticmethod
    def _token_hash(token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16)

    @staticmethod
    def _validate_docs(docs: Sequence[WeightedDocument], hash_bits: int) -> None:
        if hash_bits != 64:
            raise ValueError("hash_bits 目前仅支持 64")
        if not docs:
            raise ValueError("docs 不能为空")

        doc_ids = set()
        for doc in docs:
            if not doc.doc_id:
                raise ValueError("doc_id 不能为空")
            if doc.doc_id in doc_ids:
                raise ValueError("doc_id 在同一批次内必须唯一")
            doc_ids.add(doc.doc_id)

            if not doc.tokens:
                raise ValueError("tokens 不能为空")
            if len(doc.tokens) != len(doc.weights):
                raise ValueError("tokens 与 weights 长度必须一致")

    @classmethod
    def _weighted_simhash_python(cls, doc: WeightedDocument, hash_bits: int) -> int:
        vector = [0.0] * hash_bits
        for token, weight in zip(doc.tokens, doc.weights):
            h = cls._token_hash(token)
            for i in range(hash_bits):
                bit = 1.0 if (h >> i) & 1 else -1.0
                vector[i] += bit * float(weight)

        fingerprint = 0
        for i, v in enumerate(vector):
            if v > 0.0:
                fingerprint |= 1 << i
        return fingerprint

    def compute_simhash(
        self,
        docs: Sequence[WeightedDocument],
        hash_bits: int = 64,
    ) -> List[SimHashFingerprint]:
        self._validate_docs(docs, hash_bits)

        if self._cpp_module is not None and hasattr(self._cpp_module, "compute_simhash_v1"):
            token_batches = [doc.tokens for doc in docs]
            weight_batches = [doc.weights for doc in docs]
            values = self._cpp_module.compute_simhash_v1(token_batches, weight_batches, hash_bits)
            return [
                SimHashFingerprint(doc_id=doc.doc_id, hash_bits=hash_bits, fingerprint=int(v))
                for doc, v in zip(docs, values)
            ]

        return [
            SimHashFingerprint(
                doc_id=doc.doc_id,
                hash_bits=hash_bits,
                fingerprint=self._weighted_simhash_python(doc, hash_bits),
            )
            for doc in docs
        ]

    def hamming_topk(
        self,
        query_fps: Sequence[SimHashFingerprint],
        corpus_fps: Sequence[SimHashFingerprint],
        top_k: int,
        max_distance: Optional[int] = None,
        exclude_self: bool = True,
    ) -> List[TopKResult]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not query_fps:
            raise ValueError("query_fps 不能为空")
        if not corpus_fps:
            raise ValueError("corpus_fps 不能为空")

        for fp in list(query_fps) + list(corpus_fps):
            if fp.hash_bits != 64:
                raise ValueError("hash_bits 目前仅支持 64")

        start = time.perf_counter()
        used_fallback = False

        if self._cpp_module is not None and hasattr(self._cpp_module, "hamming_topk_v1"):
            query_vals = [fp.fingerprint for fp in query_fps]
            corpus_vals = [fp.fingerprint for fp in corpus_fps]
            cap = 64 if max_distance is None else max_distance
            raw = self._cpp_module.hamming_topk_v1(query_vals, corpus_vals, top_k, cap)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            results: List[TopKResult] = []
            for q_idx, pairs in enumerate(raw):
                q_fp = query_fps[q_idx]
                candidates: List[HammingCandidate] = []
                for corpus_idx, dist in pairs:
                    c_fp = corpus_fps[int(corpus_idx)]
                    if exclude_self and q_fp.doc_id == c_fp.doc_id:
                        continue
                    if max_distance is not None and int(dist) > max_distance:
                        continue
                    score = 1.0 - (float(dist) / q_fp.hash_bits)
                    candidates.append(
                        HammingCandidate(
                            doc_id=c_fp.doc_id,
                            distance=int(dist),
                            score=score,
                        )
                    )

                results.append(
                    TopKResult(
                        query_doc_id=q_fp.doc_id,
                        candidates=candidates[:top_k],
                        meta=EngineMeta(
                            engine="cpp",
                            version="hamming_topk_v1",
                            latency_ms=elapsed_ms,
                            fallback_used=False,
                        ),
                    )
                )
            return results

        used_fallback = self.prefer_cpp
        results = self._hamming_topk_python(query_fps, corpus_fps, top_k, max_distance, exclude_self)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return [
            TopKResult(
                query_doc_id=r.query_doc_id,
                candidates=r.candidates,
                meta=EngineMeta(
                    engine="python",
                    version="hamming_topk_v1",
                    latency_ms=elapsed_ms,
                    fallback_used=used_fallback,
                ),
                error=r.error,
            )
            for r in results
        ]

    @staticmethod
    def _hamming_topk_python(
        query_fps: Sequence[SimHashFingerprint],
        corpus_fps: Sequence[SimHashFingerprint],
        top_k: int,
        max_distance: Optional[int],
        exclude_self: bool,
    ) -> List[TopKResult]:
        results: List[TopKResult] = []
        for q_fp in query_fps:
            pairs = []
            for c_fp in corpus_fps:
                if exclude_self and q_fp.doc_id == c_fp.doc_id:
                    continue
                dist = (q_fp.fingerprint ^ c_fp.fingerprint).bit_count()
                if max_distance is not None and dist > max_distance:
                    continue
                pairs.append((c_fp.doc_id, dist))

            pairs.sort(key=lambda x: (x[1], x[0]))
            top_pairs = pairs[:top_k]
            candidates = [
                HammingCandidate(
                    doc_id=doc_id,
                    distance=dist,
                    score=1.0 - (dist / q_fp.hash_bits),
                )
                for doc_id, dist in top_pairs
            ]
            results.append(
                TopKResult(
                    query_doc_id=q_fp.doc_id,
                    candidates=candidates,
                    meta=EngineMeta(
                        engine="python",
                        version="hamming_topk_v1",
                        latency_ms=0.0,
                        fallback_used=False,
                    ),
                )
            )
        return results
