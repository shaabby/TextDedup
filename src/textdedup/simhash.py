from __future__ import annotations

import hashlib
import math
from pathlib import Path
from collections import Counter
from typing import Callable, Optional, Sequence

from .cache import SimHashCache, build_simhash_cache, load_simhash_cache, save_simhash_cache
from .config import TokenizationConfig
from .preprocess import TextPreprocessor, build_preprocessor


class SimHash:
    """支持语料拟合 IDF 权重的 SimHash 实现。"""

    def __init__(
        self,
        hash_bits: int = 64,
        tokenization_config: Optional[TokenizationConfig] = None,
        idf_cap: Optional[float] = 6.0,
        feature_mode: str = "token",
        char_ngram_size: int = 3,
    ) -> None:
        self.hash_bits = hash_bits
        self.idf_cap = idf_cap
        self.feature_mode = feature_mode
        self.char_ngram_size = char_ngram_size
        if self.feature_mode not in {"token", "char_ngram"}:
            raise ValueError("feature_mode 仅支持 token 或 char_ngram")
        if self.char_ngram_size <= 0:
            raise ValueError("char_ngram_size 必须大于 0")
        self.preprocessor: TextPreprocessor = build_preprocessor(tokenization_config)
        self.idf_weights: dict[str, float] = {}
        self._doc_count = 0

    def _extract_features(self, text: str) -> list[str]:
        if self.feature_mode == "token":
            return self.preprocessor.tokenize(text)

        normalized = self.preprocessor.normalize(text)
        compact = "".join(normalized.split())
        if not compact:
            return []
        n = self.char_ngram_size
        if len(compact) <= n:
            return [compact]
        return [compact[i : i + n] for i in range(len(compact) - n + 1)]

    def fit(self, texts: Sequence[str]) -> SimHash:
        return self.fit_with_progress(texts=texts)

    def fit_with_progress(
        self,
        texts: Sequence[str],
        progress_callback: Optional[Callable[[str], None]] = None,
        progress_interval: int = 10000,
    ) -> SimHash:
        if not texts:
            raise ValueError("texts 不能为空")
        if progress_interval <= 0:
            raise ValueError("progress_interval 必须大于 0")

        doc_freq: Counter[str] = Counter()
        self._doc_count = len(texts)

        if progress_callback is not None:
            progress_callback(f"SimHash-IDF 统计开始，共 {self._doc_count} 条")

        for idx, text in enumerate(texts, start=1):
            features = set(self._extract_features(text))
            if features:
                doc_freq.update(features)
            if progress_callback is not None and idx % progress_interval == 0:
                progress_callback(f"SimHash-IDF 统计 {idx}/{self._doc_count}")

        self.idf_weights = {}
        for token, freq in doc_freq.items():
            idf = math.log((1.0 + self._doc_count) / (1.0 + freq)) + 1.0
            if self.idf_cap is not None:
                idf = min(idf, self.idf_cap)
            self.idf_weights[token] = idf

        if progress_callback is not None:
            progress_callback(f"SimHash-IDF 统计完成，词项数 {len(self.idf_weights)}")
        return self

    def _token_hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def _idf_for(self, token: str) -> float:
        if token in self.idf_weights:
            return self.idf_weights[token]
        if self._doc_count > 0:
            fallback = math.log(1.0 + self._doc_count) + 1.0
            if self.idf_cap is not None:
                fallback = min(fallback, self.idf_cap)
            return fallback
        return 1.0

    def fingerprint(self, text: str) -> int:
        features = self._extract_features(text)
        if not features:
            return 0

        weights = Counter(features)
        vector = [0] * self.hash_bits

        for token, term_freq in weights.items():
            weight = float(term_freq) * self._idf_for(token) * self.preprocessor.token_weight(token)
            h = self._token_hash(token)
            for i in range(self.hash_bits):
                bit = 1 if (h >> i) & 1 else -1
                vector[i] += bit * weight

        fingerprint = 0
        for i, v in enumerate(vector):
            if v > 0:
                fingerprint |= 1 << i
        return fingerprint

    @staticmethod
    def hamming_distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def similarity(self, text_a: str, text_b: str) -> float:
        fp_a = self.fingerprint(text_a)
        fp_b = self.fingerprint(text_b)
        dist = self.hamming_distance(fp_a, fp_b)
        return 1.0 - (dist / self.hash_bits)

    def export_cache(self, fingerprints: Sequence[int]) -> SimHashCache:
        return build_simhash_cache(
            hash_bits=self.hash_bits,
            document_count=self._doc_count,
            idf_weights=self.idf_weights,
            fingerprints=fingerprints,
        )

    def save_cache(self, path: str | Path, fingerprints: Sequence[int]) -> None:
        save_simhash_cache(path, self.export_cache(fingerprints))

    @staticmethod
    def load_cache(path: str | Path) -> SimHashCache:
        return load_simhash_cache(path)
