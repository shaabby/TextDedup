from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Iterable, List


class SimHash:
    """简化版 SimHash 实现。"""

    def __init__(self, hash_bits: int = 64) -> None:
        self.hash_bits = hash_bits

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # 英文/数字按词切分，中文按单字切分，兼顾中英混合文本。
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())

    def _token_hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def fingerprint(self, text: str) -> int:
        tokens = self._tokenize(text)
        if not tokens:
            return 0

        weights = Counter(tokens)
        vector = [0] * self.hash_bits

        for token, weight in weights.items():
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
