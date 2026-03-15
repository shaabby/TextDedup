from __future__ import annotations

import re
from functools import cached_property
from dataclasses import dataclass
from typing import List, Optional

import jieba

from .config import TokenizationConfig
from .lexicon import ensure_jieba_user_dicts, load_user_dict_terms
from .stopwords import load_stopwords


@dataclass(frozen=True)
class TextPreprocessor:
    """Shared text normalization and tokenization utilities."""

    config: TokenizationConfig

    @cached_property
    def stopwords(self) -> set[str]:
        if not self.config.remove_stopwords:
            return set()
        base = load_stopwords(self.config.stopword_paths)
        custom = {
            token.lower() if self.config.lowercase else token
            for token in self.config.extra_stopwords
            if token.strip()
        }
        normalized_base = {
            token.lower() if self.config.lowercase else token
            for token in base
            if token.strip()
        }
        return normalized_base | custom

    def normalize(self, text: str) -> str:
        normalized = text.lower() if self.config.lowercase else text
        # Normalize all whitespace to stabilize feature extraction across stages.
        return " ".join(normalized.split())

    @cached_property
    def loaded_user_dict_paths(self) -> tuple[str, ...]:
        if self.config.tokenizer_backend != "jieba":
            return ()
        return ensure_jieba_user_dicts(self.config.user_dict_paths)

    @cached_property
    def user_dict_terms(self) -> set[str]:
        if self.config.tokenizer_backend != "jieba":
            return set()
        _ = self.loaded_user_dict_paths
        return load_user_dict_terms(self.config.user_dict_paths)

    def token_weight(self, token: str) -> float:
        if token in self.user_dict_terms:
            return self.config.proper_noun_weight
        return 1.0

    def tokenize(self, text: str) -> List[str]:
        normalized = self.normalize(text)
        if self.config.tokenizer_backend == "jieba":
            _ = self.loaded_user_dict_paths
            raw_tokens = jieba.lcut(normalized, cut_all=False)
            return [
                token
                for token in raw_tokens
                if token.strip()
                and re.fullmatch(self.config.token_pattern, token)
                and token not in self.stopwords
            ]
        return [
            token
            for token in re.findall(self.config.token_pattern, normalized)
            if token not in self.stopwords
        ]


def build_preprocessor(config: Optional[TokenizationConfig] = None) -> TextPreprocessor:
    return TextPreprocessor(config=config or TokenizationConfig())