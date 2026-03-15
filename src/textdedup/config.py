from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .lexicon import DEFAULT_USER_DICT_FILES
from .stopwords import DEFAULT_STOPWORD_FILES


@dataclass(frozen=True)
class DataConfig:
    """Configuration for loading externally prepared datasets."""

    data_path: str = "data/input.jsonl"
    file_format: str = "jsonl"
    text_field: str = "text"
    id_field: str = "id"

    def __post_init__(self) -> None:
        if not self.data_path:
            raise ValueError("data_path 不能为空")
        if self.file_format not in {"jsonl", "csv"}:
            raise ValueError("file_format 仅支持 jsonl 或 csv")
        if not self.text_field:
            raise ValueError("text_field 不能为空")
        if not self.id_field:
            raise ValueError("id_field 不能为空")


@dataclass(frozen=True)
class TokenizationConfig:
    """Shared tokenization knobs for offline processing components."""

    lowercase: bool = True
    token_pattern: str = r"[A-Za-z0-9]+|[\u4e00-\u9fff]+"
    tokenizer_backend: str = "jieba"
    user_dict_paths: Tuple[str, ...] = DEFAULT_USER_DICT_FILES
    proper_noun_weight: float = 0.6
    remove_stopwords: bool = True
    stopword_paths: Tuple[str, ...] = DEFAULT_STOPWORD_FILES
    extra_stopwords: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.tokenizer_backend not in {"jieba", "regex"}:
            raise ValueError("tokenizer_backend 仅支持 jieba 或 regex")
        if not (0.0 < self.proper_noun_weight <= 1.0):
            raise ValueError("proper_noun_weight 必须在 (0, 1] 范围内")


@dataclass(frozen=True)
class SimHashConfig:
    """Stage-1 candidate recall controls."""

    enabled: bool = True
    hash_bits: int = 64
    top_n: int = 1000
    hamming_threshold: Optional[int] = None
    idf_cap: Optional[float] = 6.0
    feature_mode: str = "token"
    char_ngram_size: int = 3

    def __post_init__(self) -> None:
        if self.hash_bits <= 0:
            raise ValueError("hash_bits 必须大于 0")
        if self.top_n <= 0:
            raise ValueError("top_n 必须大于 0")
        if self.hamming_threshold is not None and self.hamming_threshold < 0:
            raise ValueError("hamming_threshold 不能小于 0")
        if self.idf_cap is not None and self.idf_cap <= 0:
            raise ValueError("idf_cap 必须大于 0")
        if self.feature_mode not in {"token", "char_ngram"}:
            raise ValueError("feature_mode 仅支持 token 或 char_ngram")
        if self.char_ngram_size <= 0:
            raise ValueError("char_ngram_size 必须大于 0")


@dataclass(frozen=True)
class TfidfConfig:
    """Stage-2 TF-IDF rerank controls and vectorizer params."""

    enabled: bool = True
    top_n: int = 200
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 1
    analyzer: str = "word"
    idf_cap: Optional[float] = 6.0

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n 必须大于 0")
        if self.ngram_range[0] <= 0 or self.ngram_range[1] < self.ngram_range[0]:
            raise ValueError("ngram_range 非法")
        if self.min_df <= 0:
            raise ValueError("min_df 必须大于 0")
        if self.analyzer not in {"word", "char", "char_wb"}:
            raise ValueError("analyzer 仅支持 word、char 或 char_wb")
        if self.idf_cap is not None and self.idf_cap <= 0:
            raise ValueError("idf_cap 必须大于 0")


@dataclass(frozen=True)
class SBERTConfig:
    """Stage-3 SBERT semantic rerank controls."""

    enabled: bool = False
    model_name: str = "BAAI/bge-small-zh-v1.5"
    local_model_path: Optional[str] = "models/bge-small-zh-v1.5"
    device: str = "cpu"
    batch_size: int = 32
    top_n: int = 50
    final_threshold: float = 0.78
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if self.top_n <= 0:
            raise ValueError("top_n 必须大于 0")
        if not (0.0 <= self.final_threshold <= 1.0):
            raise ValueError("final_threshold 必须在 [0, 1] 范围内")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device 仅支持 cpu 或 cuda")


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level offline pipeline configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    simhash: SimHashConfig = field(default_factory=SimHashConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    sbert: SBERTConfig = field(default_factory=SBERTConfig)
    output_top_n: int = 20
    final_score_threshold: float = 0.78

    def __post_init__(self) -> None:
        if self.output_top_n <= 0:
            raise ValueError("output_top_n 必须大于 0")
        if not (0.0 <= self.final_score_threshold <= 1.0):
            raise ValueError("final_score_threshold 必须在 [0, 1] 范围内")

    @property
    def data_path(self) -> Path:
        return Path(self.data.data_path)