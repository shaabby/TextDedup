from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class TfidfVocabularyCache:
    document_count: int
    analyzer: str
    ngram_range: List[int]
    min_df: int
    vocabulary: Dict[str, int]
    idf: List[float]


@dataclass(frozen=True)
class SimHashCache:
    hash_bits: int
    document_count: int
    idf_weights: Dict[str, float]
    fingerprints: List[int]


@dataclass(frozen=True)
class SbertEmbeddingsCache:
    model_name: str
    document_count: int
    embedding_dim: int
    normalize_embeddings: bool
    embeddings: List[List[float]]


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_tfidf_vocabulary_cache(path: str | Path, cache: TfidfVocabularyCache) -> None:
    _write_json(path, asdict(cache))


def load_tfidf_vocabulary_cache(path: str | Path) -> TfidfVocabularyCache:
    payload = _read_json(path)
    return TfidfVocabularyCache(
        document_count=int(payload["document_count"]),
        analyzer=str(payload["analyzer"]),
        ngram_range=[int(v) for v in payload["ngram_range"]],
        min_df=int(payload["min_df"]),
        vocabulary={str(k): int(v) for k, v in payload["vocabulary"].items()},
        idf=[float(v) for v in payload["idf"]],
    )


def save_simhash_cache(path: str | Path, cache: SimHashCache) -> None:
    _write_json(path, asdict(cache))


def load_simhash_cache(path: str | Path) -> SimHashCache:
    payload = _read_json(path)
    return SimHashCache(
        hash_bits=int(payload["hash_bits"]),
        document_count=int(payload["document_count"]),
        idf_weights={str(k): float(v) for k, v in payload["idf_weights"].items()},
        fingerprints=[int(v) for v in payload["fingerprints"]],
    )


def save_sbert_embeddings_cache(path: str | Path, cache: SbertEmbeddingsCache) -> None:
    _write_json(path, asdict(cache))


def load_sbert_embeddings_cache(path: str | Path) -> SbertEmbeddingsCache:
    payload = _read_json(path)
    return SbertEmbeddingsCache(
        model_name=str(payload["model_name"]),
        document_count=int(payload["document_count"]),
        embedding_dim=int(payload["embedding_dim"]),
        normalize_embeddings=bool(payload["normalize_embeddings"]),
        embeddings=[[float(v) for v in row] for row in payload["embeddings"]],
    )


def build_simhash_cache(
    hash_bits: int,
    document_count: int,
    idf_weights: Dict[str, float],
    fingerprints: Sequence[int],
) -> SimHashCache:
    return SimHashCache(
        hash_bits=hash_bits,
        document_count=document_count,
        idf_weights={str(k): float(v) for k, v in idf_weights.items()},
        fingerprints=[int(v) for v in fingerprints],
    )


def build_sbert_embeddings_cache(
    model_name: str,
    normalize_embeddings: bool,
    embeddings: Sequence[Sequence[float]],
) -> SbertEmbeddingsCache:
    rows = [[float(v) for v in row] for row in embeddings]
    dim = len(rows[0]) if rows else 0
    return SbertEmbeddingsCache(
        model_name=model_name,
        document_count=len(rows),
        embedding_dim=dim,
        normalize_embeddings=normalize_embeddings,
        embeddings=rows,
    )