"""Tests for SbertSimilarityEngine and SBERTConfig."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from textdedup.config import SBERTConfig
from textdedup.sbert_similarity import SbertResult, SbertSimilarityEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_model(dim: int = 8):
    """Return a mock SentenceTransformer that produces deterministic embeddings."""
    rng = np.random.default_rng(42)

    def fake_encode(texts, **kwargs):
        vecs = rng.random((len(texts), dim)).astype(np.float32)
        if kwargs.get("normalize_embeddings", True):
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1.0, norms)
        return vecs

    mock = MagicMock()
    mock.encode.side_effect = fake_encode
    return mock


def _engine_with_mock(config: SBERTConfig | None = None):
    """Return (engine, mock_model)."""
    if config is None:
        config = SBERTConfig()
    engine = SbertSimilarityEngine(config=config)
    mock_model = _make_fake_model()
    engine._model = mock_model  # inject mock directly – skips _ensure_model
    return engine, mock_model


CORPUS = ["北京欢迎你", "上海是个好地方", "广州美食很多", "深圳发展很快", "天津港口很大"]


# ---------------------------------------------------------------------------
# SBERTConfig validation
# ---------------------------------------------------------------------------


class TestSBERTConfigValidation:
    def test_default_config_ok(self):
        cfg = SBERTConfig()
        assert cfg.enabled is False
        assert cfg.model_name == "BAAI/bge-small-zh-v1.5"
        assert cfg.device == "cpu"
        assert cfg.batch_size == 32
        assert cfg.top_n == 50
        assert cfg.final_threshold == 0.78
        assert cfg.normalize_embeddings is True

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            SBERTConfig(batch_size=0)

    def test_invalid_top_n(self):
        with pytest.raises(ValueError, match="top_n"):
            SBERTConfig(top_n=-1)

    def test_invalid_final_threshold_above_one(self):
        with pytest.raises(ValueError, match="final_threshold"):
            SBERTConfig(final_threshold=1.5)

    def test_invalid_device(self):
        with pytest.raises(ValueError, match="device"):
            SBERTConfig(device="tpu")

    def test_local_model_path_accepted(self):
        cfg = SBERTConfig(local_model_path="/models/bge")
        assert cfg.local_model_path == "/models/bge"


# ---------------------------------------------------------------------------
# SbertSimilarityEngine — basic contract
# ---------------------------------------------------------------------------


class TestSbertSimilarityEngineContract:
    def test_query_before_fit_raises(self):
        engine, _ = _engine_with_mock()
        with pytest.raises(RuntimeError, match="fit"):
            engine.query("test", top_k=3)

    def test_fit_empty_raises(self):
        engine, _ = _engine_with_mock()
        with pytest.raises(ValueError, match="不能为空"):
            engine.fit([])

    def test_fit_stores_texts(self):
        engine, _ = _engine_with_mock()
        engine.fit(CORPUS)
        assert engine._texts == CORPUS

    def test_fit_creates_embeddings(self):
        engine, _ = _engine_with_mock()
        engine.fit(CORPUS)
        assert engine._embeddings is not None
        assert engine._embeddings.shape == (len(CORPUS), 8)

    def test_query_invalid_top_k(self):
        engine, _ = _engine_with_mock()
        engine.fit(CORPUS)
        with pytest.raises(ValueError, match="top_k"):
            engine.query("查询", top_k=0)

    def test_query_returns_sbert_results(self):
        engine, _ = _engine_with_mock()
        engine.fit(CORPUS)
        results = engine.query("北京", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, SbertResult) for r in results)

    def test_query_results_descending_score(self):
        engine, _ = _engine_with_mock()
        engine.fit(CORPUS)
        results = engine.query("城市", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_result_fields(self):
        engine, _ = _engine_with_mock()
        engine.fit(CORPUS)
        results = engine.query("北京", top_k=2)
        for r in results:
            assert 0 <= r.index < len(CORPUS)
            assert CORPUS[r.index] == r.text
            assert isinstance(r.score, float)

    def test_query_top_k_clipped_to_corpus_size(self):
        engine, _ = _engine_with_mock()
        engine.fit(CORPUS)
        results = engine.query("北京", top_k=100)
        assert len(results) == len(CORPUS)


# ---------------------------------------------------------------------------
# Normalization path
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalized_scores_bounded(self):
        """With normalize_embeddings=True, cosine scores should be in [-1, 1]."""
        cfg = SBERTConfig(normalize_embeddings=True)
        engine, _ = _engine_with_mock(cfg)
        engine.fit(CORPUS)
        results = engine.query("北京", top_k=len(CORPUS))
        for r in results:
            assert -1.0 <= r.score <= 1.0 + 1e-5

    def test_unnormalized_path(self):
        """With normalize_embeddings=False the engine computes cosine manually."""
        cfg = SBERTConfig(normalize_embeddings=False)
        engine, _ = _engine_with_mock(cfg)
        engine.fit(CORPUS)
        results = engine.query("北京", top_k=3)
        assert len(results) == 3
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Embeddings cache
# ---------------------------------------------------------------------------


class TestEmbeddingsCache:
    def test_export_before_fit_raises(self):
        engine, _ = _engine_with_mock()
        with pytest.raises(RuntimeError, match="fit"):
            engine.export_embeddings_cache()

    def test_export_cache_shape(self):
        engine, _ = _engine_with_mock()
        engine.fit(CORPUS)
        cache = engine.export_embeddings_cache()
        assert cache.model_name == "BAAI/bge-small-zh-v1.5"
        assert len(cache.embeddings) == len(CORPUS)
        assert len(cache.embeddings[0]) == 8

    def test_save_and_load_roundtrip(self, tmp_path):
        engine, _ = _engine_with_mock()
        engine.fit(CORPUS)
        cache_path = tmp_path / "emb.json"
        engine.save_embeddings_cache(cache_path)
        assert cache_path.exists()
        loaded = SbertSimilarityEngine.load_embeddings_cache(cache_path)
        assert loaded.model_name == engine.config.model_name
        np.testing.assert_allclose(
            np.array(loaded.embeddings), engine._embeddings, atol=1e-5
        )


# ---------------------------------------------------------------------------
# _ensure_model: missing package error
# ---------------------------------------------------------------------------


class TestEnsureModel:
    def test_missing_sentence_transformers_raises(self):
        cfg = SBERTConfig()
        engine = SbertSimilarityEngine(config=cfg)
        # engine._model is None → will call _ensure_model → mock ImportError
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises((RuntimeError, ImportError)):
                engine._ensure_model()

    def test_local_model_path_used_over_model_name(self):
        """_ensure_model should prefer local_model_path when set."""
        cfg = SBERTConfig(local_model_path="/local/model")
        engine = SbertSimilarityEngine(config=cfg)

        mock_st_module = MagicMock()
        mock_instance = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_instance

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            engine._model = None
            engine._ensure_model()
            call_args = mock_st_module.SentenceTransformer.call_args
            assert call_args[0][0] == "/local/model"
