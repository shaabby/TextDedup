"""TextDedup Python prototype package."""

from .cache import (
	SimHashCache,
	SbertEmbeddingsCache,
	TfidfVocabularyCache,
	load_simhash_cache,
	load_sbert_embeddings_cache,
	load_tfidf_vocabulary_cache,
	save_simhash_cache,
	save_sbert_embeddings_cache,
	save_tfidf_vocabulary_cache,
)
from .chunking import ChunkedDocument, ChunkingConfig, chunk_document, chunk_documents
from .config import DataConfig, PipelineConfig, SBERTConfig, SimHashConfig, TfidfConfig, TokenizationConfig
from .data_loader import InvalidRecord, LoadedDocument, load_documents
from .cpp_bridge import (
	CppBridge,
	EngineError,
	EngineMeta,
	HammingCandidate,
	SimHashFingerprint,
	TopKResult,
	WeightedDocument,
)
from .preprocess import TextPreprocessor, build_preprocessor
from .sbert_similarity import SbertResult, SbertSimilarityEngine
from .similarity import SimilarityEngine, pairwise_similarity
from .simhash import SimHash
from .lexicon import DEFAULT_USER_DICT_FILES
from .stopwords import DEFAULT_STOPWORD_FILES, load_stopwords
from .two_stage import TwoStageResult, TwoStageSearchEngine

__all__ = [
	"TfidfVocabularyCache",
	"SimHashCache",
	"SbertEmbeddingsCache",
	"save_tfidf_vocabulary_cache",
	"load_tfidf_vocabulary_cache",
	"save_simhash_cache",
	"load_simhash_cache",
	"save_sbert_embeddings_cache",
	"load_sbert_embeddings_cache",
	"ChunkingConfig",
	"ChunkedDocument",
	"chunk_document",
	"chunk_documents",
	"DEFAULT_STOPWORD_FILES",
	"DEFAULT_USER_DICT_FILES",
	"load_stopwords",
	"DataConfig",
	"PipelineConfig",
	"SimHashConfig",
	"TfidfConfig",
	"SBERTConfig",
	"TokenizationConfig",
	"LoadedDocument",
	"InvalidRecord",
	"load_documents",
	"TextPreprocessor",
	"build_preprocessor",
	"CppBridge",
	"WeightedDocument",
	"SimHashFingerprint",
	"HammingCandidate",
	"EngineError",
	"EngineMeta",
	"TopKResult",
	"SimilarityEngine",
	"SbertSimilarityEngine",
	"SbertResult",
	"pairwise_similarity",
	"SimHash",
	"TwoStageResult",
	"TwoStageSearchEngine",
]
