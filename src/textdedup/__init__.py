"""TextDedup Python prototype package."""

from .similarity import SimilarityEngine, pairwise_similarity
from .simhash import SimHash
from .two_stage import TwoStageResult, TwoStageSearchEngine

__all__ = [
	"SimilarityEngine",
	"pairwise_similarity",
	"SimHash",
	"TwoStageResult",
	"TwoStageSearchEngine",
]
