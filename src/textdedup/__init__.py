"""TextDedup Python prototype package."""

from .similarity import SimilarityEngine, pairwise_similarity
from .simhash import SimHash

__all__ = ["SimilarityEngine", "pairwise_similarity", "SimHash"]
