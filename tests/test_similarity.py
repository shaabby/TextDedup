from textdedup.similarity import SimilarityEngine, pairwise_similarity


def test_pairwise_similarity_identical_texts_high_score() -> None:
    score = pairwise_similarity("机器学习用于文本查重", "机器学习用于文本查重")
    print(f"pairwise score={score:.6f}")
    assert score > 0.99


def test_query_returns_most_similar_text() -> None:
    texts = [
        "今天北京天气很好，适合出行",
        "深度学习可以用于自然语言处理",
        "北京今天阳光明媚非常适合旅游",
    ]
    engine = SimilarityEngine()
    engine.fit(texts)

    result = engine.query("北京天气阳光明媚", top_k=1)[0]
    print(f"top1 index={result.index}, score={result.score:.6f}, text={result.text}")
    assert result.index in (0, 2)
    assert result.score > 0.2


def test_deduplicate_finds_near_duplicate_pairs() -> None:
    texts = [
        "A quick brown fox jumps over the lazy dog",
        "A quick brown fox jumps over a lazy dog",
        "Completely different sentence here",
    ]
    engine = SimilarityEngine()
    engine.fit(texts)

    pairs = engine.deduplicate(threshold=0.6)
    print(f"duplicate pairs={pairs}")
    assert any((i, j) == (0, 1) for i, j, _ in pairs)
