from textdedup.config import TfidfConfig, TokenizationConfig
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


def test_similarity_uses_shared_normalization() -> None:
    texts = ["HELLO   北京", "完全不相关的句子"]
    engine = SimilarityEngine(tokenization_config=TokenizationConfig(lowercase=True))
    engine.fit(texts)

    result = engine.query("hello 北京", top_k=1)[0]
    assert result.index == 0


def test_similarity_can_disable_lowercase_normalization() -> None:
    texts = ["hello world", "HELLO WORLD"]
    engine = SimilarityEngine(tokenization_config=TokenizationConfig(lowercase=False))
    engine.fit(texts)

    result = engine.query("HELLO WORLD", top_k=1)[0]
    assert result.index == 1


def test_similarity_uses_jieba_word_tokens_for_chinese() -> None:
    engine = SimilarityEngine()
    engine.fit([
        "北京大学生前来应聘",
        "上海今天下雨",
    ])

    features = set(engine.vectorizer.get_feature_names_out())
    assert "北京" in features
    assert "北京 大学生" in features or "北京大学" in features or "大学生" in features


def test_similarity_filters_default_chinese_stopwords() -> None:
    engine = SimilarityEngine()
    engine.fit([
        "她在北京的市场里翻斤斗",
        "上海今天下雨",
    ])

    features = set(engine.vectorizer.get_feature_names_out())
    assert "的" not in features
    assert "她" not in features
    assert "在" not in features
    assert "市场" in features


def test_similarity_can_load_stopwords_from_file(tmp_path) -> None:
    stopword_file = tmp_path / "custom_stopwords.txt"
    stopword_file.write_text("市场\n翻斤斗\n", encoding="utf-8")

    engine = SimilarityEngine(
        tokenization_config=TokenizationConfig(
            stopword_paths=(str(stopword_file),),
            extra_stopwords=(),
        )
    )
    engine.fit([
        "北京市场翻斤斗",
        "上海今天下雨",
    ])

    features = set(engine.vectorizer.get_feature_names_out())
    assert "市场" not in features
    assert "翻斤斗" not in features
    assert "北京" in features


def test_similarity_can_load_user_dict_from_file(tmp_path) -> None:
    user_dict = tmp_path / "custom_user_dict.txt"
    user_dict.write_text("量子云图城 200000 nz\n", encoding="utf-8")

    engine = SimilarityEngine(
        tokenization_config=TokenizationConfig(
            user_dict_paths=(str(user_dict),),
            stopword_paths=(),
            extra_stopwords=(),
        )
    )
    engine.fit([
        "量子云图城计划启动",
        "上海今天下雨",
    ])

    features = set(engine.vectorizer.get_feature_names_out())
    assert "量子云图城" in features


def test_similarity_downweights_user_dict_proper_noun_feature(tmp_path) -> None:
    user_dict = tmp_path / "custom_user_dict.txt"
    user_dict.write_text("量子云图城 200000 nz\n", encoding="utf-8")

    engine = SimilarityEngine(
        tokenization_config=TokenizationConfig(
            user_dict_paths=(str(user_dict),),
            proper_noun_weight=0.5,
            stopword_paths=(),
            extra_stopwords=(),
        )
    )
    engine.fit([
        "量子云图城计划启动",
        "普通计划启动",
    ])

    vocab = engine.vectorizer.vocabulary_
    assert engine._feature_weights is not None
    assert engine._feature_weights[vocab["量子云图城"]] < engine._feature_weights[vocab["计划"]]


def test_similarity_respects_idf_cap_scaling() -> None:
    engine = SimilarityEngine(config=TfidfConfig(idf_cap=1.2))
    engine.fit([
        "rarea",
        "rareb",
        "common",
        "common",
    ])

    vocab = engine.vectorizer.vocabulary_
    idx = vocab["rarea"]
    assert engine._feature_weights is not None
    effective_weighted_idf = engine.vectorizer.idf_[idx] * engine._feature_weights[idx]
    assert effective_weighted_idf <= 1.2


def test_similarity_can_persist_tfidf_vocabulary_cache(tmp_path) -> None:
    texts = [
        "今天北京天气很好，适合出行",
        "北京今天阳光明媚非常适合旅游",
        "深度学习可以用于自然语言处理",
    ]
    engine = SimilarityEngine()
    engine.fit(texts)

    cache_path = tmp_path / "tfidf_vocab.json"
    engine.save_vocabulary_cache(cache_path)

    loaded = SimilarityEngine.load_vocabulary_cache(cache_path)
    exported = engine.export_vocabulary_cache()

    assert loaded.document_count == len(texts)
    assert loaded.analyzer == exported.analyzer
    assert loaded.ngram_range == exported.ngram_range
    assert loaded.min_df == exported.min_df
    assert loaded.vocabulary == exported.vocabulary
    assert loaded.idf == exported.idf
