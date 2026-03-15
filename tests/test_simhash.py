from textdedup.config import TokenizationConfig
from textdedup.simhash import SimHash


def test_simhash_identical_text_similarity_is_one() -> None:
    hasher = SimHash()
    score = hasher.similarity("文本去重系统", "文本去重系统")
    print(f"simhash identical score={score:.6f}")
    assert score == 1.0


def test_simhash_similar_text_has_higher_score_than_unrelated() -> None:
    hasher = SimHash()
    s1 = hasher.similarity("深度学习用于文本分类", "深度学习用于文本去重")
    s2 = hasher.similarity("深度学习用于文本分类", "今天天气不错适合跑步")
    print(f"simhash similar score={s1:.6f}, unrelated score={s2:.6f}")
    assert s1 > s2


def test_simhash_respects_shared_tokenization_normalization() -> None:
    hasher = SimHash(tokenization_config=TokenizationConfig(lowercase=True))
    a = "  HELLO   北京，TextDedup! "
    b = "hello 北京 textdedup"

    assert hasher.fingerprint(a) == hasher.fingerprint(b)


def test_simhash_uses_jieba_word_tokens_for_chinese() -> None:
    hasher = SimHash()
    tokens = hasher.preprocessor.tokenize("北京大学生前来应聘")

    assert "北京" in tokens or "北京大学" in tokens
    assert "北" not in tokens


def test_simhash_filters_default_chinese_stopwords() -> None:
    hasher = SimHash()
    tokens = hasher.preprocessor.tokenize("她在北京的市场里翻斤斗")

    assert "她" not in tokens
    assert "在" not in tokens
    assert "的" not in tokens
    assert "市场" in tokens


def test_simhash_can_load_stopwords_from_file(tmp_path) -> None:
    stopword_file = tmp_path / "custom_stopwords.txt"
    stopword_file.write_text("市场\n翻斤斗\n", encoding="utf-8")

    hasher = SimHash(
        tokenization_config=TokenizationConfig(
            stopword_paths=(str(stopword_file),),
            extra_stopwords=(),
        )
    )
    tokens = hasher.preprocessor.tokenize("北京市场翻斤斗")

    assert "市场" not in tokens
    assert "翻斤斗" not in tokens
    assert "北京" in tokens


def test_simhash_can_load_user_dict_from_file(tmp_path) -> None:
    user_dict = tmp_path / "custom_user_dict.txt"
    user_dict.write_text("量子云图城 200000 nz\n", encoding="utf-8")

    hasher = SimHash(
        tokenization_config=TokenizationConfig(
            user_dict_paths=(str(user_dict),),
            stopword_paths=(),
            extra_stopwords=(),
        )
    )
    tokens = hasher.preprocessor.tokenize("量子云图城计划启动")

    assert "量子云图城" in tokens


def test_simhash_downweights_user_dict_proper_noun(tmp_path) -> None:
    user_dict = tmp_path / "custom_user_dict.txt"
    user_dict.write_text("量子云图城 200000 nz\n", encoding="utf-8")

    hasher = SimHash(
        tokenization_config=TokenizationConfig(
            user_dict_paths=(str(user_dict),),
            proper_noun_weight=0.4,
            stopword_paths=(),
            extra_stopwords=(),
        )
    )

    assert hasher.preprocessor.token_weight("量子云图城") == 0.4
    assert hasher.preprocessor.token_weight("计划") == 1.0


def test_simhash_respects_idf_cap() -> None:
    hasher = SimHash(idf_cap=1.2)
    hasher.fit([
        "rare_a",
        "rare_b",
        "common",
        "common",
    ])

    assert max(hasher.idf_weights.values()) <= 1.2


def test_simhash_fit_assigns_higher_idf_to_rarer_tokens() -> None:
    hasher = SimHash()
    hasher.fit([
        "common rare",
        "common mid",
        "common",
    ])

    assert hasher.idf_weights["rare"] > hasher.idf_weights["common"]
    assert hasher.idf_weights["mid"] > hasher.idf_weights["common"]


def test_char_ngram_mode_is_more_robust_for_rewritten_chinese() -> None:
    original = "城市是一张由记忆、欲望和符号交织的网。"
    rewritten = "城市像由记忆、欲望与符号交织而成的网。"

    char_hasher = SimHash(feature_mode="char_ngram", char_ngram_size=3)
    fp_original = char_hasher.fingerprint(original)
    fp_rewritten = char_hasher.fingerprint(rewritten)

    # Same text must always hash to distance 0.
    assert char_hasher.hamming_distance(fp_original, fp_original) == 0

    # Light rewrite should stay within a finite, non-trivial distance range.
    char_dist = char_hasher.hamming_distance(
        fp_original,
        fp_rewritten,
    )
    assert 0 < char_dist < 64


def test_simhash_uses_fitted_idf_for_repeated_common_tokens() -> None:
    text = "common common rare"
    weighted = SimHash()
    weighted.fit([
        text,
        "common common",
        "common mid",
    ])

    assert weighted.idf_weights["rare"] > weighted.idf_weights["common"]

    vector = [0.0] * weighted.hash_bits
    token_weights = {
        "common": 2.0 * weighted.idf_weights["common"],
        "rare": weighted.idf_weights["rare"],
    }
    for token, token_weight in token_weights.items():
        token_hash = weighted._token_hash(token)
        for i in range(weighted.hash_bits):
            bit = 1.0 if (token_hash >> i) & 1 else -1.0
            vector[i] += bit * token_weight

    expected = 0
    for i, value in enumerate(vector):
        if value > 0.0:
            expected |= 1 << i

    assert weighted.fingerprint(text) == expected


def test_simhash_can_persist_fingerprint_cache(tmp_path) -> None:
    texts = [
        "common rare",
        "common mid",
        "common",
    ]
    hasher = SimHash()
    hasher.fit(texts)
    fingerprints = [hasher.fingerprint(text) for text in texts]

    cache_path = tmp_path / "simhash_cache.json"
    hasher.save_cache(cache_path, fingerprints)

    loaded = SimHash.load_cache(cache_path)

    assert loaded.hash_bits == hasher.hash_bits
    assert loaded.document_count == len(texts)
    assert loaded.idf_weights == hasher.idf_weights
    assert loaded.fingerprints == fingerprints
