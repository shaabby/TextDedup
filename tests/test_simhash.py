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
