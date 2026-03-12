from textdedup.two_stage import TwoStageSearchEngine


def test_two_stage_query_returns_relevant_top_result() -> None:
    texts = [
        "北京今天天气晴朗，适合郊游",
        "机器学习用于文本相似度计算",
        "北京今日阳光明媚非常适合出行",
        "数据库索引优化可以提升检索效率",
    ]

    engine = TwoStageSearchEngine()
    engine.fit(texts)

    results = engine.query("北京天气阳光很好适合出游", top_k=2, candidate_k=3)
    assert len(results) == 2

    print(
        "two-stage top2=",
        [
            {
                "index": r.index,
                "score": round(r.score, 6),
                "simhash_distance": r.simhash_distance,
            }
            for r in results
        ],
    )

    assert results[0].index in (0, 2)
    assert results[0].score >= results[1].score


def test_two_stage_candidate_limit_is_respected() -> None:
    texts = [
        "苹果手机发布新款",
        "安卓手机系统更新",
        "苹果新品发布会在秋季举行",
        "足球比赛进入加时赛",
    ]

    engine = TwoStageSearchEngine()
    engine.fit(texts)

    results = engine.query("苹果发布会", top_k=3, candidate_k=2)
    print("two-stage limited candidates count=", len(results))

    assert len(results) == 2
