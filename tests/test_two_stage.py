import pytest

from textdedup.config import PipelineConfig, SBERTConfig, TfidfConfig
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

    results = engine.query("北京天气阳光很好适合出游", top_k=2, tfidf_candidate_k=4)
    assert len(results) == 2

    print(
        "two-stage top2=",
        [
            {
                "index": r.index,
                "score": round(r.score, 6),
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

    results = engine.query("苹果发布会", top_k=3, tfidf_candidate_k=2)
    print("two-stage limited candidates count=", len(results))

    assert len(results) == 2


def test_two_stage_uses_config_defaults_when_query_params_omitted() -> None:
    texts = [
        "北京今天天气晴朗，适合郊游",
        "北京今日阳光明媚非常适合出行",
        "数据库索引优化可以提升检索效率",
    ]

    config = PipelineConfig(
        output_top_n=1,
        tfidf=TfidfConfig(enabled=True, top_n=3),
    )
    engine = TwoStageSearchEngine(config=config)
    engine.fit(texts)

    results = engine.query("北京天气阳光很好适合出游")
    assert len(results) == 1


def test_two_stage_requires_tfidf() -> None:
    texts = [
        "苹果手机发布新款",
        "苹果新品发布会在秋季举行",
        "足球比赛进入加时赛",
    ]
    config = PipelineConfig(
        output_top_n=2,
        tfidf=TfidfConfig(enabled=False, top_n=2),
        sbert=SBERTConfig(enabled=False, top_n=2),
    )

    engine = TwoStageSearchEngine(config=config)
    engine.fit(texts)
    with pytest.raises(RuntimeError, match="必须启用 TF-IDF"):
        engine.query("苹果发布会")


def test_two_stage_can_enable_sbert_with_fake_backend(monkeypatch) -> None:
    class _FakeSbertEngine:
        def __init__(self, config):
            self._texts = []

        def fit(self, texts):
            self._texts = list(texts)

        def query(self, text, top_k=10):
            # Deterministic descending scores by index order.
            return [
                type("_R", (), {"index": i, "score": 1.0 - (i * 0.1), "text": self._texts[i]})
                for i in range(min(top_k, len(self._texts)))
            ]

    monkeypatch.setattr("textdedup.two_stage.SbertSimilarityEngine", _FakeSbertEngine)

    texts = [
        "北京今天天气晴朗，适合郊游",
        "北京今日阳光明媚非常适合出行",
        "数据库索引优化可以提升检索效率",
    ]
    config = PipelineConfig(
        output_top_n=2,
        tfidf=TfidfConfig(enabled=True, top_n=3),
        sbert=SBERTConfig(enabled=True, top_n=2),
    )
    engine = TwoStageSearchEngine(config=config)
    engine.fit(texts)

    results = engine.query("北京天气阳光很好适合出游")
    assert len(results) == 2
    assert results[0].sbert_score is not None
    assert results[0].score == results[0].final_score


def test_two_stage_without_sbert_returns_tfidf_scores() -> None:
    texts = [
        "卡尔维诺 描写 城市 记忆 欲望",
        "看不见的城市 记忆 符号",
        "今天的天气不错",
    ]
    config = PipelineConfig(
        output_top_n=2,
        tfidf=TfidfConfig(enabled=True, top_n=2),
        sbert=SBERTConfig(enabled=False, top_n=2),
    )

    engine = TwoStageSearchEngine(config=config)
    engine.fit(texts)
    results = engine.query("城市的记忆与符号", top_k=2, tfidf_candidate_k=3)

    assert len(results) == 2
    assert all(item.tfidf_score is not None for item in results)
    assert all(item.sbert_score is None for item in results)


def test_two_stage_tfidf_candidate_k_limits_rerank_pool() -> None:
    texts = [
        "苹果 手机 新品 发布",
        "苹果 发布会 秋季",
        "足球 比赛 加时赛",
        "数据库 索引 优化",
    ]
    config = PipelineConfig(
        output_top_n=2,
        tfidf=TfidfConfig(enabled=True, top_n=2),
        sbert=SBERTConfig(enabled=False, top_n=2),
    )

    engine = TwoStageSearchEngine(config=config)
    engine.fit(texts)
    results = engine.query("苹果 发布", top_k=2, tfidf_candidate_k=1)

    assert len(results) == 1
