from __future__ import annotations

import json
from pathlib import Path

from scripts.export_shard_artifact_report import build_report


def test_build_report_includes_three_stage_matches_by_default(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "sample.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"doc_id": "1", "title": "北京天气", "text": "北京今天天气晴朗，适合郊游"}, ensure_ascii=False),
                json.dumps({"doc_id": "2", "title": "北京出游", "text": "北京今日阳光明媚非常适合出行"}, ensure_ascii=False),
                json.dumps({"doc_id": "3", "title": "数据库", "text": "数据库索引优化可以提升检索效率"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    class _FakeTwoStageSearchEngine:
        def __init__(self, config):
            self.config = config

        def fit(self, texts):
            self._texts = list(texts)

        def query(self, text, top_k=None, candidate_k=None):
            return [
                type(
                    "_R",
                    (),
                    {
                        "index": 0,
                        "simhash_distance": 0,
                        "tfidf_score": 1.0,
                        "sbert_score": 0.95,
                        "final_score": 0.95,
                    },
                )(),
                type(
                    "_R",
                    (),
                    {
                        "index": 1,
                        "simhash_distance": 2,
                        "tfidf_score": 0.88,
                        "sbert_score": 0.9,
                        "final_score": 0.9,
                    },
                )(),
            ]

    monkeypatch.setattr(
        "scripts.export_shard_artifact_report.TwoStageSearchEngine",
        _FakeTwoStageSearchEngine,
    )

    summary = build_report(
        input_path=input_path,
        output_dir=tmp_path / "out",
        sample_docs=2,
        top_terms=5,
        top_matches=2,
    )

    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    assert report["sbert_enabled"] is True
    assert report["sbert_local_model_path"] == "models/bge-small-zh-v1.5"
    assert len(report["first_document_query_preview"]["three_stage_top_matches"]) == 2
    assert report["first_document_query_preview"]["three_stage_top_matches"][0]["sbert_score"] == 0.95
