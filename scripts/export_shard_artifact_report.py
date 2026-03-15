from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from textdedup.config import DataConfig, PipelineConfig, SBERTConfig, TfidfConfig, TokenizationConfig
from textdedup.data_loader import load_documents
from textdedup.similarity import SimilarityEngine
from textdedup.two_stage import TwoStageSearchEngine


def _top_tfidf_terms(engine: SimilarityEngine, text: str, limit: int) -> list[dict[str, float]]:
    query_input = text if engine.config.analyzer == "word" else engine.preprocessor.normalize(text)
    vector = engine.vectorizer.transform([query_input])
    if engine._feature_weights is not None:
        vector = vector.multiply(engine._feature_weights)
    row = vector.toarray()[0]
    nonzero_indices = np.flatnonzero(row)
    ranked_indices = nonzero_indices[np.argsort(-row[nonzero_indices])[:limit]]
    feature_names = engine.vectorizer.get_feature_names_out()
    return [
        {
            "term": str(feature_names[i]),
            "score": round(float(row[i]), 6),
        }
        for i in ranked_indices
    ]


def _top_idf_terms(engine: SimilarityEngine, limit: int) -> list[dict[str, float]]:
    feature_names = engine.vectorizer.get_feature_names_out()
    idf = engine.vectorizer.idf_.copy()
    if engine.config.idf_cap is not None:
        idf = np.minimum(idf, engine.config.idf_cap)
    ranked_indices = np.argsort(-idf)[:limit]
    return [
        {
            "term": str(feature_names[i]),
            "idf": round(float(idf[i]), 6),
        }
        for i in ranked_indices
    ]


def build_report(
    input_path: Path,
    output_dir: Path,
    sample_docs: int,
    top_terms: int,
    top_matches: int,
    enable_sbert: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenization = TokenizationConfig(lowercase=True)
    config = DataConfig(
        data_path=str(input_path),
        file_format="jsonl",
        text_field="text",
        id_field="doc_id",
    )

    valid_iter, invalid_iter = load_documents(config=config, tokenization_config=tokenization)
    documents = list(valid_iter)
    invalid_records = list(invalid_iter)
    if not documents:
        raise ValueError("分片内没有可用文档")

    texts = [doc.text for doc in documents]

    similarity = SimilarityEngine(tokenization_config=tokenization)
    similarity.fit(texts)
    tfidf_path = output_dir / f"{input_path.stem}.tfidf_vocab.json"
    similarity.save_vocabulary_cache(tfidf_path)

    first_doc = documents[0]
    first_query_results = similarity.query(first_doc.text, top_k=min(top_matches, len(documents)))
    preview_engine = TwoStageSearchEngine(
        config=PipelineConfig(
            tokenization=tokenization,
            tfidf=TfidfConfig(enabled=True, top_n=min(max(top_matches * 5, top_matches), len(documents))),
            sbert=SBERTConfig(enabled=enable_sbert, top_n=min(top_matches, len(documents))),
            output_top_n=min(top_matches, len(documents)),
        )
    )
    preview_engine.fit(texts)
    preview_results = preview_engine.query(first_doc.text)

    invalid_reason_counts = Counter(record.reason for record in invalid_records)
    sample_payload = []
    for index, doc in enumerate(documents[:sample_docs]):
        sample_payload.append(
            {
                "index": index,
                "doc_id": doc.id,
                "title": doc.title,
                "text_length": len(doc.text),
                "top_tfidf_terms": _top_tfidf_terms(similarity, doc.text, top_terms),
            }
        )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "user_dict_paths": list(tokenization.user_dict_paths),
        "proper_noun_weight": tokenization.proper_noun_weight,
        "stopword_paths": list(tokenization.stopword_paths),
        "valid_documents": len(documents),
        "invalid_documents": len(invalid_records),
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "sbert_enabled": enable_sbert,
        "sbert_local_model_path": preview_engine.config.sbert.local_model_path,
        "sbert_model_name": preview_engine.config.sbert.model_name,
        "tfidf_vocab_path": str(tfidf_path),
        "tfidf_vocabulary_size": len(similarity.vectorizer.vocabulary_),
        "tfidf_idf_cap": similarity.config.idf_cap,
        "top_idf_terms": _top_idf_terms(similarity, top_terms),
        "sample_documents": sample_payload,
        "first_document_query_preview": {
            "doc_id": first_doc.id,
            "title": first_doc.title,
            "tfidf_top_matches": [
                {
                    "doc_id": documents[item.index].id,
                    "title": documents[item.index].title,
                    "score": round(item.score, 6),
                }
                for item in first_query_results
            ],
            "three_stage_top_matches": [
                {
                    "doc_id": documents[item.index].id,
                    "title": documents[item.index].title,
                    "tfidf_score": None if item.tfidf_score is None else round(item.tfidf_score, 6),
                    "sbert_score": None if item.sbert_score is None else round(item.sbert_score, 6),
                    "final_score": round(item.final_score, 6),
                }
                for item in preview_results
            ],
        },
    }

    report_path = output_dir / f"{input_path.stem}.artifact_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "report_path": str(report_path),
        "tfidf_vocab_path": str(tfidf_path),
        "valid_documents": len(documents),
        "invalid_documents": len(invalid_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="导出分片的 TF-IDF/SimHash 测试产物与摘要报告")
    parser.add_argument(
        "--input",
        default="data/raw2/shards/raw_docs_part_00001.jsonl",
        help="输入分片 JSONL 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw2/test_results",
        help="输出测试结果目录",
    )
    parser.add_argument(
        "--sample-docs",
        type=int,
        default=3,
        help="报告中附带的样本文档数量",
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=10,
        help="报告中每个样本文档展示的 TF-IDF 词条数",
    )
    parser.add_argument(
        "--top-matches",
        type=int,
        default=5,
        help="报告中展示的相似结果数量",
    )
    parser.add_argument(
        "--disable-sbert",
        action="store_true",
        help="禁用默认启用的本地 SBERT 三阶段预览",
    )
    args = parser.parse_args()

    summary = build_report(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        sample_docs=args.sample_docs,
        top_terms=args.top_terms,
        top_matches=args.top_matches,
        enable_sbert=not args.disable_sbert,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()