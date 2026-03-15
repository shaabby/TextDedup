from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from textdedup.config import DataConfig, TokenizationConfig
from textdedup.data_loader import iter_jsonl_records, validate_record


def clean_raw2(
    input_path: Path,
    output_dir: Path,
    dedup_mode: str = "id_text",
) -> dict[str, object]:
    if dedup_mode not in {"none", "id", "id_text"}:
        raise ValueError("dedup_mode 仅支持 none/id/id_text")

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    cleaned_path = output_dir / f"{stem}.clean.jsonl"
    rejected_path = output_dir / f"{stem}.rejected.jsonl"
    stats_path = output_dir / f"{stem}.stats.json"

    # Ensure deterministic reruns.
    for p in (cleaned_path, rejected_path, stats_path):
        if p.exists():
            p.unlink()

    data_config = DataConfig(
        data_path=str(input_path),
        file_format="jsonl",
        text_field="text",
        id_field="doc_id",
    )
    tokenization = TokenizationConfig(lowercase=True)

    seen_ids: set[str] = set()
    seen_text_sha1: set[str] = set()

    total = 0
    accepted = 0
    rejected = 0

    with cleaned_path.open("w", encoding="utf-8") as cleaned_f, rejected_path.open(
        "w", encoding="utf-8"
    ) as rejected_f:
        for line_no, record in iter_jsonl_records(Path(data_config.data_path)):
            total += 1
            doc, error = validate_record(
                raw=record,
                line_no=line_no,
                text_field=data_config.text_field,
                id_field=data_config.id_field,
                preprocessor=tokenization,
            )

            if error is not None:
                rejected += 1
                rejected_f.write(
                    json.dumps(
                        {
                            "line_no": error.line_no,
                            "reason": error.reason,
                            "raw_id": error.raw_id,
                            "raw_url": error.raw_url,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            assert doc is not None

            if dedup_mode in {"id", "id_text"} and doc.id in seen_ids:
                rejected += 1
                rejected_f.write(
                    json.dumps(
                        {
                            "line_no": line_no,
                            "reason": "duplicate_id",
                            "raw_id": doc.id,
                            "raw_url": doc.url,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            text_sha1 = hashlib.sha1(doc.text.encode("utf-8")).hexdigest()
            if dedup_mode == "id_text" and text_sha1 in seen_text_sha1:
                rejected += 1
                rejected_f.write(
                    json.dumps(
                        {
                            "line_no": line_no,
                            "reason": "duplicate_text",
                            "raw_id": doc.id,
                            "raw_url": doc.url,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            if dedup_mode in {"id", "id_text"}:
                seen_ids.add(doc.id)
            if dedup_mode == "id_text":
                seen_text_sha1.add(text_sha1)
            accepted += 1

            cleaned_f.write(
                json.dumps(
                    {
                        "id": doc.id,
                        "doc_id": doc.id,
                        "source": doc.source,
                        "title": doc.title,
                        "language": doc.language,
                        "text": doc.text,
                        "url": doc.url,
                        "text_length": len(doc.text),
                        "text_sha1": text_sha1,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    stats = {
        "input": str(input_path),
        "dedup_mode": dedup_mode,
        "output_clean": str(cleaned_path),
        "output_rejected": str(rejected_path),
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "accept_rate": round(accepted / total, 6) if total else 0.0,
    }

    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗 data/raw2 原始 JSONL 数据")
    parser.add_argument("--input", default="data/raw2/raw_docs.jsonl", help="输入 JSONL 文件路径")
    parser.add_argument("--output-dir", default="data/raw2/clean", help="清洗输出目录")
    parser.add_argument(
        "--dedup-mode",
        default="id_text",
        choices=["none", "id", "id_text"],
        help="去重模式: none=不去重, id=仅按id去重, id_text=按id和文本去重",
    )
    args = parser.parse_args()

    clean_raw2(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        dedup_mode=args.dedup_mode,
    )


if __name__ == "__main__":
    main()
