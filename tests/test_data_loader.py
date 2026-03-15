import json
from pathlib import Path

from textdedup.config import DataConfig, TokenizationConfig
from textdedup.data_loader import iter_jsonl_records, validate_record


def test_iter_jsonl_records_streams_rows(tmp_path: Path) -> None:
    file_path = tmp_path / "input.jsonl"
    rows = [
        {"doc_id": "a", "text": "hello"},
        {"doc_id": "b", "text": "world"},
    ]
    file_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    records = list(iter_jsonl_records(file_path))
    assert len(records) == 2
    assert records[0][0] == 1
    assert records[1][1]["doc_id"] == "b"


def test_validate_record_normalizes_and_checks_required_fields() -> None:
    config = DataConfig(text_field="text", id_field="doc_id")
    tokenization = TokenizationConfig(lowercase=True)

    doc, err = validate_record(
        raw={"doc_id": "X1", "text": " Hello   WORLD ", "title": " A  B "},
        line_no=1,
        text_field=config.text_field,
        id_field=config.id_field,
        preprocessor=tokenization,
    )
    assert err is None
    assert doc is not None
    assert doc.id == "X1"
    assert doc.text == "hello world"
    assert doc.title == "a b"

    doc2, err2 = validate_record(
        raw={"doc_id": "", "text": "abc"},
        line_no=2,
        text_field=config.text_field,
        id_field=config.id_field,
        preprocessor=tokenization,
    )
    assert doc2 is None
    assert err2 is not None
    assert err2.reason == "missing_id"
