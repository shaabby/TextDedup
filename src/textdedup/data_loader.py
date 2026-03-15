from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

from .config import DataConfig, TokenizationConfig
from .preprocess import build_preprocessor


@dataclass(frozen=True)
class LoadedDocument:
    """Validated and normalized document for offline dedup pipeline."""

    id: str
    text: str
    source: str = ""
    title: str = ""
    language: str = "unknown"
    url: str = ""


@dataclass(frozen=True)
class InvalidRecord:
    """Invalid input record metadata for diagnostics."""

    line_no: int
    reason: str
    raw_id: str = ""
    raw_url: str = ""


def iter_jsonl_records(file_path: Path) -> Iterator[Tuple[int, Dict[str, object]]]:
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def iter_csv_records(file_path: Path) -> Iterator[Tuple[int, Dict[str, object]]]:
    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_no, row in enumerate(reader, start=2):
            yield row_no, dict(row)


def iter_raw_records(data_path: Path, file_format: str) -> Iterator[Tuple[int, Dict[str, object]]]:
    if file_format == "jsonl":
        yield from iter_jsonl_records(data_path)
        return
    if file_format == "csv":
        yield from iter_csv_records(data_path)
        return
    raise ValueError("file_format 仅支持 jsonl 或 csv")


def validate_record(
    raw: Dict[str, object],
    line_no: int,
    text_field: str,
    id_field: str,
    preprocessor: Optional[TokenizationConfig] = None,
) -> Tuple[Optional[LoadedDocument], Optional[InvalidRecord]]:
    raw_id = str(raw.get(id_field) or raw.get("doc_id") or "").strip()
    raw_text = str(raw.get(text_field) or "").strip()
    raw_url = str(raw.get("url") or "").strip()

    if not raw_id:
        return None, InvalidRecord(line_no=line_no, reason="missing_id", raw_url=raw_url)
    if not raw_text:
        return None, InvalidRecord(line_no=line_no, reason="missing_text", raw_id=raw_id, raw_url=raw_url)

    normalizer = build_preprocessor(preprocessor)
    text = normalizer.normalize(raw_text)
    if not text:
        return None, InvalidRecord(line_no=line_no, reason="empty_text_after_normalize", raw_id=raw_id, raw_url=raw_url)

    doc = LoadedDocument(
        id=raw_id,
        text=text,
        source=str(raw.get("source") or "").strip(),
        title=normalizer.normalize(str(raw.get("title") or "")),
        language=str(raw.get("language") or "unknown").strip() or "unknown",
        url=raw_url,
    )
    return doc, None


def load_documents(
    config: DataConfig,
    tokenization_config: Optional[TokenizationConfig] = None,
) -> Tuple[Iterator[LoadedDocument], Iterator[InvalidRecord]]:
    data_path = Path(config.data_path)

    def _iter_valid() -> Iterator[LoadedDocument]:
        for line_no, record in iter_raw_records(data_path, config.file_format):
            doc, error = validate_record(
                raw=record,
                line_no=line_no,
                text_field=config.text_field,
                id_field=config.id_field,
                preprocessor=tokenization_config,
            )
            if error is None and doc is not None:
                yield doc

    def _iter_invalid() -> Iterator[InvalidRecord]:
        for line_no, record in iter_raw_records(data_path, config.file_format):
            _, error = validate_record(
                raw=record,
                line_no=line_no,
                text_field=config.text_field,
                id_field=config.id_field,
                preprocessor=tokenization_config,
            )
            if error is not None:
                yield error

    return _iter_valid(), _iter_invalid()
