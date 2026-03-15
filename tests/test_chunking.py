from __future__ import annotations

from textdedup.chunking import (
    ChunkingConfig,
    build_paragraph_chunks,
    build_sliding_sentence_chunks,
    chunk_document,
    select_chunk_spans,
)
from textdedup.data_loader import LoadedDocument


def test_build_paragraph_chunks_splits_on_newlines() -> None:
    text = "第一段第一句。\n第二段第一句。\n第三段第一句。"
    spans = build_paragraph_chunks(text, max_chars=100)
    parts = [text[start:end] for start, end in spans]
    assert parts == ["第一段第一句。", "第二段第一句。", "第三段第一句。"]


def test_build_paragraph_chunks_splits_long_paragraph_by_sentences() -> None:
    text = "第一句很长。第二句也很长。第三句还是很长。"
    spans = build_paragraph_chunks(text, max_chars=8)
    parts = [text[start:end] for start, end in spans]
    assert parts == ["第一句很长。", "第二句也很长。", "第三句还是很长。"]


def test_build_sliding_sentence_chunks_uses_overlap() -> None:
    text = "第一句。第二句。第三句。第四句。"
    spans = build_sliding_sentence_chunks(text, window_sentences=2, stride_sentences=1)
    parts = [text[start:end] for start, end in spans]
    assert parts == ["第一句。第二句。", "第二句。第三句。", "第三句。第四句。"]


def test_hybrid_falls_back_to_sliding_when_no_paragraphs() -> None:
    text = "第一句。第二句。第三句。第四句。"
    spans = select_chunk_spans(text, ChunkingConfig(mode="hybrid", window_sentences=3, stride_sentences=2))
    parts = [text[start:end] for start, end in spans]
    assert parts == ["第一句。第二句。第三句。", "第三句。第四句。"]


def test_chunk_document_keeps_offsets_and_normalizes_text() -> None:
    document = LoadedDocument(id="doc-1", text="占位", title="标题", source="src", language="zh")
    raw_text = " 第一段。\n\n第二段。 "
    chunks = chunk_document(document, raw_text, ChunkingConfig(mode="paragraph", max_chars=50))
    assert [chunk.text for chunk in chunks] == ["第一段。", "第二段。"]
    assert chunks[0].chunk_id == "doc-1#0000"
    assert chunks[1].start_offset > chunks[0].end_offset
