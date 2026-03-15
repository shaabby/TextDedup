from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Sequence

from .config import TokenizationConfig
from .data_loader import LoadedDocument
from .preprocess import build_preprocessor

ChunkMode = Literal["paragraph", "sliding", "hybrid"]


@dataclass(frozen=True)
class ChunkingConfig:
    """Controls how long documents are split into searchable chunks."""

    mode: ChunkMode = "hybrid"
    max_chars: int = 500
    window_sentences: int = 4
    stride_sentences: int = 2

    def __post_init__(self) -> None:
        if self.mode not in {"paragraph", "sliding", "hybrid"}:
            raise ValueError("mode 仅支持 paragraph、sliding 或 hybrid")
        if self.max_chars <= 0:
            raise ValueError("max_chars 必须大于 0")
        if self.window_sentences <= 0:
            raise ValueError("window_sentences 必须大于 0")
        if self.stride_sentences <= 0:
            raise ValueError("stride_sentences 必须大于 0")


@dataclass(frozen=True)
class ChunkedDocument:
    """Searchable chunk derived from a source document."""

    chunk_id: str
    source_doc_id: str
    title: str
    source: str
    language: str
    url: str
    text: str
    chunk_index: int
    chunk_mode: ChunkMode
    start_offset: int
    end_offset: int


_SENTENCE_PATTERN = re.compile(r"[^。！？!?；;\n]+(?:[。！？!?；;]+|$)", re.S)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[^\n]+", text):
        start, end = _trim_span(text, match.start(), match.end())
        if start < end:
            spans.append((start, end))
    return spans


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _SENTENCE_PATTERN.finditer(text):
        start, end = _trim_span(text, match.start(), match.end())
        if start < end:
            spans.append((start, end))
    if spans:
        return spans
    stripped = text.strip()
    if not stripped:
        return []
    start = text.find(stripped)
    return [(start, start + len(stripped))]


def _pack_spans_by_chars(text: str, spans: Sequence[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    if not spans:
        return []

    packed: list[tuple[int, int]] = []
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if end - current_start <= max_chars:
            current_end = end
            continue
        packed.append((current_start, current_end))
        current_start, current_end = start, end
    packed.append((current_start, current_end))
    return packed


def build_paragraph_chunks(text: str, max_chars: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    for start, end in paragraph_spans(text):
        if end - start <= max_chars:
            chunks.append((start, end))
            continue
        local_chunks = _pack_spans_by_chars(text[start:end], sentence_spans(text[start:end]), max_chars)
        chunks.extend((start + local_start, start + local_end) for local_start, local_end in local_chunks)
    return chunks


def build_sliding_sentence_chunks(
    text: str,
    window_sentences: int,
    stride_sentences: int,
) -> list[tuple[int, int]]:
    spans = sentence_spans(text)
    if not spans:
        return []
    if len(spans) <= window_sentences:
        return [(spans[0][0], spans[-1][1])]

    chunks: list[tuple[int, int]] = []
    last_start = -1
    for index in range(0, len(spans), stride_sentences):
        window = spans[index : index + window_sentences]
        if not window:
            break
        start, end = window[0][0], window[-1][1]
        if start != last_start:
            chunks.append((start, end))
            last_start = start
        if index + window_sentences >= len(spans):
            break
    return chunks


def select_chunk_spans(text: str, config: ChunkingConfig) -> list[tuple[int, int]]:
    if config.mode == "paragraph":
        return build_paragraph_chunks(text, config.max_chars)
    if config.mode == "sliding":
        return build_sliding_sentence_chunks(text, config.window_sentences, config.stride_sentences)

    paragraph_chunks = build_paragraph_chunks(text, config.max_chars)
    if len(paragraph_chunks) >= 2:
        return paragraph_chunks
    return build_sliding_sentence_chunks(text, config.window_sentences, config.stride_sentences)


def chunk_document(
    document: LoadedDocument,
    raw_text: str,
    chunking_config: ChunkingConfig | None = None,
    tokenization_config: TokenizationConfig | None = None,
) -> list[ChunkedDocument]:
    chunking = chunking_config or ChunkingConfig()
    preprocessor = build_preprocessor(tokenization_config)
    chunks: list[ChunkedDocument] = []
    for index, (start, end) in enumerate(select_chunk_spans(raw_text, chunking)):
        normalized = preprocessor.normalize(raw_text[start:end])
        if not normalized:
            continue
        chunks.append(
            ChunkedDocument(
                chunk_id=f"{document.id}#{index:04d}",
                source_doc_id=document.id,
                title=document.title,
                source=document.source,
                language=document.language,
                url=document.url,
                text=normalized,
                chunk_index=index,
                chunk_mode=chunking.mode,
                start_offset=start,
                end_offset=end,
            )
        )
    return chunks


def chunk_documents(
    documents: Sequence[LoadedDocument],
    raw_texts: Sequence[str],
    chunking_config: ChunkingConfig | None = None,
    tokenization_config: TokenizationConfig | None = None,
) -> list[ChunkedDocument]:
    if len(documents) != len(raw_texts):
        raise ValueError("documents 与 raw_texts 数量必须一致")
    chunks: list[ChunkedDocument] = []
    for document, raw_text in zip(documents, raw_texts):
        chunks.extend(chunk_document(document, raw_text, chunking_config, tokenization_config))
    return chunks