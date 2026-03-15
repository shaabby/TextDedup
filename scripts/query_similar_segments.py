from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import numpy as np
import sys
import time
import tomllib
from pathlib import Path

from textdedup import (
    ChunkedDocument,
    ChunkingConfig,
    DataConfig,
    PipelineConfig,
    SBERTConfig,
    SbertSimilarityEngine,
    TfidfConfig,
    TwoStageSearchEngine,
)
from textdedup.chunking import chunk_document
from textdedup.data_loader import iter_raw_records, validate_record


def _load_toml(path: str) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _resolve_dirs_to_files(dirs: list[str], file_format: str, glob_pattern: str) -> list[str]:
    """Expand each directory entry to a sorted list of files matching glob_pattern."""
    default_glob = f"*.{file_format}"
    pattern = glob_pattern.strip() if glob_pattern.strip() else default_glob
    found: list[str] = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            raise NotADirectoryError(f"数据库目录不存在或不是文件夹: {d}")
        found.extend(sorted(str(f) for f in p.glob(pattern) if f.is_file()))
    return found


def _toml_to_arg_defaults(data: dict) -> dict:
    """Flatten TOML sections to argparse dest-name keys (non-boolean fields only)."""
    d: dict = {}
    q = data.get("query", {})
    if "text" in q and q["text"].strip():
        d["query"] = q["text"].strip()
    if "file" in q:
        d["query_file"] = q["file"]
    db = data.get("database", {})
    # `dirs` (preferred) → pass through to input_dir; legacy `inputs`/`input` → direct file list.
    if "dirs" in db:
        raw = db["dirs"]
        d["input_dir"] = raw if isinstance(raw, list) else [raw]
    if "inputs" in db:
        raw = db["inputs"]
        d["input"] = raw if isinstance(raw, list) else [raw]
    elif "input" in db:
        raw = db["input"]
        d["input"] = raw if isinstance(raw, list) else [raw]
    if "file_glob" in db:
        d["file_glob"] = db["file_glob"]
    for key in ("file_format", "text_field", "id_field"):
        if key in db:
            d[key] = db[key]
    ch = data.get("chunking", {})
    if "mode" in ch:
        d["chunk_mode"] = ch["mode"]
    for key in ("max_chars", "window_sentences", "stride_sentences"):
        if key in ch:
            d[key] = ch[key]
    pl = data.get("pipeline", {})
    for key in ("top_k", "tfidf_candidate_k", "sbert_top_n"):
        if key in pl:
            d[key] = pl[key]
    if "retrieval_mode" in pl:
        d["retrieval_mode"] = pl["retrieval_mode"]

    cache = data.get("cache", {})
    if "chunk_text_cache_file" in cache:
        d["two_stage_cache_file"] = cache["chunk_text_cache_file"]
    elif "two_stage_file" in cache:
        d["two_stage_cache_file"] = cache["two_stage_file"]
    if "chunk_text_cache_meta_file" in cache:
        d["two_stage_cache_meta_file"] = cache["chunk_text_cache_meta_file"]
    elif "two_stage_meta_file" in cache:
        d["two_stage_cache_meta_file"] = cache["two_stage_meta_file"]
    if "reuse_chunk_text_cache" in cache:
        d["reuse_two_stage_cache"] = cache["reuse_chunk_text_cache"]
    elif "reuse_two_stage_cache" in cache:
        d["reuse_two_stage_cache"] = cache["reuse_two_stage_cache"]
    if "sbert_embeddings_file" in cache:
        d["sbert_cache_file"] = cache["sbert_embeddings_file"]
    if "sbert_meta_file" in cache:
        d["sbert_cache_meta_file"] = cache["sbert_meta_file"]
    if "reuse_sbert_cache" in cache:
        d["reuse_sbert_cache"] = cache["reuse_sbert_cache"]
    out = data.get("output", {})
    if "file" in out:
        d["output_file"] = out["file"]
    rt = data.get("runtime", {})
    if "progress_interval" in rt:
        d["progress_interval"] = rt["progress_interval"]
    return d


def _read_query(args: argparse.Namespace) -> str:
    if args.query_file:
        return Path(args.query_file).read_text(encoding="utf-8").strip()
    if args.query:
        return args.query.strip()
    raise ValueError("必须通过 --query、--query-file 或配置文件 [query] 节提供查询文本")


def _progress(enabled: bool, start_ts: float, message: str) -> None:
    if not enabled:
        return
    elapsed = time.perf_counter() - start_ts
    print(f"[进度 +{elapsed:7.1f}s] {message}", file=sys.stderr)


def _build_sbert_cache_signature(
    *,
    input_paths: list[str],
    data_config: DataConfig,
    chunking_config: ChunkingConfig,
    sbert_config: SBERTConfig,
) -> dict:
    file_items: list[dict] = []
    digest = hashlib.sha256()
    for path in sorted(input_paths):
        p = Path(path)
        stat = p.stat()
        item = {
            "path": str(p),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        file_items.append(item)
        digest.update(item["path"].encode("utf-8"))
        digest.update(str(item["size"]).encode("utf-8"))
        digest.update(str(item["mtime_ns"]).encode("utf-8"))

    chunk_cfg = {
        "mode": chunking_config.mode,
        "max_chars": chunking_config.max_chars,
        "window_sentences": chunking_config.window_sentences,
        "stride_sentences": chunking_config.stride_sentences,
    }
    for k in ("mode", "max_chars", "window_sentences", "stride_sentences"):
        digest.update(f"{k}={chunk_cfg[k]}".encode("utf-8"))

    data_fields = {
        "file_format": data_config.file_format,
        "text_field": data_config.text_field,
        "id_field": data_config.id_field,
    }
    for k in ("file_format", "text_field", "id_field"):
        digest.update(f"{k}={data_fields[k]}".encode("utf-8"))

    model_ref = sbert_config.local_model_path or sbert_config.model_name
    digest.update(f"model={model_ref}".encode("utf-8"))
    digest.update(f"normalize={sbert_config.normalize_embeddings}".encode("utf-8"))

    return {
        "signature": digest.hexdigest(),
        "files": file_items,
        "chunking": chunk_cfg,
        "data_fields": data_fields,
        "model_ref": model_ref,
        "normalize_embeddings": bool(sbert_config.normalize_embeddings),
    }


def _build_two_stage_cache_signature(
    *,
    input_paths: list[str],
    data_config: DataConfig,
    chunking_config: ChunkingConfig,
) -> dict:
    digest = hashlib.sha256()
    file_items: list[dict] = []
    for path in sorted(input_paths):
        p = Path(path)
        stat = p.stat()
        item = {"path": str(p), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        file_items.append(item)
        digest.update(item["path"].encode("utf-8"))
        digest.update(str(item["size"]).encode("utf-8"))
        digest.update(str(item["mtime_ns"]).encode("utf-8"))

    digest.update(f"file_format={data_config.file_format}".encode("utf-8"))
    digest.update(f"text_field={data_config.text_field}".encode("utf-8"))
    digest.update(f"id_field={data_config.id_field}".encode("utf-8"))
    digest.update(f"chunk_mode={chunking_config.mode}".encode("utf-8"))
    digest.update(f"max_chars={chunking_config.max_chars}".encode("utf-8"))
    digest.update(f"window={chunking_config.window_sentences}".encode("utf-8"))
    digest.update(f"stride={chunking_config.stride_sentences}".encode("utf-8"))

    return {
        "signature": digest.hexdigest(),
        "files": file_items,
        "data_fields": {
            "file_format": data_config.file_format,
            "text_field": data_config.text_field,
            "id_field": data_config.id_field,
        },
        "chunking": {
            "mode": chunking_config.mode,
            "max_chars": chunking_config.max_chars,
            "window_sentences": chunking_config.window_sentences,
            "stride_sentences": chunking_config.stride_sentences,
        },
    }


def _serialize_chunk(chunk: ChunkedDocument, source_file: str) -> dict:
    payload = asdict(chunk)
    payload["source_file"] = source_file
    return payload


def _deserialize_chunks(items: list[dict]) -> tuple[list[ChunkedDocument], list[str]]:
    chunks: list[ChunkedDocument] = []
    source_files: list[str] = []
    for item in items:
        source_files.append(str(item["source_file"]))
        chunk_payload = dict(item)
        chunk_payload.pop("source_file", None)
        chunks.append(ChunkedDocument(**chunk_payload))
    return chunks, source_files


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在本地文本库中查询相似片段（可通过 --config 使用 TOML 配置文件一键查重）"
    )
    parser.add_argument("--config", metavar="FILE", help="TOML 配置文件路径")
    parser.add_argument("--query", help="直接传入查询片段")
    parser.add_argument("--query-file", help="从文件读取查询片段")
    parser.add_argument(
        "--input-dir",
        nargs="+",
        default=[],
        metavar="DIR",
        help="一个或多个文本库文件夹，自动扫描其中所有符合格式的文件",
    )
    parser.add_argument(
        "--file-glob",
        default="",
        metavar="PATTERN",
        help="文件名 glob 过滤（仅对 --input-dir 扫描生效，如 '*.clean.jsonl'）",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[],
        metavar="FILE",
        help="直接指定一个或多个 JSONL/CSV 文件路径（与 --input-dir 可同时使用）",
    )
    parser.add_argument("--file-format", default="jsonl", choices=("jsonl", "csv"))
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="doc_id")
    parser.add_argument(
        "--retrieval-mode",
        default="two-stage",
        choices=("two-stage", "sbert-only"),
        help="检索模式：two-stage（TF-IDF + SBERT）或 sbert-only（纯 SBERT）",
    )
    parser.add_argument("--top-k", type=int, default=5, help="返回的相似片段数量")
    parser.add_argument("--tfidf-candidate-k", type=int, default=800, help="进入 TF-IDF 的候选数")
    parser.add_argument(
        "--chunk-mode",
        default="sliding",
        choices=("paragraph", "sliding", "hybrid"),
        help="切片方式：段落、句子滑窗、或混合模式",
    )
    parser.add_argument("--max-chars", type=int, default=500, help="段落模式下单片最大字符数")
    parser.add_argument("--window-sentences", type=int, default=6, help="滑窗模式的窗口句数")
    parser.add_argument("--stride-sentences", type=int, default=3, help="滑窗模式的步长句数")
    parser.add_argument("--disable-tfidf", action="store_true", help="禁用第二阶段 TF-IDF")
    parser.add_argument("--disable-sbert", action="store_true", help="禁用第三阶段 SBERT")
    parser.add_argument("--sbert-top-n", type=int, default=20, help="进入 SBERT 的候选数")
    parser.add_argument(
        "--two-stage-cache-file",
        default=".cache/query_two_stage_cache.json",
        metavar="FILE",
        help="two-stage 模式缓存文件（切片文本缓存）",
    )
    parser.add_argument(
        "--two-stage-cache-meta-file",
        default=".cache/query_two_stage_cache.meta.json",
        metavar="FILE",
        help="two-stage 模式缓存元数据文件",
    )
    parser.add_argument(
        "--reuse-two-stage-cache",
        action="store_true",
        help="复用 two-stage 缓存（命中签名则跳过重建）",
    )
    parser.add_argument(
        "--rebuild-two-stage-cache",
        action="store_true",
        help="强制重建 two-stage 缓存（覆盖 --reuse-two-stage-cache）",
    )
    parser.add_argument(
        "--sbert-cache-file",
        default=".cache/query_sbert_embeddings.json",
        metavar="FILE",
        help="纯 SBERT 模式的向量缓存文件",
    )
    parser.add_argument(
        "--sbert-cache-meta-file",
        default=".cache/query_sbert_embeddings.meta.json",
        metavar="FILE",
        help="纯 SBERT 模式的缓存元数据文件",
    )
    parser.add_argument(
        "--reuse-sbert-cache",
        action="store_true",
        help="复用纯 SBERT 缓存（命中签名则跳过重新编码）",
    )
    parser.add_argument(
        "--rebuild-sbert-cache",
        action="store_true",
        help="强制重建纯 SBERT 缓存（覆盖 --reuse-sbert-cache）",
    )
    parser.add_argument("--progress-interval", type=int, default=2000, help="每处理多少条文档输出一次进度")
    parser.add_argument("--quiet-progress", action="store_true", help="关闭进度输出")
    parser.add_argument("--output-file", metavar="FILE", help="结果写入 JSON 文件（不指定则输出到终端）")
    return parser


def main() -> None:
    # Pre-parse to extract --config path before building final argparse defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = build_arg_parser()
    toml_data: dict = {}
    tfidf_enabled_from_toml = True
    sbert_enabled_from_toml = True
    quiet_progress_from_toml = False
    reuse_two_stage_cache_from_toml = False
    reuse_sbert_cache_from_toml = False

    if pre_args.config:
        toml_data = _load_toml(pre_args.config)
        non_bool_defaults = _toml_to_arg_defaults(toml_data)
        parser.set_defaults(**non_bool_defaults)
        pl = toml_data.get("pipeline", {})
        rt = toml_data.get("runtime", {})
        cache = toml_data.get("cache", {})
        tfidf_enabled_from_toml = bool(pl.get("enable_tfidf", True))
        sbert_enabled_from_toml = bool(pl.get("enable_sbert", True))
        quiet_progress_from_toml = bool(rt.get("quiet_progress", False))
        if "reuse_chunk_text_cache" in cache:
            reuse_two_stage_cache_from_toml = bool(cache.get("reuse_chunk_text_cache", False))
        else:
            reuse_two_stage_cache_from_toml = bool(cache.get("reuse_two_stage_cache", False))
        reuse_sbert_cache_from_toml = bool(cache.get("reuse_sbert_cache", False))

    args = parser.parse_args()

    # Boolean resolution: TOML is the base; CLI --disable-* always forces off.
    tfidf_enabled = tfidf_enabled_from_toml and not args.disable_tfidf
    sbert_enabled = sbert_enabled_from_toml and not args.disable_sbert
    show_progress = not (quiet_progress_from_toml or args.quiet_progress)
    reuse_two_stage_cache = (reuse_two_stage_cache_from_toml or args.reuse_two_stage_cache) and not args.rebuild_two_stage_cache
    reuse_sbert_cache = (reuse_sbert_cache_from_toml or args.reuse_sbert_cache) and not args.rebuild_sbert_cache

    if args.retrieval_mode == "sbert-only" and args.disable_sbert:
        raise ValueError("sbert-only 模式下不能使用 --disable-sbert")
    if args.retrieval_mode == "sbert-only":
        sbert_enabled = True
    if args.retrieval_mode == "two-stage" and (not tfidf_enabled) and sbert_enabled:
        raise ValueError("two-stage 模式必须启用 TF-IDF；仅启用 SBERT 请使用 --retrieval-mode sbert-only")
    if args.retrieval_mode == "two-stage" and (not tfidf_enabled) and (not sbert_enabled):
        raise ValueError("TF-IDF 与 SBERT 不能同时关闭")

    if args.progress_interval <= 0:
        raise ValueError("--progress-interval 必须大于 0")
    if args.tfidf_candidate_k <= 0:
        raise ValueError("--tfidf-candidate-k 必须大于 0")

    start_ts = time.perf_counter()

    query = _read_query(args)

    # Expand directories first, then merge with any explicit --input files.
    input_dirs: list[str] = args.input_dir if isinstance(args.input_dir, list) else [args.input_dir]
    dir_files = _resolve_dirs_to_files(input_dirs, args.file_format, args.file_glob) if input_dirs else []
    explicit_files: list[str] = args.input if isinstance(args.input, list) else ([args.input] if args.input else [])
    input_paths = dir_files + explicit_files

    if not input_paths:
        raise ValueError("请通过 --input-dir 或 --input 指定文本数据库")

    _progress(show_progress, start_ts, f"数据库扫描完成，共 {len(input_paths)} 个文件")
    # Validate explicit files (dirs already checked in _resolve_dirs_to_files).
    for p in explicit_files:
        if not Path(p).exists():
            raise FileNotFoundError(f"文本数据库文件不存在: {p}")

    # Reuse first path to build DataConfig for field names / format.
    data_config = DataConfig(
        data_path=input_paths[0],
        file_format=args.file_format,
        text_field=args.text_field,
        id_field=args.id_field,
    )
    chunking_config = ChunkingConfig(
        mode=args.chunk_mode,
        max_chars=args.max_chars,
        window_sentences=args.window_sentences,
        stride_sentences=args.stride_sentences,
    )

    retrieval_mode = args.retrieval_mode
    two_stage_signature_payload = _build_two_stage_cache_signature(
        input_paths=input_paths,
        data_config=data_config,
        chunking_config=chunking_config,
    )

    chunks: list[ChunkedDocument] = []
    chunk_source_files: list[str] = []   # parallel to chunks, records which file each chunk came from
    invalid_records = []
    total_docs = 0
    two_stage_cache_hit = False

    if retrieval_mode == "two-stage" and reuse_two_stage_cache:
        cache_file = Path(args.two_stage_cache_file)
        cache_meta_file = Path(args.two_stage_cache_meta_file)
        if cache_file.exists() and cache_meta_file.exists():
            try:
                cache_meta = _load_json(cache_meta_file)
                if cache_meta.get("signature") == two_stage_signature_payload["signature"]:
                    _progress(show_progress, start_ts, "two-stage 缓存命中，加载切片与索引")
                    cache_payload = _load_json(cache_file)
                    chunks, chunk_source_files = _deserialize_chunks(cache_payload.get("chunks", []))
                    total_docs = int(cache_payload.get("processed_documents", 0))
                    invalid_count = int(cache_payload.get("invalid_documents", 0))
                    invalid_records = ["cached_invalid_record"] * max(0, invalid_count)
                    two_stage_cache_hit = True
                else:
                    _progress(show_progress, start_ts, "two-stage 缓存签名失效，将重建")
            except Exception as exc:
                _progress(show_progress, start_ts, f"two-stage 缓存不可用，将重建（{exc}）")

    if not two_stage_cache_hit:
        for file_idx, input_path in enumerate(input_paths, start=1):
            _progress(show_progress, start_ts, f"开始读取文件 {file_idx}/{len(input_paths)}: {input_path}")
            file_docs = 0
            file_chunks_before = len(chunks)
            for line_no, raw in iter_raw_records(Path(input_path), data_config.file_format):
                doc, error = validate_record(
                    raw=raw,
                    line_no=line_no,
                    text_field=data_config.text_field,
                    id_field=data_config.id_field,
                )
                if error is not None:
                    invalid_records.append(error)
                    continue
                assert doc is not None
                raw_text = str(raw.get(data_config.text_field) or "")
                new_chunks = chunk_document(doc, raw_text=raw_text, chunking_config=chunking_config)
                chunks.extend(new_chunks)
                chunk_source_files.extend([input_path] * len(new_chunks))
                total_docs += 1
                file_docs += 1
                if total_docs % args.progress_interval == 0:
                    _progress(
                        show_progress,
                        start_ts,
                        f"已处理文档 {total_docs}，累计切片 {len(chunks)}（当前文件 {Path(input_path).name}）",
                    )
            _progress(
                show_progress,
                start_ts,
                f"完成文件 {file_idx}/{len(input_paths)}: 文档 {file_docs}，新增切片 {len(chunks) - file_chunks_before}",
            )

    if not chunks:
        raise ValueError("没有可检索的有效片段，请检查输入文件和切片参数")

    top_k = min(args.top_k, len(chunks))
    results: list[dict] = []

    if retrieval_mode == "sbert-only":
        sbert_cfg = SBERTConfig(enabled=True, top_n=top_k)
        sbert_engine = SbertSimilarityEngine(config=sbert_cfg)
        cache_file = Path(args.sbert_cache_file)
        cache_meta_file = Path(args.sbert_cache_meta_file)
        signature_payload = _build_sbert_cache_signature(
            input_paths=input_paths,
            data_config=data_config,
            chunking_config=chunking_config,
            sbert_config=sbert_cfg,
        )

        loaded_from_cache = False
        if reuse_sbert_cache and cache_file.exists() and cache_meta_file.exists():
            try:
                meta = _load_json(cache_meta_file)
                signature_ok = meta.get("signature") == signature_payload["signature"]
                count_ok = int(meta.get("chunk_count", -1)) == len(chunks)
                if signature_ok and count_ok:
                    _progress(show_progress, start_ts, "SBERT 缓存命中，加载向量")
                    cache_obj = sbert_engine.load_embeddings_cache(cache_file)
                    if cache_obj.document_count != len(chunks):
                        raise ValueError("缓存条数与切片数不一致")
                    sbert_engine._texts = [chunk.text for chunk in chunks]
                    sbert_engine._embeddings = np.asarray(cache_obj.embeddings, dtype=np.float32)
                    loaded_from_cache = True
            except Exception as exc:
                _progress(show_progress, start_ts, f"SBERT 缓存不可用，将重建（{exc}）")

        if not loaded_from_cache:
            _progress(show_progress, start_ts, f"开始 SBERT 全量编码，切片总数 {len(chunks)}")
            sbert_engine.fit([chunk.text for chunk in chunks])
            _progress(show_progress, start_ts, "SBERT 编码完成")
            if args.sbert_cache_file:
                sbert_engine.save_embeddings_cache(cache_file)
                _save_json(
                    cache_meta_file,
                    {
                        **signature_payload,
                        "chunk_count": len(chunks),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                _progress(show_progress, start_ts, f"SBERT 缓存已写入 {cache_file}")

        ranked = sbert_engine.query(query, top_k=top_k)
        _progress(show_progress, start_ts, f"SBERT 查询完成，命中结果 {len(ranked)} 条")
        for item in ranked:
            results.append(
                {
                    "index": item.index,
                    "tfidf_score": None,
                    "sbert_score": item.score,
                    "final_score": item.score,
                }
            )
    else:
        engine = TwoStageSearchEngine(
            config=PipelineConfig(
                tfidf=TfidfConfig(enabled=tfidf_enabled),
                sbert=SBERTConfig(enabled=sbert_enabled, top_n=args.sbert_top_n),
                output_top_n=top_k,
            )
        )
        chunk_texts = [chunk.text for chunk in chunks]
        if two_stage_cache_hit:
            engine.load_texts_only(chunk_texts)
            _progress(show_progress, start_ts, "已从缓存加载切片文本")
        else:
            _progress(show_progress, start_ts, f"开始构建检索索引，切片总数 {len(chunks)}")
            engine.fit(
                chunk_texts,
                progress_callback=(lambda msg: _progress(show_progress, start_ts, msg)),
                progress_interval=args.progress_interval,
            )

        if not two_stage_cache_hit and args.two_stage_cache_file:
            two_stage_cache_file = Path(args.two_stage_cache_file)
            two_stage_cache_meta_file = Path(args.two_stage_cache_meta_file)
            _save_json(
                two_stage_cache_file,
                {
                    "chunks": [
                        _serialize_chunk(chunk, source_file)
                        for chunk, source_file in zip(chunks, chunk_source_files)
                    ],
                    "processed_documents": total_docs,
                    "invalid_documents": len(invalid_records),
                },
            )
            _save_json(
                two_stage_cache_meta_file,
                {
                    **two_stage_signature_payload,
                    "chunk_count": len(chunks),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            _progress(show_progress, start_ts, f"two-stage 缓存已写入 {two_stage_cache_file}")

        _progress(show_progress, start_ts, "索引构建完成，开始执行查询")
        ranked = engine.query(
            query,
            top_k=top_k,
            tfidf_candidate_k=min(args.tfidf_candidate_k, len(chunks)),
            progress_callback=(lambda msg: _progress(show_progress, start_ts, msg)),
        )
        _progress(show_progress, start_ts, f"查询完成，命中结果 {len(ranked)} 条")
        for item in ranked:
            results.append(
                {
                    "index": item.index,
                    "tfidf_score": item.tfidf_score,
                    "sbert_score": item.sbert_score,
                    "final_score": item.final_score,
                }
            )

    payload = {
        "query": query,
        "input_paths": input_paths,
        "retrieval_mode": retrieval_mode,
        "chunk_mode": chunking_config.mode,
        "chunk_count": len(chunks),
        "tfidf_candidate_k": min(args.tfidf_candidate_k, len(chunks)),
        "processed_documents": total_docs,
        "invalid_documents": len(invalid_records),
        "tfidf_enabled": tfidf_enabled,
        "sbert_enabled": sbert_enabled,
        "results": [
            {
                "rank": rank,
                "chunk_id": chunks[item["index"]].chunk_id,
                "doc_id": chunks[item["index"]].source_doc_id,
                "source_file": chunk_source_files[item["index"]],
                "title": chunks[item["index"]].title,
                "chunk_index": chunks[item["index"]].chunk_index,
                "start_offset": chunks[item["index"]].start_offset,
                "end_offset": chunks[item["index"]].end_offset,
                "tfidf_score": None if item["tfidf_score"] is None else round(item["tfidf_score"], 6),
                "sbert_score": None if item["sbert_score"] is None else round(item["sbert_score"], 6),
                "final_score": round(item["final_score"], 6),
                "text": chunks[item["index"]].text,
            }
            for rank, item in enumerate(results, start=1)
        ],
        "elapsed_seconds": round(time.perf_counter() - start_ts, 3),
    }
    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"结果已写入 {args.output_file}", file=sys.stderr)
    else:
        print(output_text)
    _progress(show_progress, start_ts, "执行结束")


if __name__ == "__main__":
    main()