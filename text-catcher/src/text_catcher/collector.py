from __future__ import annotations

import argparse
from collections import deque
from fnmatch import fnmatch
import hashlib
import importlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse

import yaml
from bs4 import BeautifulSoup


def _bootstrap_dataset_src() -> None:
    """Allow importing modules from sibling project text-plagiarism-dataset/src."""
    project_root = Path(__file__).resolve().parents[2]
    dataset_src = project_root.parent / "text-plagiarism-dataset" / "src"
    if not dataset_src.exists():
        raise FileNotFoundError(f"missing dependency path: {dataset_src}")
    src_path = str(dataset_src)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_bootstrap_dataset_src()

JsonlStorage = importlib.import_module("storage.dataset").JsonlStorage
BaseScraper = importlib.import_module("scrapers.base").BaseScraper
Document = importlib.import_module("scrapers.base").Document


_BINARY_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".bmp",
    ".ico",
    ".tif",
    ".tiff",
    ".mp4",
    ".mp3",
    ".wav",
    ".avi",
    ".mov",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".css",
    ".js",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}


class GenericHtmlCrawler(BaseScraper):
    source_name = "site"

    def __init__(self, crawl_cfg: dict | None = None) -> None:
        super().__init__(crawl_cfg)
        cfg = crawl_cfg or {}
        # Connection timeout and read timeout can be tuned independently.
        self.connect_timeout_seconds = float(cfg.get("connect_timeout_seconds", self.timeout_seconds))

    def fetch(self, url: str):
        raise NotImplementedError

    def parse(self, html: str, page_url: str = ""):
        raise NotImplementedError

    def fetch_html(self, url: str) -> tuple[str, str]:
        retriable_statuses = {429, 500, 502, 503, 504}
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._respect_min_interval()
            try:
                response = self._session.get(
                    url,
                    timeout=(self.connect_timeout_seconds, self.timeout_seconds),
                )
                status = response.status_code
                if status in retriable_statuses:
                    retry_after = self._parse_retry_after(response.headers.get("Retry-After", ""))
                    if attempt >= self.max_retries:
                        response.raise_for_status()
                    self._sleep_before_retry(attempt, retry_after)
                    continue

                response.raise_for_status()
                content_type = str(response.headers.get("Content-Type", "")).lower()
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    raise ValueError(f"non-html content-type: {content_type or 'unknown'}")

                if not response.encoding:
                    response.encoding = response.apparent_encoding or "utf-8"
                print(f"[connected] status={status} url={response.url}")
                return response.text, response.url
            except (ValueError, Exception) as exc:  # noqa: BLE001
                last_error = exc
                if isinstance(exc, ValueError):
                    break
                if attempt >= self.max_retries:
                    break
                self._sleep_before_retry(attempt)

        raise RuntimeError(f"failed to fetch html url after retries: {url}") from last_error


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_url(url: str, keep_query: bool) -> str:
    clean_url = urldefrag(url).url
    parsed = urlparse(clean_url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    path = parsed.path or "/"
    query = parsed.query if keep_query else ""
    normalized = parsed._replace(path=path, query=query)
    return urlunparse(normalized)


def _is_binary_resource_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in _BINARY_EXTENSIONS)


def _extract_links(html: str, page_url: str, keep_query: bool) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()
    for node in soup.select("a[href]"):
        href = str(node.get("href", "")).strip()
        if not href:
            continue
        absolute = urljoin(page_url, href)
        normalized = _normalize_url(absolute, keep_query=keep_query)
        if not normalized:
            continue
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"}:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)
    return links


def _extract_text_and_title(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else "untitled"

    for node in soup(["script", "style", "noscript"]):
        node.decompose()

    main_node = soup.select_one("main") or soup.select_one("article") or soup.body or soup
    text = _normalize_whitespace(main_node.get_text(" ", strip=True))
    return title, text


def _is_allowed_domain(url: str, allowed_domains: set[str]) -> bool:
    if not allowed_domains:
        return True
    hostname = (urlparse(url).hostname or "").lower()
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in allowed_domains)


def _is_allowed_path(url: str, path_allow_patterns: list[str]) -> bool:
    if not path_allow_patterns:
        return True
    path = urlparse(url).path or "/"
    return any(fnmatch(path, pattern) for pattern in path_allow_patterns)


def _is_requestable_path(
    url: str,
    path_store_patterns: list[str],
    path_request_only_patterns: list[str],
) -> bool:
    request_patterns = [*path_store_patterns, *path_request_only_patterns]
    if not request_patterns:
        return True
    path = urlparse(url).path or "/"
    return any(fnmatch(path, pattern) for pattern in request_patterns)


def _enqueue_discovered_links(
    queue: deque,
    enqueued_hashes: set[str],
    completed_url_hashes: set[str],
    html: str,
    final_url: str,
    keep_query: bool,
    allowed_domains: set[str],
    skip_binary_resources: bool,
    source_name: str,
    language: str,
    next_depth: int,
    path_store_patterns: list[str],
    path_request_only_patterns: list[str],
) -> None:
    for link in _extract_links(html, page_url=final_url, keep_query=keep_query):
        if not _is_allowed_domain(link, allowed_domains):
            continue
        if skip_binary_resources and _is_binary_resource_url(link):
            continue
        if not _is_requestable_path(link, path_store_patterns, path_request_only_patterns):
            continue
        link_hash = _sha1_hex(link)
        if link_hash in enqueued_hashes or link_hash in completed_url_hashes:
            continue
        queue.append((source_name, link, language, next_depth))
        enqueued_hashes.add(link_hash)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sha1_hex(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_for_hash(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _load_collect_state(state_path: Path) -> set[str]:
    empty_state = {
        "completed_url_hashes": set(),
        "pending_queue": [],
    }
    if not state_path.exists():
        return empty_state
    try:
        with state_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        hashes = payload.get("completed_url_hashes", [])
        pending_queue = payload.get("pending_queue", [])
        normalized_queue = []
        for item in pending_queue:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            url = str(item.get("url", "")).strip()
            language = str(item.get("language", "unknown")).strip().lower()
            depth = int(item.get("depth", 0))
            if not source or not url:
                continue
            normalized_queue.append(
                {
                    "source": source,
                    "url": url,
                    "language": language,
                    "depth": max(0, depth),
                }
            )
        return {
            "completed_url_hashes": {str(item) for item in hashes if isinstance(item, str)},
            "pending_queue": normalized_queue,
        }
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return empty_state


def _snapshot_queue(queue: deque) -> list[dict]:
    snapshot: list[dict] = []
    for source_name, url, language, depth in queue:
        snapshot.append(
            {
                "source": str(source_name),
                "url": str(url),
                "language": str(language),
                "depth": int(depth),
            }
        )
    return snapshot


def _save_collect_state(state_path: Path, completed_url_hashes: set[str], pending_queue: list[dict]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_url_hashes": sorted(completed_url_hashes),
        "pending_queue": pending_queue,
        "updated_at": int(time.time()),
    }
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def collect(config_path: str) -> None:
    cfg = _load_config(config_path)
    storage_cfg = cfg["storage"]
    target_cfg = cfg.get("targets", {})
    crawl_cfg = cfg.get("crawl", {})
    incremental_cfg = cfg.get("incremental", {})

    concurrent_workers = max(1, int(crawl_cfg.get("concurrent_workers", 4)))
    incremental_enabled = bool(incremental_cfg.get("enabled", True))
    skip_seen_urls = bool(incremental_cfg.get("skip_seen_urls", True))
    checkpoint_every_n_urls = max(1, int(crawl_cfg.get("checkpoint_every_n_urls", 100)))
    recursive_enabled = bool(crawl_cfg.get("recursive_enabled", True))
    max_depth = max(0, int(crawl_cfg.get("max_depth", 3)))
    max_pages = max(1, int(crawl_cfg.get("max_pages", 1000)))
    keep_query = bool(crawl_cfg.get("follow_query_strings", False))
    only_html = bool(crawl_cfg.get("only_html", True))
    skip_binary_resources = bool(crawl_cfg.get("skip_binary_resources", True))
    path_store_patterns = [
        str(item).strip()
        for item in crawl_cfg.get("path_allow_patterns", [])
        if str(item).strip()
    ]
    path_request_only_patterns = [
        str(item).strip()
        for item in crawl_cfg.get("path_request_only_patterns", [])
        if str(item).strip()
    ]
    seed_always_request = bool(crawl_cfg.get("seed_always_request", True))
    allowed_domains_cfg = {str(item).strip().lower() for item in crawl_cfg.get("allowed_domains", []) if str(item).strip()}

    raw_path = storage_cfg.get("raw_path", "data/raw/raw_docs.jsonl")
    failed_urls_path = storage_cfg.get("failed_urls_path", "data/raw/failed_urls.jsonl")
    collect_state_path = Path(storage_cfg.get("collect_state_path", "data/raw/collect_state.json"))

    raw_storage = JsonlStorage(raw_path)
    failed_storage = JsonlStorage(failed_urls_path)

    existing_rows = raw_storage.read_many() if Path(raw_path).exists() else []
    existing_doc_ids = {row.get("doc_id", "") for row in existing_rows if row.get("doc_id")}
    existing_url_hashes = {_sha1_hex(row.get("url", "")) for row in existing_rows if row.get("url")}
    existing_content_hashes = {
        _sha1_hex(_normalize_for_hash(row.get("text", "")))
        for row in existing_rows
        if row.get("text")
    }

    state_payload = _load_collect_state(collect_state_path)
    completed_url_hashes = state_payload["completed_url_hashes"]
    saved_pending_queue = state_payload["pending_queue"]
    completed_url_hashes.update(existing_url_hashes)
    pending_checkpoint_updates = 0
    queue: deque | None = None

    def _mark_completed_url(url_hash: str) -> None:
        nonlocal pending_checkpoint_updates
        before_size = len(completed_url_hashes)
        completed_url_hashes.add(url_hash)
        if len(completed_url_hashes) == before_size:
            return
        pending_checkpoint_updates += 1
        if pending_checkpoint_updates >= checkpoint_every_n_urls:
            pending_queue = _snapshot_queue(queue) if queue is not None else []
            _save_collect_state(collect_state_path, completed_url_hashes, pending_queue)
            print(
                f"[checkpoint] saved_state pending={pending_checkpoint_updates} "
                f"total_completed={len(completed_url_hashes)}"
            )
            pending_checkpoint_updates = 0

    configured_domains = [
        str(item).strip()
        for item in target_cfg.get("domains", [])
        if str(item).strip()
    ]
    default_language = str(target_cfg.get("default_language", "unknown")).lower()

    collected = 0
    skipped = 0
    skipped_url = 0
    skipped_content = 0
    skipped_non_html = 0
    skipped_binary = 0
    skipped_path = 0
    skipped_request_path = 0
    skipped_request_only = 0
    failed = 0

    seeds: list[tuple[str, str, str]] = []
    for domain in configured_domains:
        candidate = domain
        if not candidate.startswith(("http://", "https://")):
            candidate = f"https://{candidate}/"
        normalized_seed = _normalize_url(candidate, keep_query=keep_query)
        if not normalized_seed:
            continue
        source_name = (urlparse(normalized_seed).hostname or "site").lower()
        seeds.append((source_name, normalized_seed, default_language))

    if not seeds:
        print("collect_done new_docs=0 skipped_existing=0 skipped_url=0 skipped_content=0 failed_urls=0 workers=0 raw_path=" f"{raw_path} failed_path={failed_urls_path} state_path={collect_state_path}")
        return

    allowed_domains = set(allowed_domains_cfg)
    if not allowed_domains:
        allowed_domains = {
            (urlparse(seed_url).hostname or "").lower()
            for _, seed_url, _ in seeds
            if (urlparse(seed_url).hostname or "").strip()
        }

    crawler = GenericHtmlCrawler(crawl_cfg)
    interrupted = False
    pending_queue_size = 0
    try:
        queue = deque()
        enqueued_hashes: set[str] = set()

        if incremental_enabled and saved_pending_queue:
            for item in saved_pending_queue:
                source_name = item["source"]
                seed_url = item["url"]
                language = item["language"]
                depth = item["depth"]
                queue.append((source_name, seed_url, language, depth))
                enqueued_hashes.add(_sha1_hex(seed_url))
            print(f"[resume] restored_pending_queue size={len(saved_pending_queue)}")
        else:
            for source_name, seed_url, language in seeds:
                seed_hash = _sha1_hex(seed_url)
                queue.append((source_name, seed_url, language, 0))
                enqueued_hashes.add(seed_hash)

        processed_pages = 0

        try:
            while queue and processed_pages < max_pages:
                source_name, current_url, language, depth = queue.popleft()
                current_hash = _sha1_hex(current_url)

                if incremental_enabled and skip_seen_urls and current_hash in completed_url_hashes:
                    skipped += 1
                    skipped_url += 1
                    continue

                if not _is_allowed_domain(current_url, allowed_domains):
                    skipped += 1
                    skipped_url += 1
                    continue

                if not (depth == 0 and seed_always_request):
                    if not _is_requestable_path(current_url, path_store_patterns, path_request_only_patterns):
                        skipped += 1
                        skipped_request_path += 1
                        _mark_completed_url(current_hash)
                        continue

                if skip_binary_resources and _is_binary_resource_url(current_url):
                    skipped += 1
                    skipped_binary += 1
                    _mark_completed_url(current_hash)
                    continue

                try:
                    html, final_url = crawler.fetch_html(current_url)
                except Exception as fetch_error:  # noqa: BLE001
                    failed += 1
                    failed_storage.append_one(
                        {
                            "source": source_name,
                            "url": current_url,
                            "error": str(fetch_error),
                            "phase": "fetch_or_parse",
                            "ts": int(time.time()),
                        }
                    )
                    print(f"[warn] failed url={current_url} source={source_name}: {fetch_error}")
                    continue

                if only_html and not html:
                    skipped += 1
                    skipped_non_html += 1
                    _mark_completed_url(current_hash)
                    continue

                title, text = _extract_text_and_title(html)
                should_collect = _is_allowed_path(final_url, path_store_patterns)
                is_request_only = (not should_collect) and _is_allowed_path(final_url, path_request_only_patterns)
                text_hash = _sha1_hex(_normalize_for_hash(text)) if text else ""
                doc_key = f"{final_url}|{title}|{text[:256]}"
                doc_id = crawler._stable_doc_id(source_name, doc_key)

                if is_request_only:
                    skipped += 1
                    skipped_request_only += 1
                    _mark_completed_url(current_hash)
                    if not recursive_enabled or depth >= max_depth:
                        continue
                    _enqueue_discovered_links(
                        queue=queue,
                        enqueued_hashes=enqueued_hashes,
                        completed_url_hashes=completed_url_hashes,
                        html=html,
                        final_url=final_url,
                        keep_query=keep_query,
                        allowed_domains=allowed_domains,
                        skip_binary_resources=skip_binary_resources,
                        source_name=source_name,
                        language=language,
                        next_depth=depth + 1,
                        path_store_patterns=path_store_patterns,
                        path_request_only_patterns=path_request_only_patterns,
                    )
                    continue

                if not should_collect:
                    skipped += 1
                    skipped_path += 1
                    _mark_completed_url(current_hash)
                    if not recursive_enabled or depth >= max_depth:
                        continue
                    _enqueue_discovered_links(
                        queue=queue,
                        enqueued_hashes=enqueued_hashes,
                        completed_url_hashes=completed_url_hashes,
                        html=html,
                        final_url=final_url,
                        keep_query=keep_query,
                        allowed_domains=allowed_domains,
                        skip_binary_resources=skip_binary_resources,
                        source_name=source_name,
                        language=language,
                        next_depth=depth + 1,
                        path_store_patterns=path_store_patterns,
                        path_request_only_patterns=path_request_only_patterns,
                    )
                    continue

                if not text:
                    skipped += 1
                    _mark_completed_url(current_hash)
                    continue
                if doc_id in existing_doc_ids:
                    skipped += 1
                    _mark_completed_url(current_hash)
                    continue
                if text_hash and text_hash in existing_content_hashes:
                    skipped += 1
                    skipped_content += 1
                    _mark_completed_url(current_hash)
                    continue

                doc = Document(
                    doc_id=doc_id,
                    source=source_name,
                    title=title,
                    language=language,
                    text=text,
                    url=final_url,
                )
                raw_storage.append_one(doc.to_dict())
                existing_doc_ids.add(doc_id)
                if text_hash:
                    existing_content_hashes.add(text_hash)
                collected += 1
                processed_pages += 1

                _mark_completed_url(current_hash)
                print(f"[collect] source={source_name} depth={depth} url={final_url} docs=1")

                if not recursive_enabled or depth >= max_depth:
                    continue

                _enqueue_discovered_links(
                    queue=queue,
                    enqueued_hashes=enqueued_hashes,
                    completed_url_hashes=completed_url_hashes,
                    html=html,
                    final_url=final_url,
                    keep_query=keep_query,
                    allowed_domains=allowed_domains,
                    skip_binary_resources=skip_binary_resources,
                    source_name=source_name,
                    language=language,
                    next_depth=depth + 1,
                    path_store_patterns=path_store_patterns,
                    path_request_only_patterns=path_request_only_patterns,
                )
        except KeyboardInterrupt:
            interrupted = True
            print("[interrupt] keyboard interrupt received, saving state and pending queue...")
    finally:
        pending_queue = _snapshot_queue(queue) if queue is not None else []
        pending_queue_size = len(pending_queue)
        _save_collect_state(collect_state_path, completed_url_hashes, pending_queue)
        crawler.close()

    if interrupted:
        print(
            "collect_done "
            f"new_docs={collected} skipped_existing={skipped} "
            f"skipped_url={skipped_url} skipped_content={skipped_content} "
            f"skipped_non_html={skipped_non_html} skipped_binary={skipped_binary} skipped_path={skipped_path} "
            f"skipped_request_path={skipped_request_path} skipped_request_only={skipped_request_only} failed_urls={failed} "
            f"workers={concurrent_workers} raw_path={raw_path} "
            f"failed_path={failed_urls_path} state_path={collect_state_path} "
            f"pending_queue={pending_queue_size} status=interrupted"
        )
        print("[done] crawl interrupted and state saved")
        return

    print(
        "collect_done "
        f"new_docs={collected} skipped_existing={skipped} "
        f"skipped_url={skipped_url} skipped_content={skipped_content} "
        f"skipped_non_html={skipped_non_html} skipped_binary={skipped_binary} skipped_path={skipped_path} "
        f"skipped_request_path={skipped_request_path} skipped_request_only={skipped_request_only} failed_urls={failed} "
        f"workers={concurrent_workers} raw_path={raw_path} "
        f"failed_path={failed_urls_path} state_path={collect_state_path} pending_queue=0 status=finished"
    )
    print("[done] crawl finished")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect-only dataset pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config yaml")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    collect(args.config)


if __name__ == "__main__":
    main()
