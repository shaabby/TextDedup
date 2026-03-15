from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .lexicon import resolve_repo_path

DEFAULT_STOPWORD_FILES = (
    "data/stopwords/zh_common.txt",
    "data/stopwords/zh_domain.txt",
)


def resolve_stopword_path(path: str | Path) -> Path:
    return resolve_repo_path(path)


def load_stopwords(paths: Iterable[str | Path]) -> set[str]:
    stopwords: set[str] = set()
    for path in paths:
        resolved = resolve_stopword_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"停用词文件不存在: {resolved}")
        for line in resolved.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            stopwords.add(token)
    return stopwords