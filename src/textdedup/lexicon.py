from __future__ import annotations

from pathlib import Path
from typing import Iterable

import jieba

DEFAULT_USER_DICT_FILES = (
    "data/dicts/zh_user_dict.txt",
)

_LOADED_JIEBA_USER_DICTS: set[str] = set()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return repo_root() / raw_path


def ensure_jieba_user_dicts(paths: Iterable[str | Path]) -> tuple[str, ...]:
    resolved_paths: list[str] = []
    for path in paths:
        resolved = resolve_repo_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"用户词典文件不存在: {resolved}")
        resolved_key = str(resolved)
        if resolved_key not in _LOADED_JIEBA_USER_DICTS:
            jieba.load_userdict(str(resolved))
            _LOADED_JIEBA_USER_DICTS.add(resolved_key)
        resolved_paths.append(resolved_key)
    return tuple(resolved_paths)


def load_user_dict_terms(paths: Iterable[str | Path]) -> set[str]:
    terms: set[str] = set()
    for path in paths:
        resolved = resolve_repo_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"用户词典文件不存在: {resolved}")
        for line in resolved.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            # jieba userdict format: <word> <freq> <tag>
            terms.add(token.split()[0])
    return terms