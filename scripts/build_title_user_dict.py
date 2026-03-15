from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from textdedup.stopwords import load_stopwords

GENERIC_TITLE_TERMS = {
    "电子书在线阅读",
    "在线阅读",
    "世界名著网",
    "代表著作",
    "小说",
    "杂文集",
    "英国",
    "第章",
}

PERSON_SUFFIXES = ("小姐", "先生", "太太", "夫人", "爵士", "勋爵", "上尉", "中尉", "少佐", "医生", "牧师")
PLACE_SUFFIXES = ("广场", "大街", "林荫道", "市场", "学校", "教堂", "花园", "公馆", "庄园", "镇", "城", "郡", "街")


def _iter_titles(input_path: Path) -> list[str]:
    titles: list[str] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        title = str(payload.get("title") or "").strip()
        if title:
            titles.append(title)
    return titles


def _add_candidate(counter: Counter[str], token: str) -> None:
    cleaned = token.strip("《》()（）[]【】_-.· ")
    if not cleaned or cleaned in GENERIC_TITLE_TERMS:
        return
    if not re.fullmatch(r"[\u4e00-\u9fff]{2,16}", cleaned):
        return
    counter[cleaned] += 1


def extract_title_candidates(titles: list[str], stopwords: set[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    person_pattern = rf"([\u4e00-\u9fff]{{2,10}}(?:{'|'.join(PERSON_SUFFIXES)}))"
    place_pattern = rf"([\u4e00-\u9fff]{{2,12}}(?:{'|'.join(PLACE_SUFFIXES)}))"
    for title in titles:
        for match in re.findall(r"《([\u4e00-\u9fff]{2,16})》", title):
            _add_candidate(counter, match)
        for match in re.findall(person_pattern, title):
            _add_candidate(counter, match)
        for match in re.findall(place_pattern, title):
            _add_candidate(counter, match)
    return counter


def write_candidates(counter: Counter[str], output_path: Path, min_count: int, min_length: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [
        token
        for token, count in sorted(counter.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
        if count >= min_count and len(token) >= min_length
    ]
    content = "\n".join(f"{token} 180000 nz" for token in candidates)
    output_path.write_text(content + ("\n" if content else ""), encoding="utf-8")
    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="从标题字段抽取中文专名候选，生成 jieba 用户词典候选文件")
    parser.add_argument("--input", default="data/raw2/shards/raw_docs_part_00001.jsonl", help="输入 JSONL 分片")
    parser.add_argument("--output", default="data/dicts/zh_title_candidates.txt", help="输出用户词典候选文件")
    parser.add_argument("--min-count", type=int, default=1, help="最低出现次数")
    parser.add_argument("--min-length", type=int, default=2, help="最低词长")
    args = parser.parse_args()

    titles = _iter_titles(Path(args.input))
    stopwords = load_stopwords(("data/stopwords/zh_common.txt", "data/stopwords/zh_domain.txt"))
    counter = extract_title_candidates(titles, stopwords)
    written = write_candidates(counter, Path(args.output), min_count=args.min_count, min_length=args.min_length)
    print(json.dumps({"input": args.input, "output": args.output, "titles": len(titles), "candidates": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()