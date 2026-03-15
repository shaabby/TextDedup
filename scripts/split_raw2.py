from __future__ import annotations

import argparse
import json
from pathlib import Path


def split_jsonl(
    input_path: Path,
    output_dir: Path,
    lines_per_shard: int,
    prefix: str = "raw_docs_part",
) -> dict[str, object]:
    if lines_per_shard <= 0:
        raise ValueError("lines_per_shard 必须大于 0")
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    total_lines = 0
    current_lines = 0
    shard_file = None
    shard_path = None
    shard_paths: list[str] = []

    try:
        with input_path.open("r", encoding="utf-8") as src:
            for raw_line in src:
                line = raw_line.rstrip("\n")
                if not line:
                    continue

                if shard_file is None or current_lines >= lines_per_shard:
                    if shard_file is not None:
                        shard_file.close()

                    shard_idx += 1
                    current_lines = 0
                    shard_path = output_dir / f"{prefix}_{shard_idx:05d}.jsonl"
                    shard_paths.append(str(shard_path))
                    shard_file = shard_path.open("w", encoding="utf-8")

                shard_file.write(line)
                shard_file.write("\n")
                current_lines += 1
                total_lines += 1
    finally:
        if shard_file is not None:
            shard_file.close()

    stats = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "lines_per_shard": lines_per_shard,
        "total_lines": total_lines,
        "shard_count": shard_idx,
        "first_shard": shard_paths[0] if shard_paths else "",
        "last_shard": shard_paths[-1] if shard_paths else "",
    }
    stats_path = output_dir / "split_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="将大型 JSONL 文件流式拆分为多个小文件")
    parser.add_argument(
        "--input",
        default="data/raw2/raw_docs.jsonl",
        help="输入 JSONL 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw2/shards",
        help="输出分片目录",
    )
    parser.add_argument(
        "--lines-per-shard",
        type=int,
        default=1000,
        help="每个分片的最大行数",
    )
    args = parser.parse_args()

    stats = split_jsonl(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        lines_per_shard=args.lines_per_shard,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()