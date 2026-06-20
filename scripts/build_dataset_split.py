from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any


PATH_KEYS = ("image", "lidar")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge extracted bag datasets into train/val JSONL manifests."
    )
    parser.add_argument("--output-dir", required=True, help="Directory that will contain train.jsonl and val.jsonl.")
    parser.add_argument("--train", nargs="+", required=True, help="Extracted dataset directories for training.")
    parser.add_argument("--val", nargs="+", required=True, help="Extracted dataset directories for validation.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = merge_dirs([Path(p) for p in args.train], output_dir)
    val_rows = merge_dirs([Path(p) for p in args.val], output_dir)

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)

    print_summary("train", train_rows, output_dir / "train.jsonl")
    print_summary("val", val_rows, output_dir / "val.jsonl")


def merge_dirs(dataset_dirs: list[Path], output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        manifest = dataset_dir / "manifest.jsonl"
        if not manifest.exists():
            raise SystemExit(f"manifest not found: {manifest}")
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                rows.append(rebase_paths(sample, dataset_dir, output_dir))
    return rows


def rebase_paths(sample: dict[str, Any], dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    sample = dict(sample)
    for key in PATH_KEYS:
        if not sample.get(key):
            continue
        source_path = dataset_dir / str(sample[key])
        sample[key] = relative_posix(source_path, output_dir)
    return sample


def relative_posix(path: Path, start: Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(start).resolve()).replace(os.sep, "/")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def print_summary(name: str, rows: list[dict[str, Any]], path: Path) -> None:
    laps = Counter(row.get("lap_index", "missing") for row in rows)
    print(f"{name}: {len(rows)} samples -> {path}")
    print(f"{name} lap_index: {dict(sorted(laps.items()))}")


if __name__ == "__main__":
    main()
