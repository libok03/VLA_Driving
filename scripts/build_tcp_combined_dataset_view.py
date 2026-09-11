#!/usr/bin/env python3
"""Create a zero-copy combined view of reviewed MORAI and SSD teacher runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def link_children(source: Path, destination: Path, suffix: str | None = None) -> int:
    count = 0
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if suffix is not None and item.suffix != suffix:
            continue
        if suffix is None and not item.is_dir():
            continue
        target = destination / item.name
        resolved = item.resolve()
        if target.is_symlink() and target.resolve() == resolved:
            count += 1
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"conflicting combined-view entry: {target}")
        target.symlink_to(resolved, target_is_directory=item.is_dir())
        count += 1
    return count


def read_splits(path: Path) -> dict[str, list[str]]:
    value = json.loads(path.read_text())
    return value["splits"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-root", type=Path, required=True)
    parser.add_argument("--human-split", type=Path, required=True)
    parser.add_argument("--human-controls", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-split", type=Path, required=True)
    parser.add_argument("--teacher-controls", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-controls", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    human_runs = link_children(args.human_root, args.output_root)
    teacher_runs = link_children(args.teacher_root, args.output_root)
    human_controls = link_children(args.human_controls, args.output_controls, ".npz")
    teacher_controls = link_children(args.teacher_controls, args.output_controls, ".npz")

    human = read_splits(args.human_split)
    teacher = read_splits(args.teacher_split)
    combined = {
        split: sorted(set(human.get(split, [])) | set(teacher.get(split, [])))
        for split in ("train", "val", "test")
    }
    payload = {
        "schema_version": 1,
        "split_unit": "run",
        "strategy": "preserve_source_splits_zero_copy_union",
        "data_root": str(args.output_root),
        "sources": {
            "human": str(args.human_root),
            "teacher": str(args.teacher_root),
        },
        "splits": combined,
    }
    (args.output_root / "split.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "human_runs_linked": human_runs,
        "teacher_runs_linked": teacher_runs,
        "human_controls_linked": human_controls,
        "teacher_controls_linked": teacher_controls,
        "split_run_counts": {key: len(value) for key, value in combined.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
