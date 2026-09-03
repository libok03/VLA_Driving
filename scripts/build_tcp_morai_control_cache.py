#!/usr/bin/env python3
"""Align raw MORAI applied controls to the processed V9 frame timeline.

The raw bags are never modified.  One compact control file is written per
source run and segmented DRIVE/STOP/AVOID runs reuse it by source_run_id.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

TOPIC = "/morai/ego_vehicle_status"


def stamp_ns(message, fallback: int) -> int:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return fallback
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else fallback


def read_status(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with AnyReader([path]) as reader:
        origin = int(reader.start_time)
        connections = [c for c in reader.connections if c.topic == TOPIC]
        if not connections:
            raise RuntimeError(f"{path}: missing {TOPIC}")
        for connection, bag_time, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, connection.msgtype)
            rows.append((stamp_ns(msg, int(bag_time)), float(msg.accel), float(msg.brake), float(msg.steer)))
    value = np.asarray(rows, dtype=np.float64)
    # Keep both clocks. Processed legacy runs use absolute ROS time, while the
    # newer converter stores seconds relative to rosbag start.
    return value[:, 0] / 1e9, np.column_stack((value[:, 1:], (value[:, 0] - origin) / 1e9))


def frame_times(run_dir: Path) -> np.ndarray:
    result = []
    manifest = json.loads((run_dir / "frame_chunks.json").read_text())
    entries = manifest.get("frame_chunks", manifest.get("chunks", manifest))
    for item in entries:
        with np.load(run_dir / item["file"], allow_pickle=False) as chunk:
            result.append(np.asarray(chunk["timestamp"], dtype=np.float64))
    return np.concatenate(result)


def nearest(source_time: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_time, query).clip(0, len(source_time) - 1)
    left = (right - 1).clip(0, len(source_time) - 1)
    return np.where(abs(source_time[right] - query) < abs(source_time[left] - query), right, left)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, action="append", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = {}
    for root in args.search_root:
        for path in root.rglob("*.bag"):
            raw.setdefault(path.name, path)
    groups: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for run_dir in args.data_root.iterdir():
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            groups[manifest["source_run_id"]].append((run_dir, manifest))

    index = {"schema_version": 1, "steer_scale_deg": 40.0, "sources": {}, "missing": []}
    for count, (source_id, runs) in enumerate(sorted(groups.items()), 1):
        name = Path(runs[0][1]["source_bag"]).name
        bag = raw.get(name)
        if bag is None:
            index["missing"].append({"source_run_id": source_id, "bag": name})
            continue
        absolute, status = read_status(bag)
        timestamps = frame_times(runs[0][0])
        status_time = status[:, 3] if np.nanmax(abs(timestamps)) < 1e7 else absolute
        selected = nearest(status_time, timestamps)
        lag = np.abs(status_time[selected] - timestamps)
        throttle, brake, steer_deg = status[selected, 0], status[selected, 1], status[selected, 2]
        signed_accel = np.where(brake > throttle, -brake, throttle)
        control = np.column_stack((np.clip(signed_accel, -1, 1), np.clip(steer_deg / 40.0, -1, 1))).astype(np.float32)
        out = args.output_dir / f"{source_id}.npz"
        np.savez_compressed(out, control=control, throttle=throttle.astype(np.float32), brake=brake.astype(np.float32), steer_deg=steer_deg.astype(np.float32), timestamp=timestamps, alignment_error_s=lag.astype(np.float32))
        index["sources"][source_id] = {"file": out.name, "bag": str(bag), "frames": len(control), "max_alignment_error_s": float(lag.max()), "mean_alignment_error_s": float(lag.mean())}
        if count % 10 == 0:
            print(f"processed {count}/{len(groups)} sources", flush=True)
    (args.output_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps({"available_sources": len(index["sources"]), "missing_sources": len(index["missing"])}, sort_keys=True))


if __name__ == "__main__":
    main()
