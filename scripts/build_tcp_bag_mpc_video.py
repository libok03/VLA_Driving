#!/usr/bin/env python3
"""Run TCP over one converted MORAI bag and render an MPC-speed preview MP4."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from multimodal_planner_v9.data import ACTION_NAMES
from scripts.build_tcp_morai_gallery import render_panel
from tcp_morai_finetune.data import TCPMoraiDataset, WAYPOINT_HORIZONS_S
from tcp_morai_finetune.model import TCPMorai
from tcp_morai_finetune.train_full_policy import beta_action


def matching_runs(data_root: Path, source_bag: Path) -> list[str]:
    wanted = source_bag.resolve()
    selected: list[str] = []
    for run_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = Path(str(manifest.get("source_bag", "")))
        same_path = candidate.exists() and candidate.resolve() == wanted
        same_name = candidate.name == source_bag.name
        if same_path or same_name:
            selected.append(run_dir.name)
    if not selected:
        raise ValueError(f"no converted runs found for {source_bag}")
    return selected


def temporal_segment_speeds(waypoints_tcp: np.ndarray) -> np.ndarray:
    points = np.vstack((np.zeros((1, 2), dtype=np.float32), waypoints_tcp))
    times = np.asarray((0.0, *WAYPOINT_HORIZONS_S), dtype=np.float32)
    return np.linalg.norm(np.diff(points, axis=0), axis=1) / np.diff(times)


def put_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    scale: float = 0.48,
    color: tuple[int, int, int] = (30, 30, 30),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def speed_strip(
    panel: np.ndarray,
    segment_speed_mps: np.ndarray,
    filtered_speed_mps: float,
    current_speed_mps: float,
    control_gt: np.ndarray | None = None,
    control_pred: np.ndarray | None = None,
) -> None:
    left, top, right, bottom = 720, 560, 1280, 720
    panel[top:bottom, left:right] = 255
    maximum_kph = max(
        60.0,
        float(segment_speed_mps.max(initial=0.0) * 3.6) * 1.15,
        current_speed_mps * 3.6 * 1.15,
    )
    chart_left, chart_right = left + 18, right - 18
    baseline = bottom - 22
    chart_top = top + 42
    cv2.line(panel, (chart_left, baseline), (chart_right, baseline), (180, 180, 180), 1)
    for index, value in enumerate(segment_speed_mps):
        x0 = chart_left + index * (chart_right - chart_left) // 4 + 16
        x1 = chart_left + (index + 1) * (chart_right - chart_left) // 4 - 16
        y = int(baseline - (float(value) * 3.6 / maximum_kph) * (baseline - chart_top))
        cv2.rectangle(panel, (x0, y), (x1, baseline), (30, 135, 235), -1)
        put_text(panel, f"{value * 3.6:.1f}", (x0 + 4, max(y - 5, chart_top + 12)), scale=0.40)
        put_text(panel, f"{WAYPOINT_HORIZONS_S[index]:g}s", (x0 + 14, baseline + 17), scale=0.36)
    put_text(
        panel,
        f"TCP temporal speed | filtered MPC entry={filtered_speed_mps * 3.6:.1f} km/h"
        f" | ego={current_speed_mps * 3.6:.1f} km/h",
        (left + 18, top + 24),
        scale=0.46,
        thickness=1,
    )
    if control_gt is not None and control_pred is not None:
        put_text(panel, f"Direct control GT/pred | accel {control_gt[0]:+.2f}/{control_pred[0]:+.2f}  steer {control_gt[1]:+.2f}/{control_pred[1]:+.2f}", (left + 18, top + 48), scale=0.42)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--source-bag", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--full-policy-control-cache", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--sample-start", type=int)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--time-start", type=float)
    parser.add_argument("--time-end", type=float)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.ema_alpha <= 1.0:
        raise ValueError("ema-alpha must be in (0,1]")
    run_ids = matching_runs(args.data_root, args.source_bag)
    dataset = TCPMoraiDataset(args.data_root, run_ids, control_cache=args.full_policy_control_cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TCPMorai()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device).eval()

    records: list[dict[str, Any]] = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            target_point = batch["target_point"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                if args.full_policy_control_cache:
                    output = model.forward_original(image, state, target_point)
                    prediction = output["waypoints"]
                    control_prediction = beta_action(output["action_alpha"], output["action_beta"])
                else:
                    prediction = model(image, state, target_point)["waypoints"]
            target = batch["waypoints"].to(device, non_blocking=True)
            errors = torch.linalg.vector_norm(prediction - target, dim=-1).float().cpu().numpy()
            prediction_np = prediction.float().cpu().numpy()
            target_np = target.float().cpu().numpy()
            for index in range(len(prediction_np)):
                records.append(
                    {
                        "dataset_index": offset + index,
                        "run_id": str(batch["run_id"][index]),
                        "sample_id": int(batch["sample_id"][index]),
                        "state": ACTION_NAMES[int(batch["action_state"][index])],
                        "gps_blackout": bool(batch["gps_blackout"][index]),
                        "speed_mps": float(batch["speed_normalized"][index]) * 12.0,
                        "prediction_tcp": prediction_np[index],
                        "target_tcp": target_np[index],
                        "errors_m": errors[index],
                        "ade_m": float(errors[index].mean()),
                        "fde_m": float(errors[index, -1]),
                        "rank": "timeline",
                    }
                )
                if args.full_policy_control_cache:
                    records[-1]["control_pred"] = control_prediction[index].float().cpu().numpy()
                    records[-1]["control_gt"] = batch["current_control"][index].float().cpu().numpy()
            offset += len(prediction_np)

    timestamp_cache: dict[Path, np.ndarray] = {}
    for row in records:
        run_index, sample_index = dataset.lookup[row["dataset_index"]]
        run = dataset.runs[run_index]
        frame_index = int(run.current_frame_idx[sample_index])
        chunk_index = run._chunk_index(frame_index)
        chunk_info = run.chunks[chunk_index]
        timestamps = timestamp_cache.get(chunk_info.path)
        if timestamps is None:
            with np.load(chunk_info.path, allow_pickle=False) as chunk:
                timestamps = np.asarray(chunk["timestamp"], dtype=np.float64)
            timestamp_cache[chunk_info.path] = timestamps
        row["timestamp_s"] = float(timestamps[frame_index - chunk_info.start])
    records.sort(key=lambda row: row["timestamp_s"])
    if args.sample_start is not None:
        records = [row for row in records if row["sample_id"] >= args.sample_start]
    if args.sample_end is not None:
        records = [row for row in records if row["sample_id"] <= args.sample_end]
    if args.time_start is not None:
        records = [row for row in records if row["timestamp_s"] >= args.time_start]
    if args.time_end is not None:
        records = [row for row in records if row["timestamp_s"] <= args.time_end]
    if not records:
        raise ValueError("sample range selected no records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (1280, 720),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {args.output}")

    csv_path = args.output.with_suffix(".csv")
    filtered_speed: float | None = None
    csv_rows: list[dict[str, Any]] = []
    for frame_number, row in enumerate(records):
        segment_speed = temporal_segment_speeds(row["prediction_tcp"])
        raw_entry = float(segment_speed[0])
        filtered_speed = (
            raw_entry
            if filtered_speed is None
            else args.ema_alpha * raw_entry + (1.0 - args.ema_alpha) * filtered_speed
        )
        panel = render_panel(dataset, row)
        speed_strip(panel, segment_speed, filtered_speed, row["speed_mps"], row.get("control_gt"), row.get("control_pred"))
        cv2.rectangle(panel, (0, 405), (720, 460), (20, 20, 20), -1)
        put_text(
            panel,
            f"BAG TIMELINE {frame_number + 1:03d}/{len(records):03d} | GT {row['state']}"
            f" | ADE {row['ade_m']:.2f}m",
            (14, 443),
            scale=0.72,
            color=(255, 255, 255),
            thickness=2,
        )
        writer.write(panel)
        csv_rows.append(
            {
                "frame": frame_number,
                "sample_id": row["sample_id"],
                "timestamp_s": row["timestamp_s"],
                "gt_state": row["state"],
                "ego_speed_kph": row["speed_mps"] * 3.6,
                "tcp_speed_0_0p6_kph": segment_speed[0] * 3.6,
                "tcp_speed_0p6_1p0_kph": segment_speed[1] * 3.6,
                "tcp_speed_1p0_1p6_kph": segment_speed[2] * 3.6,
                "tcp_speed_1p6_2p0_kph": segment_speed[3] * 3.6,
                "filtered_mpc_entry_speed_kph": filtered_speed * 3.6,
                "ade_m": row["ade_m"],
                "fde_2s_m": row["fde_m"],
                **({
                    "gt_accel": row["control_gt"][0], "pred_accel": row["control_pred"][0],
                    "gt_steer": row["control_gt"][1], "pred_steer": row["control_pred"][1],
                } if "control_gt" in row else {}),
            }
        )
    writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)
    summary = {
        "source_bag": str(args.source_bag.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "frames": len(records),
        "fps": args.fps,
        "duration_seconds": len(records) / args.fps,
        "run_ids": run_ids,
        "sample_range": [records[0]["sample_id"], records[-1]["sample_id"]],
        "timestamp_range_s": [records[0]["timestamp_s"], records[-1]["timestamp_s"]],
        "state_counts": {
            name: sum(row["state"] == name for row in records) for name in ACTION_NAMES
        },
        "speed_policy": (
            "TCP temporal waypoint spacing, first-segment EMA for MPC-entry preview; "
            "production MPC must still apply map, curvature, TTC, acceleration and stop constraints"
        ),
        "video": str(args.output.resolve()),
        "csv": str(csv_path.resolve()),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
