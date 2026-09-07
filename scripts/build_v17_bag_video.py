#!/usr/bin/env python3
"""Render one MORAI run with V17 predictions as an MP4.

The video is an open-loop diagnostic: it shows raw V17 heads and action
probabilities. Runtime queueing, smoothing, MPC, and safety overrides are not
applied here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodal_planner_v9.data import ACTION_NAMES
from multimodal_planner_v9.model import ModelConfig
from multimodal_planner_v17_spatial30.data import (
    GOAL_NORMALIZATION_M,
    GoalSpatialCandidateDataset,
)
from multimodal_planner_v17_spatial30.model import GoalSpatialCandidatePlannerV17


ACTION_COLORS = {
    "DRIVE": (72, 210, 72),
    "STOP": (64, 64, 255),
    "AVOID": (40, 190, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def lidar_image(bev: np.ndarray) -> np.ndarray:
    occupancy, height, density = bev.astype(np.uint8)
    image = cv2.applyColorMap(height, cv2.COLORMAP_TURBO)
    brightness = 0.25 + 0.75 * density.astype(np.float32) / 255.0
    image = np.clip(image.astype(np.float32) * brightness[..., None], 0, 255)
    image[occupancy == 0] *= 0.16
    return cv2.resize(image.astype(np.uint8), (360, 360))


def project(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    u = np.clip((points[:, 1] + 9.0) / 18.0 * 639.0, 0, 639)
    v = np.clip(359.0 - (points[:, 0] + 3.0) / 33.0 * 359.0, 0, 359)
    return np.rint(np.stack((u, v), axis=-1)).astype(np.int32)


def path_panel(
    drive_path: np.ndarray,
    avoid_path: np.ndarray,
    goal_xy: np.ndarray,
    selected_name: str,
) -> np.ndarray:
    canvas = np.full((360, 640, 3), 24, dtype=np.uint8)
    for lateral in (-8.0, -4.0, 0.0, 4.0, 8.0):
        line = project(np.asarray(((-3.0, lateral), (30.0, lateral))))
        cv2.line(canvas, tuple(line[0]), tuple(line[1]), (48, 48, 48), 1)
    for forward in (0.0, 3.0, 6.0, 10.0, 15.0, 22.0, 30.0):
        line = project(np.asarray(((forward, -9.0), (forward, 9.0))))
        cv2.line(canvas, tuple(line[0]), tuple(line[1]), (48, 48, 48), 1)
    for label, path, color in (
        ("DRIVE", drive_path, (255, 185, 65)),
        ("AVOID", avoid_path, (70, 255, 95)),
    ):
        points = np.vstack((np.zeros((1, 2), np.float32), path))
        uv = project(points)
        thickness = 4 if label == selected_name else 2
        cv2.polylines(canvas, [uv], False, color, thickness, cv2.LINE_AA)
        for point in uv[1:]:
            cv2.circle(canvas, tuple(point), 3, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (16 if label == "DRIVE" else 126, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(project(goal_xy.reshape(1, 2))[0]), 7, (40, 240, 255), -1)
    cv2.circle(canvas, tuple(project(np.zeros((1, 2), np.float32))[0]), 6, (255, 255, 255), -1)
    return canvas


def render(
    frame: dict[str, np.ndarray],
    drive_path: np.ndarray,
    avoid_path: np.ndarray,
    speed: np.ndarray,
    probabilities: np.ndarray,
    goal_xy: np.ndarray,
    truth: str,
    sample_id: int,
    index: int,
    total: int,
) -> np.ndarray:
    selected = ACTION_NAMES[int(probabilities.argmax())]
    canvas = np.full((720, 1280, 3), 18, dtype=np.uint8)
    canvas[:360, :640] = cv2.resize(cv2.cvtColor(frame["front"], cv2.COLOR_RGB2BGR), (640, 360))
    canvas[:360, 640:1000] = lidar_image(frame["lidar_bev"])
    canvas[420:660, :320] = cv2.resize(cv2.cvtColor(frame["left"], cv2.COLOR_RGB2BGR), (320, 240))
    canvas[420:660, 320:640] = cv2.resize(cv2.cvtColor(frame["right"], cv2.COLOR_RGB2BGR), (320, 240))
    canvas[360:, 640:] = path_panel(drive_path, avoid_path, goal_xy, selected)

    color = ACTION_COLORS[selected]
    cv2.rectangle(canvas, (0, 0), (640, 55), (0, 0, 0), -1)
    cv2.putText(canvas, f"V17 PRED {selected} | GT {truth}",
                (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.80, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "LEFT", (12, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 235, 235), 2)
    cv2.putText(canvas, "RIGHT", (332, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 235, 235), 2)
    lines = [
        f"frame {index + 1}/{total} | sample={sample_id}",
        f"P DRIVE={probabilities[0]:.3f}",
        f"P STOP ={probabilities[1]:.3f}",
        f"P AVOID={probabilities[2]:.3f}",
        f"goal=({goal_xy[0]:.1f}, {goal_xy[1]:.1f}) m",
        "speed@3/6/10/15/22/30m:",
        " ".join(f"{value * 3.6:.1f}" for value in speed) + " km/h",
        "raw V17 heads; queue/MPC/safety not applied",
    ]
    for line_index, line in enumerate(lines):
        cv2.putText(canvas, line, (1010, 28 + 23 * line_index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    if args.stride < 1 or args.max_frames < 1:
        raise ValueError("stride and max-frames must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_values = dict(checkpoint["model_config"])
    config_values.update(pretrained_camera=False, freeze_camera_backbone=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = GoalSpatialCandidatePlannerV17(ModelConfig(**config_values)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    dataset = GoalSpatialCandidateDataset(
        args.data_root, [args.run_id], blackout_weight=1.0, stop_weight=1.0,
        drive_weight=1.0, avoidance_weight=1.0, photometric_augmentation=False,
        action_drive_weight=1.0, action_stop_weight=1.0, action_avoid_weight=1.0,
        target_policy="morai_route_residual",
    )
    selected = list(range(0, len(dataset), args.stride))[:args.max_frames]
    if not selected:
        raise ValueError("selected run contains no samples")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {args.output}")

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for video_index, dataset_index in enumerate(selected):
            item = dataset[dataset_index]
            outputs = model(
                item["front"].unsqueeze(0).to(device),
                item["left"].unsqueeze(0).to(device),
                item["right"].unsqueeze(0).to(device),
                item["lidar_bev"].unsqueeze(0).to(device),
                item["goal_point"].unsqueeze(0).to(device),
            )
            probabilities = outputs["action_probabilities"][0].cpu().numpy()
            drive_path = outputs["drive_path_xy_m"][0].cpu().numpy()
            avoid_path = outputs["avoid_path_xy_m"][0].cpu().numpy()
            speed = outputs["target_speed_mps"][0].cpu().numpy()
            run_index, sample_index = dataset.lookup[dataset_index]
            run = dataset.runs[run_index]
            frame = run.frame(int(run.current_frame_idx[sample_index]))
            goal_xy = item["goal_point"].numpy() * GOAL_NORMALIZATION_M
            truth = ACTION_NAMES[int(item["action_state"])]
            panel = render(
                frame, drive_path, avoid_path, speed, probabilities, goal_xy, truth,
                int(item["sample_id"]), video_index, len(selected),
            )
            writer.write(panel)
            rows.append({
                "video_frame": video_index, "sample_id": int(item["sample_id"]),
                "gt_action": truth, "pred_action": ACTION_NAMES[int(probabilities.argmax())],
                "p_drive": float(probabilities[0]), "p_stop": float(probabilities[1]),
                "p_avoid": float(probabilities[2]), "mean_speed_kph": float(speed.mean() * 3.6),
            })
    writer.release()
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "run_id": args.run_id,
        "frames": len(rows), "fps": args.fps, "duration_s": len(rows) / args.fps,
        "video": str(args.output.resolve()), "csv": str(csv_path.resolve()),
        "note": "Open-loop raw V17 candidates; runtime queue, MPC and safety are not applied.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
