#!/usr/bin/env python3
"""Render qualitative MORAI validation examples for a TCP checkpoint."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from multimodal_planner_v9.data import (
    ACTION_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_split_manifest,
)
from tcp_morai_finetune.data import TCPMoraiDataset, WAYPOINT_HORIZONS_S
from tcp_morai_finetune.model import TCPMorai
from tcp_morai_finetune.train_full_policy import beta_action


COLORS = {
    "route": (150, 150, 150),
    "gt": (210, 90, 30),
    "prediction": (20, 130, 235),
}


def tcp_to_morai(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    converted = np.empty_like(value)
    converted[..., 0] = -value[..., 1]
    converted[..., 1] = -value[..., 0]
    return converted


def input_image(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    value = value * IMAGENET_STD + IMAGENET_MEAN
    return np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)


def project(points: np.ndarray, max_forward: float, lateral_extent: float) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    u = 30.0 + (value[:, 1] + lateral_extent) / (2.0 * lateral_extent) * 500.0
    v = 520.0 - value[:, 0] / max_forward * 470.0
    return np.rint(np.stack((u, v), axis=1)).astype(np.int32)


def path_plot(route: np.ndarray, gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    canvas = np.full((560, 560, 3), 255, dtype=np.uint8)
    forward_max = max(10.0, float(np.max(np.concatenate((route[:, 0], gt[:, 0], prediction[:, 0])))) * 1.12)
    lateral_extent = max(
        4.0,
        float(np.max(np.abs(np.concatenate((route[:, 1], gt[:, 1], prediction[:, 1]))))) * 1.25,
    )
    for forward in np.linspace(0.0, forward_max, 6):
        points = project(np.asarray(((forward, -lateral_extent), (forward, lateral_extent))), forward_max, lateral_extent)
        cv2.line(canvas, tuple(points[0]), tuple(points[1]), (225, 225, 225), 1)
        cv2.putText(canvas, f"{forward:.0f}m", (3, int(points[0, 1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)
    center = project(np.asarray(((0.0, 0.0), (forward_max, 0.0))), forward_max, lateral_extent)
    cv2.line(canvas, tuple(center[0]), tuple(center[1]), (205, 205, 205), 1)
    paths = (
        ("route", route[:, :2], 2),
        ("gt", np.vstack((np.zeros((1, 2), dtype=np.float32), gt)), 4),
        ("prediction", np.vstack((np.zeros((1, 2), dtype=np.float32), prediction)), 4),
    )
    for name, points, width in paths:
        cv2.polylines(canvas, [project(points, forward_max, lateral_extent)], False, COLORS[name], width, cv2.LINE_AA)
    for point in project(gt, forward_max, lateral_extent):
        cv2.circle(canvas, tuple(point), 5, COLORS["gt"], -1, cv2.LINE_AA)
    for point in project(prediction, forward_max, lateral_extent):
        cv2.circle(canvas, tuple(point), 5, COLORS["prediction"], -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(project(np.zeros((1, 2)), forward_max, lateral_extent)[0]), 7, (30, 30, 30), -1)
    labels = (("Route", "route"), ("GT", "gt"), ("TCP Pred", "prediction"))
    x = 185
    for label, name in labels:
        cv2.line(canvas, (x, 25), (x + 24, 25), COLORS[name], 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 30, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[name], 1, cv2.LINE_AA)
        x += 110
    return canvas


def put_lines(canvas: np.ndarray, lines: list[str], x: int, y: int) -> None:
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (35, 35, 35), 1, cv2.LINE_AA)
        y += 25


def render_panel(dataset: TCPMoraiDataset, row: dict[str, Any]) -> np.ndarray:
    sample = dataset[row["dataset_index"]]
    run_index, sample_index = dataset.lookup[row["dataset_index"]]
    run = dataset.runs[run_index]
    frame = run.frame(int(run.current_frame_idx[sample_index]))
    # Display the original 640x360 camera frame without the TCP 900x256 model
    # resize. This keeps the audit video human-readable while inference still
    # receives the exact tensor defined by the checkpoint input contract.
    camera_rgb = np.asarray(frame["front"], dtype=np.uint8)
    camera_bgr = cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(camera_bgr, (720, 405), interpolation=cv2.INTER_LINEAR)
    camera = np.full((720, 720, 3), 245, dtype=np.uint8)
    camera[:405] = resized
    route = np.asarray(frame["route"], dtype=np.float32)[:, :2]
    gt = tcp_to_morai(row["target_tcp"])
    prediction = tcp_to_morai(row["prediction_tcp"])
    plot = path_plot(route, gt, prediction)
    canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
    canvas[:, :720] = camera
    canvas[:560, 720:] = plot
    cv2.rectangle(canvas, (0, 405), (720, 460), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        f"GT {row['state']} | {row['rank'].upper()} EXAMPLE | ADE {row['ade_m']:.2f}m",
        (14, 443),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    horizon_text = "  ".join(
        f"{h:g}s:{error:.2f}m"
        for h, error in zip(WAYPOINT_HORIZONS_S, row["errors_m"])
    )
    put_lines(
        canvas,
        [
            f"run: {row['run_id']}",
            f"sample: {row['sample_id']}  speed: {row['speed_mps'] * 3.6:.1f} km/h",
            f"ADE: {row['ade_m']:.3f} m   FDE(2s): {row['fde_m']:.3f} m",
            horizon_text,
            f"GPS blackout: {bool(row['gps_blackout'])}",
            *( [f"control GT/pred acc: {row['control_gt'][0]:+.3f} / {row['control_pred'][0]:+.3f}",
                f"control GT/pred steer: {row['control_gt'][1]:+.3f} / {row['control_pred'][1]:+.3f}"] if "control_gt" in row else [] ),
            "Axes shown in MORAI ego frame: forward / left",
        ],
        740,
        585,
    )
    return canvas


def diverse_quantiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["ade_m"])
    requests = (
        ("best", 0.03),
        ("best", 0.15),
        ("median", 0.45),
        ("median", 0.55),
        ("worst", 0.90),
        ("worst", 1.00),
    )
    selected: list[dict[str, Any]] = []
    used_runs: set[str] = set()
    for rank, quantile in requests:
        center = int(round(quantile * (len(ordered) - 1)))
        candidates = sorted(range(len(ordered)), key=lambda index: abs(index - center))
        chosen = next(
            (
                ordered[index]
                for index in candidates
                if ordered[index]["run_id"] not in used_runs
            ),
            ordered[center],
        )
        copy = dict(chosen)
        copy["rank"] = rank
        selected.append(copy)
        used_runs.add(copy["run_id"])
    return selected


def all_ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every row while retaining useful best/median/worst filters."""
    if not rows:
        return []
    values = np.asarray([row["ade_m"] for row in rows], dtype=np.float32)
    low, high = np.quantile(values, (0.15, 0.90))
    selected = []
    for row in rows:
        copy = dict(row)
        copy["rank"] = (
            "best"
            if row["ade_m"] <= low
            else "worst" if row["ade_m"] >= high else "median"
        )
        selected.append(copy)
    return sorted(selected, key=lambda row: (row["run_id"], row["sample_id"]))


def write_html(output: Path, selected: list[tuple[Path, dict[str, Any]]], epoch: int) -> None:
    cards = []
    for path, row in selected:
        cards.append(
            f'<figure class="{row["rank"]}"><a href="images/{path.name}">'
            f'<img loading="lazy" src="images/{path.name}"></a>'
            f'<figcaption>{html.escape(row["state"])} · {row["rank"]} · '
            f'ADE {row["ade_m"]:.3f}m · FDE {row["fde_m"]:.3f}m · '
            f'{html.escape(row["run_id"])}/{row["sample_id"]}</figcaption></figure>'
        )
    (output / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>TCP MORAI examples</title>
<style>
body {{ font-family:sans-serif; margin:22px; background:#f5f5f5; color:#222; }}
a {{ color:#1769aa; }} .grid {{ display:grid; grid-template-columns:repeat(2,minmax(480px,1fr)); gap:14px; }}
figure {{ margin:0; padding:8px; background:white; border:2px solid #aaa; }}
figure.best {{ border-color:#2b9b52; }} figure.worst {{ border-color:#d34a42; }}
img {{ width:100%; height:auto; }} figcaption {{ margin-top:6px; }}
</style><h1>TCP MORAI Epoch {epoch} validation examples</h1>
<p>Cards are grouped by GT state and colored by ADE rank. TCP itself does not predict the state.</p>
<div class="grid">{''.join(cards)}</div>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--full-policy-control-cache", type=Path)
    parser.add_argument(
        "--source-group",
        help="Use every processed run from this run_manifest source_group.",
    )
    parser.add_argument(
        "--render-all-state",
        choices=tuple(ACTION_NAMES),
        help="Render every sample of one GT state instead of six examples per state.",
    )
    parser.add_argument(
        "--input-state",
        choices=tuple(ACTION_NAMES),
        help="Load only run directories whose manifest action_name matches this state.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    epoch = int(checkpoint["epoch"]) + 1
    model = TCPMorai()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device).eval()
    if args.source_group:
        run_ids = []
        for run_dir in sorted(path for path in args.data_root.iterdir() if path.is_dir()):
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if str(manifest.get("source_group", "")) == args.source_group:
                if args.input_state and str(manifest.get("action_name", "")) != args.input_state:
                    continue
                run_ids.append(run_dir.name)
        if not run_ids:
            raise ValueError(f"no runs found for source_group={args.source_group}")
    else:
        splits = load_split_manifest(args.split_manifest)
        run_ids = splits["val"]
    if args.full_policy_control_cache:
        available = {path.stem for path in args.full_policy_control_cache.glob("*.npz")}
        run_ids = [run_id for run_id in run_ids if json.loads((args.data_root / run_id / "run_manifest.json").read_text())["source_run_id"] in available]
    dataset = TCPMoraiDataset(args.data_root, run_ids, control_cache=args.full_policy_control_cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            target_point = batch["target_point"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
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
                rows.append(
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
                    }
                )
                if args.full_policy_control_cache:
                    rows[-1]["control_pred"] = control_prediction[index].float().cpu().numpy()
                    rows[-1]["control_gt"] = batch["current_control"][index].float().cpu().numpy()
            offset += len(prediction_np)
            if offset % 500 < len(prediction_np):
                print(f"inference {offset}/{len(dataset)}", flush=True)

    chosen: list[dict[str, Any]] = []
    if args.render_all_state:
        chosen = all_ranked(
            [row for row in rows if row["state"] == args.render_all_state]
        )
    else:
        for state_name in ACTION_NAMES:
            state_rows = [row for row in rows if row["state"] == state_name]
            if not state_rows:
                continue
            chosen.extend(diverse_quantiles(state_rows))
    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[Path, dict[str, Any]]] = []
    for sequence, row in enumerate(chosen):
        filename = f"{sequence:02d}_{row['state']}_{row['rank']}_{row['run_id']}_s{row['sample_id']:05d}.jpg"
        path = images_dir / filename
        cv2.imwrite(str(path), render_panel(dataset, row), [cv2.IMWRITE_JPEG_QUALITY, 94])
        rendered.append((path, row))
    write_html(args.output, rendered, epoch)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": epoch,
        "validation_samples": len(rows),
        "rendered": len(rendered),
        "selection": (
            f"all {args.render_all_state} samples"
            if args.render_all_state
            else "two diverse best, median, and worst ADE examples per GT state"
        ),
        "source_group": args.source_group,
        "input_state": args.input_state,
        "state_counts": {name: sum(row["state"] == name for row in rows) for name in ACTION_NAMES},
        "images": [str(path.relative_to(args.output)) for path, _ in rendered],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
