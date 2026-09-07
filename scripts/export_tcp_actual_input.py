#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.build_tcp_bag_mpc_video import matching_runs
from scripts.build_tcp_morai_gallery import input_image
from tcp_morai_finetune.data import TCPMoraiDataset
from multimodal_planner_v9.data import ACTION_NAMES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--source-bag", type=Path, required=True)
    p.add_argument("--control-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--state", choices=tuple(ACTION_NAMES))
    args = p.parse_args()
    runs = matching_runs(args.data_root, args.source_bag)
    dataset = TCPMoraiDataset(args.data_root, runs, control_cache=args.control_cache)
    # Use the earliest sample in bag time rather than label-directory order.
    candidates = []
    for index, (run_index, sample_index) in enumerate(dataset.lookup):
        run = dataset.runs[run_index]
        candidates.append((int(run.sample_id[sample_index]), index, run_index, sample_index))
    if args.state:
        candidates = [row for row in candidates if ACTION_NAMES[int(dataset.action_labels[row[1]])] == args.state]
        if not candidates:
            raise ValueError(f"no {args.state} samples")
        candidates.sort()
        sample_id, dataset_index, run_index, sample_index = candidates[len(candidates) // 2]
    else:
        sample_id, dataset_index, run_index, sample_index = min(candidates)
    sample = dataset[dataset_index]
    actual_rgb = input_image(sample["image"])
    run = dataset.runs[run_index]
    original_rgb = run.frame(int(run.current_frame_idx[sample_index]))["front"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.state.lower()}" if args.state else ""
    actual_path = args.output_dir / f"actual_model_input_3x256x900{suffix}.png"
    original_path = args.output_dir / f"original_processed_front_640x360{suffix}.png"
    cv2.imwrite(str(actual_path), cv2.cvtColor(actual_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(original_path), cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))

    canvas = np.full((720, 980, 3), 255, dtype=np.uint8)
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    actual_bgr = cv2.cvtColor(actual_rgb, cv2.COLOR_RGB2BGR)
    canvas[45:405, 170:810] = original_bgr
    canvas[454:710, 40:940] = actual_bgr
    cv2.putText(canvas, "PROCESSED FRONT SOURCE 640x360", (170, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (20,20,20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "ACTUAL TCP INPUT 900x256 (no crop; anisotropic resize)", (40, 440), cv2.FONT_HERSHEY_SIMPLEX, .7, (20,20,20), 2, cv2.LINE_AA)
    comparison = args.output_dir / f"original_vs_actual_tcp_input{suffix}.png"
    cv2.imwrite(str(comparison), canvas)
    print(json.dumps({"sample_id": sample_id, "run_id": run.run_id, "original": str(original_path.resolve()), "actual_input": str(actual_path.resolve()), "comparison": str(comparison.resolve()), "actual_shape_hwc": list(actual_rgb.shape)}, sort_keys=True))


if __name__ == "__main__":
    main()
