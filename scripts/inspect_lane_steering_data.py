from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import statistics

from vla_driving.data.lane_steering_dataset import LaneSteeringDataset
from vla_driving.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect direct steering labels before training.")
    parser.add_argument("--config", default="configs/lane_steering.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    label_cfg = cfg["labels"]
    manifest_key = "train_manifest" if args.split == "train" else "val_manifest"
    dataset = LaneSteeringDataset(
        data_root=data_cfg["data_root"],
        manifest_path=data_cfg[manifest_key],
        perception_dim=data_cfg["perception_dim"],
        lidar_size=data_cfg["lidar_size"],
        steering_gain=label_cfg["steering_gain"],
        steering_limit=label_cfg["steering_limit"],
        near_waypoint_index=label_cfg.get("near_waypoint_index", 0),
        far_waypoint_index=label_cfg.get("far_waypoint_index", 4),
        lidar_max_range=data_cfg.get("lidar_max_range", 10.0),
        max_samples=args.max_samples,
    )
    values = [float(dataset[idx]["steering"]) for idx in range(len(dataset))]
    signs = Counter("pos" if v > 1.0 else "neg" if v < -1.0 else "zero" for v in values)
    print(f"samples: {len(values)}")
    print(f"steering min/max/mean: {min(values):.3f} {max(values):.3f} {statistics.mean(values):.3f}")
    print(f"steering signs: {dict(signs)}")


if __name__ == "__main__":
    main()
