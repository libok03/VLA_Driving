from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_motor_control import build_dataset
from vla_driving.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NPZ caches for motor-control training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        dataset = build_dataset(cfg, split, max_samples=args.max_samples)
        if len(dataset) == 0:
            raise SystemExit(f"No samples found for {split}.")
        cache_path = output_dir / f"{split}.npz"
        write_cache(dataset, cache_path, compressed=args.compressed)
        print(f"{split}: {len(dataset)} samples -> {cache_path}")


def write_cache(dataset, cache_path: Path, compressed: bool) -> None:
    first = dataset[0]
    lidar_key = "lidar_summary" if "lidar_summary" in first else "lidar"
    perception = np.empty((len(dataset), first["perception"].numel()), dtype=np.float32)
    lidar = np.empty((len(dataset), first[lidar_key].numel()), dtype=np.float32)
    target = np.empty((len(dataset), first["target"].numel()), dtype=np.float32)

    for idx in tqdm(range(len(dataset)), desc=cache_path.stem):
        sample = dataset[idx]
        perception[idx] = sample["perception"].numpy()
        lidar[idx] = sample[lidar_key].numpy()
        target[idx] = sample["target"].numpy()

    save = np.savez_compressed if compressed else np.savez
    save(cache_path, perception=perception, lidar=lidar, target=target)


if __name__ == "__main__":
    main()
