from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class LaneSteeringDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        perception_dim: int,
        lidar_size: int,
        steering_gain: float,
        steering_limit: float,
        near_waypoint_index: int = 0,
        far_waypoint_index: int = 4,
        lidar_max_range: float = 10.0,
        max_samples: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.samples = self._load_manifest(manifest_path)
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        self.perception_dim = int(perception_dim)
        self.lidar_size = int(lidar_size)
        self.steering_gain = float(steering_gain)
        self.steering_limit = float(steering_limit)
        self.near_waypoint_index = int(near_waypoint_index)
        self.far_waypoint_index = int(far_waypoint_index)
        self.lidar_max_range = float(lidar_max_range)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        perception = self._fit_vector(sample["perception"], self.perception_dim)
        lidar = np.load(self.data_root / sample["lidar"]).astype(np.float32)
        lidar = self._fit_vector(lidar, self.lidar_size)
        lidar_summary = summarize_lidar(lidar, self.lidar_max_range)
        steering = self._steering_label(sample)
        return {
            "perception": torch.from_numpy(perception),
            "lidar_summary": torch.from_numpy(lidar_summary),
            "steering": torch.tensor(steering, dtype=torch.float32),
        }

    @staticmethod
    def _load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
        with Path(manifest_path).open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _fit_vector(values: Any, size: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        fitted = np.zeros(size, dtype=np.float32)
        fitted[: min(size, values.shape[0])] = values[:size]
        return fitted

    def _steering_label(self, sample: dict[str, Any]) -> float:
        if "steering" in sample:
            return float(np.clip(sample["steering"], -self.steering_limit, self.steering_limit))
        waypoints = np.asarray(sample["future_waypoints"], dtype=np.float32)
        far_idx = int(np.clip(self.far_waypoint_index, 0, waypoints.shape[0] - 1))
        near_idx = int(np.clip(self.near_waypoint_index, 0, waypoints.shape[0] - 1))
        lateral_delta = float(waypoints[far_idx, 1] - waypoints[near_idx, 1])
        return float(np.clip(lateral_delta * self.steering_gain, -self.steering_limit, self.steering_limit))


def summarize_lidar(ranges: np.ndarray, max_range: float = 10.0) -> np.ndarray:
    values = np.asarray(ranges, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, max_range)
    if values.size == 0:
        return np.zeros(5, dtype=np.float32)

    sectors = [
        _sector(values, -15.0, 15.0),
        _sector(values, 15.0, 55.0),
        _sector(values, -55.0, -15.0),
        _sector(values, 55.0, 105.0),
        _sector(values, -105.0, -55.0),
    ]
    summary = np.asarray([_robust_min(sector, max_range) for sector in sectors], dtype=np.float32)
    return summary / max(max_range, 1e-6)


def _sector(values: np.ndarray, start_deg: float, end_deg: float) -> np.ndarray:
    count = values.shape[0]
    angles = np.linspace(-180.0, 180.0, count, endpoint=False, dtype=np.float32)
    mask = (angles >= start_deg) & (angles < end_deg)
    return values[mask]


def _robust_min(values: np.ndarray, default: float) -> float:
    valid = values[values > 0.02]
    if valid.size == 0:
        return float(default)
    return float(np.percentile(valid, 10.0))
