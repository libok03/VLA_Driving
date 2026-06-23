from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from vla_driving.data.lane_steering_dataset import summarize_lidar


class MotorControlDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        perception_dim: int,
        lidar_size: int,
        steering_limit: float,
        speed_limit: float,
        lidar_max_range: float = 10.0,
        max_samples: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.samples = self._load_manifest(manifest_path)
        self.samples = [sample for sample in self.samples if "steering" in sample and "speed" in sample]
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        self.perception_dim = int(perception_dim)
        self.lidar_size = int(lidar_size)
        self.steering_limit = float(steering_limit)
        self.speed_limit = float(speed_limit)
        self.lidar_max_range = float(lidar_max_range)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        perception = self._fit_vector(sample["perception"], self.perception_dim)
        lidar = np.load(self.data_root / sample["lidar"]).astype(np.float32)
        lidar = self._fit_vector(lidar, self.lidar_size)
        lidar_summary = summarize_lidar(lidar, self.lidar_max_range)
        steering = float(np.clip(sample["steering"], -self.steering_limit, self.steering_limit))
        speed = float(np.clip(sample["speed"], 0.0, self.speed_limit))
        target = np.asarray([steering, speed], dtype=np.float32)
        return {
            "perception": torch.from_numpy(perception),
            "lidar_summary": torch.from_numpy(lidar_summary),
            "target": torch.from_numpy(target),
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
