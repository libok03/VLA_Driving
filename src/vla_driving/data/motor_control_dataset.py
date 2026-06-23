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
        use_lidar_summary: bool = True,
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
        self.use_lidar_summary = bool(use_lidar_summary)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        perception = self._fit_vector(sample["perception"], self.perception_dim)
        lidar = np.load(self.data_root / sample["lidar"]).astype(np.float32)
        lidar = self._fit_vector(lidar, self.lidar_size)
        steering = float(np.clip(sample["steering"], -self.steering_limit, self.steering_limit))
        speed = float(np.clip(sample["speed"], 0.0, self.speed_limit))
        target = np.asarray([steering, speed], dtype=np.float32)
        item = {
            "perception": torch.from_numpy(perception),
            "target": torch.from_numpy(target),
        }
        if self.use_lidar_summary:
            item["lidar_summary"] = torch.from_numpy(summarize_lidar(lidar, self.lidar_max_range))
        else:
            lidar = np.nan_to_num(lidar, nan=0.0, posinf=0.0, neginf=0.0)
            lidar = np.clip(lidar, 0.0, self.lidar_max_range) / max(self.lidar_max_range, 1e-6)
            item["lidar"] = torch.from_numpy(lidar)
        return item

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
