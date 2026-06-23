from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class MotorTemporalDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        perception_dim: int,
        lidar_size: int,
        sequence_length: int,
        steering_limit: float,
        speed_limit: float,
        lidar_max_range: float = 10.0,
        max_samples: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.sequence_length = int(sequence_length)
        self.perception_dim = int(perception_dim)
        self.lidar_size = int(lidar_size)
        self.steering_limit = float(steering_limit)
        self.speed_limit = float(speed_limit)
        self.lidar_max_range = float(lidar_max_range)

        samples = self._load_manifest(manifest_path)
        samples = [sample for sample in samples if "steering" in sample and "speed" in sample]
        if max_samples > 0:
            samples = samples[:max_samples]
        self.samples = samples
        self.features, self.targets, self.bag_keys = self._load_arrays(samples)
        self.indices = self._valid_indices()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        end_idx = self.indices[index]
        start_idx = end_idx - self.sequence_length + 1
        return {
            "sequence": torch.from_numpy(self.features[start_idx : end_idx + 1]),
            "target": torch.from_numpy(self.targets[end_idx]),
        }

    @staticmethod
    def _load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
        with Path(manifest_path).open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _load_arrays(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        features = np.zeros((len(samples), self.perception_dim + self.lidar_size), dtype=np.float32)
        targets = np.zeros((len(samples), 2), dtype=np.float32)
        bag_keys: list[str] = []

        for idx, sample in enumerate(samples):
            perception = self._fit_vector(sample["perception"], self.perception_dim)
            lidar = np.load(self.data_root / sample["lidar"]).astype(np.float32)
            lidar = self._fit_vector(lidar, self.lidar_size)
            lidar = np.nan_to_num(lidar, nan=0.0, posinf=0.0, neginf=0.0)
            lidar = np.clip(lidar, 0.0, self.lidar_max_range) / max(self.lidar_max_range, 1e-6)
            steering = float(np.clip(sample["steering"], -self.steering_limit, self.steering_limit))
            speed = float(np.clip(sample["speed"], 0.0, self.speed_limit))

            features[idx, : self.perception_dim] = perception
            features[idx, self.perception_dim :] = lidar
            targets[idx] = [steering, speed]
            bag_keys.append(self._bag_key(str(sample["lidar"])))

        return features, targets, bag_keys

    def _valid_indices(self) -> list[int]:
        valid: list[int] = []
        for end_idx in range(self.sequence_length - 1, len(self.samples)):
            start_idx = end_idx - self.sequence_length + 1
            bag_key = self.bag_keys[end_idx]
            if all(self.bag_keys[idx] == bag_key for idx in range(start_idx, end_idx + 1)):
                valid.append(end_idx)
        return valid

    @staticmethod
    def _bag_key(lidar_path: str) -> str:
        parts = Path(lidar_path).parts
        for part in parts:
            if part.startswith("rosbag2_"):
                return part
        return ""

    @staticmethod
    def _fit_vector(values: Any, size: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        fitted = np.zeros(size, dtype=np.float32)
        fitted[: min(size, values.shape[0])] = values[:size]
        return fitted
