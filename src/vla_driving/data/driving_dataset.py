from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vla_driving.perception import PerceptionExtractor


class DrivingDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        image_size: tuple[int, int],
        perception_dim: int,
        lidar_size: int,
        route_points: int,
        waypoint_count: int,
        waypoint_dim: int = 2,
    ) -> None:
        self.data_root = Path(data_root)
        self.samples = self._load_manifest(manifest_path)
        self.lidar_size = lidar_size
        self.route_points = route_points
        self.waypoint_count = waypoint_count
        self.waypoint_dim = waypoint_dim
        self.perception_dim = perception_dim
        self.perception = PerceptionExtractor({"yolo_enabled": False, "dim": perception_dim})
        _ = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        perception = self._load_perception(sample)
        lidar = np.load(self.data_root / sample["lidar"]).astype(np.float32)
        lidar = self._fit_vector(lidar, self.lidar_size)
        route = self._fit_points(np.asarray(sample.get("route", []), dtype=np.float32), self.route_points)
        waypoints = self._fit_points(
            np.asarray(sample["future_waypoints"], dtype=np.float32),
            self.waypoint_count,
            self.waypoint_dim,
        )

        return {
            "perception": torch.from_numpy(perception),
            "lidar": torch.from_numpy(lidar),
            "pose": torch.tensor(self._build_state(sample), dtype=torch.float32),
            "route": torch.from_numpy(route),
            "waypoints": torch.from_numpy(waypoints),
        }

    @staticmethod
    def _load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
        path = Path(manifest_path)
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _load_perception(self, sample: dict[str, Any]) -> np.ndarray:
        if "perception" in sample:
            return self._fit_vector(np.asarray(sample["perception"], dtype=np.float32), self.perception_dim)
        image = Image.open(self.data_root / sample["image"]).convert("RGB")
        return self.perception.extract(np.asarray(image))

    @staticmethod
    def _fit_vector(values: np.ndarray, size: int) -> np.ndarray:
        fitted = np.zeros(size, dtype=np.float32)
        length = min(values.shape[0], size)
        fitted[:length] = values[:length]
        return fitted

    @staticmethod
    def _fit_points(points: np.ndarray, count: int, dim: int = 2) -> np.ndarray:
        fitted = np.zeros((count, dim), dtype=np.float32)
        if points.size == 0:
            return fitted
        length = min(points.shape[0], count)
        width = min(points.shape[1], dim)
        fitted[:length, :width] = points[:length, :width]
        return fitted

    @staticmethod
    def _build_state(sample: dict[str, Any]) -> list[float]:
        pose = list(sample["pose"])
        if len(pose) >= 4:
            x, y, yaw = pose[0], pose[1], pose[3]
        elif len(pose) == 3:
            x, y, yaw = pose
        else:
            raise ValueError("pose must be [x, y, yaw] or legacy [x, y, z, yaw]")
        lap_index = float(sample.get("lap_index", 0.0))
        return [x, y, yaw, lap_index]
