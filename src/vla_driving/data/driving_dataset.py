from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DrivingDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        image_size: tuple[int, int],
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
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(self.data_root / sample["image"]).convert("RGB")
        lidar = np.load(self.data_root / sample["lidar"]).astype(np.float32)
        lidar = self._fit_vector(lidar, self.lidar_size)
        route = self._fit_points(np.asarray(sample.get("route", []), dtype=np.float32), self.route_points)
        waypoints = self._fit_points(
            np.asarray(sample["future_waypoints"], dtype=np.float32),
            self.waypoint_count,
            self.waypoint_dim,
        )

        return {
            "image": self.image_transform(image),
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
        route_mode = DrivingDataset._route_mode_id(sample.get("route_mode_id", sample.get("route_mode", 0.0)))
        return [x, y, yaw, route_mode]

    @staticmethod
    def _route_mode_id(value: Any) -> float:
        if isinstance(value, str):
            return {"main": 0.0, "shortcut": 1.0}.get(value, 0.0)
        return float(value)
