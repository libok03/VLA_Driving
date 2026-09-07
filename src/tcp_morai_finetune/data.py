from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from multimodal_planner_v9.data import (
    ACTION_AVOID,
    ACTION_COUNT,
    ACTION_DRIVE,
    ACTION_NAMES,
    ACTION_STOP,
    RunStore,
    _apply_photometric_augmentation,
    _image_tensor,
    _sample_photometric_augmentation,
)


# Official TCP predicts four future frames saved at roughly 2 Hz.  MORAI's raw
# labels are 5 Hz, so these are the closest *existing* samples to
# 0.5/1.0/1.5/2.0 s.  No trajectory interpolation or label correction is used.
WAYPOINT_INDICES = np.asarray((2, 4, 7, 9), dtype=np.int64)
WAYPOINT_HORIZONS_S = (0.6, 1.0, 1.6, 2.0)


def _morai_xy_to_tcp(points: np.ndarray) -> np.ndarray:
    """Map MORAI [forward, left] to TCP [right, negative-forward]."""
    value = np.asarray(points, dtype=np.float32)
    converted = np.empty_like(value)
    converted[..., 0] = -value[..., 1]
    converted[..., 1] = -value[..., 0]
    return converted


def _center_crop_resize(image: np.ndarray, size: int = 256) -> np.ndarray:
    height, width = image.shape[:2]
    crop = min(height, width)
    top = (height - crop) // 2
    left = (width - crop) // 2
    value = image[top : top + crop, left : left + crop]
    return cv2.resize(value, (size, size), interpolation=cv2.INTER_AREA)


def _route_target(route: np.ndarray, lookahead_m: float = 10.0) -> np.ndarray:
    points = np.asarray(route, dtype=np.float32)[:, :2]
    closest = int(np.square(points).sum(axis=1).argmin())
    forward = points[closest:]
    if len(forward) < 2:
        return points[-1].copy()
    segment = np.linalg.norm(np.diff(forward, axis=0), axis=1)
    cumulative = np.concatenate((np.zeros(1, dtype=np.float32), np.cumsum(segment)))
    index = int(np.searchsorted(cumulative, lookahead_m, side="left"))
    return forward[min(index, len(forward) - 1)].copy()


def _route_command(route: np.ndarray, lookahead_m: float = 20.0) -> np.ndarray:
    target = _route_target(route, lookahead_m)
    angle = float(np.arctan2(target[1], max(target[0], 1.0e-3)))
    # Official TCP order: LEFT, RIGHT, STRAIGHT, LANEFOLLOW,
    # CHANGELANELEFT, CHANGELANERIGHT.  V9 Local Route has no explicit CARLA
    # road command, so curvature provides the non-privileged adapter.
    command = 3
    if angle > np.deg2rad(12.0):
        command = 0
    elif angle < -np.deg2rad(12.0):
        command = 1
    one_hot = np.zeros(6, dtype=np.float32)
    one_hot[command] = 1.0
    return one_hot


class TCPMoraiDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        run_ids: list[str],
        photometric_augmentation: bool = False,
        augmentation_profile: str = "standard",
        control_cache: Path | None = None,
    ) -> None:
        if augmentation_profile not in {"standard", "strong"}:
            raise ValueError(f"unknown augmentation profile: {augmentation_profile}")
        self.data_root = Path(data_root)
        self.photometric_augmentation = bool(photometric_augmentation)
        self.augmentation_profile = augmentation_profile
        self.control_cache = Path(control_cache) if control_cache is not None else None
        self._control_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._chunk_cache: OrderedDict[tuple[str, int], dict[str, np.ndarray]] = (
            OrderedDict()
        )
        self.runs = [
            RunStore(
                self.data_root / run_id,
                cache_chunks=6,
                shared_cache=self._chunk_cache,
            )
            for run_id in run_ids
        ]
        self.source_run_ids = []
        for run in self.runs:
            manifest = json.loads((run.run_dir / "run_manifest.json").read_text())
            self.source_run_ids.append(str(manifest["source_run_id"]))
        if self.control_cache is not None:
            missing = [source for source in self.source_run_ids if not (self.control_cache / f"{source}.npz").exists()]
            if missing:
                raise FileNotFoundError(f"missing control cache for {len(missing)} runs; first={missing[0]}")
        self.lookup = [
            (run_index, sample_index)
            for run_index, run in enumerate(self.runs)
            for sample_index in range(len(run))
        ]
        labels = []
        for run_index, sample_index in self.lookup:
            action = self.runs[run_index].action_state
            if action is None:
                raise KeyError(f"{self.runs[run_index].run_dir}: action_state is required")
            labels.append(int(action[sample_index]))
        self.action_labels = np.asarray(labels, dtype=np.int64)
        if not np.isin(self.action_labels, np.arange(ACTION_COUNT)).all():
            raise ValueError("action_state contains an invalid class")

    def __len__(self) -> int:
        return len(self.lookup)

    def sampling_weights(
        self,
        fractions: tuple[float, float, float] = (0.65, 0.25, 0.10),
    ) -> np.ndarray:
        target = np.asarray(fractions, dtype=np.float64)
        if target.shape != (ACTION_COUNT,) or np.any(target <= 0.0):
            raise ValueError("fractions must contain three positive values")
        target /= target.sum()
        counts = np.bincount(self.action_labels, minlength=ACTION_COUNT)
        if np.any(counts == 0):
            raise ValueError(f"every class is required, got {counts.tolist()}")
        weights = target[self.action_labels] / counts[self.action_labels]
        return (weights / weights.mean()).astype(np.float64)

    def __getitem__(self, item: int) -> dict[str, Any]:
        run_index, sample_index = self.lookup[item]
        run = self.runs[run_index]
        frame_index = int(run.current_frame_idx[sample_index])
        chunk_index = run._chunk_index(frame_index)
        chunk_info = run.chunks[chunk_index]
        chunk = run._load_chunk(chunk_index)
        local_index = frame_index - chunk_info.start

        image = run._decode_image(chunk, "front", local_index)
        if self.photometric_augmentation:
            image = _apply_photometric_augmentation(
                image,
                _sample_photometric_augmentation(self.augmentation_profile),
                variant=0,
            )
        # The original TCP direct-control attention is fixed to an 8x29
        # ResNet feature map, corresponding to a 256x900 front image. Existing
        # trajectory/state jobs retain their historical 256x256 crop.
        image = (
            cv2.resize(image, (900, 256), interpolation=cv2.INTER_AREA)
            if self.control_cache is not None
            else _center_crop_resize(image)
        )
        route = np.asarray(chunk["route"][local_index], dtype=np.float32)
        target_point_morai = _route_target(route)
        target_point = _morai_xy_to_tcp(target_point_morai)
        command = _route_command(route)
        speed_mps = abs(float(chunk["vehicle"][local_index, 0]))
        state = np.concatenate(
            (
                np.asarray((speed_mps / 12.0,), dtype=np.float32),
                target_point.astype(np.float32),
                command,
            )
        )
        target = np.asarray(run.target[sample_index], dtype=np.float32)
        waypoints = _morai_xy_to_tcp(target[WAYPOINT_INDICES, :2])
        action = int(self.action_labels[item])
        result = {
            "image": _image_tensor(image),
            "state": torch.from_numpy(state),
            "target_point": torch.from_numpy(target_point.astype(np.float32)),
            "waypoints": torch.from_numpy(waypoints),
            "speed_normalized": torch.tensor(speed_mps / 12.0, dtype=torch.float32),
            "action_state": torch.tensor(action, dtype=torch.long),
            "gps_blackout": torch.tensor(bool(run.gps_blackout[sample_index])),
            "run_id": run.run_id,
            "sample_id": int(run.sample_id[sample_index]),
        }
        if self.control_cache is not None:
            source = self.source_run_ids[run_index]
            controls = self._control_cache.pop(source, None)
            if controls is None:
                with np.load(self.control_cache / f"{source}.npz", allow_pickle=False) as cache:
                    controls = np.asarray(cache["control"], dtype=np.float32)
            self._control_cache[source] = controls
            while len(self._control_cache) > 8:
                self._control_cache.popitem(last=False)
            future_indices = run.future_frame_idx[sample_index, WAYPOINT_INDICES]
            result["current_control"] = torch.from_numpy(controls[frame_index].copy())
            result["future_control"] = torch.from_numpy(controls[future_indices].copy())
        return result

    def summary(self) -> dict[str, Any]:
        counts = np.bincount(self.action_labels, minlength=ACTION_COUNT)
        return {
            "runs": len(self.runs),
            "samples": len(self),
            "action_counts": dict(zip(ACTION_NAMES, counts.tolist())),
            "input": {
                "front": [3, 256, 900] if self.control_cache is not None else [3, 256, 256],
                "speed": "vehicle[0] m/s normalized by 12",
                "target_point": (
                    "10m Local Route lookahead converted from MORAI "
                    "[forward,left] to TCP [right,negative-forward]"
                ),
                "command": "6-way TCP one-hot derived from 20m route curvature",
            },
            "target": {
                "waypoints": [4, 2],
                "horizons_s": list(WAYPOINT_HORIZONS_S),
                "source_indices": WAYPOINT_INDICES.tolist(),
                "source_dt_s": 0.2,
                "policy": (
                    "raw relative_x/y without geometric correction, then "
                    "axis-mapped to TCP [right,negative-forward]"
                ),
            },
            "controls": None if self.control_cache is None else {
                "current": [2], "future": [4, 2],
                "order": ["signed_acceleration", "normalized_steering"],
                "source": "raw /morai/ego_vehicle_status aligned by timestamp",
            },
            "augmentation": {
                "enabled": self.photometric_augmentation,
                "profile": self.augmentation_profile,
            },
        }


__all__ = [
    "TCPMoraiDataset",
    "WAYPOINT_HORIZONS_S",
    "WAYPOINT_INDICES",
    "_morai_xy_to_tcp",
    "_route_command",
    "_route_target",
]
