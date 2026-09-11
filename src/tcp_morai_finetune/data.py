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

# DataLoader already parallelizes JPEG decode/resize across worker processes.
# Prevent every worker from creating another OpenCV thread pool and
# oversubscribing the 12-thread host CPU.
cv2.setNumThreads(0)


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


def _route_command(route: np.ndarray, lookahead_m: float = 30.0) -> np.ndarray:
    """Infer only TCP's high-level command from longer route context.

    The TCP target point remains the existing 10 m point and waypoint/control
    targets are unchanged.  Looking farther here prevents a vehicle waiting on
    the straight approach to a junction from being labelled LANEFOLLOW merely
    because the actual turn starts beyond the old 20 m command horizon.
    """
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


# Keep the official six-dimensional TCP command contract so published TCP
# checkpoints remain loadable.  MORAI does not use CARLA's lane-change
# commands; reserve CHANGELANELEFT (index 4) as the explicit obstacle-avoidance
# command.  The runtime meaning is therefore AVOID, not a generic lane change.
TCP_COMMAND_AVOID = 4


def _remove_short_left_runs(commands: np.ndarray, min_samples: int = 8) -> np.ndarray:
    """Suppress brief curvature-triggered LEFT commands.

    Converted MORAI samples are 4 Hz.  A real junction LEFT remains active for
    several seconds, while ordinary bends only cross the 12 degree threshold
    for a handful of samples.  Runs shorter than 2 seconds are LANEFOLLOW.
    """
    output = np.asarray(commands, dtype=np.int64).copy()
    index = 0
    while index < len(output):
        if output[index] != 0:  # TCP command index 0 = LEFT
            index += 1
            continue
        end = index + 1
        while end < len(output) and output[end] == 0:
            end += 1
        if end - index < min_samples:
            output[index:end] = 3  # TCP command index 3 = LANEFOLLOW
        index = end
    return output


def _traffic_light_points(node_file: Path) -> np.ndarray:
    nodes = json.loads(Path(node_file).read_text())
    points = [
        node["point"][:2]
        for node in nodes
        if node.get("traffic_light_id")
        and not str(node["traffic_light_id"]).upper().startswith("LCS")
    ]
    return np.asarray(points, dtype=np.float64)


def _apply_signal_straight(
    commands: np.ndarray,
    pose_xy: np.ndarray,
    signal_xy: np.ndarray,
    approach_m: float = 30.0,
    exit_m: float = 10.0,
    capture_radius_m: float = 5.0,
) -> np.ndarray:
    """Override only signalized-intersection passages with TCP STRAIGHT."""
    output = np.asarray(commands, dtype=np.int64).copy()
    signal_window = _signal_window_mask(
        pose_xy,
        signal_xy,
        approach_m=approach_m,
        exit_m=exit_m,
        capture_radius_m=capture_radius_m,
    )
    # A signal does not imply a straight maneuver. Preserve genuine LEFT and
    # RIGHT, and promote only otherwise-LANEFOLLOW samples.
    output[signal_window & (output == 3)] = 2
    return output


def _signal_window_mask(
    pose_xy: np.ndarray,
    signal_xy: np.ndarray,
    approach_m: float = 30.0,
    exit_m: float = 10.0,
    capture_radius_m: float = 5.0,
) -> np.ndarray:
    """Mark samples surrounding an actually traversed MGeo traffic signal."""
    pose_xy = np.asarray(pose_xy, dtype=np.float64)
    signal_xy = np.asarray(signal_xy, dtype=np.float64)
    output = np.zeros(len(pose_xy), dtype=bool)
    if len(output) == 0 or len(signal_xy) == 0:
        return output
    progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pose_xy, axis=0), axis=1))]
    nearest = np.sqrt(
        np.min(np.sum((pose_xy[:, None, :] - signal_xy[None, :, :]) ** 2, axis=2), axis=1)
    )
    hits = np.flatnonzero(nearest <= capture_radius_m)
    if len(hits) == 0:
        return output
    # Collapse consecutive samples around one physical signal passage.
    groups = np.split(hits, np.flatnonzero(np.diff(hits) > 4) + 1)
    for group in groups:
        event = int(group[np.argmin(nearest[group])])
        begin = int(np.searchsorted(progress, progress[event] - approach_m, side="left"))
        end = int(np.searchsorted(progress, progress[event] + exit_m, side="right"))
        output[begin:end] = True
    return output


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
        self.command_labels: list[np.ndarray] = []
        self.signal_window_labels: list[np.ndarray] = []
        node_file = Path(__file__).resolve().parents[1] / "external_models/acca2026_mgeo/node_set.json"
        signal_xy = _traffic_light_points(node_file) if node_file.exists() else np.empty((0, 2))
        for run in self.runs:
            run_commands = []
            run_pose_xy = []
            has_pose = True
            for frame_index in run.current_frame_idx:
                frame_index = int(frame_index)
                chunk_index = run._chunk_index(frame_index)
                chunk_info = run.chunks[chunk_index]
                chunk = run._load_chunk(chunk_index)
                route = np.asarray(
                    chunk["route"][frame_index - chunk_info.start], dtype=np.float32
                )
                command_pose_key = (
                    "localization_pose" if "localization_pose" in chunk else "pose"
                )
                if command_pose_key in chunk:
                    run_pose_xy.append(
                        np.asarray(
                            chunk[command_pose_key][frame_index - chunk_info.start, :2],
                            dtype=np.float64,
                        )
                    )
                else:
                    has_pose = False
                run_commands.append(int(np.argmax(_route_command(route))))
            filtered = _remove_short_left_runs(np.asarray(run_commands, dtype=np.int64))
            signal_window = (
                _signal_window_mask(np.asarray(run_pose_xy), signal_xy)
                if has_pose else np.zeros(len(filtered), dtype=bool)
            )
            self.signal_window_labels.append(signal_window)
            self.command_labels.append(
                _apply_signal_straight(filtered, np.asarray(run_pose_xy), signal_xy)
                if has_pose else filtered
            )
        labels = []
        for run_index, sample_index in self.lookup:
            action = self.runs[run_index].action_state
            if action is None:
                raise KeyError(f"{self.runs[run_index].run_dir}: action_state is required")
            labels.append(int(action[sample_index]))
        self.action_labels = np.asarray(labels, dtype=np.int64)
        self.signal_window = np.concatenate(self.signal_window_labels).astype(bool)
        self.signal_visible = self.signal_window.copy()
        # Samples immediately preceding an AVOID interval still retain their
        # original DRIVE label and human trajectory/control targets.  Mark
        # them separately so training can learn anticipatory steering instead
        # of seeing mostly already-active avoidance frames.
        avoid_starts: dict[str, list[int]] = {}
        for run_index, run in enumerate(self.runs):
            if not run.run_id.endswith("__AVOID") or not len(run.current_frame_idx):
                continue
            frames = np.sort(np.asarray(run.current_frame_idx, dtype=np.int64))
            starts = frames[np.r_[True, np.diff(frames) > 1]]
            avoid_starts.setdefault(self.source_run_ids[run_index], []).extend(starts.tolist())
        self.avoid_approach = np.zeros(len(self.lookup), dtype=bool)
        for item, (run_index, sample_index) in enumerate(self.lookup):
            if self.action_labels[item] != ACTION_DRIVE:
                continue
            frame = int(self.runs[run_index].current_frame_idx[sample_index])
            self.avoid_approach[item] = any(
                start - 12 <= frame < start
                for start in avoid_starts.get(self.source_run_ids[run_index], ())
            )
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

    def hard_event_sampling_weights(
        self,
        fractions: tuple[float, ...] = (0.25, 0.25, 0.20, 0.15, 0.15),
    ) -> np.ndarray:
        """Balance AVOID approach/active, signal DRIVE/STOP, and replay."""
        target = np.asarray(fractions, dtype=np.float64)
        if target.shape not in {(4,), (5,)} or np.any(target < 0.0) or not np.any(target > 0.0):
            raise ValueError("fractions must contain four legacy or five approach-aware values")
        target /= target.sum()
        if len(target) == 4:
            category = np.full(len(self), 3, dtype=np.int64)
            category[self.signal_visible & (self.action_labels == ACTION_STOP)] = 2
            category[self.signal_visible & (self.action_labels == ACTION_DRIVE)] = 1
            category[self.action_labels == ACTION_AVOID] = 0
        else:
            category = np.full(len(self), 4, dtype=np.int64)
            category[self.signal_visible & (self.action_labels == ACTION_STOP)] = 3
            category[self.signal_visible & (self.action_labels == ACTION_DRIVE)] = 2
            category[self.action_labels == ACTION_AVOID] = 1
            category[self.avoid_approach] = 0
        counts = np.bincount(category, minlength=len(target))
        if np.any((counts == 0) & (target > 0.0)):
            raise ValueError(f"requested hard-event category is empty, got {counts.tolist()}")
        weights = target[category] / counts[category]
        return (weights / weights.mean()).astype(np.float64)

    def set_signal_visibility(self, visible: np.ndarray) -> None:
        value = np.asarray(visible, dtype=bool)
        if value.shape != (len(self),):
            raise ValueError(
                f"signal visibility must have shape {(len(self),)}, got {value.shape}"
            )
        # A visual detection is never allowed to expand beyond the MGeo signal
        # passage window used to mine candidates.
        self.signal_visible = value & self.signal_window

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
        # Keep MORAI's native 640x360 aspect ratio for full-policy training.
        # The original TCP implementation resized this image to 900x256 to
        # obtain an 8x29 feature map, which visibly stretched lanes, traffic
        # lights, and obstacles.  TCPMorai now adapts its learned attention
        # map to the native ResNet feature-map size instead.
        image = (
            image
            if self.control_cache is not None
            else _center_crop_resize(image)
        )
        route = np.asarray(chunk["route"][local_index], dtype=np.float32)
        target_point_morai = _route_target(route)
        target_point = _morai_xy_to_tcp(target_point_morai)
        action = int(self.action_labels[item])
        command_index = int(self.command_labels[run_index][sample_index])
        # Preserve the previously reviewed TCP command map verbatim and
        # override only the per-bag, manually labelled AVOID interval.
        # AVOID approach samples remain useful to the sampler, but retain their
        # original command unless the reviewed bag label itself says AVOID.
        if action == ACTION_AVOID:
            command_index = TCP_COMMAND_AVOID
        command = np.zeros(6, dtype=np.float32)
        command[command_index] = 1.0
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
        result = {
            "image": _image_tensor(image),
            "state": torch.from_numpy(state),
            "target_point": torch.from_numpy(target_point.astype(np.float32)),
            "waypoints": torch.from_numpy(waypoints),
            "speed_normalized": torch.tensor(speed_mps / 12.0, dtype=torch.float32),
            "action_state": torch.tensor(action, dtype=torch.long),
            "gps_blackout": torch.tensor(bool(run.gps_blackout[sample_index])),
            "signal_window": torch.tensor(bool(self.signal_window[item])),
            "signal_visible": torch.tensor(bool(self.signal_visible[item])),
            "avoid_approach": torch.tensor(bool(self.avoid_approach[item])),
            "command_index": torch.tensor(command_index, dtype=torch.long),
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
                "front": [3, 360, 640] if self.control_cache is not None else [3, 256, 256],
                "speed": "vehicle[0] m/s normalized by 12",
                "target_point": (
                    "10m Local Route lookahead converted from MORAI "
                    "[forward,left] to TCP [right,negative-forward]"
                ),
                "command": (
                    "6-way TCP one-hot derived from up to 30m route context; "
                    "command context only, target point remains 10m"
                ),
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
    "_remove_short_left_runs",
    "_apply_signal_straight",
    "_signal_window_mask",
    "_traffic_light_points",
    "_route_target",
]
