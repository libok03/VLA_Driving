from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SCALES = np.asarray([50.0, 50.0, np.pi, 20.0], dtype=np.float32)
TARGET_FIELDS = (
    "relative_x",
    "relative_y",
    "relative_yaw",
    "future_speed",
)


@dataclass(frozen=True)
class PhotometricAugmentation:
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    hue_shift: float = 0.0
    fog_strength: float = 0.0
    gamma: float = 1.0
    red_gain: float = 1.0
    blue_gain: float = 1.0
    blur_sigma: float = 0.0
    noise_std: float = 0.0
    jpeg_quality: int = 100
    vignette_strength: float = 0.0
    noise_seed: int = 0


def _sample_photometric_augmentation(
    profile: str = "standard",
) -> PhotometricAugmentation:
    if profile not in {"standard", "strong"}:
        raise ValueError(f"unknown augmentation profile: {profile}")
    strong = profile == "strong"
    jitter_enabled = bool(torch.rand(()) < (0.95 if strong else 0.8))
    fog_enabled = bool(torch.rand(()) < (0.35 if strong else 0.25))

    def uniform(low: float, high: float) -> float:
        return float(torch.empty(()).uniform_(low, high))

    if not strong:
        return PhotometricAugmentation(
            brightness=uniform(0.8, 1.2) if jitter_enabled else 1.0,
            contrast=uniform(0.8, 1.2) if jitter_enabled else 1.0,
            saturation=uniform(0.85, 1.15) if jitter_enabled else 1.0,
            hue_shift=uniform(-0.03, 0.03) if jitter_enabled else 0.0,
            fog_strength=uniform(0.08, 0.28) if fog_enabled else 0.0,
        )
    blur_enabled = bool(torch.rand(()) < 0.25)
    noise_enabled = bool(torch.rand(()) < 0.30)
    jpeg_enabled = bool(torch.rand(()) < 0.20)
    vignette_enabled = bool(torch.rand(()) < 0.25)
    return PhotometricAugmentation(
        brightness=uniform(0.55, 1.45) if jitter_enabled else 1.0,
        contrast=uniform(0.65, 1.40) if jitter_enabled else 1.0,
        saturation=uniform(0.65, 1.35) if jitter_enabled else 1.0,
        # Keep hue bounded because red/green signal semantics must not change.
        hue_shift=uniform(-0.025, 0.025) if jitter_enabled else 0.0,
        fog_strength=uniform(0.05, 0.38) if fog_enabled else 0.0,
        gamma=uniform(0.70, 1.45),
        red_gain=uniform(0.85, 1.15),
        blue_gain=uniform(0.85, 1.15),
        blur_sigma=uniform(0.35, 1.20) if blur_enabled else 0.0,
        noise_std=uniform(0.005, 0.030) if noise_enabled else 0.0,
        jpeg_quality=int(torch.randint(45, 91, ()).item()) if jpeg_enabled else 100,
        vignette_strength=(
            uniform(0.08, 0.32) if vignette_enabled else 0.0
        ),
        noise_seed=int(torch.randint(0, 2**31 - 1, ()).item()),
    )


def _apply_photometric_augmentation(
    image: np.ndarray,
    augmentation: PhotometricAugmentation,
    variant: int = 0,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("augmentation expects uint8 RGB [H,W,3]")
    value = image.astype(np.float32) / 255.0
    value *= augmentation.brightness
    mean = value.mean(axis=(0, 1), keepdims=True)
    value = (value - mean) * augmentation.contrast + mean
    luminance = (
        0.299 * value[..., 0:1]
        + 0.587 * value[..., 1:2]
        + 0.114 * value[..., 2:3]
    )
    value = np.clip(
        luminance + augmentation.saturation * (value - luminance),
        0.0,
        1.0,
    )
    if augmentation.hue_shift != 0.0:
        hsv = cv2.cvtColor(
            np.rint(value * 255.0).astype(np.uint8),
            cv2.COLOR_RGB2HSV,
        )
        hue = hsv[..., 0].astype(np.int16)
        hue = (hue + round(augmentation.hue_shift * 180.0)) % 180
        hsv[..., 0] = hue.astype(np.uint8)
        value = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    value = np.power(np.clip(value, 0.0, 1.0), augmentation.gamma)
    value *= np.asarray(
        [augmentation.red_gain, 1.0, augmentation.blue_gain],
        dtype=np.float32,
    )
    if augmentation.fog_strength > 0.0:
        vertical = np.linspace(
            1.0,
            0.0,
            value.shape[0],
            dtype=np.float32,
        )[:, None, None]
        veil = augmentation.fog_strength * (0.65 + 0.35 * vertical)
        fog_color = np.asarray([0.92, 0.94, 0.96], dtype=np.float32)
        value = value * (1.0 - veil) + fog_color * veil
    if augmentation.vignette_strength > 0.0:
        height, width = value.shape[:2]
        yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        radius_squared = np.clip(xx * xx + yy * yy, 0.0, 1.0)
        vignette = 1.0 - augmentation.vignette_strength * radius_squared
        value *= vignette[..., None]
    if augmentation.noise_std > 0.0:
        rng = np.random.default_rng(augmentation.noise_seed + int(variant))
        value += rng.normal(
            0.0,
            augmentation.noise_std,
            size=value.shape,
        ).astype(np.float32)
    output = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    if augmentation.blur_sigma > 0.0:
        output = cv2.GaussianBlur(
            output,
            (0, 0),
            sigmaX=augmentation.blur_sigma,
            sigmaY=augmentation.blur_sigma,
        )
    if augmentation.jpeg_quality < 100:
        encoded, buffer = cv2.imencode(
            ".jpg",
            output,
            [cv2.IMWRITE_JPEG_QUALITY, augmentation.jpeg_quality],
        )
        if encoded:
            decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if decoded is not None:
                output = decoded
    return output


def _deterministic_run_split(
    run_ids: list[str],
    seed: int,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> dict[str, list[str]]:
    if not run_ids:
        raise ValueError("no run IDs were provided")
    ids = sorted(run_ids)
    random.Random(seed).shuffle(ids)
    count = len(ids)
    test_count = max(1, round(count * (1.0 - train_ratio - val_ratio)))
    val_count = max(1, round(count * val_ratio))
    train_count = count - val_count - test_count
    if train_count < 1:
        raise ValueError("not enough runs for train/validation/test split")
    return {
        "train": sorted(ids[:train_count]),
        "val": sorted(ids[train_count : train_count + val_count]),
        "test": sorted(ids[train_count + val_count :]),
    }


def save_split_manifest(
    path: Path,
    splits: dict[str, list[str]],
    seed: int,
    data_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "split_unit": "run",
        "seed": seed,
        "data_root": str(data_root.resolve()),
        "splits": splits,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split_unit") != "run":
        raise ValueError(f"{path}: split_unit must be 'run'")
    return payload["splits"]


@dataclass(frozen=True)
class FrameChunk:
    path: Path
    start: int
    end: int


class RunStore:
    def __init__(
        self,
        run_dir: Path,
        cache_chunks: int = 3,
        shared_cache: OrderedDict[tuple[str, int], dict[str, np.ndarray]] | None = None,
        allow_legacy_target_fields: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.run_id = run_dir.name
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.action_label_policy = str(
                manifest.get(
                    "action_label_policy",
                    "reviewed explicit action_state",
                )
            )
            self.dataset_version = str(manifest.get("dataset_version", "unknown"))
        else:
            self.action_label_policy = "reviewed explicit action_state"
            self.dataset_version = "unknown"
        with np.load(run_dir / "sample_index.npz", allow_pickle=False) as sample:
            self.sample_id = np.asarray(sample["sample_id"], dtype=np.int64)
            self.current_frame_idx = np.asarray(
                sample["current_frame_idx"],
                dtype=np.int64,
            )
            self.history_frame_idx = np.asarray(
                sample["history_frame_idx"],
                dtype=np.int64,
            )
            self.future_frame_idx = np.asarray(
                sample["future_frame_idx"],
                dtype=np.int64,
            )
            self.gps_blackout = np.asarray(
                sample["gps_blackout"],
                dtype=np.bool_,
            )
            self.action_state = (
                np.asarray(sample["action_state"], dtype=np.int64)
                if "action_state" in sample.files
                else None
            )
            if all(field in sample.files for field in TARGET_FIELDS):
                self.target = np.stack(
                    [
                        np.asarray(sample[field], dtype=np.float32)
                        for field in TARGET_FIELDS
                    ],
                    axis=-1,
                )
                self.target_schema = "v9_relative_fields"
            elif allow_legacy_target_fields and {
                "future_xy",
                "future_yaw",
                "future_speed",
            }.issubset(sample.files):
                future_xy = np.asarray(sample["future_xy"], dtype=np.float32)
                self.target = np.stack(
                    (
                        future_xy[..., 0],
                        future_xy[..., 1],
                        np.asarray(sample["future_yaw"], dtype=np.float32),
                        np.asarray(sample["future_speed"], dtype=np.float32),
                    ),
                    axis=-1,
                )
                self.target_schema = "legacy_fields_without_numeric_correction"
            else:
                missing = [
                    field for field in TARGET_FIELDS if field not in sample.files
                ]
                raise KeyError(
                    f"{run_dir / 'sample_index.npz'}: missing target fields "
                    f"{missing}"
                )
            if self.target.shape != (len(self.sample_id), 20, 4):
                raise ValueError(
                    f"{run_dir / 'sample_index.npz'}: target shape "
                    f"{self.target.shape} is invalid"
                )
            if not np.isfinite(self.target).all():
                raise ValueError(f"{run_dir}: target contains non-finite values")
        chunk_manifest = json.loads(
            (run_dir / "frame_chunks.json").read_text(encoding="utf-8")
        )
        entries = chunk_manifest.get(
            "frame_chunks",
            chunk_manifest.get("chunks", chunk_manifest),
        )
        self.chunks = [
            FrameChunk(
                run_dir / item["file"],
                int(item["start_frame"]),
                int(item["end_frame_exclusive"]),
            )
            for item in entries
        ]
        self.cache_chunks = cache_chunks
        self.shared_cache = shared_cache
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.sample_id)

    def _chunk_index(self, frame_index: int) -> int:
        candidate = min(frame_index // 100, len(self.chunks) - 1)
        chunk = self.chunks[candidate]
        if chunk.start <= frame_index < chunk.end:
            return candidate
        for index, chunk in enumerate(self.chunks):
            if chunk.start <= frame_index < chunk.end:
                return index
        raise IndexError(f"{self.run_id}: frame {frame_index} is out of range")

    def _load_chunk(self, chunk_index: int) -> dict[str, np.ndarray]:
        cache: OrderedDict[Any, dict[str, np.ndarray]]
        cache_key: Any
        if self.shared_cache is not None:
            cache = self.shared_cache
            cache_key = (str(self.run_dir), chunk_index)
        else:
            cache = self._cache
            cache_key = chunk_index
        cached = cache.pop(cache_key, None)
        if cached is not None:
            cache[cache_key] = cached
            return cached
        keys = (
            "lidar_bev",
            "imu",
            "vehicle",
            "health",
            "route",
            "mgeo",
            "front_jpeg_data",
            "front_jpeg_offsets",
            "left_jpeg_data",
            "left_jpeg_offsets",
            "right_jpeg_data",
            "right_jpeg_offsets",
        )
        with np.load(self.chunks[chunk_index].path, allow_pickle=False) as frame:
            loaded = {key: np.asarray(frame[key]) for key in keys}
        cache[cache_key] = loaded
        while len(cache) > self.cache_chunks:
            cache.popitem(last=False)
        return loaded

    @staticmethod
    def _decode_image(
        chunk: dict[str, np.ndarray],
        camera: str,
        local_index: int,
    ) -> np.ndarray:
        data = chunk[f"{camera}_jpeg_data"]
        offsets = chunk[f"{camera}_jpeg_offsets"]
        encoded = data[offsets[local_index] : offsets[local_index + 1]]
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to decode {camera} JPEG")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def frame(self, frame_index: int) -> dict[str, np.ndarray]:
        chunk_index = self._chunk_index(frame_index)
        info = self.chunks[chunk_index]
        local_index = frame_index - info.start
        chunk = self._load_chunk(chunk_index)
        return {
            "front": self._decode_image(chunk, "front", local_index),
            "left": self._decode_image(chunk, "left", local_index),
            "right": self._decode_image(chunk, "right", local_index),
            "lidar_bev": chunk["lidar_bev"][local_index],
            "imu": chunk["imu"][local_index],
            "vehicle": chunk["vehicle"][local_index],
            "health": chunk["health"][local_index],
            "route": chunk["route"][local_index],
            "mgeo": chunk["mgeo"][local_index],
        }


def _image_tensor(image: np.ndarray) -> torch.Tensor:
    value = image.astype(np.float32) / 255.0
    value = (value - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))


def _ego_vector(frame: dict[str, np.ndarray]) -> np.ndarray:
    vehicle = np.asarray(frame["vehicle"], dtype=np.float32).copy()
    imu = np.asarray(frame["imu"], dtype=np.float32).copy()
    health = np.asarray(frame["health"], dtype=np.float32).copy()
    vehicle /= np.asarray([30.0, 30.0, 10.0, 0.7, 10.0], dtype=np.float32)
    imu /= np.asarray([5.0, 5.0, 5.0, 20.0, 20.0, 20.0], dtype=np.float32)
    health[1] = np.clip(health[1] / 15.0, 0.0, 1.0)
    health[4] = np.clip(health[4], 0.0, 1.0)
    return np.concatenate((vehicle, imu, health)).astype(np.float32)


class PlannerDatasetBase(Dataset):
    def __init__(
        self,
        data_root: Path,
        run_ids: list[str],
        blackout_weight: float = 2.0,
        max_samples: int = 0,
        seed: int = 2026,
        allow_legacy_target_fields: bool = False,
        photometric_augmentation: bool = False,
        augmentation_profile: str = "standard",
    ) -> None:
        # One bounded LRU is shared across all runs. A cache per run scales as
        # O(number_of_runs) and becomes tens of GB for Bench2Drive Base.
        self._chunk_cache: OrderedDict[
            tuple[str, int], dict[str, np.ndarray]
        ] = OrderedDict()
        self.runs = [
            RunStore(
                data_root / run_id,
                cache_chunks=6,
                shared_cache=self._chunk_cache,
                allow_legacy_target_fields=allow_legacy_target_fields,
            )
            for run_id in run_ids
        ]
        self.lookup = [
            (run_index, sample_index)
            for run_index, run in enumerate(self.runs)
            for sample_index in range(len(run))
        ]
        if max_samples > 0 and len(self.lookup) > max_samples:
            rng = random.Random(seed)
            rng.shuffle(self.lookup)
            self.lookup = self.lookup[:max_samples]
        self.blackout_weight = float(blackout_weight)
        self.photometric_augmentation = photometric_augmentation
        if augmentation_profile not in {"standard", "strong"}:
            raise ValueError(
                f"unknown augmentation profile: {augmentation_profile}"
            )
        self.augmentation_profile = augmentation_profile
        self.blackout_samples = sum(
            int(self.runs[run_index].gps_blackout[sample_index])
            for run_index, sample_index in self.lookup
        )

    @property
    def mean_sample_weight(self) -> float:
        if not self.lookup:
            return 1.0
        blackout_fraction = self.blackout_samples / len(self.lookup)
        return 1.0 + (self.blackout_weight - 1.0) * blackout_fraction

    def __len__(self) -> int:
        return len(self.lookup)

    def __getitem__(self, item: int) -> dict[str, Any]:
        run_index, sample_index = self.lookup[item]
        run = self.runs[run_index]
        frames = [
            run.frame(int(index))
            for index in run.history_frame_idx[sample_index]
        ]
        current = frames[-1]
        if self.photometric_augmentation:
            augmentation = _sample_photometric_augmentation(
                self.augmentation_profile
            )
            for frame_index, frame in enumerate(frames):
                for camera_index, camera in enumerate(
                    ("front", "left", "right")
                ):
                    frame[camera] = _apply_photometric_augmentation(
                        frame[camera],
                        augmentation,
                        variant=frame_index * 3 + camera_index,
                    )
        route = np.asarray(current["route"], dtype=np.float32).copy()
        route[:, :2] /= 50.0
        target = run.target[sample_index].copy() / TARGET_SCALES
        blackout = bool(run.gps_blackout[sample_index])
        return {
            "front": torch.stack(
                [_image_tensor(frame["front"]) for frame in frames]
            ),
            "left": torch.stack(
                [_image_tensor(frame["left"]) for frame in frames]
            ),
            "right": torch.stack(
                [_image_tensor(frame["right"]) for frame in frames]
            ),
            "lidar_bev": torch.stack(
                [
                    torch.from_numpy(
                        np.ascontiguousarray(
                            frame["lidar_bev"].astype(np.float32) / 255.0
                        )
                    )
                    for frame in frames
                ]
            ),
            "ego": torch.stack(
                [torch.from_numpy(_ego_vector(frame)) for frame in frames]
            ),
            "mgeo": torch.from_numpy(
                np.ascontiguousarray(current["mgeo"].astype(np.float32))
            ),
            "local_route": torch.from_numpy(np.ascontiguousarray(route)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "sample_weight": torch.tensor(
                self.blackout_weight if blackout else 1.0,
                dtype=torch.float32,
            ),
            "gps_blackout": torch.tensor(blackout),
            "run_id": run.run_id,
            "sample_id": int(run.sample_id[sample_index]),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "runs": len(self.runs),
            "samples": len(self.lookup),
            "blackout_samples": self.blackout_samples,
            "blackout_weight": self.blackout_weight,
            "mean_sample_weight": self.mean_sample_weight,
            "target_fields": list(TARGET_FIELDS),
            "target_schemas": sorted({run.target_schema for run in self.runs}),
            "target_shape": [20, 4],
            "target_scales": TARGET_SCALES.tolist(),
            "label_policy": "provided_relative_values_without_correction",
            "augmentation": {
                "enabled": self.photometric_augmentation,
                "profile": self.augmentation_profile,
                "scope": "train_only_shared_across_5_frames_and_3_cameras",
                "color_jitter_probability": (
                    0.95 if self.augmentation_profile == "strong" else 0.8
                ),
                "fog_probability": (
                    0.35 if self.augmentation_profile == "strong" else 0.25
                ),
                "strong_effects": (
                    [
                        "gamma",
                        "white_balance",
                        "gaussian_blur",
                        "sensor_noise",
                        "jpeg_compression",
                        "vignette",
                    ]
                    if self.augmentation_profile == "strong"
                    else []
                ),
            },
        }


AVOIDANCE_LATERAL_THRESHOLD_M = 0.75
SPATIAL_ANCHORS_M = np.asarray(
    (
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.5,
        13.0,
        14.5,
        16.0,
        18.0,
        20.0,
    ),
    dtype=np.float32,
)
SPATIAL_ANCHOR_COUNT = int(SPATIAL_ANCHORS_M.size)
DEFAULT_TURN_EVENT_THRESHOLD_RAD = np.deg2rad(12.0)
DEFAULT_TURN_LOOKAHEAD_M = 40.0
# V9 learns only a non-positive speed residual.  Keep the external reference
# identical across legacy MORAI samples and future pretraining datasets so the
# delta-v target has one unambiguous meaning.
FIXED_BASE_SPEED_KPH = 60.0
FIXED_BASE_SPEED_MPS = FIXED_BASE_SPEED_KPH / 3.6
# Backward-compatible names for callers importing the old constants.
NORMAL_BASE_SPEED_MPS = FIXED_BASE_SPEED_MPS
SPEED_ZONE_BASE_SPEED_MPS = FIXED_BASE_SPEED_MPS
ACTION_DRIVE = 0
ACTION_STOP = 1
ACTION_AVOID = 2
ACTION_COUNT = 3
ACTION_NAMES = ("DRIVE", "STOP", "AVOID")
TARGET_POLICY_RAW_DRIVE_AVOID = "raw_drive_avoid"
TARGET_POLICY_MORAI_ROUTE_RESIDUAL = "morai_route_residual"
TARGET_POLICIES = (
    TARGET_POLICY_RAW_DRIVE_AVOID,
    TARGET_POLICY_MORAI_ROUTE_RESIDUAL,
)


def project_target_to_route_np(
    route: np.ndarray,
    physical_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project unchanged future XY onto Local Route as progress and signed d."""
    route_value = np.asarray(route, dtype=np.float32)
    target_value = np.asarray(physical_target, dtype=np.float32)
    if route_value.shape != (64, 4):
        raise ValueError(f"route must have shape [64,4], got {route_value.shape}")
    if target_value.shape != (20, 4):
        raise ValueError(
            "physical_target must have shape [20,4], "
            f"got {target_value.shape}"
        )
    points = route_value[:, :2]
    segments = points[1:] - points[:-1]
    lengths = np.maximum(np.linalg.norm(segments, axis=-1), 1.0e-4)
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(lengths, dtype=np.float32))
    )
    origin_fraction = np.clip(
        np.sum((-points[:-1]) * segments, axis=-1) / np.square(lengths),
        0.0,
        1.0,
    )
    origin_projected = points[:-1] + origin_fraction[:, None] * segments
    origin_segment = int(
        np.argmin(np.sum(np.square(origin_projected), axis=-1))
    )
    origin_s = (
        cumulative[origin_segment]
        + origin_fraction[origin_segment] * lengths[origin_segment]
    )
    target_xy = target_value[:, :2]
    delta = target_xy[:, None, :] - points[None, :-1, :]
    fraction = np.clip(
        np.sum(delta * segments[None, :, :], axis=-1)
        / np.square(lengths)[None, :],
        0.0,
        1.0,
    )
    projected = points[None, :-1, :] + fraction[..., None] * segments[None]
    residual = target_xy[:, None, :] - projected
    closest = np.argmin(np.sum(np.square(residual), axis=-1), axis=1)
    row = np.arange(target_xy.shape[0])
    progress = (
        cumulative[closest]
        + fraction[row, closest] * lengths[closest]
        - origin_s
    )
    unit = segments[closest] / lengths[closest, None]
    selected_residual = residual[row, closest]
    lateral = (
        unit[:, 0] * selected_residual[:, 1]
        - unit[:, 1] * selected_residual[:, 0]
    )
    return progress.astype(np.float32), lateral.astype(np.float32)


def temporal_residual_to_spatial_np(
    progress_m: np.ndarray,
    lateral_m: np.ndarray,
    anchors_m: np.ndarray = SPATIAL_ANCHORS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate unchanged temporal lateral residual at reached stations."""
    progress = np.asarray(progress_m, dtype=np.float32).reshape(-1)
    lateral = np.asarray(lateral_m, dtype=np.float32).reshape(-1)
    anchors = np.asarray(anchors_m, dtype=np.float32).reshape(-1)
    if progress.shape != lateral.shape:
        raise ValueError("progress and lateral arrays must have the same shape")
    if not np.isfinite(progress).all() or not np.isfinite(lateral).all():
        raise ValueError("progress and lateral targets must be finite")
    if np.any(np.diff(anchors) <= 0.0):
        raise ValueError("anchors must be strictly increasing")
    pairs = [(0.0, 0.0)]
    furthest = 0.0
    for s_value, d_value in zip(progress.tolist(), lateral.tolist()):
        if s_value < furthest - 1.0e-3 or s_value < 0.0:
            continue
        furthest = max(furthest, float(s_value))
        if abs(float(s_value) - pairs[-1][0]) <= 1.0e-3:
            pairs[-1] = (float(s_value), float(d_value))
        else:
            pairs.append((float(s_value), float(d_value)))
    valid = anchors <= furthest + 1.0e-3
    target = np.zeros_like(anchors, dtype=np.float32)
    if valid.any() and len(pairs) >= 2:
        source_s = np.asarray([item[0] for item in pairs], dtype=np.float32)
        source_d = np.asarray([item[1] for item in pairs], dtype=np.float32)
        target[valid] = np.interp(anchors[valid], source_s, source_d)
    return target, valid.astype(np.bool_)


def deterministic_run_split(
    run_ids: list[str],
    seed: int,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> dict[str, list[str]]:
    """Split label run directories by their shared source bag.

    Desktop conversion creates up to three ``__DRIVE/__STOP/__AVOID`` run
    directories per source bag. Keeping those siblings in one split prevents
    identical frame chunks from leaking across train/validation/test.
    """
    grouped: dict[str, list[str]] = {}
    for run_id in run_ids:
        source_id, separator, suffix = run_id.rpartition("__")
        group_id = (
            source_id
            if separator and suffix in ACTION_NAMES
            else run_id
        )
        grouped.setdefault(group_id, []).append(run_id)
    group_splits = _deterministic_run_split(
        list(grouped),
        seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    return {
        split: sorted(
            run_id
            for group_id in group_ids
            for run_id in grouped[group_id]
        )
        for split, group_ids in group_splits.items()
    }


def _validate_mgeo_np(mgeo: np.ndarray) -> np.ndarray:
    value = np.asarray(mgeo, dtype=np.float32)
    if value.shape != (64, 8):
        raise ValueError(f"mgeo must have shape [64,8], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("mgeo must contain only finite values")
    return value


def mgeo_base_speed_profile_mps_np(mgeo: np.ndarray) -> np.ndarray:
    """Return the fixed external 60 km/h profile for legacy samples.

    MGeo is validated for compatibility with the existing NPZ contract, but
    no MGeo field is allowed to change the base speed.  Raw future speed stays
    untouched and is converted to a non-positive residual elsewhere.
    """
    _validate_mgeo_np(mgeo)
    return np.full(64, FIXED_BASE_SPEED_MPS, dtype=np.float32)


def base_speed_profile_to_anchors_np(
    profile_mps: np.ndarray,
    local_route: np.ndarray,
    anchors_m: np.ndarray = SPATIAL_ANCHORS_M,
) -> np.ndarray:
    """Interpolate a Local-Route-aligned speed profile at spatial anchors.

    New preprocessing should write a speed value aligned with every Local
    Route point. Existing v001 data uses the corresponding 64 MGeo flags as a
    compatibility adapter before raw MGeo is discarded.
    """
    profile = np.asarray(profile_mps, dtype=np.float32).reshape(-1)
    route = np.asarray(local_route, dtype=np.float32)
    anchors = np.asarray(anchors_m, dtype=np.float32).reshape(-1)
    if profile.shape != (64,):
        raise ValueError(f"base speed profile must have shape [64], got {profile.shape}")
    if route.shape != (64, 4):
        raise ValueError(f"local_route must have shape [64,4], got {route.shape}")
    if not np.isfinite(profile).all() or np.any(profile <= 0.0):
        raise ValueError("base speed profile must contain finite positive values")
    if not np.isfinite(route).all():
        raise ValueError("local_route must contain only finite values")

    points = route[:, :2]
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    safe_lengths = np.maximum(lengths, 1.0e-4)
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(safe_lengths))
    )
    projection_fraction = np.clip(
        -(points[:-1] * segments).sum(axis=1) / np.square(safe_lengths),
        0.0,
        1.0,
    )
    projected = points[:-1] + projection_fraction[:, None] * segments
    closest = int(np.square(projected).sum(axis=1).argmin())
    origin_s = (
        cumulative[closest]
        + projection_fraction[closest] * safe_lengths[closest]
    )
    query_s = origin_s + np.maximum(anchors, 0.0)
    return np.interp(
        query_s,
        cumulative,
        profile,
        left=float(profile[0]),
        right=float(profile[-1]),
    ).astype(np.float32)


def mgeo_base_speed_mps_np(mgeo: np.ndarray) -> float:
    """Return the fixed external 60 km/h scalar for legacy callers."""
    _validate_mgeo_np(mgeo)
    return FIXED_BASE_SPEED_MPS


def route_turn_event_np(
    route: np.ndarray,
    threshold_rad: float = DEFAULT_TURN_EVENT_THRESHOLD_RAD,
    lookahead_m: float = DEFAULT_TURN_LOOKAHEAD_M,
) -> bool:
    """Detect a meaningful near-route heading change without altering the route."""
    value = np.asarray(route, dtype=np.float32)
    if value.shape != (64, 4):
        raise ValueError(f"route must have shape [64,4], got {value.shape}")
    points = value[:, :2]
    start = int(np.argmin(np.square(points).sum(axis=1)))
    forward = points[start:]
    if len(forward) < 3:
        return False
    step = np.diff(forward, axis=0)
    length = np.linalg.norm(step, axis=1)
    valid = length > 1.0e-3
    if int(valid.sum()) < 2:
        return False
    cumulative = np.cumsum(length)
    selected = valid & (cumulative <= lookahead_m)
    if int(selected.sum()) < 2:
        selected = valid
    heading = np.unwrap(np.arctan2(step[selected, 1], step[selected, 0]))
    change = np.max(np.abs(heading - heading[0]))
    return bool(change >= threshold_rad)


def temporal_speed_delta_to_spatial_np(
    progress_m: np.ndarray,
    future_speed_mps: np.ndarray,
    base_speed_mps: float | np.ndarray,
    *,
    event: bool,
    anchors_m: np.ndarray = SPATIAL_ANCHORS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Project raw future speed reduction onto reached route stations.

    The source speed is never corrected. Positive residuals are clipped to zero
    because V9 may only reduce the externally supplied MGeo maximum speed.
    ``event`` controls whether the supplied raw future speed is projected.
    V9 action datasets pass ``True`` for DRIVE and AVOID. STOP remains in the
    action-classification loss but is masked from both residual heads.
    """
    progress = np.asarray(progress_m, dtype=np.float32).reshape(-1)
    speed = np.asarray(future_speed_mps, dtype=np.float32).reshape(-1)
    anchors = np.asarray(anchors_m, dtype=np.float32).reshape(-1)
    if progress.shape != speed.shape:
        raise ValueError("progress and speed must have the same shape")
    base = np.asarray(base_speed_mps, dtype=np.float32)
    if base.ndim == 0:
        base = np.full_like(anchors, float(base))
    else:
        base = base.reshape(-1)
    if base.shape != anchors.shape:
        raise ValueError("base_speed_mps must be scalar or match the anchor shape")
    if not np.isfinite(base).all() or np.any(base <= 0.0):
        raise ValueError("base_speed_mps must contain finite positive values")
    if not np.isfinite(progress).all() or not np.isfinite(speed).all():
        raise ValueError("speed target inputs must be finite")
    if not event:
        return np.zeros_like(anchors), np.zeros_like(anchors, dtype=np.bool_)

    pairs: list[tuple[float, float]] = []
    furthest = 0.0
    for s_value, speed_value in zip(progress.tolist(), speed.tolist()):
        if s_value < furthest - 1.0e-3 or s_value < 0.0:
            continue
        furthest = max(furthest, float(s_value))
        if pairs and abs(float(s_value) - pairs[-1][0]) <= 1.0e-3:
            pairs[-1] = (float(s_value), float(speed_value))
        else:
            pairs.append((float(s_value), float(speed_value)))

    valid = anchors <= furthest + 1.0e-3
    target = np.zeros_like(anchors, dtype=np.float32)
    if valid.any() and pairs:
        source_s = np.asarray([item[0] for item in pairs], dtype=np.float32)
        source_speed = np.asarray([item[1] for item in pairs], dtype=np.float32)
        interpolated_speed = np.interp(anchors[valid], source_s, source_speed)
        target[valid] = np.clip(
            interpolated_speed - base[valid],
            -base[valid],
            0.0,
        )
    return target, valid.astype(np.bool_)


def apply_action_target_policy_np(
    action_state: int,
    lateral_target_m: np.ndarray,
    lateral_valid: np.ndarray,
    speed_target_mps: np.ndarray,
    speed_valid: np.ndarray,
    *,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the runtime-aligned regression contract for one V9 sample.

    ``raw_drive_avoid`` preserves the original Bench2Drive supervision:
    DRIVE and AVOID follow the recorded future trajectory while STOP is
    classification-only. ``morai_route_residual`` makes the fixed Local Route
    the explicit DRIVE target (zero delta-d/delta-v at every spatial anchor),
    preserves the recorded residual only for AVOID, and keeps STOP out of both
    regression heads. The model still emits both candidates for every state;
    the downstream state machine alone decides whether to apply them.
    """
    if policy not in TARGET_POLICIES:
        raise ValueError(f"unknown V9 target policy: {policy!r}")
    if action_state not in (ACTION_DRIVE, ACTION_STOP, ACTION_AVOID):
        raise ValueError(f"invalid action_state: {action_state}")

    lateral = np.asarray(lateral_target_m, dtype=np.float32).copy()
    lateral_mask = np.asarray(lateral_valid, dtype=np.bool_).copy()
    speed = np.asarray(speed_target_mps, dtype=np.float32).copy()
    speed_mask = np.asarray(speed_valid, dtype=np.bool_).copy()
    if not (
        lateral.shape == lateral_mask.shape == speed.shape == speed_mask.shape
    ):
        raise ValueError("spatial targets and masks must share one shape")

    if action_state == ACTION_STOP:
        lateral_mask[:] = False
        speed_mask[:] = False
    elif (
        policy == TARGET_POLICY_MORAI_ROUTE_RESIDUAL
        and action_state == ACTION_DRIVE
    ):
        lateral[:] = 0.0
        speed[:] = 0.0
        # DRIVE means the complete 20 m Local Route is the desired candidate,
        # independent of how far the recorded vehicle happened to travel.
        lateral_mask[:] = True
        speed_mask[:] = True
    return lateral, lateral_mask, speed, speed_mask


class PlannerDataset(PlannerDatasetBase):
    """V9 samples with explicit DRIVE/STOP/AVOID conditional supervision."""

    def __init__(
        self,
        *args: Any,
        action_drive_weight: float = 1.0,
        action_stop_weight: float = 0.0,
        action_avoid_weight: float = 6.0,
        target_policy: str = TARGET_POLICY_RAW_DRIVE_AVOID,
        **kwargs: Any,
    ) -> None:
        if target_policy not in TARGET_POLICIES:
            raise ValueError(
                f"target_policy must be one of {TARGET_POLICIES}, got "
                f"{target_policy!r}"
            )
        self.target_policy = target_policy
        action_weights = np.asarray(
            (action_drive_weight, action_stop_weight, action_avoid_weight),
            dtype=np.float32,
        )
        if (
            not np.isfinite(action_weights).all()
            or action_weights[ACTION_DRIVE] <= 0.0
            or action_weights[ACTION_STOP] < 0.0
            or action_weights[ACTION_AVOID] <= 0.0
        ):
            raise ValueError(
                "DRIVE/AVOID regression weights must be positive and the "
                "STOP regression weight must be non-negative"
            )
        # These legacy weighting/threshold arguments remain accepted so older
        # launch scripts keep working. V9 uses reviewed action_state directly;
        # it never derives DRIVE/STOP/AVOID from trajectory heuristics.
        for legacy_name in (
            "stop_weight",
            "drive_weight",
            "avoidance_weight",
            "avoidance_threshold_m",
            "stop_speed_threshold_mps",
            "stop_max_endpoint_distance_m",
        ):
            kwargs.pop(legacy_name, None)
        super().__init__(*args, **kwargs)
        progress_targets = []
        temporal_lateral_targets = []
        for run_index, sample_index in self.lookup:
            run = self.runs[run_index]
            frame_index = int(run.current_frame_idx[sample_index])
            chunk_index = run._chunk_index(frame_index)
            chunk_info = run.chunks[chunk_index]
            chunk = run._load_chunk(chunk_index)
            route = np.asarray(
                chunk["route"][frame_index - chunk_info.start],
                dtype=np.float32,
            )
            progress, lateral = project_target_to_route_np(
                route,
                run.target[sample_index],
            )
            progress_targets.append(progress)
            temporal_lateral_targets.append(lateral)
        self.route_progress_targets = np.asarray(
            progress_targets,
            dtype=np.float32,
        )
        self.lateral_residual_targets = np.asarray(
            temporal_lateral_targets,
            dtype=np.float32,
        )
        lateral_targets = []
        lateral_masks = []
        for progress, lateral in zip(
            self.route_progress_targets,
            self.lateral_residual_targets,
        ):
            target, mask = temporal_residual_to_spatial_np(
                progress,
                lateral,
                anchors_m=SPATIAL_ANCHORS_M,
            )
            lateral_targets.append(target)
            lateral_masks.append(mask)
        self.spatial_lateral_targets = np.asarray(
            lateral_targets,
            dtype=np.float32,
        )
        self.spatial_lateral_masks = np.asarray(
            lateral_masks,
            dtype=np.bool_,
        )
        action_by_run: dict[str, np.ndarray] = {}
        missing_action_runs = []
        for run in self.runs:
            with np.load(run.run_dir / "sample_index.npz", allow_pickle=False) as data:
                if "action_state" not in data.files:
                    missing_action_runs.append(run.run_id)
                    continue
                labels = np.asarray(data["action_state"], dtype=np.int64)
                if labels.shape != (len(run),):
                    raise ValueError(
                        f"{run.run_id}: action_state must have shape "
                        f"[{len(run)}], got {labels.shape}"
                    )
                if np.any((labels < 0) | (labels >= ACTION_COUNT)):
                    raise ValueError(
                        f"{run.run_id}: action_state must use "
                        "0=DRIVE, 1=STOP, 2=AVOID"
                    )
                action_by_run[run.run_id] = labels
        if missing_action_runs:
            preview = ", ".join(missing_action_runs[:3])
            raise RuntimeError(
                "V9 requires reviewed action_state labels in every "
                "sample_index.npz (0=DRIVE, 1=STOP, 2=AVOID); missing in "
                f"{len(missing_action_runs)} run(s), including {preview}"
            )

        targets = []
        masks = []
        action_labels = []
        regression_labels = []
        base_speed_profiles = []
        spatial_base_speed_profiles = []
        for item, (run_index, sample_index) in enumerate(self.lookup):
            run = self.runs[run_index]
            frame_index = int(run.current_frame_idx[sample_index])
            chunk_index = run._chunk_index(frame_index)
            chunk_info = run.chunks[chunk_index]
            chunk = run._load_chunk(chunk_index)
            local_index = frame_index - chunk_info.start
            route = np.asarray(chunk["route"][local_index], dtype=np.float32)
            mgeo = np.asarray(chunk["mgeo"][local_index], dtype=np.float32)
            action = int(action_by_run[run.run_id][sample_index])
            regression_active = action != ACTION_STOP
            base_speed_profile = mgeo_base_speed_profile_mps_np(mgeo)
            spatial_base_speed = base_speed_profile_to_anchors_np(
                base_speed_profile,
                route,
            )
            target, valid = temporal_speed_delta_to_spatial_np(
                self.route_progress_targets[item],
                run.target[sample_index, :, 3],
                spatial_base_speed,
                event=regression_active,
            )
            targets.append(target)
            masks.append(valid)
            action_labels.append(action)
            regression_labels.append(regression_active)
            base_speed_profiles.append(base_speed_profile)
            spatial_base_speed_profiles.append(spatial_base_speed)

        self.spatial_speed_delta_targets_mps = np.asarray(targets, dtype=np.float32)
        self.spatial_speed_masks = np.asarray(masks, dtype=np.bool_)
        self.action_labels = np.asarray(action_labels, dtype=np.int64)
        self.avoid_action_labels = self.action_labels == ACTION_AVOID
        self.regression_labels = np.asarray(regression_labels, dtype=np.bool_)
        for item, action in enumerate(self.action_labels.tolist()):
            (
                self.spatial_lateral_targets[item],
                self.spatial_lateral_masks[item],
                self.spatial_speed_delta_targets_mps[item],
                self.spatial_speed_masks[item],
            ) = apply_action_target_policy_np(
                action,
                self.spatial_lateral_targets[item],
                self.spatial_lateral_masks[item],
                self.spatial_speed_delta_targets_mps[item],
                self.spatial_speed_masks[item],
                policy=self.target_policy,
            )
        self.action_weights = action_weights
        blackout_multiplier = np.asarray(
            [
                (
                    self.blackout_weight
                    if bool(self.runs[run_index].gps_blackout[sample_index])
                    else 1.0
                )
                for run_index, sample_index in self.lookup
            ],
            dtype=np.float32,
        )
        # STOP uses zero regression weight. Cross-entropy class weights are
        # independent, so STOP still trains the action classifier.
        self.v9_sample_weights = (
            action_weights[self.action_labels] * blackout_multiplier
        ).astype(np.float32)
        self._mean_v9_sample_weight = float(
            self.v9_sample_weights.mean() if len(self.v9_sample_weights) else 1.0
        )
        self.base_speed_profiles_mps = np.asarray(
            base_speed_profiles, dtype=np.float32
        )
        self.spatial_base_speed_profiles_mps = np.asarray(
            spatial_base_speed_profiles, dtype=np.float32
        )

    @property
    def mean_sample_weight(self) -> float:
        if hasattr(self, "_mean_v9_sample_weight"):
            return self._mean_v9_sample_weight
        return super().mean_sample_weight

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = super().__getitem__(item)
        # V9 uses MGeo only during dataset initialization to adapt the legacy
        # external base-speed profile. Raw MGeo never reaches the model.
        sample.pop("mgeo")
        sample["action_state"] = torch.tensor(
            int(self.action_labels[item]),
            dtype=torch.long,
        )
        sample["avoidance"] = torch.tensor(
            bool(self.avoid_action_labels[item])
        )
        sample["sample_weight"] = torch.tensor(
            float(self.v9_sample_weights[item]),
            dtype=torch.float32,
        )
        sample["target_route_progress_m"] = torch.from_numpy(
            self.route_progress_targets[item].copy()
        )
        sample["target_lateral_residual_m"] = torch.from_numpy(
            self.lateral_residual_targets[item].copy()
        )
        sample["target_spatial_lateral_m"] = torch.from_numpy(
            self.spatial_lateral_targets[item].copy()
        )
        sample["target_spatial_valid"] = torch.from_numpy(
            self.spatial_lateral_masks[item].copy()
        )
        sample["base_speed_profile_mps"] = torch.from_numpy(
            self.base_speed_profiles_mps[item].copy()
        )
        sample["target_spatial_speed_delta_mps"] = torch.from_numpy(
            self.spatial_speed_delta_targets_mps[item].copy()
        )
        sample["target_spatial_speed_valid"] = torch.from_numpy(
            self.spatial_speed_masks[item].copy()
        )
        sample["speed_event"] = torch.tensor(
            bool(self.regression_labels[item])
        )
        sample["regression_active"] = torch.tensor(
            bool(self.regression_labels[item])
        )
        sample["base_speed_mps"] = torch.from_numpy(
            self.spatial_base_speed_profiles_mps[item].copy()
        )
        return sample

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        for key in (
            "motion_state_names",
            "motion_state_counts",
            "motion_state_sample_weights",
            "avoidance_count",
            "avoidance_fraction",
            "avoidance_threshold_m",
            "avoidance_weight",
            "mean_combined_sample_weight",
            "mean_final_sample_weight",
        ):
            summary.pop(key, None)
        counts = np.bincount(self.action_labels, minlength=ACTION_COUNT)
        summary.update(
            {
                "architecture_target": (
                    "DRIVE_STOP_AVOID_gate_plus_AVOID_delta_d_delta_v"
                ),
                "action_names": list(ACTION_NAMES),
                "action_counts": {
                    name: int(counts[index])
                    for index, name in enumerate(ACTION_NAMES)
                },
                "action_sample_weights": {
                    name: float(self.action_weights[index])
                    for index, name in enumerate(ACTION_NAMES)
                },
                "action_label_policy": (
                    sorted({run.action_label_policy for run in self.runs})
                ),
                "dataset_versions": sorted(
                    {run.dataset_version for run in self.runs}
                ),
                "speed_target_policy": (
                    (
                        "fixed base_speed=60km/h; DRIVE delta_v=0 at all anchors; "
                        "AVOID delta_v=clip(raw_future_speed-base_speed,-base,0); "
                        "STOP is classification-only"
                    )
                    if self.target_policy == TARGET_POLICY_MORAI_ROUTE_RESIDUAL
                    else (
                        "fixed base_speed=60km/h; "
                        "delta_v=clip(raw_future_speed-base_speed,-base,0); "
                        "supervised on DRIVE and AVOID; STOP is classification-only"
                    )
                ),
                "lateral_target_policy": (
                    (
                        "DRIVE delta_d=0 at all anchors (Local Route target); "
                        "AVOID uses raw trajectory residual; "
                        "STOP is classification-only"
                    )
                    if self.target_policy == TARGET_POLICY_MORAI_ROUTE_RESIDUAL
                    else (
                        "raw trajectory residual supervised on DRIVE and AVOID; "
                        "STOP is classification-only"
                    )
                ),
                "target_policy": self.target_policy,
                "model_map_inputs": (
                    "Local Route only; fixed 60km/h base_speed_profile_mps[64] "
                    "is an external non-learned output constraint"
                ),
                "mean_v9_sample_weight": self.mean_sample_weight,
                "spatial_anchors_m": SPATIAL_ANCHORS_M.tolist(),
                "valid_labels_per_anchor": (
                    self.spatial_lateral_masks.sum(axis=0).tolist()
                ),
                "mean_valid_anchors_per_sample": float(
                    self.spatial_lateral_masks.sum(axis=1).mean()
                ),
            }
        )
        return summary


__all__ = [
    "AVOIDANCE_LATERAL_THRESHOLD_M",
    "ACTION_AVOID",
    "ACTION_COUNT",
    "ACTION_DRIVE",
    "ACTION_NAMES",
    "TARGET_POLICIES",
    "TARGET_POLICY_MORAI_ROUTE_RESIDUAL",
    "TARGET_POLICY_RAW_DRIVE_AVOID",
    "FIXED_BASE_SPEED_KPH",
    "FIXED_BASE_SPEED_MPS",
    "ACTION_STOP",
    "DEFAULT_TURN_EVENT_THRESHOLD_RAD",
    "DEFAULT_TURN_LOOKAHEAD_M",
    "NORMAL_BASE_SPEED_MPS",
    "PlannerDataset",
    "SPEED_ZONE_BASE_SPEED_MPS",
    "SPATIAL_ANCHOR_COUNT",
    "SPATIAL_ANCHORS_M",
    "apply_action_target_policy_np",
    "base_speed_profile_to_anchors_np",
    "deterministic_run_split",
    "load_split_manifest",
    "mgeo_base_speed_mps_np",
    "mgeo_base_speed_profile_mps_np",
    "route_turn_event_np",
    "save_split_manifest",
    "temporal_speed_delta_to_spatial_np",
]
