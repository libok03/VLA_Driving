from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from vla_driving.planning.lap_counter import LapCounter
from vla_driving.utils.config import load_config


class Ros2BagExtractor:
    def __init__(
        self,
        cfg: dict[str, Any],
        bag_path: str | Path,
        output_dir: str | Path,
        sample_hz: float,
        image_quality: int,
        storage_id: str,
        waypoints_topic: str,
        require_waypoints: bool,
    ) -> None:
        self.cfg = cfg
        self.bag_path = Path(bag_path)
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "images"
        self.lidar_dir = self.output_dir / "lidar"
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.sample_period_ns = int(1_000_000_000 / sample_hz)
        self.image_quality = int(image_quality)
        self.storage_id = storage_id
        self.waypoints_topic = waypoints_topic
        self.require_waypoints = require_waypoints

        data_cfg = cfg["data"]
        self.lidar_size = int(data_cfg["lidar_size"])
        self.route_points = int(data_cfg["route_points"])
        self.waypoint_count = int(data_cfg["waypoint_count"])
        route_cfg = cfg["route"]
        self.lap_counter = LapCounter(
            gate_a=tuple(route_cfg["finish_gate_a"]),
            gate_b=tuple(route_cfg["finish_gate_b"]),
            forward_yaw=route_cfg["finish_forward_yaw"],
            total_laps=route_cfg["total_laps"],
            cooldown_s=route_cfg["lap_cooldown_s"],
            arm_distance_m=route_cfg["lap_arm_distance_m"],
        )
        self.shortcut_allowed_laps = set(int(v) for v in route_cfg.get("shortcut_allowed_laps", []))

        self.topics = dict(cfg["ros2"]["topics"])
        if self.waypoints_topic:
            self.topics["future_waypoints"] = self.waypoints_topic

        self.image: PILImage.Image | None = None
        self.lidar = np.zeros(self.lidar_size, dtype=np.float32)
        self.has_lidar = False
        self.pose: tuple[float, float, float, float] | None = None
        self.route = np.zeros((self.route_points, 2), dtype=np.float32)
        self.route_mode_id = 0
        self.future_waypoints: np.ndarray | None = None
        self.next_sample_time_ns: int | None = None
        self.sample_index = 0

    def run(self) -> int:
        from rclpy.serialization import deserialize_message
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
        from rosidl_runtime_py.utilities import get_message

        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.lidar_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self.manifest_path.unlink()

        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=str(self.bag_path), storage_id=self.storage_id),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )

        topic_types = {
            topic_metadata.name: topic_metadata.type for topic_metadata in reader.get_all_topics_and_types()
        }
        message_types = {
            topic: get_message(type_name)
            for topic, type_name in topic_types.items()
            if topic in set(self.topics.values())
        }

        while reader.has_next():
            topic, raw_data, timestamp_ns = reader.read_next()
            if topic not in message_types:
                continue

            msg = deserialize_message(raw_data, message_types[topic])
            self._update_state(topic, msg, timestamp_ns)

            if self.next_sample_time_ns is None:
                self.next_sample_time_ns = timestamp_ns
            if timestamp_ns >= self.next_sample_time_ns:
                if self._has_required_state():
                    self._write_sample(timestamp_ns)
                self.next_sample_time_ns = timestamp_ns + self.sample_period_ns

        return self.sample_index

    def _update_state(self, topic: str, msg: Any, timestamp_ns: int) -> None:
        if topic == self.topics["image"]:
            self.image = self._image_msg_to_pil(msg)
        elif topic == self.topics["lidar"]:
            self.lidar = self._fit_lidar(msg.ranges)
            self.has_lidar = True
        elif topic == self.topics["odom"]:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            yaw = self._yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)
            self.pose = (float(position.x), float(position.y), float(position.z), yaw)
            lap_state = self.lap_counter.update(
                x=float(position.x),
                y=float(position.y),
                yaw=yaw,
                timestamp_s=timestamp_ns / 1_000_000_000.0,
            )
            self.route_mode_id = self._route_mode_for_lap(lap_state.lap_count)
        elif topic == self.topics["local_route"]:
            route = np.zeros((self.route_points, 2), dtype=np.float32)
            for idx, pose_stamped in enumerate(msg.poses[: self.route_points]):
                route[idx] = [pose_stamped.pose.position.x, pose_stamped.pose.position.y]
            self.route = route
        elif topic == self.topics.get("future_waypoints"):
            self.future_waypoints = self._fit_waypoints(msg.data)

    def _has_required_state(self) -> bool:
        if self.image is None or self.pose is None or not self.has_lidar:
            return False
        if self.require_waypoints and self.future_waypoints is None:
            return False
        return True

    def _write_sample(self, timestamp_ns: int) -> None:
        if self.image is None or self.pose is None:
            return

        stem = f"{self.sample_index:06d}"
        image_path = self.image_dir / f"{stem}.jpg"
        lidar_path = self.lidar_dir / f"{stem}.npy"
        self.image.save(image_path, quality=self.image_quality)
        np.save(lidar_path, self.lidar)

        sample: dict[str, Any] = {
            "image": image_path.relative_to(self.output_dir).as_posix(),
            "lidar": lidar_path.relative_to(self.output_dir).as_posix(),
            "pose": [float(v) for v in self.pose],
            "route_mode": self._route_mode_name(self.route_mode_id),
            "route_mode_id": self.route_mode_id,
            "route": self.route.astype(float).tolist(),
            "stamp": timestamp_ns / 1_000_000_000.0,
        }
        if self.future_waypoints is not None:
            sample["future_waypoints"] = self.future_waypoints.astype(float).tolist()

        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample, separators=(",", ":")) + "\n")

        self.sample_index += 1

    def _fit_lidar(self, ranges: Any) -> np.ndarray:
        values = np.asarray(ranges, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        fitted = np.zeros(self.lidar_size, dtype=np.float32)
        fitted[: min(self.lidar_size, values.shape[0])] = values[: self.lidar_size]
        return fitted

    def _fit_waypoints(self, values: Any) -> np.ndarray:
        flat = np.asarray(values, dtype=np.float32)
        points = flat[: flat.shape[0] - (flat.shape[0] % 2)].reshape(-1, 2)
        fitted = np.zeros((self.waypoint_count, 2), dtype=np.float32)
        fitted[: min(self.waypoint_count, points.shape[0])] = points[: self.waypoint_count]
        return fitted

    @staticmethod
    def _image_msg_to_pil(msg: Any) -> PILImage.Image:
        channels = Ros2BagExtractor._channels_for_encoding(msg.encoding)
        array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        image = array[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if msg.encoding.lower() == "bgr8":
            image = image[:, :, ::-1].copy()
        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        return PILImage.fromarray(image.astype(np.uint8), mode="RGB")

    @staticmethod
    def _channels_for_encoding(encoding: str) -> int:
        encoding = encoding.lower()
        if encoding in {"rgb8", "bgr8"}:
            return 3
        if encoding in {"mono8", "8uc1"}:
            return 1
        raise ValueError(f"Unsupported image encoding: {encoding}")

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _route_mode_name(route_mode_id: int) -> str:
        return "shortcut" if route_mode_id == 1 else "main"

    def _route_mode_for_lap(self, lap_count: int) -> int:
        return 1 if lap_count in self.shortcut_allowed_laps else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract VLA Driving dataset samples from a ROS2 bag.")
    parser.add_argument("bag_path", help="Path to a ROS2 bag directory.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="data/ros2_bag")
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--image-quality", type=int, default=95)
    parser.add_argument("--storage-id", default="sqlite3", choices=["sqlite3", "mcap"])
    parser.add_argument(
        "--waypoints-topic",
        default="",
        help="Optional Float32MultiArray label topic with flattened future waypoint x,y pairs.",
    )
    parser.add_argument(
        "--require-waypoints",
        action="store_true",
        help="Only write samples after the waypoint label topic has been received.",
    )
    args = parser.parse_args()

    if args.sample_hz <= 0.0:
        raise SystemExit("--sample-hz must be greater than zero.")

    try:
        import rclpy  # noqa: F401
        import rosbag2_py  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ROS2 Python packages are required. Run this inside a sourced ROS2 environment."
        ) from exc

    cfg = load_config(Path(args.config))
    count = Ros2BagExtractor(
        cfg=cfg,
        bag_path=args.bag_path,
        output_dir=args.output_dir,
        sample_hz=args.sample_hz,
        image_quality=args.image_quality,
        storage_id=args.storage_id,
        waypoints_topic=args.waypoints_topic,
        require_waypoints=args.require_waypoints,
    ).run()
    print(f"Wrote {count} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
