from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from vla_driving.perception import PerceptionExtractor
from vla_driving.planning.lap_counter import LapCounter
from vla_driving.utils.config import load_config
from vla_driving.utils.geometry import world_to_ego, wrap_angle


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
        generate_waypoints_from_odom: bool,
        future_step_s: float,
        generate_route_from_odom: bool,
        route_step_s: float,
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
        self.generate_waypoints_from_odom = generate_waypoints_from_odom
        self.future_step_ns = int(future_step_s * 1_000_000_000)
        self.generate_route_from_odom = generate_route_from_odom
        self.route_step_ns = int(route_step_s * 1_000_000_000)
        self.odom_yaw_offset = float(cfg.get("ros2", {}).get("odom_yaw_offset", 0.0))

        data_cfg = cfg["data"]
        self.lidar_size = int(data_cfg["lidar_size"])
        self.route_points = int(data_cfg["route_points"])
        self.waypoint_count = int(data_cfg["waypoint_count"])
        self.waypoint_dim = int(data_cfg.get("waypoint_dim", 2))
        self.perception_dim = int(data_cfg.get("perception_dim", 32))
        perception_cfg = dict(cfg.get("ros2", {}).get("perception", {}))
        perception_cfg["dim"] = self.perception_dim
        self.perception_extractor = PerceptionExtractor(perception_cfg, dim=self.perception_dim)
        route_cfg = cfg["route"]
        self.lap_counter = LapCounter(
            gate_a=tuple(route_cfg["finish_gate_a"]),
            gate_b=tuple(route_cfg["finish_gate_b"]),
            forward_yaw=route_cfg["finish_forward_yaw"],
            total_laps=route_cfg["total_laps"],
            cooldown_s=route_cfg["lap_cooldown_s"],
            arm_distance_m=route_cfg["lap_arm_distance_m"],
            trigger_mode=route_cfg.get("lap_trigger_mode", "gate"),
            trigger_center=route_cfg.get("lap_trigger_center"),
            trigger_radius_m=route_cfg.get("lap_trigger_radius_m", 3.0),
        )
        self.topics = dict(cfg["ros2"]["topics"])
        if self.waypoints_topic:
            self.topics["future_waypoints"] = self.waypoints_topic

        self.image: PILImage.Image | None = None
        self.perception = np.zeros(self.perception_dim, dtype=np.float32)
        self.has_perception = False
        self.lidar = np.zeros(self.lidar_size, dtype=np.float32)
        self.has_lidar = False
        self.pose: tuple[float, float, float] | None = None
        self.route = np.zeros((self.route_points, 2), dtype=np.float32)
        self.lap_index = 0
        self.future_waypoints: np.ndarray | None = None
        self.odom_trajectory: list[tuple[int, float, float, float]] = []
        self.odom_times: list[int] = []
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

        reader = self._open_reader(SequentialReader, StorageOptions, ConverterOptions)

        topic_types = {
            topic_metadata.name: topic_metadata.type for topic_metadata in reader.get_all_topics_and_types()
        }
        message_types = {
            topic: get_message(type_name)
            for topic, type_name in topic_types.items()
            if topic in set(self.topics.values())
        }

        if self.generate_waypoints_from_odom or self.generate_route_from_odom:
            self._collect_odom_trajectory(reader, deserialize_message, message_types)
            reader = self._open_reader(SequentialReader, StorageOptions, ConverterOptions)

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

    def _open_reader(self, reader_cls: Any, storage_options_cls: Any, converter_options_cls: Any) -> Any:
        reader = reader_cls()
        reader.open(
            storage_options_cls(uri=str(self.bag_path), storage_id=self.storage_id),
            converter_options_cls(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
        return reader

    def _collect_odom_trajectory(
        self,
        reader: Any,
        deserialize_message: Any,
        message_types: dict[str, Any],
    ) -> None:
        odom_topic = self.topics["odom"]
        if odom_topic not in message_types:
            return

        trajectory: list[tuple[int, float, float, float]] = []
        while reader.has_next():
            topic, raw_data, timestamp_ns = reader.read_next()
            if topic != odom_topic:
                continue
            msg = deserialize_message(raw_data, message_types[topic])
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            yaw = self._odom_yaw(orientation)
            trajectory.append((timestamp_ns, float(position.x), float(position.y), yaw))

        self.odom_trajectory = trajectory
        self.odom_times = [item[0] for item in trajectory]

    def _update_state(self, topic: str, msg: Any, timestamp_ns: int) -> None:
        if topic == self.topics["image"]:
            self.image = self._image_msg_to_pil(msg)
            self.perception = self.perception_extractor.extract(np.asarray(self.image))
            self.has_perception = True
        elif topic == self.topics.get("perception_features"):
            self.perception = self._fit_vector(msg.data, self.perception_dim)
            self.has_perception = True
        elif topic == self.topics["lidar"]:
            self.lidar = self._fit_lidar(msg.ranges)
            self.has_lidar = True
        elif topic == self.topics["odom"]:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            yaw = self._odom_yaw(orientation)
            self.pose = (float(position.x), float(position.y), yaw)
            lap_state = self.lap_counter.update(
                x=float(position.x),
                y=float(position.y),
                yaw=yaw,
                timestamp_s=timestamp_ns / 1_000_000_000.0,
            )
            self.lap_index = lap_state.lap_count
        elif topic == self.topics["local_route"]:
            route = np.zeros((self.route_points, 2), dtype=np.float32)
            for idx, pose_stamped in enumerate(msg.poses[: self.route_points]):
                route[idx] = [pose_stamped.pose.position.x, pose_stamped.pose.position.y]
            self.route = route
        elif topic == self.topics.get("future_waypoints"):
            self.future_waypoints = self._fit_waypoints(msg.data)

    def _has_required_state(self) -> bool:
        if not self.has_perception or self.pose is None or not self.has_lidar:
            return False
        if self.require_waypoints and self.future_waypoints is None:
            return False
        return True

    def _write_sample(self, timestamp_ns: int) -> None:
        if self.pose is None:
            return

        future_waypoints = self.future_waypoints
        if self.generate_waypoints_from_odom:
            future_waypoints = self._future_waypoints_from_odom(timestamp_ns)
            if future_waypoints is None:
                return
        route = self.route
        if self.generate_route_from_odom:
            route = self._future_route_from_odom(timestamp_ns)
            if route is None:
                return

        stem = f"{self.sample_index:06d}"
        lidar_path = self.lidar_dir / f"{stem}.npy"
        image_path = self.image_dir / f"{stem}.jpg"
        image_relpath = ""
        if self.image is not None:
            self.image.save(image_path, quality=self.image_quality)
            image_relpath = image_path.relative_to(self.output_dir).as_posix()
        np.save(lidar_path, self.lidar)

        sample: dict[str, Any] = {
            "perception": self.perception.astype(float).tolist(),
            "lidar": lidar_path.relative_to(self.output_dir).as_posix(),
            "pose": [float(v) for v in self.pose],
            "lap_index": self.lap_index,
            "route": route.astype(float).tolist(),
            "stamp": timestamp_ns / 1_000_000_000.0,
        }
        if image_relpath:
            sample["image"] = image_relpath
        if future_waypoints is not None:
            sample["future_waypoints"] = future_waypoints.astype(float).tolist()

        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample, separators=(",", ":")) + "\n")

        self.sample_index += 1

    def _fit_lidar(self, ranges: Any) -> np.ndarray:
        values = np.asarray(ranges, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return self._fit_vector(values, self.lidar_size)

    @staticmethod
    def _fit_vector(values: Any, size: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        fitted = np.zeros(size, dtype=np.float32)
        fitted[: min(size, values.shape[0])] = values[:size]
        return fitted

    def _fit_waypoints(self, values: Any) -> np.ndarray:
        flat = np.asarray(values, dtype=np.float32)
        usable = flat.shape[0] - (flat.shape[0] % self.waypoint_dim)
        points = flat[:usable].reshape(-1, self.waypoint_dim)
        fitted = np.zeros((self.waypoint_count, self.waypoint_dim), dtype=np.float32)
        fitted[: min(self.waypoint_count, points.shape[0])] = points[: self.waypoint_count]
        return fitted

    def _future_waypoints_from_odom(self, timestamp_ns: int) -> np.ndarray | None:
        if self.pose is None or len(self.odom_trajectory) < 2:
            return None

        waypoints = np.zeros((self.waypoint_count, self.waypoint_dim), dtype=np.float32)
        points_ego, future_speeds = self._future_ego_points_from_odom(
            timestamp_ns=timestamp_ns,
            count=self.waypoint_count,
            step_ns=self.future_step_ns,
            include_speed=True,
        )
        if points_ego is None or future_speeds is None:
            return None
        waypoints[:, :2] = points_ego[:, :2]
        if self.waypoint_dim >= 3:
            waypoints[:, 2] = np.asarray(future_speeds, dtype=np.float32)
        return waypoints

    def _future_route_from_odom(self, timestamp_ns: int) -> np.ndarray | None:
        points_ego, _ = self._future_ego_points_from_odom(
            timestamp_ns=timestamp_ns,
            count=self.route_points,
            step_ns=self.route_step_ns,
            include_speed=False,
        )
        return points_ego

    def _future_ego_points_from_odom(
        self,
        timestamp_ns: int,
        count: int,
        step_ns: int,
        include_speed: bool,
    ) -> tuple[np.ndarray | None, list[float] | None]:
        if self.pose is None or len(self.odom_trajectory) < 2:
            return None, None

        current_x, current_y, current_yaw = self.pose
        future_xy: list[tuple[float, float]] = []
        future_speeds: list[float] = []

        for offset in range(1, count + 1):
            target_time_ns = timestamp_ns + offset * step_ns
            idx = bisect_left(self.odom_times, target_time_ns)
            if idx >= len(self.odom_trajectory):
                return None, None
            _, x, y, _ = self.odom_trajectory[idx]
            future_xy.append((x, y))
            if include_speed:
                elapsed_s = (offset * step_ns) / 1_000_000_000.0
                future_speeds.append(math.hypot(x - current_x, y - current_y) / max(elapsed_s, 1e-6))

        points_ego = world_to_ego(
            np.asarray(future_xy, dtype=np.float32),
            (current_x, current_y, current_yaw),
        ).astype(np.float32)
        return points_ego, future_speeds

    def _odom_yaw(self, orientation: Any) -> float:
        yaw = self._yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)
        return wrap_angle(yaw + self.odom_yaw_offset)

    def _speed_at_odom_index(self, idx: int) -> float:
        if idx <= 0 or idx >= len(self.odom_trajectory):
            return 0.0
        t0, x0, y0, _ = self.odom_trajectory[idx - 1]
        t1, x1, y1, _ = self.odom_trajectory[idx]
        dt = (t1 - t0) / 1_000_000_000.0
        if dt <= 1e-6:
            return 0.0
        distance = math.hypot(x1 - x0, y1 - y0)
        return float(distance / dt)

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
        help="Optional Float32MultiArray label topic with flattened future waypoint x,y,speed triples.",
    )
    parser.add_argument(
        "--require-waypoints",
        action="store_true",
        help="Only write samples after the waypoint label topic has been received.",
    )
    parser.add_argument(
        "--generate-waypoints-from-odom",
        action="store_true",
        help="Generate future waypoint labels from future odometry instead of reading a label topic.",
    )
    parser.add_argument(
        "--future-step-s",
        type=float,
        default=0.2,
        help="Time spacing, in seconds, between generated future odometry waypoints.",
    )
    parser.add_argument(
        "--generate-route-from-odom",
        action="store_true",
        help="Generate local route input from future odometry instead of reading /local_route.",
    )
    parser.add_argument(
        "--route-step-s",
        type=float,
        default=0.2,
        help="Time spacing, in seconds, between generated local route points.",
    )
    args = parser.parse_args()

    if args.sample_hz <= 0.0:
        raise SystemExit("--sample-hz must be greater than zero.")
    if args.future_step_s <= 0.0:
        raise SystemExit("--future-step-s must be greater than zero.")
    if args.route_step_s <= 0.0:
        raise SystemExit("--route-step-s must be greater than zero.")

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
        generate_waypoints_from_odom=args.generate_waypoints_from_odom,
        future_step_s=args.future_step_s,
        generate_route_from_odom=args.generate_route_from_odom,
        route_step_s=args.route_step_s,
    ).run()
    print(f"Wrote {count} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
