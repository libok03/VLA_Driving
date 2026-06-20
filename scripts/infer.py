from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from vla_driving.control.pure_pursuit import PurePursuitController
from vla_driving.models.lightweight_transfuser import LightweightTransFuser
from vla_driving.perception import PerceptionExtractor
from vla_driving.planning.lap_counter import LapCounter
from vla_driving.utils.config import load_config
from vla_driving.utils.geometry import wrap_angle


class Ros2InferenceNode:
    def __init__(self, cfg: dict, checkpoint: str) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, LaserScan
        from nav_msgs.msg import Odometry, Path
        from std_msgs.msg import Float32, Float32MultiArray

        class _Node(Node):
            pass

        self.rclpy = rclpy
        self.msg_types = {
            "Image": Image,
            "LaserScan": LaserScan,
            "Odometry": Odometry,
            "Path": Path,
            "Float32": Float32,
            "Float32MultiArray": Float32MultiArray,
        }
        self.node = _Node("vla_driving_inference")
        self.cfg = cfg
        self.device = self._resolve_device(cfg["device"])
        self.model = LightweightTransFuser(**cfg["model"]).to(self.device)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.eval()

        data_cfg = cfg["data"]
        self.lidar_size = data_cfg["lidar_size"]
        self.route_points = data_cfg["route_points"]
        perception_cfg = dict(cfg["ros2"].get("perception", {}))
        perception_cfg["dim"] = data_cfg["perception_dim"]
        self.perception_extractor = PerceptionExtractor(perception_cfg, dim=data_cfg["perception_dim"])
        self.controller = PurePursuitController(**cfg["control"])
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

        self.perception_tensor: torch.Tensor | None = None
        self.lidar_tensor: torch.Tensor | None = None
        self.pose: tuple[float, float, float] | None = None
        self.odom_yaw_offset = float(cfg.get("ros2", {}).get("odom_yaw_offset", 0.0))
        self.route = np.zeros((self.route_points, 2), dtype=np.float32)
        self.last_stamp_s = 0.0
        self.recent_waypoints: deque[np.ndarray] = deque(maxlen=3)

        topics = cfg["ros2"]["topics"]
        qos = int(cfg["ros2"].get("qos_depth", 10))
        self.node.create_subscription(Image, topics["image"], self._on_image, qos)
        self.node.create_subscription(LaserScan, topics["lidar"], self._on_lidar, qos)
        self.node.create_subscription(Odometry, topics["odom"], self._on_odom, qos)
        self.node.create_subscription(Path, topics["local_route"], self._on_route, qos)
        self.steering_pub = self.node.create_publisher(Float32, topics["steering_cmd"], qos)
        self.speed_pub = self.node.create_publisher(Float32, topics["speed_cmd"], qos)
        self.waypoints_pub = self.node.create_publisher(Float32MultiArray, topics["waypoints"], qos)
        period = 1.0 / float(cfg["ros2"].get("inference_hz", 10.0))
        self.node.create_timer(period, self._tick)

    def spin(self) -> None:
        self.rclpy.spin(self.node)
        self.node.destroy_node()

    def _on_image(self, msg) -> None:
        self.perception_tensor = self._image_msg_to_perception(msg)
        self.last_stamp_s = self._stamp_to_seconds(msg.header.stamp)

    def _on_lidar(self, msg) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=0.0, neginf=0.0)
        fitted = np.zeros(self.lidar_size, dtype=np.float32)
        fitted[: min(self.lidar_size, ranges.shape[0])] = ranges[: self.lidar_size]
        self.lidar_tensor = torch.from_numpy(fitted).unsqueeze(0).to(self.device)
        self.last_stamp_s = self._stamp_to_seconds(msg.header.stamp)

    def _on_odom(self, msg) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = wrap_angle(
            self._yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)
            + self.odom_yaw_offset
        )
        self.pose = (float(position.x), float(position.y), yaw)
        self.last_stamp_s = self._stamp_to_seconds(msg.header.stamp)

    def _on_route(self, msg) -> None:
        route = np.zeros((self.route_points, 2), dtype=np.float32)
        for idx, pose_stamped in enumerate(msg.poses[: self.route_points]):
            route[idx] = [pose_stamped.pose.position.x, pose_stamped.pose.position.y]
        self.route = route

    def _tick(self) -> None:
        if self.perception_tensor is None or self.lidar_tensor is None or self.pose is None:
            return

        x, y, yaw = self.pose
        lap_state = self.lap_counter.update(
            x=x,
            y=y,
            yaw=yaw,
            timestamp_s=self.last_stamp_s,
        )
        state = torch.tensor(
            [[x, y, yaw, float(lap_state.lap_count)]],
            dtype=torch.float32,
            device=self.device,
        )
        route = torch.from_numpy(self.route).unsqueeze(0).to(self.device)

        with torch.no_grad():
            waypoints = self.model(self.perception_tensor, self.lidar_tensor, state, route)[0].cpu().numpy()
        if self.recent_waypoints:
            waypoints = 0.7 * waypoints + 0.3 * np.mean(np.stack(self.recent_waypoints), axis=0)
        self.recent_waypoints.append(waypoints)

        steering = 0.0 if lap_state.finished else self.controller.steer_from_waypoints(waypoints)
        speed = 0.0 if lap_state.finished else self._speed_from_waypoints(waypoints)
        self._publish(steering, speed, waypoints)

    def _publish(self, steering: float, speed: float, waypoints: np.ndarray) -> None:
        Float32 = self.msg_types["Float32"]
        Float32MultiArray = self.msg_types["Float32MultiArray"]
        steering_msg = Float32()
        steering_msg.data = float(steering)
        self.steering_pub.publish(steering_msg)

        speed_msg = Float32()
        speed_msg.data = float(speed)
        self.speed_pub.publish(speed_msg)

        waypoints_msg = Float32MultiArray()
        waypoints_msg.data = waypoints.astype(np.float32).reshape(-1).tolist()
        self.waypoints_pub.publish(waypoints_msg)

    def _image_msg_to_perception(self, msg) -> torch.Tensor:
        channels = self._channels_for_encoding(msg.encoding)
        array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        image = array[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if msg.encoding.lower() == "bgr8":
            image = image[:, :, ::-1].copy()
        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        features = self.perception_extractor.extract(image.astype(np.uint8))
        return torch.from_numpy(features).unsqueeze(0).to(self.device)

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
    def _stamp_to_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _speed_from_waypoints(waypoints: np.ndarray) -> float:
        if waypoints.shape[1] < 3:
            return 0.0
        return float(max(waypoints[0, 2], 0.0))

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 topic-based VLA Driving inference node.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    try:
        import rclpy
    except ImportError as exc:
        raise SystemExit("ROS2 rclpy is required. Run this script inside a sourced ROS2 environment.") from exc

    cfg = load_config(Path(args.config))
    rclpy.init()
    node = Ros2InferenceNode(cfg, args.checkpoint)
    try:
        node.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
