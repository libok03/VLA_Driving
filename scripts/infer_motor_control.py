from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vla_driving.data.lane_steering_dataset import summarize_lidar
from vla_driving.models.motor_control import MotorControlMLP
from vla_driving.perception import PerceptionExtractor
from vla_driving.utils.config import load_config


class MotorControlInferenceNode:
    def __init__(self, cfg: dict, checkpoint: str) -> None:
        import rclpy
        from rclpy.node import Node
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs.msg import Image, LaserScan
        from std_msgs.msg import Float32, Float32MultiArray

        class _Node(Node):
            pass

        self.rclpy = rclpy
        self.msg_types = {
            "Image": Image,
            "LaserScan": LaserScan,
            "Float32": Float32,
            "Float32MultiArray": Float32MultiArray,
        }
        self.node = _Node("vla_motor_control_inference")
        self.cfg = cfg
        self.device = self._resolve_device(cfg["device"])
        self.model = MotorControlMLP(**cfg["model"]).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.eval()

        data_cfg = cfg["data"]
        self.perception_dim = int(data_cfg["perception_dim"])
        self.lidar_size = int(data_cfg["lidar_size"])
        self.lidar_max_range = float(data_cfg.get("lidar_max_range", 10.0))
        perception_cfg = dict(cfg["ros2"].get("perception", {}))
        perception_cfg["dim"] = self.perception_dim
        self.perception_extractor = PerceptionExtractor(perception_cfg, dim=self.perception_dim)

        control_cfg = cfg.get("control", {})
        labels_cfg = cfg.get("labels", {})
        self.steering_output_gain = float(control_cfg.get("steering_output_gain", 1.0))
        self.speed_output_gain = float(control_cfg.get("speed_output_gain", 1.0))
        self.motor_max_angle = float(control_cfg.get("motor_max_angle", labels_cfg.get("steering_limit", 100.0)))
        self.motor_max_speed = float(control_cfg.get("motor_max_speed", labels_cfg.get("speed_limit", 20.0)))
        self.motor_min_speed = float(control_cfg.get("motor_min_speed", 0.0))
        self.motor_msg_type = control_cfg.get("motor_msg_type", "xycar_msgs/msg/XycarMotor")

        self.perception_tensor: torch.Tensor | None = None
        self.lidar_summary_tensor: torch.Tensor | None = None

        topics = cfg["ros2"]["topics"]
        qos = int(cfg["ros2"].get("qos_depth", 10))
        self.node.create_subscription(Image, topics["image"], self._on_image, qos)
        self.node.create_subscription(
            Float32MultiArray,
            topics.get("perception_features", "/vla_driving/perception_features"),
            self._on_perception_features,
            qos,
        )
        self.node.create_subscription(LaserScan, topics["lidar"], self._on_lidar, qos)
        self.steering_pub = self.node.create_publisher(Float32, topics["steering_cmd"], qos)
        self.speed_pub = self.node.create_publisher(Float32, topics["speed_cmd"], qos)
        motor_msg_cls = get_message(str(self.motor_msg_type))
        self.motor_msg_cls = motor_msg_cls
        self.motor_pub = self.node.create_publisher(
            motor_msg_cls,
            control_cfg.get("motor_topic", topics.get("motor_cmd", "/xycar_motor")),
            qos,
        )

        period = 1.0 / float(cfg["ros2"].get("inference_hz", 20.0))
        self.node.create_timer(period, self._tick)
        self.node.get_logger().info(
            f"motor control inference ready: checkpoint={checkpoint}, topic={topics.get('motor_cmd', '/xycar_motor')}"
        )

    def spin(self) -> None:
        self.rclpy.spin(self.node)
        self.node.destroy_node()

    def _on_image(self, msg) -> None:
        self.perception_tensor = self._image_msg_to_perception(msg)

    def _on_perception_features(self, msg) -> None:
        fitted = np.zeros(self.perception_dim, dtype=np.float32)
        values = np.asarray(msg.data, dtype=np.float32)
        fitted[: min(fitted.shape[0], values.shape[0])] = values[: fitted.shape[0]]
        self.perception_tensor = torch.from_numpy(fitted).unsqueeze(0).to(self.device)

    def _on_lidar(self, msg) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        fitted = np.zeros(self.lidar_size, dtype=np.float32)
        fitted[: min(fitted.shape[0], ranges.shape[0])] = ranges[: fitted.shape[0]]
        summary = summarize_lidar(fitted, max_range=self.lidar_max_range)
        self.lidar_summary_tensor = torch.from_numpy(summary).unsqueeze(0).to(self.device)

    def _tick(self) -> None:
        if self.perception_tensor is None or self.lidar_summary_tensor is None:
            return
        with torch.no_grad():
            output = self.model(self.perception_tensor, self.lidar_summary_tensor)[0].cpu().numpy()
        steering = float(output[0]) * self.steering_output_gain
        speed = float(output[1]) * self.speed_output_gain
        steering = float(np.clip(steering, -self.motor_max_angle, self.motor_max_angle))
        speed = float(np.clip(speed, self.motor_min_speed, self.motor_max_speed))
        self._publish(steering, speed)

    def _publish(self, steering: float, speed: float) -> None:
        Float32 = self.msg_types["Float32"]
        steering_msg = Float32()
        steering_msg.data = float(steering)
        self.steering_pub.publish(steering_msg)

        speed_msg = Float32()
        speed_msg.data = float(speed)
        self.speed_pub.publish(speed_msg)

        motor_msg = self.motor_msg_cls()
        self._assign_motor_field(motor_msg, ("angle", "steering", "steer"), steering)
        self._assign_motor_field(motor_msg, ("speed", "velocity", "throttle"), speed)
        self.motor_pub.publish(motor_msg)

    @staticmethod
    def _assign_motor_field(msg, names: tuple[str, ...], value: float) -> None:
        for name in names:
            if not hasattr(msg, name):
                continue
            current = getattr(msg, name)
            if isinstance(current, int):
                setattr(msg, name, int(round(value)))
            else:
                setattr(msg, name, type(current)(value))
            return

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
    def _resolve_device(name: str) -> torch.device:
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 direct motor angle/speed inference node.")
    parser.add_argument("--config", default="configs/motor_control.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/motor_control_2026_06_22/best.pt")
    args = parser.parse_args()

    try:
        import rclpy
    except ImportError as exc:
        raise SystemExit("ROS2 rclpy is required. Run this script inside a sourced ROS2 environment.") from exc

    cfg = load_config(Path(args.config))
    rclpy.init()
    node = MotorControlInferenceNode(cfg, args.checkpoint)
    try:
        node.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
