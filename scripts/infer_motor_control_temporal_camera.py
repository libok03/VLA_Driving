from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vla_driving.models.motor_temporal_camera import MotorTemporalCameraGRU
from vla_driving.utils.config import load_config


class MotorTemporalCameraInferenceNode:
    IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, cfg: dict, checkpoint: str) -> None:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs.msg import Image, LaserScan
        from std_msgs.msg import Float32

        class _Node(Node):
            pass

        self.rclpy = rclpy
        self.msg_types = {"Float32": Float32, "Image": Image, "LaserScan": LaserScan, "Odometry": Odometry}
        self.node = _Node("vla_motor_temporal_camera_inference")
        self.cfg = cfg
        self.device = self._resolve_device(cfg["device"])
        self.model = MotorTemporalCameraGRU(**cfg["model"]).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.eval()

        data_cfg = cfg["data"]
        self.image_size = tuple(int(v) for v in data_cfg["image_size"])
        self.lidar_size = int(data_cfg["lidar_size"])
        self.pose_dim = int(data_cfg.get("pose_dim", 4))
        self.sequence_length = int(data_cfg["sequence_length"])
        self.lidar_max_range = float(data_cfg.get("lidar_max_range", 10.0))

        control_cfg = cfg.get("control", {})
        labels_cfg = cfg.get("labels", {})
        self.steering_output_gain = float(control_cfg.get("steering_output_gain", 1.0))
        self.speed_output_gain = float(control_cfg.get("speed_output_gain", 1.0))
        self.motor_max_angle = float(control_cfg.get("motor_max_angle", labels_cfg.get("steering_limit", 100.0)))
        self.motor_max_speed = float(control_cfg.get("motor_max_speed", labels_cfg.get("speed_limit", 20.0)))
        self.motor_min_speed = float(control_cfg.get("motor_min_speed", 0.0))
        self.motor_msg_type = control_cfg.get("motor_msg_type", "xycar_msgs/msg/XycarMotor")

        self.image: np.ndarray | None = None
        self.lidar: np.ndarray | None = None
        self.pose: np.ndarray | None = None
        self.pose_origin: np.ndarray | None = None
        self.sequence: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=self.sequence_length)

        topics = cfg["ros2"]["topics"]
        qos = int(cfg["ros2"].get("qos_depth", 10))
        self.node.create_subscription(Image, topics.get("image", "/usb_cam/image_raw/front"), self._on_image, qos)
        self.node.create_subscription(LaserScan, topics.get("lidar", "/scan"), self._on_lidar, qos)
        self.node.create_subscription(Odometry, topics.get("pose", "/scan_odom_map"), self._on_odometry, qos)

        self.steering_pub = self.node.create_publisher(Float32, topics.get("steering_cmd", "/vla_driving/steering"), qos)
        self.speed_pub = self.node.create_publisher(Float32, topics.get("speed_cmd", "/vla_driving/speed"), qos)
        motor_msg_cls = get_message(str(self.motor_msg_type))
        self.motor_msg_cls = motor_msg_cls
        self.motor_pub = self.node.create_publisher(
            motor_msg_cls,
            control_cfg.get("motor_topic", topics.get("motor_cmd", "/xycar_motor")),
            qos,
        )

        period = 1.0 / float(cfg["ros2"].get("inference_hz", 20.0))
        self.node.create_timer(period, self._tick)
        self.node.get_logger().info(f"camera temporal motor inference ready: checkpoint={checkpoint}")

    def spin(self) -> None:
        self.rclpy.spin(self.node)
        self.node.destroy_node()

    def _on_image(self, msg) -> None:
        self.image = self._image_msg_to_tensor_array(msg)

    def _on_lidar(self, msg) -> None:
        ranges = self._fit_vector(msg.ranges, self.lidar_size)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=0.0, neginf=0.0)
        self.lidar = np.clip(ranges, 0.0, self.lidar_max_range) / max(self.lidar_max_range, 1e-6)

    def _on_odometry(self, msg) -> None:
        pose = msg.pose.pose
        yaw = self._yaw_from_quaternion(
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        self.pose = self._pose_features(float(pose.position.x), float(pose.position.y), yaw)

    def _tick(self) -> None:
        if self.image is None or self.lidar is None:
            return
        pose = self.pose if self.pose is not None else self._empty_pose()
        self.sequence.append((self.image, self.lidar, pose))
        if len(self.sequence) < self.sequence_length:
            return

        images, lidars, poses = zip(*self.sequence)
        image_tensor = torch.from_numpy(np.stack(images)).unsqueeze(0).to(self.device)
        lidar_tensor = torch.from_numpy(np.stack(lidars)).unsqueeze(0).to(self.device)
        pose_tensor = torch.from_numpy(np.stack(poses)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor, lidar_tensor, pose_tensor)[0].cpu().numpy()
        steering = float(np.clip(output[0] * self.steering_output_gain, -self.motor_max_angle, self.motor_max_angle))
        speed = float(np.clip(output[1] * self.speed_output_gain, self.motor_min_speed, self.motor_max_speed))
        self._publish(steering, speed)

    def _image_msg_to_tensor_array(self, msg) -> np.ndarray:
        channels = self._channels_for_encoding(msg.encoding)
        array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        image = array[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if msg.encoding.lower() == "bgr8":
            image = image[:, :, ::-1].copy()
        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        try:
            import cv2

            image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        except ImportError:
            from PIL import Image as PILImage

            image = np.asarray(PILImage.fromarray(image).resize((self.image_size[1], self.image_size[0])))
        image = image.astype(np.float32) / 255.0
        image = (image - self.IMAGE_MEAN) / self.IMAGE_STD
        return image.transpose(2, 0, 1).astype(np.float32)

    def _pose_features(self, x: float, y: float, yaw: float) -> np.ndarray:
        if self.pose_dim <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.pose_origin is None:
            self.pose_origin = np.asarray([x, y], dtype=np.float32)
        if self.pose_dim == 4:
            rel = np.asarray([x, y], dtype=np.float32) - self.pose_origin
            return np.asarray([rel[0], rel[1], np.sin(yaw), np.cos(yaw)], dtype=np.float32)
        return self._fit_vector([x, y, yaw], self.pose_dim)

    def _empty_pose(self) -> np.ndarray:
        if self.pose_dim == 4:
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return np.zeros(self.pose_dim, dtype=np.float32)

    def _publish(self, steering: float, speed: float) -> None:
        Float32 = self.msg_types["Float32"]
        steering_msg = Float32()
        steering_msg.data = steering
        self.steering_pub.publish(steering_msg)
        speed_msg = Float32()
        speed_msg.data = speed
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

    @staticmethod
    def _fit_vector(values, size: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        fitted = np.zeros(size, dtype=np.float32)
        fitted[: min(size, values.shape[0])] = values[:size]
        return fitted

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
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 temporal camera+LiDAR+pose motor inference node.")
    parser.add_argument("--config", default="configs/motor_control_temporal_camera.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/motor_control_temporal_camera_2026_06_25/best.pt")
    args = parser.parse_args()

    try:
        import rclpy
    except ImportError as exc:
        raise SystemExit("ROS2 rclpy is required. Run this script inside a sourced ROS2 environment.") from exc

    cfg = load_config(Path(args.config))
    rclpy.init()
    node = MotorTemporalCameraInferenceNode(cfg, args.checkpoint)
    try:
        node.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
