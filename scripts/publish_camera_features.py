from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vla_driving.perception import PerceptionExtractor
from vla_driving.utils.config import load_config


class CameraFeatureNode:
    def __init__(self, cfg: dict, image_topic: str, feature_topic: str) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float32MultiArray

        class _Node(Node):
            pass

        self.rclpy = rclpy
        self.msg_types = {
            "Image": Image,
            "Float32MultiArray": Float32MultiArray,
        }
        self.node = _Node("vla_camera_feature_publisher")

        data_cfg = cfg["data"]
        perception_cfg = dict(cfg.get("ros2", {}).get("perception", {}))
        perception_cfg["dim"] = int(data_cfg["perception_dim"])
        self.extractor = PerceptionExtractor(perception_cfg, dim=int(data_cfg["perception_dim"]))

        qos = int(cfg["ros2"].get("qos_depth", 10))
        self.pub = self.node.create_publisher(Float32MultiArray, feature_topic, qos)
        self.node.create_subscription(Image, image_topic, self._on_image, qos)
        self.node.get_logger().info(f"camera features: {image_topic} -> {feature_topic}")

    def spin(self) -> None:
        self.rclpy.spin(self.node)
        self.node.destroy_node()

    def _on_image(self, msg) -> None:
        image = self._image_msg_to_rgb(msg)
        features = self.extractor.extract(image)
        out = self.msg_types["Float32MultiArray"]()
        out.data = features.astype(np.float32).tolist()
        self.pub.publish(out)

    @staticmethod
    def _image_msg_to_rgb(msg) -> np.ndarray:
        channels = CameraFeatureNode._channels_for_encoding(msg.encoding)
        array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        image = array[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if msg.encoding.lower() == "bgr8":
            image = image[:, :, ::-1].copy()
        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        return image.astype(np.uint8)

    @staticmethod
    def _channels_for_encoding(encoding: str) -> int:
        encoding = encoding.lower()
        if encoding in {"rgb8", "bgr8"}:
            return 3
        if encoding in {"mono8", "8uc1"}:
            return 1
        raise ValueError(f"Unsupported image encoding: {encoding}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish compact VLA perception features from a camera topic.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--image-topic", default="")
    parser.add_argument("--feature-topic", default="")
    args = parser.parse_args()

    try:
        import rclpy
    except ImportError as exc:
        raise SystemExit("ROS2 rclpy is required. Run this script inside a sourced ROS2 environment.") from exc

    cfg = load_config(Path(args.config))
    topics = cfg["ros2"]["topics"]
    image_topic = args.image_topic or topics["image"]
    feature_topic = args.feature_topic or topics["perception_features"]

    rclpy.init()
    node = CameraFeatureNode(cfg, image_topic=image_topic, feature_topic=feature_topic)
    try:
        node.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
