from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
from pathlib import Path
from typing import Any

import numpy as np


class CdrReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        if len(data) < 4:
            raise ValueError("CDR payload too small")
        self.endian = "<" if data[1] == 1 else ">"
        self.offset = 4

    def align(self, size: int) -> None:
        padding = (-self.offset) % size
        self.offset += padding

    def u32(self) -> int:
        self.align(4)
        value = struct.unpack_from(self.endian + "I", self.data, self.offset)[0]
        self.offset += 4
        return int(value)

    def i32(self) -> int:
        self.align(4)
        value = struct.unpack_from(self.endian + "i", self.data, self.offset)[0]
        self.offset += 4
        return int(value)

    def f32(self) -> float:
        self.align(4)
        value = struct.unpack_from(self.endian + "f", self.data, self.offset)[0]
        self.offset += 4
        return float(value)

    def f64(self) -> float:
        self.align(8)
        value = struct.unpack_from(self.endian + "d", self.data, self.offset)[0]
        self.offset += 8
        return float(value)

    def string(self) -> str:
        size = self.u32()
        raw = self.data[self.offset : self.offset + size]
        self.offset += size
        return raw.rstrip(b"\x00").decode("utf-8", errors="replace")

    def f32_array(self) -> np.ndarray:
        size = self.u32()
        self.align(4)
        byte_count = size * 4
        values = np.frombuffer(self.data, dtype=self.endian + "f4", count=size, offset=self.offset).astype(np.float32)
        self.offset += byte_count
        return values


def read_header(reader: CdrReader) -> None:
    reader.i32()
    reader.u32()
    reader.string()


def parse_float32_multi_array(data: bytes) -> np.ndarray:
    reader = CdrReader(data)
    dim_count = reader.u32()
    for _ in range(dim_count):
        reader.string()
        reader.u32()
        reader.u32()
    reader.u32()
    return reader.f32_array()


def parse_laser_scan(data: bytes) -> np.ndarray:
    reader = CdrReader(data)
    read_header(reader)
    for _ in range(7):
        reader.f32()
    ranges = reader.f32_array()
    return ranges


def parse_xycar_motor(data: bytes) -> tuple[float, float]:
    reader = CdrReader(data)
    read_header(reader)
    angle = reader.f32()
    speed = reader.f32()
    return angle, speed


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def parse_pose_stamped(data: bytes) -> np.ndarray:
    reader = CdrReader(data)
    read_header(reader)
    x = reader.f64()
    y = reader.f64()
    reader.f64()
    qx = reader.f64()
    qy = reader.f64()
    qz = reader.f64()
    qw = reader.f64()
    return np.asarray([x, y, yaw_from_quaternion(qx, qy, qz, qw)], dtype=np.float32)


def parse_odometry(data: bytes) -> np.ndarray:
    reader = CdrReader(data)
    read_header(reader)
    reader.string()
    x = reader.f64()
    y = reader.f64()
    reader.f64()
    qx = reader.f64()
    qy = reader.f64()
    qz = reader.f64()
    qw = reader.f64()
    return np.asarray([x, y, yaw_from_quaternion(qx, qy, qz, qw)], dtype=np.float32)


def parse_pose(data: bytes, topic_type: str) -> np.ndarray:
    if topic_type == "nav_msgs/msg/Odometry":
        return parse_odometry(data)
    if topic_type in {"geometry_msgs/msg/PoseStamped", "geometry_msgs/msg/PoseWithCovarianceStamped"}:
        return parse_pose_stamped(data)
    raise ValueError(f"Unsupported pose topic type: {topic_type}")


def fit_vector(values: Any, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    fitted = np.zeros(size, dtype=np.float32)
    fitted[: min(size, values.shape[0])] = values[:size]
    return fitted


def db3_path_for_bag(bag_dir: Path) -> Path:
    files = sorted(bag_dir.glob("*.db3"))
    if not files:
        raise SystemExit(f"No .db3 file found in {bag_dir}")
    return files[0]


def topic_map(con: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    return {
        int(row[0]): (str(row[1]), str(row[2]))
        for row in con.execute("select id, name, type from topics")
    }


def extract_bag(
    bag_dir: Path,
    output_dir: Path,
    sample_hz: float,
    perception_topic: str,
    lidar_topic: str,
    motor_topic: str,
    pose_topic: str,
    perception_dim: int,
    lidar_size: int,
) -> int:
    db_path = db3_path_for_bag(bag_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir = output_dir / "lidar"
    lidar_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    period_ns = int(1_000_000_000 / sample_hz)
    perception: np.ndarray | None = None
    lidar: np.ndarray | None = None
    steering: float | None = None
    speed: float | None = None
    pose: np.ndarray | None = None
    next_sample_ns: int | None = None
    sample_index = 0

    con = sqlite3.connect(db_path)
    topics = topic_map(con)
    wanted = {perception_topic, lidar_topic, motor_topic}
    if pose_topic:
        wanted.add(pose_topic)
    query = """
        select topic_id, timestamp, data
        from messages
        order by timestamp asc
    """
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for topic_id, timestamp_ns, raw in con.execute(query):
            topic_name, topic_type = topics[int(topic_id)]
            if topic_name not in wanted:
                continue
            try:
                if topic_name == perception_topic:
                    perception = fit_vector(parse_float32_multi_array(raw), perception_dim)
                elif topic_name == lidar_topic:
                    lidar = fit_vector(parse_laser_scan(raw), lidar_size)
                elif topic_name == motor_topic:
                    steering, speed = parse_xycar_motor(raw)
                elif topic_name == pose_topic:
                    pose = parse_pose(raw, topic_type)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse {topic_name} at {timestamp_ns} in {bag_dir}") from exc

            if next_sample_ns is None:
                next_sample_ns = int(timestamp_ns)
            if int(timestamp_ns) < next_sample_ns:
                continue
            if perception is None or lidar is None or steering is None or speed is None:
                next_sample_ns = int(timestamp_ns) + period_ns
                continue
            if pose_topic and pose is None:
                next_sample_ns = int(timestamp_ns) + period_ns
                continue

            stem = f"{sample_index:06d}"
            lidar_path = lidar_dir / f"{stem}.npy"
            np.save(lidar_path, lidar)
            sample = {
                "perception": perception.astype(float).tolist(),
                "lidar": lidar_path.relative_to(output_dir).as_posix(),
                "steering": float(steering),
                "speed": float(speed),
                "stamp": int(timestamp_ns) / 1_000_000_000.0,
            }
            if pose_topic and pose is not None:
                sample["pose"] = pose.astype(float).tolist()
            manifest.write(json.dumps(sample, separators=(",", ":")) + "\n")
            sample_index += 1
            next_sample_ns = int(timestamp_ns) + period_ns
    con.close()
    return sample_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract motor-control samples from sqlite3 ROS2 bags without ROS2.")
    parser.add_argument("bag_path", help="Path to a ROS2 sqlite3 bag directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--perception-topic", default="/vla_driving/perception_features")
    parser.add_argument("--lidar-topic", default="/scan")
    parser.add_argument("--motor-topic", default="/xycar_motor")
    parser.add_argument("--pose-topic", default="", help="Optional nav_msgs/Odometry or PoseStamped topic.")
    parser.add_argument("--perception-dim", type=int, default=32)
    parser.add_argument("--lidar-size", type=int, default=360)
    args = parser.parse_args()

    if args.sample_hz <= 0.0:
        raise SystemExit("--sample-hz must be greater than zero.")
    count = extract_bag(
        bag_dir=Path(args.bag_path),
        output_dir=Path(args.output_dir),
        sample_hz=args.sample_hz,
        perception_topic=args.perception_topic,
        lidar_topic=args.lidar_topic,
        motor_topic=args.motor_topic,
        pose_topic=args.pose_topic,
        perception_dim=args.perception_dim,
        lidar_size=args.lidar_size,
    )
    print(f"Wrote {count} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
