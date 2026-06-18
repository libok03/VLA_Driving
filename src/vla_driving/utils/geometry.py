from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_to_ego(points_xy: np.ndarray, pose_xy_yaw: tuple[float, float, float]) -> np.ndarray:
    x, y, yaw = pose_xy_yaw
    shifted = points_xy - np.array([x, y], dtype=np.float32)
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return shifted @ rot.T
