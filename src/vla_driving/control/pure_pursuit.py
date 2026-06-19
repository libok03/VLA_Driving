from __future__ import annotations

import math

import numpy as np


class PurePursuitController:
    def __init__(
        self,
        wheel_base: float = 0.32,
        lookahead_distance: float = 1.5,
        max_steer_rad: float = 0.6,
    ) -> None:
        self.wheel_base = wheel_base
        self.lookahead_distance = lookahead_distance
        self.max_steer_rad = max_steer_rad

    def steer_from_waypoints(self, waypoints: np.ndarray) -> float:
        if waypoints.size == 0:
            return 0.0

        points_xy = waypoints[:, :2]
        distances = np.linalg.norm(points_xy, axis=1)
        idx = int(np.argmin(np.abs(distances - self.lookahead_distance)))
        target_x, target_y = points_xy[idx]
        lookahead = max(float(np.linalg.norm([target_x, target_y])), 1e-3)
        curvature = 2.0 * target_y / (lookahead * lookahead)
        steering = math.atan(self.wheel_base * curvature)
        return float(np.clip(steering, -self.max_steer_rad, self.max_steer_rad))
