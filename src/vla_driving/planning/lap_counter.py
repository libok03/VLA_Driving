from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vla_driving.utils.geometry import wrap_angle


@dataclass(frozen=True)
class LapState:
    lap_count: int
    laps_remaining: int
    finished: bool
    armed: bool


class LapCounter:
    """Counts completed laps using a directed start/finish gate."""

    def __init__(
        self,
        gate_a: tuple[float, float],
        gate_b: tuple[float, float],
        forward_yaw: float,
        total_laps: int = 3,
        min_progress: float = 0.7,
        cooldown_s: float = 3.0,
        arm_distance_m: float = 1.0,
        yaw_tolerance_rad: float = np.deg2rad(60.0),
    ) -> None:
        self.gate_a = np.asarray(gate_a, dtype=np.float32)
        self.gate_b = np.asarray(gate_b, dtype=np.float32)
        self.forward_yaw = forward_yaw
        self.total_laps = total_laps
        self.min_progress = min_progress
        self.cooldown_s = cooldown_s
        self.arm_distance_m = arm_distance_m
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.lap_count = 0
        self.finished = False
        self.armed = False
        self.prev_side: float | None = None
        self.last_cross_time = -1e9

    def update(self, x: float, y: float, yaw: float, timestamp_s: float, lap_progress: float) -> LapState:
        point = np.asarray([x, y], dtype=np.float32)
        side = self._side_of_gate(point)

        if self._distance_to_gate(point) > self.arm_distance_m:
            self.armed = True

        if self.prev_side is None:
            self.prev_side = side
            return self.state

        crossed = self.prev_side != 0.0 and side != 0.0 and np.sign(side) != np.sign(self.prev_side)
        direction_ok = abs(wrap_angle(yaw - self.forward_yaw)) <= self.yaw_tolerance_rad
        cooldown_ok = (timestamp_s - self.last_cross_time) >= self.cooldown_s
        progress_ok = lap_progress >= self.min_progress

        if (
            not self.finished
            and self.armed
            and crossed
            and direction_ok
            and cooldown_ok
            and progress_ok
        ):
            self.lap_count += 1
            self.last_cross_time = timestamp_s
            self.armed = False
            self.finished = self.lap_count >= self.total_laps

        self.prev_side = side
        return self.state

    @property
    def state(self) -> LapState:
        return LapState(
            lap_count=self.lap_count,
            laps_remaining=max(self.total_laps - self.lap_count, 0),
            finished=self.finished,
            armed=self.armed,
        )

    def reset(self) -> None:
        self.lap_count = 0
        self.finished = False
        self.armed = False
        self.prev_side = None
        self.last_cross_time = -1e9

    def _side_of_gate(self, point: np.ndarray) -> float:
        gate = self.gate_b - self.gate_a
        rel = point - self.gate_a
        return float(gate[0] * rel[1] - gate[1] * rel[0])

    def _distance_to_gate(self, point: np.ndarray) -> float:
        gate = self.gate_b - self.gate_a
        denom = float(np.dot(gate, gate))
        if denom <= 1e-9:
            return float(np.linalg.norm(point - self.gate_a))
        t = float(np.clip(np.dot(point - self.gate_a, gate) / denom, 0.0, 1.0))
        nearest = self.gate_a + t * gate
        return float(np.linalg.norm(point - nearest))
