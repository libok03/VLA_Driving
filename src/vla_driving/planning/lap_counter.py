from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vla_driving.utils.geometry import wrap_angle


@dataclass(frozen=True)
class LapState:
    lap_count: int
    finished: bool
    armed: bool


class LapCounter:
    """Counts completed laps using either a directed gate or radius trigger."""

    def __init__(
        self,
        gate_a: tuple[float, float],
        gate_b: tuple[float, float],
        forward_yaw: float,
        total_laps: int = 3,
        cooldown_s: float = 3.0,
        arm_distance_m: float = 1.0,
        yaw_tolerance_rad: float = np.deg2rad(60.0),
        trigger_mode: str = "gate",
        trigger_center: tuple[float, float] | None = None,
        trigger_radius_m: float = 3.0,
    ) -> None:
        self.gate_a = np.asarray(gate_a, dtype=np.float32)
        self.gate_b = np.asarray(gate_b, dtype=np.float32)
        self.forward_yaw = forward_yaw
        self.total_laps = total_laps
        self.cooldown_s = cooldown_s
        self.arm_distance_m = arm_distance_m
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.trigger_mode = trigger_mode
        self.trigger_center = (
            np.asarray(trigger_center, dtype=np.float32) if trigger_center is not None else None
        )
        self.trigger_radius_m = trigger_radius_m
        self.lap_count = 0
        self.finished = False
        self.armed = False
        self.prev_side: float | None = None
        self.last_cross_time = -1e9

    def update(self, x: float, y: float, yaw: float, timestamp_s: float) -> LapState:
        point = np.asarray([x, y], dtype=np.float32)
        if self.trigger_mode == "radius":
            return self._update_radius(point, timestamp_s)
        return self._update_gate(point, yaw, timestamp_s)

    def _update_radius(self, point: np.ndarray, timestamp_s: float) -> LapState:
        if self.trigger_center is None:
            self.trigger_center = point.copy()
            return self.state

        distance = float(np.linalg.norm(point - self.trigger_center))
        was_armed = self.armed
        if distance > self.trigger_radius_m:
            self.armed = True

        entered_trigger = distance <= self.trigger_radius_m
        cooldown_ok = (timestamp_s - self.last_cross_time) >= self.cooldown_s
        if not self.finished and was_armed and entered_trigger and cooldown_ok:
            self.lap_count += 1
            self.last_cross_time = timestamp_s
            self.armed = False
            self.finished = self.lap_count >= self.total_laps
        return self.state

    def _update_gate(self, point: np.ndarray, yaw: float, timestamp_s: float) -> LapState:
        side = self._side_of_gate(point)
        was_armed = self.armed

        if self._distance_to_gate(point) > self.arm_distance_m:
            self.armed = True

        if self.prev_side is None:
            self.prev_side = side
            return self.state

        crossed = self.prev_side != 0.0 and side != 0.0 and np.sign(side) != np.sign(self.prev_side)
        direction_ok = abs(wrap_angle(yaw - self.forward_yaw)) <= self.yaw_tolerance_rad
        cooldown_ok = (timestamp_s - self.last_cross_time) >= self.cooldown_s

        if not self.finished and was_armed and crossed and direction_ok and cooldown_ok:
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
