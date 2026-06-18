from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vla_driving.utils.geometry import world_to_ego


@dataclass(frozen=True)
class LocalRoute:
    points_ego: np.ndarray
    lap_progress: float
    nearest_index: int


class RouteProvider:
    """Slices global route waypoints into fixed-size ego-frame route points."""

    def __init__(
        self,
        main_route_xy: np.ndarray,
        shortcut_route_xy: np.ndarray | None = None,
        route_points: int = 10,
    ) -> None:
        self.main_route_xy = self._validate_route(main_route_xy)
        self.shortcut_route_xy = (
            self._validate_route(shortcut_route_xy) if shortcut_route_xy is not None else None
        )
        self.route_points = route_points

    def get_local_route(
        self,
        x: float,
        y: float,
        yaw: float,
        route_mode: str = "main",
    ) -> LocalRoute:
        route = self._select_route(route_mode)
        nearest_idx = int(np.argmin(np.linalg.norm(route - np.array([x, y], dtype=np.float32), axis=1)))
        indices = (nearest_idx + np.arange(self.route_points)) % len(route)
        points_ego = world_to_ego(route[indices], (x, y, yaw))
        lap_progress = nearest_idx / max(len(route) - 1, 1)
        return LocalRoute(
            points_ego=points_ego.astype(np.float32),
            lap_progress=float(lap_progress),
            nearest_index=nearest_idx,
        )

    def _select_route(self, route_mode: str) -> np.ndarray:
        if route_mode == "shortcut" and self.shortcut_route_xy is not None:
            return self.shortcut_route_xy
        return self.main_route_xy

    @staticmethod
    def _validate_route(route_xy: np.ndarray) -> np.ndarray:
        route = np.asarray(route_xy, dtype=np.float32)
        if route.ndim != 2 or route.shape[1] != 2 or route.shape[0] < 2:
            raise ValueError("Route must have shape (N, 2) with at least two points.")
        return route
