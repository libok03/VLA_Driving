from __future__ import annotations

import numpy as np

from vla_driving.planning.route_provider import RouteProvider


def test_route_provider_returns_fixed_ego_points() -> None:
    route = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    provider = RouteProvider(route, route_points=3)

    local = provider.get_local_route(x=1.1, y=0.0, yaw=0.0)

    assert local.points_ego.shape == (3, 2)
    assert local.nearest_index == 1
    assert local.route_progress > 0.0


def test_route_provider_uses_shortcut_when_requested() -> None:
    main = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    shortcut = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    provider = RouteProvider(main, shortcut, route_points=2)

    local = provider.get_local_route(x=0.0, y=1.0, yaw=0.0, route_mode="shortcut")

    assert np.allclose(local.points_ego[0], [0.0, 0.0])
