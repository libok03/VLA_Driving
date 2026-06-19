from __future__ import annotations

import torch

from vla_driving.models.lightweight_transfuser import LightweightTransFuser


def test_lightweight_transfuser_forward_shape() -> None:
    model = LightweightTransFuser(
        route_points=10,
        waypoint_count=5,
        waypoint_dim=3,
        pose_dim=5,
        fusion_type="transformer",
    )
    image = torch.randn(2, 3, 160, 320)
    lidar = torch.randn(2, 360)
    pose = torch.randn(2, 5)
    route = torch.randn(2, 10, 2)
    output = model(image, lidar, pose, route)
    assert output.shape == (2, 5, 3)


def test_mlp_fusion_forward_shape() -> None:
    model = LightweightTransFuser(
        route_points=10,
        waypoint_count=5,
        waypoint_dim=3,
        pose_dim=5,
        fusion_type="mlp",
    )
    image = torch.randn(2, 3, 160, 320)
    lidar = torch.randn(2, 360)
    pose = torch.randn(2, 5)
    route = torch.randn(2, 10, 2)
    output = model(image, lidar, pose, route)
    assert output.shape == (2, 5, 3)
