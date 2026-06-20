from __future__ import annotations

import torch

from vla_driving.models.lightweight_transfuser import LightweightTransFuser


def test_lightweight_transfuser_forward_shape() -> None:
    model = LightweightTransFuser(
        route_points=10,
        waypoint_count=5,
        waypoint_dim=3,
        pose_dim=4,
        perception_dim=32,
        fusion_type="transformer",
    )
    perception = torch.randn(2, 32)
    lidar = torch.randn(2, 360)
    pose = torch.randn(2, 4)
    route = torch.randn(2, 10, 2)
    output = model(perception, lidar, pose, route)
    assert output.shape == (2, 5, 3)


def test_mlp_fusion_forward_shape() -> None:
    model = LightweightTransFuser(
        route_points=10,
        waypoint_count=5,
        waypoint_dim=3,
        pose_dim=4,
        perception_dim=32,
        fusion_type="mlp",
    )
    perception = torch.randn(2, 32)
    lidar = torch.randn(2, 360)
    pose = torch.randn(2, 4)
    route = torch.randn(2, 10, 2)
    output = model(perception, lidar, pose, route)
    assert output.shape == (2, 5, 3)
