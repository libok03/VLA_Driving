from __future__ import annotations

import numpy as np
import torch

from vla_driving.data.lane_steering_dataset import summarize_lidar
from vla_driving.models.lane_steering import LaneSteeringMLP


def test_lane_steering_model_shape() -> None:
    model = LaneSteeringMLP(perception_dim=32, lidar_summary_dim=5)
    perception = torch.randn(4, 32)
    lidar_summary = torch.randn(4, 5)
    output = model(perception, lidar_summary)
    assert output.shape == (4,)


def test_lidar_summary_shape_and_range() -> None:
    ranges = np.ones(360, dtype=np.float32) * 5.0
    ranges[175:185] = 1.0
    summary = summarize_lidar(ranges, max_range=10.0)
    assert summary.shape == (5,)
    assert np.all(summary >= 0.0)
    assert np.all(summary <= 1.0)
    assert summary[0] < 0.2
