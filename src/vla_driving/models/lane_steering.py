from __future__ import annotations

import torch
from torch import nn


class LaneSteeringMLP(nn.Module):
    """Small direct steering regressor for lane-following experiments."""

    def __init__(
        self,
        perception_dim: int = 32,
        lidar_summary_dim: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.05,
        output_scale: float = 50.0,
    ) -> None:
        super().__init__()
        self.output_scale = float(output_scale)
        input_dim = perception_dim + lidar_summary_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, perception: torch.Tensor, lidar_summary: torch.Tensor) -> torch.Tensor:
        features = torch.cat([perception, lidar_summary], dim=1)
        return self.net(features).squeeze(1) * self.output_scale
