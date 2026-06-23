from __future__ import annotations

import torch
from torch import nn


class MotorControlMLP(nn.Module):
    """Small direct regressor for xycar motor angle and speed."""

    def __init__(
        self,
        perception_dim: int = 32,
        lidar_summary_dim: int = 5,
        hidden_dim: int = 96,
        dropout: float = 0.05,
        steering_scale: float = 100.0,
        speed_scale: float = 20.0,
    ) -> None:
        super().__init__()
        self.register_buffer("output_scale", torch.tensor([steering_scale, speed_scale], dtype=torch.float32))
        input_dim = perception_dim + lidar_summary_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )

    def forward(self, perception: torch.Tensor, lidar_summary: torch.Tensor) -> torch.Tensor:
        features = torch.cat([perception, lidar_summary], dim=1)
        output = self.net(features) * self.output_scale
        speed = torch.clamp(output[:, 1], min=0.0)
        return torch.stack([output[:, 0], speed], dim=1)
