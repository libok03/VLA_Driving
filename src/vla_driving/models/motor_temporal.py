from __future__ import annotations

import torch
from torch import nn


class MotorTemporalGRU(nn.Module):
    """Temporal regressor for xycar motor angle and speed."""

    def __init__(
        self,
        perception_dim: int = 32,
        lidar_size: int = 360,
        pose_dim: int = 0,
        hidden_dim: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.1,
        steering_scale: float = 100.0,
        speed_scale: float = 20.0,
    ) -> None:
        super().__init__()
        self.register_buffer("output_scale", torch.tensor([steering_scale, speed_scale], dtype=torch.float32))
        input_dim = int(perception_dim) + int(lidar_size) + int(pose_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = sequence.shape
        encoded = self.encoder(sequence.reshape(batch * steps, dim)).reshape(batch, steps, -1)
        output, _ = self.gru(encoded)
        pred = self.head(output[:, -1]) * self.output_scale
        speed = torch.clamp(pred[:, 1], min=0.0)
        return torch.stack([pred[:, 0], speed], dim=1)
