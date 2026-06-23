from __future__ import annotations

import torch
from torch import nn


class MotorControlMLP(nn.Module):
    """Small direct regressor for xycar motor angle and speed."""

    def __init__(
        self,
        perception_dim: int = 32,
        lidar_summary_dim: int = 5,
        lidar_size: int | None = None,
        hidden_dim: int = 96,
        dropout: float = 0.05,
        steering_scale: float = 100.0,
        speed_scale: float = 20.0,
    ) -> None:
        super().__init__()
        self.register_buffer("output_scale", torch.tensor([steering_scale, speed_scale], dtype=torch.float32))
        lidar_dim = lidar_summary_dim if lidar_size is None else int(lidar_size)
        input_dim = perception_dim + lidar_dim
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

    def forward(self, perception: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        features = torch.cat([perception, lidar], dim=1)
        output = self.net(features) * self.output_scale
        speed = torch.clamp(output[:, 1], min=0.0)
        return torch.stack([output[:, 0], speed], dim=1)


class MotorControlAttention(nn.Module):
    """Attention regressor that treats LaserScan bins as ordered range tokens."""

    def __init__(
        self,
        perception_dim: int = 32,
        lidar_size: int = 360,
        token_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_dim: int = 128,
        dropout: float = 0.1,
        steering_scale: float = 100.0,
        speed_scale: float = 20.0,
    ) -> None:
        super().__init__()
        self.lidar_size = int(lidar_size)
        self.register_buffer("output_scale", torch.tensor([steering_scale, speed_scale], dtype=torch.float32))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.perception_token = nn.Sequential(
            nn.Linear(perception_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.lidar_value = nn.Linear(1, token_dim)
        self.lidar_position = nn.Parameter(torch.zeros(1, self.lidar_size, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 2),
            nn.Tanh(),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.lidar_position, std=0.02)

    def forward(self, perception: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        if lidar.shape[1] != self.lidar_size:
            raise ValueError(f"Expected lidar size {self.lidar_size}, got {lidar.shape[1]}")
        batch = perception.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        perception_token = self.perception_token(perception).unsqueeze(1)
        lidar_tokens = self.lidar_value(lidar.unsqueeze(-1)) + self.lidar_position
        tokens = torch.cat([cls, perception_token, lidar_tokens], dim=1)
        encoded = self.encoder(tokens)
        output = self.head(encoded[:, 0]) * self.output_scale
        speed = torch.clamp(output[:, 1], min=0.0)
        return torch.stack([output[:, 0], speed], dim=1)


def build_motor_control_model(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    model_type = str(cfg.pop("type", "mlp")).lower()
    if model_type == "mlp":
        return MotorControlMLP(**cfg)
    if model_type in {"attention", "transformer"}:
        cfg.pop("lidar_summary_dim", None)
        return MotorControlAttention(**cfg)
    raise ValueError(f"Unknown motor control model type: {model_type}")
