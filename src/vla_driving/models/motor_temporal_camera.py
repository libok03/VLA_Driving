from __future__ import annotations

import torch
from torch import nn


class CameraEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class MotorTemporalCameraGRU(nn.Module):
    def __init__(
        self,
        lidar_size: int = 360,
        pose_dim: int = 4,
        image_feature_dim: int = 256,
        hidden_dim: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.1,
        steering_scale: float = 100.0,
        speed_scale: float = 20.0,
    ) -> None:
        super().__init__()
        self.register_buffer("output_scale", torch.tensor([steering_scale, speed_scale], dtype=torch.float32))
        self.camera_encoder = CameraEncoder(image_feature_dim)
        input_dim = int(image_feature_dim) + int(lidar_size) + int(pose_dim)
        self.frame_encoder = nn.Sequential(
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

    def forward(self, image: torch.Tensor, lidar: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = image.shape
        image_features = self.camera_encoder(image.reshape(batch * steps, channels, height, width))
        image_features = image_features.reshape(batch, steps, -1)
        frame = torch.cat([image_features, lidar, pose], dim=2)
        encoded = self.frame_encoder(frame.reshape(batch * steps, -1)).reshape(batch, steps, -1)
        output, _ = self.gru(encoded)
        pred = self.head(output[:, -1]) * self.output_scale
        speed = torch.clamp(pred[:, 1], min=0.0)
        return torch.stack([pred[:, 0], speed], dim=1)
