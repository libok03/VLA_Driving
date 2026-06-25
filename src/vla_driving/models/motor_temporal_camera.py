from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class CameraEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
        self.projection = nn.Linear(backbone.fc.in_features, hidden_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image).flatten(1)
        return self.projection(features)


class MotorTemporalCameraGRU(nn.Module):
    def __init__(
        self,
        lidar_size: int = 360,
        pose_dim: int = 4,
        image_feature_dim: int = 256,
        camera_pretrained: bool = False,
        hidden_dim: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.1,
        steering_scale: float = 100.0,
        speed_scale: float = 20.0,
    ) -> None:
        super().__init__()
        self.register_buffer("output_scale", torch.tensor([steering_scale, speed_scale], dtype=torch.float32))
        self.camera_encoder = CameraEncoder(image_feature_dim, pretrained=camera_pretrained)
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
