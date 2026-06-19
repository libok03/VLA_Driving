from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class LidarEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        if lidar.ndim == 2:
            lidar = lidar.unsqueeze(1)
        return self.net(lidar)


class PoseRouteEncoder(nn.Module):
    def __init__(self, pose_dim: int, route_points: int, hidden_dim: int = 128) -> None:
        super().__init__()
        input_dim = pose_dim + route_points * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pose: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([pose, route.flatten(start_dim=1)], dim=1))


class MLPFusionHead(nn.Module):
    def __init__(self, hidden_dim: int, waypoint_count: int, waypoint_dim: int, dropout: float) -> None:
        super().__init__()
        self.waypoint_count = waypoint_count
        self.waypoint_dim = waypoint_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, waypoint_count * waypoint_dim),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        fused = torch.cat(features, dim=1)
        return self.net(fused).view(-1, self.waypoint_count, self.waypoint_dim)


class TransformerFusionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        waypoint_count: int,
        waypoint_dim: int,
        dropout: float,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.waypoint_count = waypoint_count
        self.waypoint_dim = waypoint_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.modality_embedding = nn.Parameter(torch.zeros(1, 4, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, waypoint_count * waypoint_dim),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        batch_size = features[0].shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.stack(features, dim=1)
        tokens = torch.cat([cls, tokens], dim=1) + self.modality_embedding
        fused = self.encoder(tokens)[:, 0]
        return self.head(fused).view(-1, self.waypoint_count, self.waypoint_dim)


class LightweightTransFuser(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        lidar_size: int = 360,
        pose_dim: int = 4,
        route_points: int = 10,
        waypoint_count: int = 5,
        waypoint_dim: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        fusion_type: str = "transformer",
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        _ = lidar_size
        self.image_encoder = ImageEncoder(image_channels, hidden_dim)
        self.lidar_encoder = LidarEncoder(hidden_dim)
        self.pose_route_encoder = PoseRouteEncoder(pose_dim, route_points, hidden_dim)
        if fusion_type == "mlp":
            self.fusion = MLPFusionHead(hidden_dim, waypoint_count, waypoint_dim, dropout)
        elif fusion_type == "transformer":
            self.fusion = TransformerFusionHead(
                hidden_dim=hidden_dim,
                waypoint_count=waypoint_count,
                waypoint_dim=waypoint_dim,
                dropout=dropout,
                layers=transformer_layers,
                heads=transformer_heads,
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
        pose: torch.Tensor,
        route: torch.Tensor,
    ) -> torch.Tensor:
        image_feature = self.image_encoder(image)
        lidar_feature = self.lidar_encoder(lidar)
        pose_route_feature = self.pose_route_encoder(pose, route)
        return self.fusion([image_feature, lidar_feature, pose_route_feature])
