from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from multimodal_planner_v9.data import ACTION_COUNT, SPATIAL_ANCHORS_M


XY_SCALE_M = 50.0
MAX_LATERAL_RESIDUAL_M = 5.0


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 192
    history_frames: int = 5
    horizon: int = 20
    attention_heads: int = 4
    spatial_layers: int = 2
    route_layers: int = 2
    dropout: float = 0.1
    pretrained_camera: bool = True
    freeze_camera_backbone: bool = True


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        normalized_key_value = self.kv_norm(key_value)
        attended, _ = self.attention(
            self.token_norm(tokens),
            normalized_key_value,
            normalized_key_value,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.ffn(self.ffn_norm(tokens))


class SharedCameraEncoder(nn.Module):
    """One ImageNet ResNet18 shared by all three camera views."""

    def __init__(self, dim: int, pretrained: bool, frozen: bool) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        network = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(network.children())[:-2])
        self.projection = nn.Conv2d(512, dim, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.camera_id = nn.Parameter(torch.zeros(3, 1, dim))
        nn.init.normal_(self.camera_id, std=0.02)
        self.output_norm = nn.LayerNorm(dim)
        self.frozen = frozen
        self.set_backbone_trainable(not frozen)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.frozen = not trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        if self.frozen:
            self.backbone.eval()

    def train(self, mode: bool = True) -> SharedCameraEncoder:
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def encode_view(
        self,
        images: torch.Tensor,
        camera_index: int,
    ) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                feature = self.backbone(images)
        else:
            feature = self.backbone(images)
        feature = self.pool(self.projection(feature))
        tokens = feature.flatten(2).transpose(1, 2)
        return self.output_norm(tokens + self.camera_id[camera_index])

    def forward(
        self,
        front: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                self.encode_view(front, 0),
                self.encode_view(left, 1),
                self.encode_view(right, 2),
            ),
            dim=1,
        )


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class LightBEVEncoder(nn.Module):
    """Four-stage VLP16 BEV encoder with multi-scale 4x4 token fusion."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        # BEV rows map x=[50,-10] m to [0,255]. Rows 214 onward are x<0,
        # which a front-bumper VLP16 sees only as ego-body reflections. Mask
        # them inside the model so Bench, MORAI training, and ROS inference use
        # the identical front-only visibility region.
        visibility = torch.ones((1, 1, 256, 256), dtype=torch.float32)
        visibility[..., 214:, :] = 0.0
        self.register_buffer(
            "front_visibility_mask",
            visibility,
            persistent=False,
        )
        self.stage1 = nn.Sequential(
            ConvNormAct(3, 32, 2),
            ConvNormAct(32, 32),
        )
        self.stage2 = nn.Sequential(
            ConvNormAct(32, 64, 2),
            ConvNormAct(64, 64),
        )
        self.stage3 = nn.Sequential(
            ConvNormAct(64, 128, 2),
            ConvNormAct(128, 128),
            ConvNormAct(128, 128),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(128, 192, 2),
            ConvNormAct(192, 192),
            ConvNormAct(192, 192),
            ConvNormAct(192, 192),
        )
        branch_dim = dim // 3
        self.scale2 = nn.Conv2d(64, branch_dim, 1)
        self.scale3 = nn.Conv2d(128, branch_dim, 1)
        self.scale4 = nn.Conv2d(192, dim - 2 * branch_dim, 1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.norm = nn.LayerNorm(dim)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        visibility = self.front_visibility_mask
        if visibility.shape[-2:] != bev.shape[-2:]:
            visibility = F.interpolate(
                visibility,
                size=bev.shape[-2:],
                mode="nearest",
            )
        bev = bev * visibility.to(device=bev.device, dtype=bev.dtype)
        stage1 = self.stage1(bev)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        fused = torch.cat(
            (
                self.pool(self.scale2(stage2)),
                self.pool(self.scale3(stage3)),
                self.pool(self.scale4(stage4)),
            ),
            dim=1,
        )
        return self.norm(fused.flatten(2).transpose(1, 2))


class EgoLocalizationEncoder(nn.Module):
    def __init__(self, input_dim: int, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, ego: torch.Tensor) -> torch.Tensor:
        return self.net(ego).unsqueeze(1)


class InputConditionedSlotGenerator(nn.Module):
    """Generate eight dynamic queries from the current sensor tokens."""

    def __init__(self, dim: int, slots: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.slot_scores = nn.Linear(dim, slots, bias=False)

    def forward(self, key_value: torch.Tensor) -> torch.Tensor:
        weights = self.slot_scores(self.norm(key_value)).transpose(1, 2)
        return torch.matmul(torch.softmax(weights, dim=-1), key_value)


class LocalRouteEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, route: torch.Tensor) -> torch.Tensor:
        return self.net(route)


def _route_geometry(
    local_route: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return route points, segments, lengths, and ego projection progress."""
    if local_route.ndim != 3 or local_route.shape[-2:] != (64, 4):
        raise ValueError(
            "local_route must have shape [batch,64,4], "
            f"got {tuple(local_route.shape)}"
        )
    points_m = local_route[..., :2] * XY_SCALE_M
    segments = points_m[:, 1:] - points_m[:, :-1]
    lengths = torch.linalg.vector_norm(segments, dim=-1).clamp_min(1.0e-4)
    cumulative = torch.cat(
        (
            torch.zeros_like(lengths[:, :1]),
            torch.cumsum(lengths, dim=1),
        ),
        dim=1,
    )
    start = points_m[:, :-1]
    projection_fraction = (
        -(start * segments).sum(dim=-1) / lengths.square()
    ).clamp(0.0, 1.0)
    projected = start + projection_fraction.unsqueeze(-1) * segments
    closest_segment = projected.square().sum(dim=-1).argmin(dim=1)
    batch_index = torch.arange(points_m.shape[0], device=points_m.device)
    origin_s = (
        cumulative[batch_index, closest_segment]
        + projection_fraction[batch_index, closest_segment]
        * lengths[batch_index, closest_segment]
    )
    return points_m, segments, lengths, origin_s


def interpolate_route_by_progress(
    local_route: torch.Tensor,
    forward_progress_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate route XY and tangent yaw at ego-relative stations."""
    if forward_progress_m.ndim != 2:
        raise ValueError(
            "forward_progress_m must have shape [batch,anchors], "
            f"got {tuple(forward_progress_m.shape)}"
        )
    points_m, segments, lengths, origin_s = _route_geometry(local_route)
    cumulative = torch.cat(
        (
            torch.zeros_like(lengths[:, :1]),
            torch.cumsum(lengths, dim=1),
        ),
        dim=1,
    )
    absolute_s = origin_s.unsqueeze(1) + forward_progress_m.clamp_min(0.0)
    route_end = cumulative[:, -1:]
    search_s = torch.minimum(absolute_s, route_end)
    segment_index = torch.searchsorted(
        cumulative.contiguous(),
        search_s.contiguous(),
        right=True,
    ) - 1
    segment_index = segment_index.clamp(0, segments.shape[1] - 1)
    gather_xy = segment_index.unsqueeze(-1).expand(-1, -1, 2)
    segment_start = torch.gather(points_m[:, :-1], 1, gather_xy)
    segment = torch.gather(segments, 1, gather_xy)
    segment_length = torch.gather(lengths, 1, segment_index)
    segment_s = torch.gather(cumulative[:, :-1], 1, segment_index)
    fraction = ((search_s - segment_s) / segment_length).clamp(0.0, 1.0)
    xy_m = segment_start + fraction.unsqueeze(-1) * segment
    overflow = (absolute_s - route_end).clamp_min(0.0)
    last_unit = segments[:, -1] / lengths[:, -1:].clamp_min(1.0e-4)
    xy_m = xy_m + overflow.unsqueeze(-1) * last_unit.unsqueeze(1)
    yaw_rad = torch.atan2(segment[..., 1], segment[..., 0])
    return xy_m, yaw_rad


def path_curvature_from_yaw(
    yaw_rad: torch.Tensor,
    stations_m: torch.Tensor,
) -> torch.Tensor:
    """Return wrapped finite-difference curvature at fixed path stations."""
    if yaw_rad.shape != stations_m.shape or yaw_rad.ndim != 2:
        raise ValueError("yaw_rad and stations_m must share shape [batch,anchors]")
    delta_yaw = torch.atan2(
        torch.sin(yaw_rad[:, 1:] - yaw_rad[:, :-1]),
        torch.cos(yaw_rad[:, 1:] - yaw_rad[:, :-1]),
    )
    delta_s = (stations_m[:, 1:] - stations_m[:, :-1]).clamp_min(1.0e-3)
    curvature = delta_yaw / delta_s
    return torch.cat((torch.zeros_like(curvature[:, :1]), curvature), dim=1)


class SpatialResidualSpeedPlannerV9(nn.Module):
    """Standalone V9 DRIVE/STOP/AVOID and near-field Δd/Δv model."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        dim = cfg.hidden_dim

        self.camera_encoder = SharedCameraEncoder(
            dim,
            cfg.pretrained_camera,
            cfg.freeze_camera_backbone,
        )
        self.lidar_encoder = LightBEVEncoder(dim)
        self.ego_encoder = EgoLocalizationEncoder(input_dim=16, dim=dim)
        self.slot_generator = InputConditionedSlotGenerator(dim, slots=8)
        self.spatial_fusion = nn.ModuleList(
            CrossAttentionBlock(dim, cfg.attention_heads, cfg.dropout)
            for _ in range(cfg.spatial_layers)
        )
        self.temporal_gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
        )
        self.route_encoder = LocalRouteEncoder(dim)
        self.route_fusion = nn.ModuleList(
            CrossAttentionBlock(dim, cfg.attention_heads, cfg.dropout)
            for _ in range(cfg.route_layers)
        )
        self.output_norm = nn.LayerNorm(dim)

        self.lateral_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 2, len(SPATIAL_ANCHORS_M)),
        )
        nn.init.zeros_(self.lateral_head[-1].weight)
        nn.init.zeros_(self.lateral_head[-1].bias)
        self.register_buffer(
            "spatial_anchors_m",
            torch.as_tensor(SPATIAL_ANCHORS_M.copy()),
            persistent=True,
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim, ACTION_COUNT),
        )
        self.speed_path_encoder = nn.Sequential(
            nn.Linear(6, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.speed_delta_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 2, 1),
        )
        nn.init.zeros_(self.speed_delta_head[-1].weight)
        nn.init.constant_(self.speed_delta_head[-1].bias, -4.0)

    def set_camera_backbone_trainable(self, trainable: bool) -> None:
        self.camera_encoder.set_backbone_trainable(trainable)

    def encode_frames(
        self,
        front: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        lidar_bev: torch.Tensor,
        ego: torch.Tensor,
    ) -> torch.Tensor:
        batch, history = front.shape[:2]
        flat = batch * history
        camera_tokens = self.camera_encoder(
            front.reshape(flat, *front.shape[2:]),
            left.reshape(flat, *left.shape[2:]),
            right.reshape(flat, *right.shape[2:]),
        )
        lidar_tokens = self.lidar_encoder(
            lidar_bev.reshape(flat, *lidar_bev.shape[2:])
        )
        ego_tokens = self.ego_encoder(ego.reshape(flat, -1))
        key_value = torch.cat((camera_tokens, lidar_tokens, ego_tokens), dim=1)
        if key_value.shape[1] != 65:
            raise RuntimeError(
                f"spatial fusion must receive KV65, got {key_value.shape}"
            )
        slots = self.slot_generator(key_value)
        for layer in self.spatial_fusion:
            slots = layer(slots, key_value)
        return slots.reshape(batch, history, 8, self.config.hidden_dim)

    def _context(
        self,
        front: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        lidar_bev: torch.Tensor,
        ego: torch.Tensor,
        local_route: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_slots = self.encode_frames(front, left, right, lidar_bev, ego)
        batch, history, slots, dim = frame_slots.shape
        temporal_input = frame_slots.permute(0, 2, 1, 3).reshape(
            batch * slots,
            history,
            dim,
        )
        _, hidden = self.temporal_gru(temporal_input)
        fused = hidden[-1].reshape(batch, slots, dim)
        route_tokens = self.route_encoder(local_route)
        for layer in self.route_fusion:
            fused = layer(fused, route_tokens)
        fused = self.output_norm(fused)
        return fused.mean(dim=1), fused

    def _base_speed_at_anchors(
        self,
        base_speed_profile_mps: torch.Tensor,
        local_route: torch.Tensor,
        stations_m: torch.Tensor,
    ) -> torch.Tensor:
        if (
            base_speed_profile_mps.ndim != 2
            or base_speed_profile_mps.shape[1] != 64
        ):
            raise ValueError(
                "base_speed_profile_mps must have shape [batch,64], "
                f"got {tuple(base_speed_profile_mps.shape)}"
            )
        if not bool(torch.isfinite(base_speed_profile_mps).all()):
            raise ValueError("base_speed_profile_mps must be finite")
        if not bool((base_speed_profile_mps > 0.0).all()):
            raise ValueError("base_speed_profile_mps must be positive")
        _, _, lengths, origin_s = _route_geometry(local_route)
        cumulative = torch.cat(
            (
                torch.zeros_like(lengths[:, :1]),
                torch.cumsum(lengths, dim=1),
            ),
            dim=1,
        )
        query_s = origin_s.unsqueeze(1) + stations_m.clamp_min(0.0)
        search_s = torch.minimum(query_s, cumulative[:, -1:])
        segment_index = torch.searchsorted(
            cumulative.contiguous(),
            search_s.contiguous(),
            right=True,
        ) - 1
        segment_index = segment_index.clamp(0, 62)
        segment_s = torch.gather(cumulative[:, :-1], 1, segment_index)
        segment_length = torch.gather(lengths, 1, segment_index)
        fraction = ((search_s - segment_s) / segment_length).clamp(0.0, 1.0)
        start_speed = torch.gather(
            base_speed_profile_mps[:, :-1],
            1,
            segment_index,
        )
        end_speed = torch.gather(
            base_speed_profile_mps[:, 1:],
            1,
            segment_index,
        )
        return start_speed + fraction * (end_speed - start_speed)

    def forward(
        self,
        front: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        lidar_bev: torch.Tensor,
        ego: torch.Tensor,
        base_speed_profile_mps: torch.Tensor,
        local_route: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context, fused = self._context(
            front,
            left,
            right,
            lidar_bev,
            ego,
            local_route,
        )
        lateral_residual_m = (
            torch.tanh(self.lateral_head(context)) * MAX_LATERAL_RESIDUAL_M
        )
        stations = self.spatial_anchors_m.to(
            device=context.device,
            dtype=context.dtype,
        ).unsqueeze(0).expand(context.shape[0], -1)
        base_xy_m, base_yaw = interpolate_route_by_progress(local_route, stations)
        normal = torch.stack((-torch.sin(base_yaw), torch.cos(base_yaw)), dim=-1)
        path_xy_m = base_xy_m + lateral_residual_m.unsqueeze(-1) * normal
        with_origin = torch.cat(
            (torch.zeros_like(path_xy_m[:, :1]), path_xy_m),
            dim=1,
        )
        step = with_origin[:, 1:] - with_origin[:, :-1]
        yaw = torch.atan2(step[..., 1], step[..., 0])
        curvature = path_curvature_from_yaw(yaw, stations)

        base_speed_mps = self._base_speed_at_anchors(
            base_speed_profile_mps,
            local_route,
            stations,
        )
        path_features = torch.cat(
            (
                (stations / float(SPATIAL_ANCHORS_M[-1])).unsqueeze(-1),
                path_xy_m / XY_SCALE_M,
                (yaw / torch.pi).unsqueeze(-1),
                (curvature / 0.2).clamp(-5.0, 5.0).unsqueeze(-1),
                (
                    lateral_residual_m / MAX_LATERAL_RESIDUAL_M
                ).unsqueeze(-1),
            ),
            dim=-1,
        )
        path_tokens = self.speed_path_encoder(path_features.detach())
        speed_tokens = path_tokens + context.unsqueeze(1)
        reduction_fraction = torch.sigmoid(
            self.speed_delta_head(speed_tokens).squeeze(-1)
        )
        speed_delta_mps = -reduction_fraction * base_speed_mps
        candidate_spatial_speed_mps = torch.clamp(
            base_speed_mps + speed_delta_mps,
            min=0.0,
        )
        action_logits = self.action_head(context)
        action_probabilities = torch.softmax(action_logits, dim=-1)

        base_with_origin = torch.cat(
            (torch.zeros_like(base_xy_m[:, :1]), base_xy_m),
            dim=1,
        )
        base_step = base_with_origin[:, 1:] - base_with_origin[:, :-1]
        base_path_yaw = torch.atan2(base_step[..., 1], base_step[..., 0])
        base_spatial_path = torch.cat(
            (
                base_xy_m / XY_SCALE_M,
                (base_path_yaw / torch.pi).unsqueeze(-1),
            ),
            dim=-1,
        )
        candidate_spatial_path = torch.cat(
            (
                path_xy_m / XY_SCALE_M,
                (yaw / torch.pi).unsqueeze(-1),
            ),
            dim=-1,
        )
        return {
            "lateral_residual_m": lateral_residual_m,
            "speed_delta_mps": speed_delta_mps,
            "base_speed_mps": base_speed_mps,
            "candidate_spatial_speed_mps": candidate_spatial_speed_mps,
            "spatial_stations_m": stations,
            "base_spatial_path": base_spatial_path,
            "base_spatial_path_xy_m": base_xy_m,
            "base_spatial_path_yaw_rad": base_path_yaw,
            "candidate_spatial_path": candidate_spatial_path,
            "candidate_spatial_path_xy_m": path_xy_m,
            "candidate_spatial_path_yaw_rad": yaw,
            "spatial_path_curvature_per_m": curvature,
            "speed_path_tokens": path_tokens,
            "action_logits": action_logits,
            "action_probabilities": action_probabilities,
            "action_prediction": action_logits.argmax(dim=-1),
            "spatial_tokens": fused,
        }

    def load_v8_state_dict(
        self,
        source_state: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Import only shape-compatible shared encoder weights from V8."""
        own = self.state_dict()
        transferable = {
            name: value
            for name, value in source_state.items()
            if name in own
            and own[name].shape == value.shape
            and name != "spatial_anchors_m"
            and not name.startswith(
                (
                    "speed_delta_head.",
                    "mgeo_encoder.",
                    "state_head.",
                    "lateral_head.",
                )
            )
        }
        incompatible = self.load_state_dict(transferable, strict=False)
        allowed_missing = {
            name
            for name in own
            if name == "spatial_anchors_m"
            or name.startswith(
                (
                    "speed_path_encoder.",
                    "speed_delta_head.",
                    "lateral_head.",
                    "action_head.",
                )
            )
        }
        if (
            set(incompatible.missing_keys) != allowed_missing
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "V9 transfer mismatch: "
                f"missing={sorted(incompatible.missing_keys)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        return {
            "loaded_parameters": len(transferable),
            "fresh_parameters": sorted(allowed_missing),
        }

    def parameter_counts(self) -> dict[str, int]:
        counts = {
            "camera": sum(
                parameter.numel()
                for parameter in self.camera_encoder.parameters()
            ),
            "lidar": sum(
                parameter.numel()
                for parameter in self.lidar_encoder.parameters()
            ),
            "ego": sum(
                parameter.numel()
                for parameter in self.ego_encoder.parameters()
            ),
            "lateral_head": sum(
                parameter.numel()
                for parameter in self.lateral_head.parameters()
            ),
            "learned_speed_parameters": sum(
                parameter.numel()
                for module in (self.speed_path_encoder, self.speed_delta_head)
                for parameter in module.parameters()
            ),
            "action_head": sum(
                parameter.numel()
                for parameter in self.action_head.parameters()
            ),
        }
        counts["total"] = sum(
            parameter.numel() for parameter in self.parameters()
        )
        counts["trainable"] = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return counts


__all__ = [
    "MAX_LATERAL_RESIDUAL_M",
    "ModelConfig",
    "SpatialResidualSpeedPlannerV9",
    "XY_SCALE_M",
    "interpolate_route_by_progress",
    "path_curvature_from_yaw",
]
