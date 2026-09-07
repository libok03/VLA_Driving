from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Beta

from multimodal_planner_v9.data import FIXED_BASE_SPEED_MPS
from .model import (
    PATH_X_MAX_M, PATH_X_MIN_M, PATH_Y_MAX_M, PATH_Y_MIN_M,
)


BETA_EPS = 1.0e-3


def _unit_path(path: torch.Tensor) -> torch.Tensor:
    x = (path[..., 0] - PATH_X_MIN_M) / (PATH_X_MAX_M - PATH_X_MIN_M)
    y = (path[..., 1] - PATH_Y_MIN_M) / (PATH_Y_MAX_M - PATH_Y_MIN_M)
    return torch.stack((x, y), -1).clamp(BETA_EPS, 1.0 - BETA_EPS)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value).to(value.dtype)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def goal_trajectory_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    action_weight: float = 0.5,
    path_nll_weight: float = 0.02,
    position_weight: float = 1.0,
    step_weight: float = 0.5,
    acceleration_weight: float = 0.25,
    speed_nll_weight: float = 0.02,
    speed_weight: float = 1.0,
    speed_step_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Distribution math stays FP32 even when sensor encoders use autocast.
    path = outputs["trajectory_xy_m"].float()
    speed = outputs["target_speed_mps"].float()
    target_path = batch["target_path_xy_m"].float()
    target_speed = batch["target_speed_mps"].float()
    path_valid = batch["target_path_valid"].bool()
    speed_valid = batch["target_speed_valid"].bool()
    path_alpha = outputs["path_alpha"].float().clamp(1.001, 100.0)
    path_beta = outputs["path_beta"].float().clamp(1.001, 100.0)
    speed_alpha = outputs["speed_alpha"].float().clamp(1.001, 100.0)
    speed_beta = outputs["speed_beta"].float().clamp(1.001, 100.0)

    path_unit = _unit_path(target_path)
    speed_unit = (target_speed / FIXED_BASE_SPEED_MPS).clamp(
        BETA_EPS, 1.0 - BETA_EPS
    )
    path_nll = _masked_mean(-Beta(path_alpha, path_beta).log_prob(path_unit), path_valid)
    speed_nll = _masked_mean(
        -Beta(speed_alpha, speed_beta).log_prob(speed_unit), speed_valid
    )
    position = _masked_mean(
        F.smooth_l1_loss(path / PATH_X_MAX_M, target_path / PATH_X_MAX_M, reduction="none"),
        path_valid,
    )
    origin = torch.zeros_like(path[:, :1])
    pred_step = torch.diff(torch.cat((origin, path), 1), dim=1)
    target_step = torch.diff(torch.cat((origin, target_path), 1), dim=1)
    step = _masked_mean(
        F.smooth_l1_loss(pred_step / PATH_X_MAX_M, target_step / PATH_X_MAX_M, reduction="none"),
        path_valid,
    )
    acceleration = _masked_mean(
        F.smooth_l1_loss(
            torch.diff(pred_step, dim=1) / PATH_X_MAX_M,
            torch.diff(target_step, dim=1) / PATH_X_MAX_M,
            reduction="none",
        ),
        path_valid[:, 1:],
    )
    speed_mean = _masked_mean(
        F.smooth_l1_loss(
            speed / FIXED_BASE_SPEED_MPS,
            target_speed / FIXED_BASE_SPEED_MPS,
            reduction="none",
        ),
        speed_valid,
    )
    speed_step = _masked_mean(
        F.smooth_l1_loss(
            torch.diff(speed, dim=1) / FIXED_BASE_SPEED_MPS,
            torch.diff(target_speed, dim=1) / FIXED_BASE_SPEED_MPS,
            reduction="none",
        ),
        speed_valid[:, 1:],
    )
    action = F.cross_entropy(outputs["action_logits"].float(), batch["action_state"])

    total = (
        action_weight * action
        + path_nll_weight * path_nll
        + position_weight * position
        + step_weight * step
        + acceleration_weight * acceleration
        + speed_nll_weight * speed_nll
        + speed_weight * speed_mean
        + speed_step_weight * speed_step
    )
    terms = {
        "loss": total.detach(), "action": action.detach(),
        "path_nll": path_nll.detach(), "position": position.detach(),
        "step": step.detach(), "acceleration": acceleration.detach(),
        "speed_nll": speed_nll.detach(), "speed": speed_mean.detach(),
        "speed_step": speed_step.detach(),
        "path_mae_m": _masked_mean((path - target_path).abs(), path_valid).detach(),
        "speed_mae_mps": _masked_mean((speed - target_speed).abs(), speed_valid).detach(),
    }
    return total, terms


__all__ = ["BETA_EPS", "goal_trajectory_loss"]
