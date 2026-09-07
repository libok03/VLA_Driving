from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Beta

from multimodal_planner_v9.data import ACTION_AVOID, ACTION_DRIVE, FIXED_BASE_SPEED_MPS
from .data import GOAL_LOOKAHEAD_M
from .model import LATERAL_SCALE_M


BETA_EPS = 1.0e-3
PATH_SCALE = torch.tensor(
    [GOAL_LOOKAHEAD_M, LATERAL_SCALE_M], dtype=torch.float32
)


def _expand_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    return expanded.expand_as(value).to(value.dtype)


def _masked_per_sample(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = _expand_mask(mask, value)
    reduce_dims = tuple(range(1, value.ndim))
    return (value * expanded).sum(reduce_dims) / expanded.sum(reduce_dims).clamp_min(1.0)


def _weighted_mean(
    per_sample: torch.Tensor,
    active: torch.Tensor,
    sample_weight: torch.Tensor,
    normalizer: float,
) -> torch.Tensor:
    value = per_sample * active.to(per_sample.dtype) * sample_weight
    return value.mean() / max(float(normalizer), 1.0e-8)


def _path_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    scale: torch.Tensor = PATH_SCALE,
) -> dict[str, torch.Tensor]:
    scale = scale.to(prediction.device, prediction.dtype)
    position_value = F.smooth_l1_loss(
        prediction / scale, target / scale, reduction="none"
    )
    position = _masked_per_sample(position_value, valid)

    origin = torch.zeros_like(prediction[:, :1])
    pred_step = torch.diff(torch.cat((origin, prediction), 1), dim=1)
    target_step = torch.diff(torch.cat((origin, target), 1), dim=1)
    step_valid = valid.clone()
    step_valid[:, 1:] &= valid[:, :-1]
    step_value = F.smooth_l1_loss(
        pred_step / scale, target_step / scale, reduction="none"
    )
    step = _masked_per_sample(step_value, step_valid)

    pred_acc = torch.diff(pred_step, dim=1)
    target_acc = torch.diff(target_step, dim=1)
    acc_valid = step_valid[:, 1:] & step_valid[:, :-1]
    acceleration = _masked_per_sample(
        F.smooth_l1_loss(
            pred_acc / scale, target_acc / scale, reduction="none"
        ),
        acc_valid,
    )

    endpoint_index = valid.to(torch.long).sum(1).sub(1).clamp_min(0)
    batch_index = torch.arange(prediction.shape[0], device=prediction.device)
    endpoint_value = F.smooth_l1_loss(
        prediction[batch_index, endpoint_index] / scale,
        target[batch_index, endpoint_index] / scale,
        reduction="none",
    ).mean(-1)
    endpoint_value = endpoint_value * valid.any(1).to(endpoint_value.dtype)

    abs_error = _masked_per_sample((prediction - target).abs(), valid)
    euclidean = _masked_per_sample(torch.linalg.vector_norm(prediction - target, dim=-1), valid)
    return {
        "position": position,
        "step": step,
        "acceleration": acceleration,
        "endpoint": endpoint_value,
        "mae": abs_error,
        "ade": euclidean,
    }


def candidate_trajectory_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    sample_weight_normalizer: float,
    *,
    action_weight: float = 0.5,
    position_weight: float = 1.0,
    step_weight: float = 0.5,
    acceleration_weight: float = 0.25,
    endpoint_weight: float = 0.5,
    speed_weight: float = 1.0,
    speed_step_weight: float = 0.2,
    speed_nll_weight: float = 0.02,
    drive_class_weight: float = 1.0,
    stop_class_weight: float = 1.0,
    avoid_class_weight: float = 1.0,
    path_scale: torch.Tensor = PATH_SCALE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    action_state = batch["action_state"].long()
    sample_weight = batch["sample_weight"].float()
    drive_valid = batch["target_drive_path_valid"].bool()
    avoid_valid = batch["target_avoid_path_valid"].bool()
    speed_valid = batch["target_speed_valid"].bool()
    drive_active = action_state == ACTION_DRIVE
    avoid_active = action_state == ACTION_AVOID
    speed_active = drive_active | avoid_active

    drive = _path_terms(
        outputs["drive_path_xy_m"].float(),
        batch["target_drive_path_xy_m"].float(), drive_valid,
        path_scale,
    )
    avoid = _path_terms(
        outputs["avoid_path_xy_m"].float(),
        batch["target_avoid_path_xy_m"].float(), avoid_valid,
        path_scale,
    )

    def weighted_candidate(name: str) -> torch.Tensor:
        return _weighted_mean(
            drive[name], drive_active, sample_weight, sample_weight_normalizer
        ) + _weighted_mean(
            avoid[name], avoid_active, sample_weight, sample_weight_normalizer
        )

    position = weighted_candidate("position")
    step = weighted_candidate("step")
    acceleration = weighted_candidate("acceleration")
    endpoint = weighted_candidate("endpoint")
    speed = outputs["target_speed_mps"].float()
    target_speed = batch["target_speed_mps"].float()
    speed_value = _masked_per_sample(
        F.smooth_l1_loss(
            speed / FIXED_BASE_SPEED_MPS,
            target_speed / FIXED_BASE_SPEED_MPS,
            reduction="none",
        ),
        speed_valid,
    )
    speed_loss = _weighted_mean(
        speed_value, speed_active, sample_weight, sample_weight_normalizer
    )
    speed_step_valid = speed_valid[:, 1:] & speed_valid[:, :-1]
    speed_step_value = _masked_per_sample(
        F.smooth_l1_loss(
            torch.diff(speed, dim=1) / FIXED_BASE_SPEED_MPS,
            torch.diff(target_speed, dim=1) / FIXED_BASE_SPEED_MPS,
            reduction="none",
        ),
        speed_step_valid,
    )
    speed_step = _weighted_mean(
        speed_step_value, speed_active, sample_weight, sample_weight_normalizer
    )
    speed_unit = (target_speed / FIXED_BASE_SPEED_MPS).clamp(
        BETA_EPS, 1.0 - BETA_EPS
    )
    speed_nll_value = _masked_per_sample(
        -Beta(
            outputs["speed_alpha"].float().clamp(1.001, 100.0),
            outputs["speed_beta"].float().clamp(1.001, 100.0),
        ).log_prob(speed_unit),
        speed_valid,
    )
    speed_nll = _weighted_mean(
        speed_nll_value, speed_active, sample_weight, sample_weight_normalizer
    )

    class_weights = torch.tensor(
        (drive_class_weight, stop_class_weight, avoid_class_weight),
        device=action_state.device, dtype=torch.float32,
    )
    action = F.cross_entropy(
        outputs["action_logits"].float(), action_state, weight=class_weights
    )
    # A sensor/fusion/trajectory-only pretraining stage sets both speed
    # weights to zero.  In that mode the auxiliary Beta NLL must also be
    # disabled, otherwise the supposedly excluded speed task still changes
    # the shared representation through its probabilistic head.
    effective_speed_nll_weight = (
        speed_nll_weight
        if speed_weight > 0.0 or speed_step_weight > 0.0
        else 0.0
    )
    total = (
        action_weight * action
        + position_weight * position
        + step_weight * step
        + acceleration_weight * acceleration
        + endpoint_weight * endpoint
        + speed_weight * speed_loss
        + speed_step_weight * speed_step
        + effective_speed_nll_weight * speed_nll
    )

    drive_mae = _weighted_mean(drive["mae"], drive_active, torch.ones_like(sample_weight), 1.0)
    avoid_mae = _weighted_mean(avoid["mae"], avoid_active, torch.ones_like(sample_weight), 1.0)
    drive_ade = _weighted_mean(drive["ade"], drive_active, torch.ones_like(sample_weight), 1.0)
    avoid_ade = _weighted_mean(avoid["ade"], avoid_active, torch.ones_like(sample_weight), 1.0)
    speed_abs = _masked_per_sample((speed - target_speed).abs(), speed_valid)
    speed_mae = _weighted_mean(
        speed_abs, speed_active, torch.ones_like(sample_weight), 1.0
    )
    terms = {
        "loss": total.detach(), "action": action.detach(),
        "position": position.detach(), "step": step.detach(),
        "acceleration": acceleration.detach(), "endpoint": endpoint.detach(),
        "speed": speed_loss.detach(),
        "speed_step": speed_step.detach(), "speed_nll": speed_nll.detach(),
        "effective_speed_nll_weight": torch.as_tensor(
            effective_speed_nll_weight, device=total.device
        ),
        "drive_path_mae_m": drive_mae.detach(),
        "avoid_path_mae_m": avoid_mae.detach(),
        "drive_ade_m": drive_ade.detach(), "avoid_ade_m": avoid_ade.detach(),
        "speed_mae_mps": speed_mae.detach(),
        "drive_fraction": drive_active.float().mean().detach(),
        "avoid_fraction": avoid_active.float().mean().detach(),
    }
    return total, terms


__all__ = ["candidate_trajectory_loss"]
