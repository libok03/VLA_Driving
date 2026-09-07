from __future__ import annotations

import torch
import torch.nn.functional as F

from multimodal_planner_v9.data import ACTION_COUNT


LATERAL_SCALE_M = 5.0
SPEED_SCALE_MPS = 20.0


def _masked_sample_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    count = mask.sum(dim=1).clamp_min(1.0)
    return (value * mask).sum(dim=1) / count


def _weighted_mean(
    value: torch.Tensor,
    sample_weight: torch.Tensor,
    normalizer: float,
) -> torch.Tensor:
    return (value * sample_weight).mean() / max(float(normalizer), 1.0e-8)


def planner_loss(
    outputs: dict[str, torch.Tensor],
    target_lateral_m: torch.Tensor,
    target_lateral_valid: torch.Tensor,
    target_speed_delta_mps: torch.Tensor,
    target_speed_valid: torch.Tensor,
    action_state: torch.Tensor,
    sample_weight: torch.Tensor,
    sample_weight_normalizer: float = 1.0,
    *,
    lateral_weight: float = 1.0,
    lateral_step_weight: float = 0.5,
    lateral_acceleration_weight: float = 0.25,
    inactive_lateral_prior_weight: float = 0.0,
    speed_weight: float = 1.0,
    speed_step_weight: float = 0.2,
    inactive_speed_prior_weight: float = 0.0,
    action_weight: float = 0.5,
    drive_class_weight: float = 1.0,
    stop_class_weight: float = 4.0,
    avoid_class_weight: float = 6.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    lateral = outputs["lateral_residual_m"]
    speed = outputs["speed_delta_mps"]
    action_logits = outputs["action_logits"]
    if lateral.shape != target_lateral_m.shape:
        raise ValueError("lateral prediction and target shapes must match")
    if speed.shape != target_speed_delta_mps.shape:
        raise ValueError("speed prediction and target shapes must match")
    if action_logits.shape != (lateral.shape[0], ACTION_COUNT):
        raise ValueError("action_logits must have shape [batch,3]")
    if action_state.shape != (lateral.shape[0],):
        raise ValueError("action_state must have shape [batch]")

    lateral_valid = target_lateral_valid.to(lateral.dtype)
    speed_valid = target_speed_valid.to(speed.dtype)

    lateral_values = F.smooth_l1_loss(
        lateral / LATERAL_SCALE_M,
        target_lateral_m / LATERAL_SCALE_M,
        reduction="none",
    )
    lateral_per_sample = _masked_sample_mean(lateral_values, lateral_valid)
    lateral_loss = _weighted_mean(
        lateral_per_sample,
        sample_weight,
        sample_weight_normalizer,
    )

    lateral_step = lateral[:, 1:] - lateral[:, :-1]
    target_lateral_step = target_lateral_m[:, 1:] - target_lateral_m[:, :-1]
    lateral_step_valid = lateral_valid[:, 1:] * lateral_valid[:, :-1]
    lateral_step_values = F.smooth_l1_loss(
        lateral_step / LATERAL_SCALE_M,
        target_lateral_step / LATERAL_SCALE_M,
        reduction="none",
    )
    lateral_step_loss = _weighted_mean(
        _masked_sample_mean(lateral_step_values, lateral_step_valid),
        sample_weight,
        sample_weight_normalizer,
    )

    lateral_acc = lateral_step[:, 1:] - lateral_step[:, :-1]
    target_lateral_acc = (
        target_lateral_step[:, 1:] - target_lateral_step[:, :-1]
    )
    lateral_acc_valid = (
        lateral_step_valid[:, 1:] * lateral_step_valid[:, :-1]
    )
    lateral_acc_values = F.smooth_l1_loss(
        lateral_acc / LATERAL_SCALE_M,
        target_lateral_acc / LATERAL_SCALE_M,
        reduction="none",
    )
    lateral_acceleration_loss = _weighted_mean(
        _masked_sample_mean(lateral_acc_values, lateral_acc_valid),
        sample_weight,
        sample_weight_normalizer,
    )

    inactive_lateral_prior = lateral.new_zeros(())

    speed_values = F.smooth_l1_loss(
        speed / SPEED_SCALE_MPS,
        target_speed_delta_mps / SPEED_SCALE_MPS,
        reduction="none",
    )
    speed_loss = _weighted_mean(
        _masked_sample_mean(speed_values, speed_valid),
        sample_weight,
        sample_weight_normalizer,
    )
    speed_step = speed[:, 1:] - speed[:, :-1]
    target_speed_step = (
        target_speed_delta_mps[:, 1:] - target_speed_delta_mps[:, :-1]
    )
    speed_step_valid = speed_valid[:, 1:] * speed_valid[:, :-1]
    speed_step_values = F.smooth_l1_loss(
        speed_step / SPEED_SCALE_MPS,
        target_speed_step / SPEED_SCALE_MPS,
        reduction="none",
    )
    speed_step_loss = _weighted_mean(
        _masked_sample_mean(speed_step_values, speed_step_valid),
        sample_weight,
        sample_weight_normalizer,
    )
    inactive_speed_prior = speed.new_zeros(())

    class_weights = torch.as_tensor(
        (drive_class_weight, stop_class_weight, avoid_class_weight),
        dtype=lateral.dtype,
        device=lateral.device,
    )
    action_loss = F.cross_entropy(
        action_logits,
        action_state,
        weight=class_weights,
    )
    total = (
        lateral_weight * lateral_loss
        + lateral_step_weight * lateral_step_loss
        + lateral_acceleration_weight * lateral_acceleration_loss
        + inactive_lateral_prior_weight * inactive_lateral_prior
        + speed_weight * speed_loss
        + speed_step_weight * speed_step_loss
        + inactive_speed_prior_weight * inactive_speed_prior
        + action_weight * action_loss
    )

    lateral_count = lateral_valid.sum().clamp_min(1.0)
    speed_count = speed_valid.sum().clamp_min(1.0)
    terms = {
        "loss": total.detach(),
        "action": action_loss.detach(),
        "lateral": lateral_loss.detach(),
        "lateral_step": lateral_step_loss.detach(),
        "lateral_acceleration": lateral_acceleration_loss.detach(),
        "inactive_lateral_prior": inactive_lateral_prior.detach(),
        "speed": speed_loss.detach(),
        "speed_step": speed_step_loss.detach(),
        "inactive_speed_prior": inactive_speed_prior.detach(),
        "avoid_fraction": (action_state == 2).to(lateral.dtype).mean().detach(),
        "lateral_mae_m": (
            ((lateral - target_lateral_m).abs() * lateral_valid).sum()
            / lateral_count
        ).detach(),
        "speed_delta_mae_mps": (
            ((speed - target_speed_delta_mps).abs() * speed_valid).sum()
            / speed_count
        ).detach(),
        "lateral_valid_anchor_fraction": lateral_valid.mean().detach(),
        "speed_valid_anchor_fraction": speed_valid.mean().detach(),
    }
    return total, terms


__all__ = ["planner_loss"]
