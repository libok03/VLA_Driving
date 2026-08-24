from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from multimodal_planner_v9 import train as training
from multimodal_planner_v9.data import ACTION_AVOID, ACTION_DRIVE, ACTION_NAMES
from multimodal_planner_v9.metrics import ActionMetricAccumulator
from .data import PATH_ANCHORS_M, WAYPOINT_HORIZONS_S, GoalCandidateDataset
from .losses import candidate_trajectory_loss
from .model import GoalCandidatePlannerV16


def forward_batch(
    model: GoalCandidatePlannerV16, batch: dict[str, Any]
) -> dict[str, torch.Tensor]:
    return model(
        batch["front"], batch["left"], batch["right"],
        batch["lidar_bev"], batch["goal_point"],
    )


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    normalizer: float,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return candidate_trajectory_loss(
        outputs,
        batch,
        normalizer,
        action_weight=weights["action_weight"],
        position_weight=weights["lateral_weight"],
        step_weight=weights["lateral_step_weight"],
        acceleration_weight=weights["lateral_acceleration_weight"],
        speed_weight=weights["speed_weight"],
        speed_step_weight=weights["speed_step_weight"],
        drive_class_weight=weights["drive_class_weight"],
        stop_class_weight=weights["stop_class_weight"],
        avoid_class_weight=weights["avoid_class_weight"],
    )


def _empty_state_metric() -> dict[str, Any]:
    return {
        "coordinate_abs_sum": 0.0,
        "euclidean_sum": 0.0,
        "longitudinal_abs_sum": 0.0,
        "lateral_abs_sum": 0.0,
        "point_count": 0,
        "fde_sum": np.zeros(len(PATH_ANCHORS_M), dtype=np.float64),
        "fde_count": np.zeros(len(PATH_ANCHORS_M), dtype=np.int64),
    }


def evaluate_v16(
    model: GoalCandidatePlannerV16,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    normalizer: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    action_metrics = ActionMetricAccumulator()
    state_metrics = {
        "DRIVE": _empty_state_metric(),
        "AVOID": _empty_state_metric(),
    }
    speed_abs_sum = 0.0
    speed_count = 0

    with torch.inference_mode():
        for batch in loader:
            batch = training.move_batch(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = forward_batch(model, batch)
                _, terms = compute_loss(outputs, batch, normalizer, weights)
            batch_count = outputs["drive_path_xy_m"].shape[0]
            count += batch_count
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value) * batch_count
            action_metrics.update(outputs["action_logits"], batch["action_state"])

            for action_index, name, pred_key, target_key, valid_key in (
                (
                    ACTION_DRIVE, "DRIVE", "drive_path_xy_m",
                    "target_drive_path_xy_m", "target_drive_path_valid",
                ),
                (
                    ACTION_AVOID, "AVOID", "avoid_path_xy_m",
                    "target_avoid_path_xy_m", "target_avoid_path_valid",
                ),
            ):
                sample_mask = batch["action_state"] == action_index
                if not bool(sample_mask.any()):
                    continue
                prediction = outputs[pred_key][sample_mask].float()
                target = batch[target_key][sample_mask].float()
                valid = batch[valid_key][sample_mask].bool()
                error = prediction - target
                distance = torch.linalg.vector_norm(error, dim=-1)
                metric = state_metrics[name]
                metric["coordinate_abs_sum"] += float(
                    (error.abs() * valid.unsqueeze(-1)).sum()
                )
                metric["euclidean_sum"] += float((distance * valid).sum())
                metric["longitudinal_abs_sum"] += float(
                    (error[..., 0].abs() * valid).sum()
                )
                metric["lateral_abs_sum"] += float(
                    (error[..., 1].abs() * valid).sum()
                )
                metric["point_count"] += int(valid.sum())
                for anchor_index in range(len(PATH_ANCHORS_M)):
                    anchor_valid = valid[:, anchor_index]
                    metric["fde_sum"][anchor_index] += float(
                        distance[anchor_valid, anchor_index].sum()
                    )
                    metric["fde_count"][anchor_index] += int(anchor_valid.sum())

            speed_valid = batch["target_speed_valid"].bool()
            speed_abs_sum += float(
                (
                    (outputs["target_speed_mps"].float() - batch["target_speed_mps"].float()).abs()
                    * speed_valid
                ).sum()
            )
            speed_count += int(speed_valid.sum())

    result = {name: value / max(count, 1) for name, value in totals.items()}
    action_result = action_metrics.result()
    result["action_accuracy"] = action_result["accuracy"]
    result["action_macro_f1"] = action_result["macro_f1"]
    result["action_metrics"] = action_result
    result["speed_mae_mps"] = speed_abs_sum / max(speed_count, 1)
    result["state_path_metrics"] = {}
    total_coordinate = 0.0
    total_coordinate_count = 0
    for name, raw in state_metrics.items():
        points = max(int(raw["point_count"]), 1)
        fde = {
            f"fde_at_{float(seconds):g}s_m": (
                float(raw["fde_sum"][index])
                / max(int(raw["fde_count"][index]), 1)
            )
            for index, seconds in enumerate(WAYPOINT_HORIZONS_S)
        }
        result["state_path_metrics"][name] = {
            "valid_points": int(raw["point_count"]),
            "coordinate_mae_m": float(raw["coordinate_abs_sum"]) / (2 * points),
            "ade_m": float(raw["euclidean_sum"]) / points,
            "longitudinal_mae_m": float(raw["longitudinal_abs_sum"]) / points,
            "lateral_mae_m": float(raw["lateral_abs_sum"]) / points,
            **fde,
        }
        total_coordinate += float(raw["coordinate_abs_sum"])
        total_coordinate_count += 2 * int(raw["point_count"])
    result["path_mae_m"] = total_coordinate / max(total_coordinate_count, 1)
    drive_metric = result["state_path_metrics"]["DRIVE"]
    avoid_metric = result["state_path_metrics"]["AVOID"]
    result["drive_path_mae_m"] = drive_metric["coordinate_mae_m"]
    result["avoid_path_mae_m"] = avoid_metric["coordinate_mae_m"]
    result["drive_ade_m"] = drive_metric["ade_m"]
    result["avoid_ade_m"] = avoid_metric["ade_m"]
    # Checkpoint selection must follow the tasks that are actually active.
    # Bench2Drive representation pretraining disables action and speed; in
    # that stage selecting by their deliberately untrained metrics would pick
    # an arbitrary checkpoint instead of the best trajectory representation.
    result["optimization_loss"] = result["loss"]
    active_state_ades = [
        metric["ade_m"]
        for metric in (drive_metric, avoid_metric)
        if metric["valid_points"] > 0
    ]
    balanced_ade = sum(active_state_ades) / max(len(active_state_ades), 1)
    representation_only = (
        weights["action_weight"] <= 0.0
        and weights["speed_weight"] <= 0.0
        and weights["speed_step_weight"] <= 0.0
    )
    if representation_only:
        result["loss"] = balanced_ade
        result["selection_policy"] = "balanced_drive_avoid_ade"
    else:
        result["loss"] = (
            balanced_ade
            + 0.25 * result["speed_mae_mps"]
            + 2.0 * (1.0 - result["action_macro_f1"])
        )
        result["selection_policy"] = (
            "balanced_ade_plus_speed_mae_plus_action_macro_f1"
        )
    result["selection_score"] = result["loss"]
    return result


def set_training_phase(
    model: GoalCandidatePlannerV16,
    epoch: int,
    head_warmup_epochs: int,
    backbone_warmup_epochs: int,
) -> str:
    if epoch < head_warmup_epochs:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module in (
            model.goal_encoder, model.route_fusion,
            model.drive_path_decoder, model.avoid_path_decoder,
            model.speed_beta_head, model.action_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
        model.set_camera_backbone_trainable(False)
        return "single_goal_and_tcp6_candidate_heads"
    for parameter in model.parameters():
        parameter.requires_grad = True
    backbone_trainable = epoch >= backbone_warmup_epochs
    model.set_camera_backbone_trainable(backbone_trainable)
    return "full" if backbone_trainable else "full_except_camera_backbone"


def main() -> None:
    training.PlannerDataset = GoalCandidateDataset
    training.SpatialResidualSpeedPlannerV9 = GoalCandidatePlannerV16
    training.SPATIAL_ANCHORS_M = PATH_ANCHORS_M
    training.forward_batch = forward_batch
    training.compute_loss = compute_loss
    training.evaluate = evaluate_v16
    training.set_training_phase = set_training_phase
    training.main()


if __name__ == "__main__":
    main()
