"""Training entry point for goal-conditioned direct Beta trajectory V13 (60m Goal Lookahead) with Per-Epoch Modality Ablation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from multimodal_planner_v9.metrics import ActionMetricAccumulator
from multimodal_planner_v10.data import TEMPORAL_ANCHORS_S
from multimodal_planner_v9 import train as training
from .data import GoalTrajectoryDataset
from .losses import goal_trajectory_loss
from .model import GoalTrajectoryPlannerV13


def forward_batch(
    model: GoalTrajectoryPlannerV13, batch: dict[str, Any]
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
    del normalizer
    return goal_trajectory_loss(
        outputs,
        batch,
        action_weight=weights["action_weight"],
        position_weight=weights["lateral_weight"],
        step_weight=weights["lateral_step_weight"],
        acceleration_weight=weights["lateral_acceleration_weight"],
        speed_weight=weights["speed_weight"],
        speed_step_weight=weights["speed_step_weight"],
    )


def eval_single_condition(
    model: GoalTrajectoryPlannerV13,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    weights: dict[str, float],
    condition: str,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    action_metrics = ActionMetricAccumulator()

    with torch.inference_mode():
        for batch in loader:
            batch = training.move_batch(batch, device)

            # Apply modality ablation masking
            if condition == "no_cameras":
                batch["front"] = torch.zeros_like(batch["front"])
                batch["left"] = torch.zeros_like(batch["left"])
                batch["right"] = torch.zeros_like(batch["right"])
            elif condition == "no_lidar":
                batch["lidar_bev"] = torch.zeros_like(batch["lidar_bev"])
            elif condition == "no_goal":
                batch["goal_point"] = torch.zeros_like(batch["goal_point"])
            elif condition == "goal_only":
                batch["front"] = torch.zeros_like(batch["front"])
                batch["left"] = torch.zeros_like(batch["left"])
                batch["right"] = torch.zeros_like(batch["right"])
                batch["lidar_bev"] = torch.zeros_like(batch["lidar_bev"])

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = forward_batch(model, batch)
                _, terms = compute_loss(outputs, batch, 1.0, weights)

            action_metrics.update(outputs["action_logits"], batch["action_state"])
            batch_count = outputs["trajectory_xy_m"].shape[0]
            count += batch_count
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value) * batch_count

    result = {name: value / max(count, 1) for name, value in totals.items()}
    act_res = action_metrics.result()
    result["action_accuracy"] = act_res["accuracy"]
    result["action_macro_f1"] = act_res["macro_f1"]
    result["action_metrics"] = act_res
    return result


def evaluate_v13(
    model: GoalTrajectoryPlannerV13,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    normalizer: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    """Evaluates full validation baseline and performs per-epoch Modality Ablations."""
    del normalizer
    ablation_conditions = ["full", "no_cameras", "no_lidar", "no_goal", "goal_only"]
    ablation_results = {}

    for condition in ablation_conditions:
        ablation_results[condition] = eval_single_condition(
            model, loader, device, amp_enabled, weights, condition
        )

    main_result = dict(ablation_results["full"])
    main_result["ablation_summary"] = {
        cond: {
            "loss": float(res["loss"]),
            "path_mae_m": float(res["path_mae_m"]),
            "speed_mae_mps": float(res["speed_mae_mps"]),
            "action_accuracy": float(res["action_accuracy"]),
            "action_macro_f1": float(res["action_macro_f1"]),
        }
        for cond, res in ablation_results.items()
    }

    print("\n" + "=" * 75, flush=True)
    print(" === MODALITY ABLATION EVALUATION SUMMARY (Per-Epoch Validation) ===", flush=True)
    print(f"{'Condition':<14} | {'Val Loss':<9} | {'Path MAE (m)':<12} | {'Speed MAE':<10} | {'Action Acc':<10} | {'Macro F1':<8}", flush=True)
    print("-" * 75, flush=True)
    for cond in ablation_conditions:
        res = ablation_results[cond]
        print(
            f"{cond:<14} | {res['loss']:<9.4f} | {res['path_mae_m']:<12.4f} | "
            f"{res['speed_mae_mps']:<10.4f} | {res['action_accuracy']*100:<9.2f}% | "
            f"{res['action_macro_f1']:<8.4f}",
            flush=True,
        )
    print("=" * 75 + "\n", flush=True)

    return main_result


def set_training_phase(
    model: GoalTrajectoryPlannerV13,
    epoch: int,
    head_warmup_epochs: int,
    backbone_warmup_epochs: int,
) -> str:
    if epoch < head_warmup_epochs:
        for parameter in model.parameters():
            parameter.requires_grad = False
        modules = [
            model.goal_encoder, model.route_fusion, model.path_beta_head,
            model.speed_beta_head, model.action_head,
        ]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        model.set_camera_backbone_trainable(False)
        return "goal_fusion_and_three_heads"
    for parameter in model.parameters():
        parameter.requires_grad = True
    backbone_trainable = epoch >= backbone_warmup_epochs
    model.set_camera_backbone_trainable(backbone_trainable)
    return "full" if backbone_trainable else "full_except_camera_backbone"


def main() -> None:
    training.PlannerDataset = GoalTrajectoryDataset
    training.SpatialResidualSpeedPlannerV9 = GoalTrajectoryPlannerV13
    training.SPATIAL_ANCHORS_M = TEMPORAL_ANCHORS_S
    training.forward_batch = forward_batch
    training.compute_loss = compute_loss
    training.evaluate = evaluate_v13
    training.set_training_phase = set_training_phase
    training.main()


if __name__ == "__main__":
    main()
