from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from multimodal_planner_v9.data import (
    ACTION_NAMES,
    SPATIAL_ANCHORS_M,
    TARGET_POLICIES,
    TARGET_POLICY_MORAI_ROUTE_RESIDUAL,
    PlannerDataset,
    deterministic_run_split,
    load_split_manifest,
    save_split_manifest,
)
from multimodal_planner_v9.losses import planner_loss
from multimodal_planner_v9.metrics import (
    ActionMetricAccumulator,
    SpatialResidualMetricAccumulator,
    SpatialSpeedMetricAccumulator,
)
from multimodal_planner_v9.model import ModelConfig, SpatialResidualSpeedPlannerV9


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_runs(data_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir()
        and (path / "sample_index.npz").is_file()
        and (path / "frame_chunks.json").is_file()
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }


def batch_identity(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": [str(value) for value in batch["run_id"]],
        "sample_id": batch["sample_id"].detach().cpu().tolist(),
        "gps_blackout": batch["gps_blackout"].detach().cpu().tolist(),
        "action_state": batch["action_state"].detach().cpu().tolist(),
        "avoidance": batch["avoidance"].detach().cpu().tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_outputs/multimodal_planner_v9"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("multimodal_planner_v9/splits/desktop_v001.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--backbone-warmup-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--blackout-weight", type=float, default=2.0)
    parser.add_argument(
        "--stop-sample-weight",
        type=float,
        default=0.0,
        help="STOP regression weight; keep 0 to exclude STOP from delta_d/delta_v",
    )
    parser.add_argument("--drive-sample-weight", type=float, default=1.0)
    parser.add_argument("--avoidance-weight", type=float, default=6.0)
    parser.add_argument("--avoidance-threshold-m", type=float, default=0.75)
    parser.add_argument("--stop-class-weight", type=float, default=4.0)
    parser.add_argument("--drive-class-weight", type=float, default=1.0)
    parser.add_argument("--avoid-class-weight", type=float, default=6.0)
    parser.add_argument("--stop-speed-threshold", type=float, default=0.6)
    parser.add_argument("--stop-max-endpoint-distance", type=float, default=10.0)
    parser.add_argument(
        "--target-policy",
        choices=TARGET_POLICIES,
        default="raw_drive_avoid",
        help=(
            "Regression supervision contract. morai_route_residual uses "
            "Local Route (zero delta) for DRIVE, recorded residuals for "
            "AVOID, and classification-only STOP."
        ),
    )
    parser.add_argument("--lateral-weight", type=float, default=1.0)
    parser.add_argument("--lateral-step-weight", type=float, default=0.5)
    parser.add_argument("--lateral-acceleration-weight", type=float, default=0.25)
    parser.add_argument("--inactive-lateral-prior-weight", type=float, default=0.0)
    parser.add_argument("--action-weight", type=float, default=0.5)
    parser.add_argument("--speed-weight", type=float, default=1.0)
    parser.add_argument("--speed-step-weight", type=float, default=0.2)
    parser.add_argument("--inactive-speed-prior-weight", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument(
        "--action-balanced-sampler",
        action="store_true",
        help=(
            "Sample DRIVE/STOP/AVOID at the configured target fractions, "
            "using inverse class frequency with replacement."
        ),
    )
    parser.add_argument("--sampler-drive-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--sampler-stop-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--sampler-avoid-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--allow-legacy-target-fields", action="store_true")
    parser.add_argument("--no-photometric-augmentation", action="store_true")
    parser.add_argument(
        "--augmentation-profile",
        choices=("standard", "strong"),
        default="standard",
        help=(
            "Photometric augmentation strength. Use strong for MORAI "
            "fine-tuning; validation is never augmented."
        ),
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def forward_batch(
    model: SpatialResidualSpeedPlannerV9,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    return model(
        batch["front"],
        batch["left"],
        batch["right"],
        batch["lidar_bev"],
        batch["ego"],
        batch["base_speed_profile_mps"],
        batch["local_route"],
    )


def loss_kwargs(
    args: argparse.Namespace,
    training: bool,
) -> dict[str, float]:
    return {
        "lateral_weight": args.lateral_weight,
        "lateral_step_weight": args.lateral_step_weight,
        "lateral_acceleration_weight": args.lateral_acceleration_weight,
        "inactive_lateral_prior_weight": args.inactive_lateral_prior_weight,
        "action_weight": args.action_weight,
        "stop_class_weight": args.stop_class_weight if training else 1.0,
        "drive_class_weight": args.drive_class_weight if training else 1.0,
        "avoid_class_weight": args.avoid_class_weight if training else 1.0,
        "speed_weight": args.speed_weight,
        "speed_step_weight": args.speed_step_weight,
        "inactive_speed_prior_weight": args.inactive_speed_prior_weight,
    }


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    normalizer: float,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return planner_loss(
        outputs,
        batch["target_spatial_lateral_m"],
        batch["target_spatial_valid"],
        batch["target_spatial_speed_delta_mps"],
        batch["target_spatial_speed_valid"],
        batch["action_state"],
        batch["sample_weight"],
        normalizer,
        **weights,
    )


def action_balanced_sample_weights(
    action_labels: torch.Tensor,
    target_fractions: tuple[float, float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.as_tensor(action_labels, dtype=torch.long, device="cpu")
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("action labels must be a non-empty rank-1 tensor")
    counts = torch.bincount(labels, minlength=len(ACTION_NAMES))
    if bool((counts == 0).any()):
        missing = [
            ACTION_NAMES[index]
            for index, count in enumerate(counts.tolist())
            if count == 0
        ]
        raise ValueError(
            "action-balanced sampling requires every action class; missing "
            + ", ".join(missing)
        )
    if target_fractions is None:
        targets = torch.ones(len(ACTION_NAMES), dtype=torch.double)
    else:
        targets = torch.as_tensor(target_fractions, dtype=torch.double)
        if (
            targets.shape != (len(ACTION_NAMES),)
            or not bool(torch.isfinite(targets).all())
            or bool((targets <= 0.0).any())
        ):
            raise ValueError(
                "sampler action fractions must be finite positive values"
            )
    # Preserve the historical equal-sampler scale (aggregate class mass 1.0)
    # while allowing non-uniform target ratios.
    targets = targets / targets.sum() * len(ACTION_NAMES)
    weights = (targets / counts.to(torch.double))[labels]
    return weights, counts


def make_action_balanced_sampler(
    dataset: PlannerDataset,
    seed: int,
    target_fractions: tuple[float, float, float] | None = None,
) -> tuple[WeightedRandomSampler, dict[str, Any], float]:
    weights, counts = action_balanced_sample_weights(
        torch.from_numpy(dataset.action_labels),
        target_fractions,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    sample_weights = torch.from_numpy(dataset.v9_sample_weights).to(torch.double)
    labels = torch.from_numpy(dataset.action_labels).to(torch.long)
    class_mean_sample_weights = torch.stack(
        [sample_weights[labels == index].mean() for index in range(len(ACTION_NAMES))]
    )
    aggregate_mass = torch.stack(
        [weights[labels == index].sum() for index in range(len(ACTION_NAMES))]
    )
    aggregate_mass /= aggregate_mass.sum()
    # The sampler may target a non-uniform action distribution. Normalize
    # regression weights by the weight expected under that actual sampling
    # distribution, not by an unweighted mean over the three classes. With
    # 65/25/10 sampling and DRIVE/STOP/AVOID weights 1/0/6, for example, the
    # correct normalizer is 1.25 rather than (1+0+6)/3 = 2.33.
    normalizer = float((class_mean_sample_weights * aggregate_mass).sum())
    summary = {
        "enabled": True,
        "policy": "target_action_fraction_with_replacement",
        "epoch_samples": len(dataset),
        "source_counts": {
            name: int(counts[index])
            for index, name in enumerate(ACTION_NAMES)
        },
        "expected_action_fraction": {
            name: float(aggregate_mass[index])
            for index, name in enumerate(ACTION_NAMES)
        },
        "sample_weight_normalizer": normalizer,
    }
    return sampler, summary, normalizer


def summarize_running_metrics(
    running: dict[str, float], sample_count: int
) -> dict[str, float]:
    """Convert accumulated batch sums into correctly conditioned metrics.

    V16 candidate metrics are zero for samples belonging to another action.
    Their accumulated numerator is therefore correct, but dividing it by all
    samples makes AVOID look smaller by exactly ``avoid_fraction``.  Preserve
    the global denominator for losses and use state counts only for candidate
    errors.
    """
    denominator = max(sample_count, 1)
    metrics = {name: value / denominator for name, value in running.items()}
    for state in ("drive", "avoid"):
        state_count = running.get(f"{state}_fraction", 0.0)
        if state_count <= 0.0:
            continue
        for suffix in ("path_mae_m", "ade_m"):
            name = f"{state}_{suffix}"
            if name in running:
                metrics[name] = running[name] / state_count
    moving_count = (
        running.get("drive_fraction", 0.0)
        + running.get("avoid_fraction", 0.0)
    )
    if moving_count > 0.0 and "speed_mae_mps" in running:
        metrics["speed_mae_mps"] = running["speed_mae_mps"] / moving_count
    return metrics


@torch.no_grad()
def evaluate(
    model: SpatialResidualSpeedPlannerV9,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    normalizer: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    lateral_metrics = {
        "all": SpatialResidualMetricAccumulator(),
        "blackout": SpatialResidualMetricAccumulator(),
        "non_blackout": SpatialResidualMetricAccumulator(),
    }
    speed_metrics = {
        "all": SpatialSpeedMetricAccumulator(),
        "blackout": SpatialSpeedMetricAccumulator(),
        "non_blackout": SpatialSpeedMetricAccumulator(),
    }
    action_metrics = ActionMetricAccumulator()
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = forward_batch(model, batch)
            _, terms = compute_loss(outputs, batch, normalizer, weights)
        tensors = {**outputs, **terms}
        bad = [
            name
            for name, value in tensors.items()
            if isinstance(value, torch.Tensor)
            and not bool(torch.isfinite(value).all())
        ]
        if bad:
            raise FloatingPointError(
                "non-finite validation tensors: "
                + json.dumps({"bad": bad, **batch_identity(batch)}, sort_keys=True)
            )
        lateral_metrics["all"].update(
            outputs["lateral_residual_m"],
            batch["target_spatial_lateral_m"],
            batch["target_spatial_valid"],
            batch["avoidance"],
        )
        speed_metrics["all"].update(
            outputs["speed_delta_mps"],
            batch["target_spatial_speed_delta_mps"],
            batch["target_spatial_speed_valid"],
            batch["speed_event"],
        )
        blackout = batch["gps_blackout"].bool()
        for name, mask in (("blackout", blackout), ("non_blackout", ~blackout)):
            if not bool(mask.any()):
                continue
            lateral_metrics[name].update(
                outputs["lateral_residual_m"][mask],
                batch["target_spatial_lateral_m"][mask],
                batch["target_spatial_valid"][mask],
                batch["avoidance"][mask],
            )
            speed_metrics[name].update(
                outputs["speed_delta_mps"][mask],
                batch["target_spatial_speed_delta_mps"][mask],
                batch["target_spatial_speed_valid"][mask],
                batch["speed_event"][mask],
            )
        action_metrics.update(outputs["action_logits"], batch["action_state"])
        batch_count = outputs["lateral_residual_m"].shape[0]
        count += batch_count
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_count
    result = {name: value / max(count, 1) for name, value in totals.items()}
    result["spatial_residual_metrics"] = {
        name: accumulator.result()
        for name, accumulator in lateral_metrics.items()
    }
    result["spatial_speed_metrics"] = {
        name: accumulator.result()
        for name, accumulator in speed_metrics.items()
    }
    result["action_metrics"] = action_metrics.result()
    return result


def make_optimizer(
    model: SpatialResidualSpeedPlannerV9,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    backbone = list(model.camera_encoder.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    other = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": other, "lr": args.lr},
            {"params": backbone, "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )


def set_training_phase(
    model: SpatialResidualSpeedPlannerV9,
    epoch: int,
    head_warmup_epochs: int,
    backbone_warmup_epochs: int,
) -> str:
    if epoch < head_warmup_epochs:
        for parameter in model.parameters():
            parameter.requires_grad = False
        head_modules = [
            model.speed_path_encoder,
            model.speed_delta_head,
            model.lateral_head,
            model.action_head,
        ]
        regression_ego_adapter = getattr(model, "regression_ego_adapter", None)
        if regression_ego_adapter is not None:
            head_modules.append(regression_ego_adapter)
        for module in head_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        model.set_camera_backbone_trainable(False)
        return "new_action_and_avoid_heads_only"
    for parameter in model.parameters():
        parameter.requires_grad = True
    backbone_trainable = epoch >= backbone_warmup_epochs
    model.set_camera_backbone_trainable(backbone_trainable)
    return "full" if backbone_trainable else "full_except_camera_backbone"


def build_dataset(
    args: argparse.Namespace,
    run_ids: list[str],
    training: bool,
) -> PlannerDataset:
    return PlannerDataset(
        args.data_root,
        run_ids,
        blackout_weight=args.blackout_weight if training else 1.0,
        stop_weight=1.0,
        drive_weight=1.0,
        avoidance_weight=1.0,
        avoidance_threshold_m=args.avoidance_threshold_m,
        max_samples=args.max_train_samples if training else args.max_val_samples,
        seed=args.seed if training else args.seed + 1,
        allow_legacy_target_fields=args.allow_legacy_target_fields,
        photometric_augmentation=(
            training and not args.no_photometric_augmentation
        ),
        augmentation_profile=args.augmentation_profile,
        stop_speed_threshold_mps=args.stop_speed_threshold,
        stop_max_endpoint_distance_m=args.stop_max_endpoint_distance,
        action_drive_weight=args.drive_sample_weight if training else 1.0,
        action_stop_weight=args.stop_sample_weight if training else 0.0,
        action_avoid_weight=args.avoidance_weight if training else 1.0,
        target_policy=args.target_policy,
    )


def main() -> None:
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_ids = discover_runs(args.data_root)
    if args.split_manifest.exists():
        splits = load_split_manifest(args.split_manifest)
    else:
        splits = deterministic_run_split(run_ids, args.seed)
        save_split_manifest(args.split_manifest, splits, args.seed, args.data_root)
    missing = set(sum(splits.values(), [])) - set(run_ids)
    if missing:
        raise RuntimeError(f"split manifest references missing runs: {sorted(missing)}")

    train_dataset = build_dataset(args, splits["train"], True)
    val_dataset = build_dataset(args, splits["val"], False)
    print("train_dataset", json.dumps(train_dataset.summary(), sort_keys=True))
    print("val_dataset", json.dumps(val_dataset.summary(), sort_keys=True))
    train_sampler = None
    train_sample_weight_normalizer = train_dataset.mean_sample_weight
    sampler_summary: dict[str, Any] = {
        "enabled": False,
        "policy": "uniform_sample_shuffle",
        "epoch_samples": len(train_dataset),
        "sample_weight_normalizer": train_sample_weight_normalizer,
    }
    if args.action_balanced_sampler:
        (
            train_sampler,
            sampler_summary,
            train_sample_weight_normalizer,
        ) = make_action_balanced_sampler(
            train_dataset,
            args.seed,
            (
                args.sampler_drive_fraction,
                args.sampler_stop_fraction,
                args.sampler_avoid_fraction,
            ),
        )
    print("train_sampler", json.dumps(sampler_summary, sort_keys=True))
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available() and not args.cpu,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_kwargs
    )
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    initialization = None
    if args.resume:
        initialization = torch.load(args.resume, map_location=device, weights_only=False)
    elif args.init_checkpoint:
        initialization = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
    if initialization is not None:
        values = dict(initialization["model_config"])
        values["pretrained_camera"] = False
        values["freeze_camera_backbone"] = True
        config = ModelConfig(**values)
    else:
        config = ModelConfig(
            pretrained_camera=not args.no_pretrained,
            freeze_camera_backbone=True,
        )
    model = SpatialResidualSpeedPlannerV9(config).to(device)
    if args.resume:
        saved_anchors = initialization["model_state"].get("spatial_anchors_m")
        expected_anchors = model.spatial_anchors_m.detach().cpu()
        if (
            saved_anchors is None
            or saved_anchors.shape != expected_anchors.shape
            or not torch.allclose(
                saved_anchors.detach().cpu(),
                expected_anchors,
                atol=1.0e-6,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "resume checkpoint uses an incompatible spatial-anchor "
                "schedule; start a new output run for the 0.5--20 m "
                "near-field anchors"
            )
    initialization_report = None
    if args.init_checkpoint:
        source_architecture = str(initialization.get("architecture", ""))
        target_architecture = getattr(
            model,
            "architecture_name",
            "multimodal_planner_v9_DRIVE_STOP_AVOID",
        )
        if source_architecture == target_architecture:
            model.load_state_dict(initialization["model_state"], strict=True)
            initialization_report = {
                "loaded_parameters": len(initialization["model_state"]),
                "fresh_parameters": [],
                "policy": (
                    "full same-architecture model transfer; optimizer and "
                    "scheduler restart"
                ),
                "checkpoint": str(args.init_checkpoint.resolve()),
                "source_epoch": int(initialization.get("epoch", -1)) + 1,
            }
        else:
            transfer_report = model.load_v8_state_dict(
                initialization["model_state"]
            )
            initialization_report = {
                **transfer_report,
                "policy": transfer_report.get(
                    "policy",
                    "compatible sensor/temporal weights transferred",
                ),
                "checkpoint": str(args.init_checkpoint.resolve()),
                "source_epoch": int(initialization.get("epoch", -1)) + 1,
            }
        print("initialization", json.dumps(initialization_report, sort_keys=True))

    optimizer = make_optimizer(model, args)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print("device", device)
    print("parameters", json.dumps(model.parameter_counts(), sort_keys=True))
    start_epoch = 0
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    if args.resume:
        model.load_state_dict(initialization["model_state"], strict=True)
        optimizer.load_state_dict(initialization["optimizer_state"])
        scaler.load_state_dict(initialization["scaler_state"])
        start_epoch = int(initialization["epoch"]) + 1
        best_val = float(initialization["best_val"])
        history = list(initialization.get("history", []))

    train_weights = loss_kwargs(args, True)
    val_weights = loss_kwargs(args, False)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        phase = set_training_phase(
            model,
            epoch,
            args.head_warmup_epochs,
            args.backbone_warmup_epochs,
        )
        # Full ResNet18 backprop can overflow intermediate FP16 activations
        # even when GradScaler protects the optimizer step. Keep AMP for the
        # warmup phases, but use true FP32 once the camera backbone is unfrozen.
        epoch_amp_enabled = amp_enabled and phase != "full"
        precision_policy = "amp_fp16" if epoch_amp_enabled else "fp32"
        print(
            "training_precision "
            + json.dumps(
                {
                    "epoch": epoch + 1,
                    "phase": phase,
                    "policy": precision_policy,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}
        sample_count = 0
        nonfinite_gradient_skips = 0
        accumulation_identities: list[dict[str, Any]] = []
        started = time.time()
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            accumulation_identities.append(batch_identity(batch))
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=epoch_amp_enabled,
            ):
                outputs = forward_batch(model, batch)
                loss, terms = compute_loss(
                    outputs,
                    batch,
                    train_sample_weight_normalizer,
                    train_weights,
                )
            tensors = {**outputs, **terms, "total": loss}
            bad = [
                name
                for name, value in tensors.items()
                if isinstance(value, torch.Tensor)
                and not bool(torch.isfinite(value).all())
            ]
            if bad:
                raise FloatingPointError(
                    "non-finite training tensors: "
                    + json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "bad": bad,
                            **batch_identity(batch),
                        },
                        sort_keys=True,
                    )
                )
            if epoch_amp_enabled:
                scaler.scale(loss / args.grad_accum).backward()
            else:
                (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                if epoch_amp_enabled:
                    scaler.unscale_(optimizer)
                # Outputs can still be finite while a backward kernel creates
                # a non-finite gradient. Fail before AdamW can contaminate its
                # moments or the model parameters.
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                    error_if_nonfinite=False,
                )
                if not bool(torch.isfinite(gradient_norm)):
                    nonfinite_gradient_skips += 1
                    print(
                        "nonfinite_gradient_skip "
                        + json.dumps(
                            {
                                "epoch": epoch + 1,
                                "step": step + 1,
                                "gradient_norm": str(float(gradient_norm)),
                                "accumulated_batches": accumulation_identities,
                                "skip_count": nonfinite_gradient_skips,
                                "precision_policy": precision_policy,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    # Never let one corrupt accumulated update contaminate the
                    # optimizer moments. GradScaler still needs update() after
                    # unscale_() so its scale is reduced for the next batch.
                    if epoch_amp_enabled:
                        scaler.update()
                    if nonfinite_gradient_skips > 32:
                        raise FloatingPointError(
                            "too many non-finite gradient updates in one epoch"
                        )
                elif epoch_amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_identities.clear()
            batch_count = outputs["lateral_residual_m"].shape[0]
            sample_count += batch_count
            for name, value in terms.items():
                running[name] = running.get(name, 0.0) + float(value) * batch_count
            if (step + 1) % args.log_every == 0:
                metrics = summarize_running_metrics(running, sample_count)
                print(
                    f"epoch={epoch + 1}/{args.epochs} phase={phase} "
                    f"step={step + 1}/{len(train_loader)} "
                    f"train={json.dumps(metrics, sort_keys=True)}",
                    flush=True,
                )

        train_metrics = summarize_running_metrics(running, sample_count)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            epoch_amp_enabled,
            val_dataset.mean_sample_weight,
            val_weights,
        )
        record = {
            "epoch": epoch,
            "training_phase": phase,
            "precision_policy": precision_policy,
            "nonfinite_gradient_skips": nonfinite_gradient_skips,
            "seconds": time.time() - started,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print("epoch_result", json.dumps(record, sort_keys=True), flush=True)
        metrics_dir = args.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"epoch_{epoch + 1:03d}.json").write_text(
            json.dumps(val_metrics, indent=2), encoding="utf-8"
        )
        state = {
            "architecture": getattr(
                model,
                "architecture_name",
                "multimodal_planner_v9_DRIVE_STOP_AVOID",
            ),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_val": min(best_val, val_metrics["loss"]),
            "history": history,
            "model_config": vars(config),
            "train_args": vars(args),
            "splits": splits,
            "parameter_counts": model.parameter_counts(),
            "initialization": initialization_report,
            "output_contract": (
                model.output_contract()
                if callable(getattr(model, "output_contract", None))
                else {
                "spatial_lateral_residual_m": [len(SPATIAL_ANCHORS_M)],
                "spatial_speed_delta_mps": [len(SPATIAL_ANCHORS_M)],
                "base_spatial_speed_mps": [len(SPATIAL_ANCHORS_M)],
                "candidate_spatial_speed_mps": [len(SPATIAL_ANCHORS_M)],
                "action_probabilities": [3],
                "action_prediction": ["DRIVE", "STOP", "AVOID"],
                "spatial_anchors_m": SPATIAL_ANCHORS_M.tolist(),
                "anchor_axis": (
                    "time_s"
                    if hasattr(model, "temporal_anchors_s")
                    else "route_progress_m"
                ),
                "temporal_anchors_s": (
                    model.temporal_anchors_s.detach().cpu().tolist()
                    if hasattr(model, "temporal_anchors_s")
                    else None
                ),
                "speed_constraint": "-external_base <= delta_v <= 0",
                "action_state": list(ACTION_NAMES),
                "model_routing": (
                    "classification argmax is diagnostic output only; "
                    "no path/speed hard routing"
                ),
                "runtime_queue_routing": {
                    "DRIVE": "Local Route; speed=min(external base,TTC)",
                    "STOP": "Local Route; exact speed=0",
                    "AVOID": "Local Route+delta_d; speed=min(base+delta_v,TTC)",
                },
                "runtime_speed_rule": (
                    "ordinary lead vehicles use TTC; the downstream state "
                    "machine applies learned delta_v only for AVOID"
                ),
                }
            ),
            "label_policy": (
                model.label_policy()
                if callable(getattr(model, "label_policy", None))
                else
                (
                    "explicit reviewed action_state; DRIVE targets Local Route "
                    "with zero delta_d/delta_v; AVOID supervises raw "
                    "delta_d/delta_v; STOP is classification-only; no "
                    "model-internal routing"
                )
                if args.target_policy == TARGET_POLICY_MORAI_ROUTE_RESIDUAL
                else (
                    "explicit reviewed action_state; DRIVE and AVOID supervise "
                    "raw delta_d/delta_v; STOP is classification-only; no "
                    "model-internal routing"
                )
            ),
            "loss_weighting": {
                **train_weights,
                "blackout_sample_weight": args.blackout_weight,
                "stop_sample_weight": args.stop_sample_weight,
                "drive_sample_weight": args.drive_sample_weight,
                "avoid_action_sample_weight": args.avoidance_weight,
                "train_mean_sample_weight": train_dataset.mean_sample_weight,
                "train_sample_weight_normalizer": (
                    train_sample_weight_normalizer
                ),
                "sampler": sampler_summary,
            },
        }
        torch.save(state, args.output_dir / "latest.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            state["best_val"] = best_val
            torch.save(state, args.output_dir / "best.pt")
        if (epoch + 1) % args.save_every == 0:
            torch.save(state, args.output_dir / f"epoch_{epoch + 1:03d}.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
