from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vla_driving.data.motor_temporal_image_dataset import MotorTemporalImageDataset
from vla_driving.models.motor_temporal_camera import MotorTemporalCameraGRU
from vla_driving.utils.config import load_config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_dataset(cfg: dict, split: str, max_samples: int = 0) -> MotorTemporalImageDataset:
    data_cfg = cfg["data"]
    label_cfg = cfg["labels"]
    manifest_key = "train_manifest" if split == "train" else "val_manifest"
    return MotorTemporalImageDataset(
        data_root=data_cfg["data_root"],
        manifest_path=data_cfg[manifest_key],
        image_size=tuple(data_cfg["image_size"]),
        lidar_size=data_cfg["lidar_size"],
        pose_dim=data_cfg.get("pose_dim", 4),
        sequence_length=data_cfg["sequence_length"],
        steering_limit=label_cfg["steering_limit"],
        speed_limit=label_cfg["speed_limit"],
        lidar_max_range=data_cfg.get("lidar_max_range", 10.0),
        max_samples=max_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal camera+LiDAR+pose xycar motor model.")
    parser.add_argument("--config", default="configs/motor_control_temporal_camera.yaml")
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print("device:", device)

    train_dataset = build_dataset(cfg, "train", max_samples=args.overfit_samples)
    val_dataset = build_dataset(cfg, "val")
    print("samples:", len(train_dataset), len(val_dataset))
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit("No image temporal motor samples found. Re-extract bags with --image-topic.")

    model = MotorTemporalCameraGRU(**cfg["model"]).to(device)
    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    if args.resume and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"resumed: {checkpoint_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.SmoothL1Loss()
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_memory,
    )

    best_val = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, loss_fn, device)

        print(f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)


def run_epoch(
    model: MotorTemporalCameraGRU,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    total_loss = 0.0
    total_count = 0
    for batch in tqdm(loader, leave=False):
        image = batch["image"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        pose = batch["pose"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        pred = model(image, lidar, pose)
        loss = loss_fn(pred, target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = image.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


if __name__ == "__main__":
    main()
