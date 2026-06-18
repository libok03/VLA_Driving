from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vla_driving.data.driving_dataset import DrivingDataset
from vla_driving.models.lightweight_transfuser import LightweightTransFuser
from vla_driving.utils.config import load_config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_dataset(cfg: dict, split: str) -> DrivingDataset:
    data_cfg = cfg["data"]
    manifest_key = "train_manifest" if split == "train" else "val_manifest"
    return DrivingDataset(
        data_root=data_cfg["data_root"],
        manifest_path=data_cfg[manifest_key],
        image_size=tuple(data_cfg["image_size"]),
        lidar_size=data_cfg["lidar_size"],
        route_points=data_cfg["route_points"],
        waypoint_count=data_cfg["waypoint_count"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    model = LightweightTransFuser(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    loss_fn = nn.SmoothL1Loss()

    train_loader = DataLoader(
        build_dataset(cfg, "train"),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )
    val_loader = DataLoader(
        build_dataset(cfg, "val"),
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")


def run_epoch(
    model: LightweightTransFuser,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    total_loss = 0.0
    total_count = 0
    for batch in tqdm(loader, leave=False):
        image = batch["image"].to(device)
        lidar = batch["lidar"].to(device)
        pose = batch["pose"].to(device)
        route = batch["route"].to(device)
        target = batch["waypoints"].to(device)

        pred = model(image, lidar, pose, route)
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
