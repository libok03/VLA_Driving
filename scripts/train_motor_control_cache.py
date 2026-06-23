from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_motor_control import resolve_device
from vla_driving.models.motor_control import MotorControlMLP
from vla_driving.utils.config import load_config


class CachedMotorDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cache_path: str | Path) -> None:
        cache = np.load(cache_path)
        self.perception = torch.from_numpy(cache["perception"].astype(np.float32, copy=False))
        self.lidar = torch.from_numpy(cache["lidar"].astype(np.float32, copy=False))
        self.target = torch.from_numpy(cache["target"].astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "perception": self.perception[index],
            "lidar": self.lidar[index],
            "target": self.target[index],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train direct xycar motor model from NPZ caches.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print("device:", device)

    train_dataset = CachedMotorDataset(Path(args.cache_dir) / "train.npz")
    val_dataset = CachedMotorDataset(Path(args.cache_dir) / "val.npz")
    print("samples:", len(train_dataset), len(val_dataset))
    print("feature dims:", train_dataset.perception.shape[1], train_dataset.lidar.shape[1])

    model = MotorControlMLP(**cfg["model"]).to(device)
    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    if args.resume and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"resumed: {checkpoint_path}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
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
    model: MotorControlMLP,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    total_loss = 0.0
    total_count = 0
    for batch in tqdm(loader, leave=False):
        perception = batch["perception"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        pred = model(perception, lidar)
        loss = loss_fn(pred, target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = perception.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


if __name__ == "__main__":
    main()
