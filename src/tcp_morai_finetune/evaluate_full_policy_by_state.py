from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from multimodal_planner_v9.data import ACTION_NAMES, load_split_manifest
from tcp_morai_finetune.data import TCPMoraiDataset, WAYPOINT_HORIZONS_S
from tcp_morai_finetune.model import TCPMorai
from tcp_morai_finetune.train_full_policy import beta_action, move


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--control-cache", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    split = load_split_manifest(args.split_manifest)
    available = {x.stem for x in args.control_cache.glob("*.npz")}
    run_ids = []
    for run_id in split[args.split]:
        manifest = json.loads((args.data_root / run_id / "run_manifest.json").read_text())
        if manifest["source_run_id"] in available:
            run_ids.append(run_id)
    dataset = TCPMoraiDataset(args.data_root, run_ids, control_cache=args.control_cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TCPMorai().to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    count = torch.zeros(3, dtype=torch.int64)
    distance = torch.zeros((3, 4), dtype=torch.float64)
    current_abs = torch.zeros((3, 2), dtype=torch.float64)
    future_abs = torch.zeros((3, 2), dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            output = model.forward_original(batch["image"], batch["state"], batch["target_point"])
            path_error = torch.linalg.vector_norm(output["waypoints"] - batch["waypoints"], dim=-1)
            current = (beta_action(output["action_alpha"], output["action_beta"]) - batch["current_control"]).abs()
            future = (torch.stack([beta_action(a,b) for a,b in zip(output["future_alpha"], output["future_beta"])], 1) - batch["future_control"]).abs().mean(1)
            for action in range(3):
                mask = batch["action_state"] == action
                n = int(mask.sum())
                count[action] += n
                if n:
                    distance[action] += path_error[mask].sum(0).cpu().double()
                    current_abs[action] += current[mask].sum(0).cpu().double()
                    future_abs[action] += future[mask].sum(0).cpu().double()
    states = {}
    for action, name in enumerate(ACTION_NAMES):
        n = max(int(count[action]), 1)
        horizon = distance[action] / n
        states[name] = {
            "count": int(count[action]),
            "ade_m": float(horizon.mean()),
            "fde_2s_m": float(horizon[-1]),
            "per_horizon_error_m": {f"{s:g}s": float(horizon[i]) for i,s in enumerate(WAYPOINT_HORIZONS_S)},
            "current_accel_mae": float(current_abs[action,0]/n),
            "current_steer_mae": float(current_abs[action,1]/n),
            "future_accel_mae": float(future_abs[action,0]/n),
            "future_steer_mae": float(future_abs[action,1]/n),
        }
    result = {"checkpoint": str(args.checkpoint), "checkpoint_epoch": int(payload["epoch"])+1, "split": args.split, "macro_ade_m": float(np.mean([x["ade_m"] for x in states.values()])), "states": states}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
