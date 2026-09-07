from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from multimodal_planner_v9.data import load_split_manifest
from tcp_morai_finetune.data import TCPMoraiDataset, WAYPOINT_HORIZONS_S
from tcp_morai_finetune.model import TCPMorai, load_reproduction_checkpoint


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--control-cache", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--current-control-weight", type=float, default=0.5)
    p.add_argument("--future-control-weight", type=float, default=0.5)
    p.add_argument("--speed-weight", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def beta_action(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    alpha, beta = alpha.float().clamp_min(1e-4), beta.float().clamp_min(1e-4)
    return (alpha / (alpha + beta)) * 2.0 - 1.0


class Metrics:
    def __init__(self) -> None:
        self.n = 0; self.sums: dict[str, float] = {}
        self.horizon = torch.zeros(4, dtype=torch.float64)

    def update(self, output, batch, terms) -> None:
        n = int(batch["image"].shape[0]); self.n += n
        for key, value in terms.items(): self.sums[key] = self.sums.get(key, 0.0) + float(value.detach().cpu()) * n
        distance = torch.linalg.vector_norm(output["waypoints"].detach() - batch["waypoints"], dim=-1).cpu().double()
        self.horizon += distance.sum(0)
        current = beta_action(output["action_alpha"], output["action_beta"])
        future = torch.stack([beta_action(a, b) for a, b in zip(output["future_alpha"], output["future_beta"])], 1)
        self.sums["current_control_mae"] = self.sums.get("current_control_mae", 0.0) + float(F.l1_loss(current, batch["current_control"].float()).detach().cpu()) * n
        self.sums["future_control_mae"] = self.sums.get("future_control_mae", 0.0) + float(F.l1_loss(future, batch["future_control"].float()).detach().cpu()) * n

    def compute(self):
        h = self.horizon / max(self.n, 1)
        return {**{k: v/max(self.n,1) for k,v in self.sums.items()}, "ade_m": float(h.mean()), "fde_2s_m": float(h[-1]), "per_horizon_error_m": {f"{s:g}s": float(h[i]) for i,s in enumerate(WAYPOINT_HORIZONS_S)}}


def losses(output, batch, cfg):
    waypoint = F.l1_loss(output["waypoints"], batch["waypoints"])
    speed = F.l1_loss(output["speed"].squeeze(-1), batch["speed_normalized"])
    # Direct deterministic human-control supervision. This is deliberately not
    # KL distillation: no teacher distribution participates in the objective.
    current_action = beta_action(output["action_alpha"], output["action_beta"])
    future_action = torch.stack([beta_action(a,b) for a,b in zip(output["future_alpha"],output["future_beta"])],1)
    current = F.smooth_l1_loss(current_action, batch["current_control"].float())
    future = F.smooth_l1_loss(future_action, batch["future_control"].float())
    total = waypoint + cfg.speed_weight*speed + cfg.current_control_weight*current + cfg.future_control_weight*future
    return {"loss": total, "waypoint_l1": waypoint, "speed_l1": speed, "current_control_loss": current, "future_control_loss": future}


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval(); metrics=Metrics()
    for batch in loader:
        batch=move(batch,device); output=model.forward_original(batch["image"],batch["state"],batch["target_point"])
        metrics.update(output,batch,losses(output,batch,cfg))
    return metrics.compute()


def main() -> None:
    cfg=args(); random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True,exist_ok=True)
    split=load_split_manifest(cfg.split_manifest)
    available={p.stem for p in cfg.control_cache.glob("*.npz")}
    def usable(ids):
        result=[]
        for rid in ids:
            manifest=json.loads((cfg.data_root/rid/"run_manifest.json").read_text())
            if manifest["source_run_id"] in available: result.append(rid)
        return result
    filtered={key:usable(value) for key,value in split.items()}
    (cfg.output_dir/"effective_split.json").write_text(json.dumps(filtered,indent=2))
    train=TCPMoraiDataset(cfg.data_root,filtered["train"],True,"strong",cfg.control_cache)
    val=TCPMoraiDataset(cfg.data_root,filtered["val"],False,"standard",cfg.control_cache)
    sampler=WeightedRandomSampler(torch.from_numpy(train.sampling_weights()),len(train),replacement=True,generator=torch.Generator().manual_seed(cfg.seed))
    kwargs=dict(batch_size=cfg.batch_size,num_workers=cfg.num_workers,pin_memory=torch.cuda.is_available() and not cfg.cpu,persistent_workers=cfg.num_workers>0)
    train_loader=DataLoader(train,sampler=sampler,**kwargs); val_loader=DataLoader(val,shuffle=False,**kwargs)
    device=torch.device("cpu" if cfg.cpu or not torch.cuda.is_available() else "cuda")
    model=TCPMorai(); initialization=load_reproduction_checkpoint(model,cfg.init_checkpoint)
    model.set_original_training_phase("full"); model.to(device)
    backbone=list(model.perception.parameters()); backbone_ids={id(p) for p in backbone}
    heads=[p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    optimizer=torch.optim.AdamW([{"params":heads,"lr":cfg.lr},{"params":backbone,"lr":cfg.backbone_lr}],weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.epochs)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    print("contract",json.dumps({"initialization_only":str(cfg.init_checkpoint),"teacher":None,"distillation":None,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"train_samples":len(train),"val_samples":len(val),"train_runs":len(filtered["train"]),"val_runs":len(filtered["val"])},sort_keys=True),flush=True)
    best=math.inf; history=[]
    for epoch in range(cfg.epochs):
        started=time.time(); model.set_original_training_phase("full"); model.train(); metrics=Metrics()
        for step,batch in enumerate(train_loader,1):
            batch=move(batch,device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                output=model.forward_original(batch["image"],batch["state"],batch["target_point"])
                terms=losses(output,batch,cfg)
            if not torch.isfinite(terms["loss"]): raise FloatingPointError(f"non-finite loss epoch={epoch+1} step={step}")
            scaler.scale(terms["loss"]).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.0)
            scaler.step(optimizer); scaler.update(); metrics.update(output,batch,terms)
            if step%cfg.log_every==0: print(f"epoch={epoch+1}/{cfg.epochs} phase=full step={step}/{len(train_loader)} train={json.dumps(metrics.compute(),sort_keys=True)}",flush=True)
        train_result=metrics.compute(); val_result=evaluate(model,val_loader,device,cfg); scheduler.step()
        record={"epoch":epoch,"seconds":time.time()-started,"train":train_result,"val":val_result}; history.append(record)
        print("epoch_result",json.dumps(record,sort_keys=True),flush=True)
        payload={"schema":"tcp_morai_full_policy_v1","epoch":epoch,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"history":history,"initialization":initialization,"teacher":None,"distillation":None,"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()}}
        torch.save(payload,cfg.output_dir/"latest.pt")
        score=val_result["ade_m"]+val_result["current_control_mae"]+val_result["future_control_mae"]
        if score<best: best=score; payload["selection_score"]=score; torch.save(payload,cfg.output_dir/"best.pt")
        (cfg.output_dir/"history.json").write_text(json.dumps(history,indent=2))


if __name__ == "__main__": main()
