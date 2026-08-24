from __future__ import annotations

from typing import Any

import torch

from multimodal_planner_v9.data import ACTION_COUNT, ACTION_NAMES


class SpatialResidualMetricAccumulator:
    def __init__(self, anchors: int | None = None) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.valid_count = 0
        self.sample_count = 0
        size = 0 if anchors is None else anchors
        self.anchor_abs_sum = torch.zeros(size, dtype=torch.float64)
        self.anchor_count = torch.zeros(size, dtype=torch.int64)
        self.avoidance_abs_sum = 0.0
        self.avoidance_count = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        avoidance: torch.Tensor,
    ) -> None:
        if prediction.ndim != 2 or target.shape != prediction.shape:
            raise ValueError("prediction and target must share shape [batch,anchors]")
        if valid.shape != prediction.shape:
            raise ValueError("valid must match prediction shape")
        if self.anchor_abs_sum.numel() == 0:
            self.anchor_abs_sum = torch.zeros(
                prediction.shape[1], dtype=torch.float64
            )
            self.anchor_count = torch.zeros(
                prediction.shape[1], dtype=torch.int64
            )
        elif self.anchor_abs_sum.numel() != prediction.shape[1]:
            raise ValueError("anchor count changed between metric updates")
        error = (prediction.detach().float() - target.detach().float()).cpu()
        mask = valid.detach().bool().cpu()
        absolute = error.abs()
        self.abs_sum += float(absolute[mask].sum())
        self.sq_sum += float(error[mask].square().sum())
        self.valid_count += int(mask.sum())
        self.sample_count += prediction.shape[0]
        self.anchor_abs_sum += (absolute * mask).sum(dim=0).double()
        self.anchor_count += mask.sum(dim=0)
        avoid = avoidance.detach().bool().cpu().unsqueeze(1) & mask
        self.avoidance_abs_sum += float(absolute[avoid].sum())
        self.avoidance_count += int(avoid.sum())

    def result(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "valid_anchor_count": self.valid_count,
            "lateral_mae_m": self.abs_sum / max(self.valid_count, 1),
            "lateral_rmse_m": (
                self.sq_sum / max(self.valid_count, 1)
            ) ** 0.5,
            "avoidance_lateral_mae_m": (
                self.avoidance_abs_sum / max(self.avoidance_count, 1)
            ),
            "per_anchor_mae_m": [
                float(self.anchor_abs_sum[index])
                / max(int(self.anchor_count[index]), 1)
                for index in range(len(self.anchor_count))
            ],
            "per_anchor_count": self.anchor_count.tolist(),
        }


class ActionMetricAccumulator:
    def __init__(self) -> None:
        self.confusion = torch.zeros(
            ACTION_COUNT,
            ACTION_COUNT,
            dtype=torch.int64,
        )

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = logits.detach().argmax(dim=-1).cpu()
        truth = target.detach().cpu()
        for actual, predicted in zip(truth.tolist(), prediction.tolist()):
            self.confusion[int(actual), int(predicted)] += 1

    def result(self) -> dict[str, Any]:
        total = int(self.confusion.sum())
        states = {}
        f1s = []
        for index, name in enumerate(ACTION_NAMES):
            tp = int(self.confusion[index, index])
            fn = int(self.confusion[index].sum()) - tp
            fp = int(self.confusion[:, index].sum()) - tp
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1.0e-12)
            f1s.append(f1)
            states[name] = {
                "count": int(self.confusion[index].sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        return {
            "accuracy": int(torch.diag(self.confusion).sum()) / max(total, 1),
            "macro_f1": sum(f1s) / len(f1s),
            "confusion_matrix_actual_rows_predicted_columns": (
                self.confusion.tolist()
            ),
            "actions": states,
        }


class SpatialSpeedMetricAccumulator:
    def __init__(self, anchors: int | None = None) -> None:
        self.abs_sum = 0.0
        self.valid_count = 0
        self.event_abs_sum = 0.0
        self.event_count = 0
        self.normal_abs_delta_sum = 0.0
        self.normal_count = 0
        size = 0 if anchors is None else anchors
        self.anchor_abs_sum = torch.zeros(size, dtype=torch.float64)
        self.anchor_count = torch.zeros(size, dtype=torch.int64)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        event: torch.Tensor,
    ) -> None:
        if prediction.ndim != 2 or target.shape != prediction.shape:
            raise ValueError("prediction and target must share shape [batch,anchors]")
        if valid.shape != prediction.shape:
            raise ValueError("valid must match prediction shape")
        if self.anchor_abs_sum.numel() == 0:
            self.anchor_abs_sum = torch.zeros(
                prediction.shape[1], dtype=torch.float64
            )
            self.anchor_count = torch.zeros(
                prediction.shape[1], dtype=torch.int64
            )
        elif self.anchor_abs_sum.numel() != prediction.shape[1]:
            raise ValueError("anchor count changed between metric updates")
        pred = prediction.detach().float().cpu()
        truth = target.detach().float().cpu()
        mask = valid.detach().bool().cpu()
        event_mask = event.detach().bool().cpu().unsqueeze(1) & mask
        normal_mask = ~event.detach().bool().cpu().unsqueeze(1) & mask
        absolute = (pred - truth).abs()
        self.abs_sum += float(absolute[mask].sum())
        self.valid_count += int(mask.sum())
        self.event_abs_sum += float(absolute[event_mask].sum())
        self.event_count += int(event_mask.sum())
        self.normal_abs_delta_sum += float(pred.abs()[normal_mask].sum())
        self.normal_count += int(normal_mask.sum())
        self.anchor_abs_sum += (absolute * mask).sum(dim=0).double()
        self.anchor_count += mask.sum(dim=0)

    def result(self) -> dict[str, Any]:
        return {
            "valid_anchor_count": self.valid_count,
            "speed_delta_mae_mps": self.abs_sum / max(self.valid_count, 1),
            "event_speed_delta_mae_mps": (
                self.event_abs_sum / max(self.event_count, 1)
            ),
            "normal_abs_predicted_delta_mps": (
                self.normal_abs_delta_sum / max(self.normal_count, 1)
            ),
            "per_anchor_mae_mps": [
                float(self.anchor_abs_sum[index])
                / max(int(self.anchor_count[index]), 1)
                for index in range(len(self.anchor_count))
            ],
            "per_anchor_count": self.anchor_count.tolist(),
        }


__all__ = [
    "ActionMetricAccumulator",
    "SpatialResidualMetricAccumulator",
    "SpatialSpeedMetricAccumulator",
]
