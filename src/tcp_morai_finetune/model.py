from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import resnet34


@dataclass(frozen=True)
class TCPConfig:
    pred_len: int = 4


class TCPMorai(nn.Module):
    """Official TCP topology with MORAI trajectory and direct-control paths.

    Module names and tensor shapes intentionally match OpenDriveLab/TCP so a
    published reproduction Lightning checkpoint can be transferred directly.
    Value branches remain only for checkpoint compatibility. Full-policy
    training optimizes the encoder, trajectory branch, and direct-control
    branch from MORAI human labels without teacher distillation.
    """

    def __init__(self, config: TCPConfig | None = None) -> None:
        super().__init__()
        self.config = config or TCPConfig()
        self.perception = resnet34(weights=None)

        self.measurements = nn.Sequential(
            nn.Linear(1 + 2 + 6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.join_traj = nn.Sequential(
            nn.Linear(128 + 1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.join_ctrl = nn.Sequential(
            nn.Linear(128 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.speed_branch = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        # MORAI action labels supervise this head directly.  It is intentionally
        # attached only to TCP's camera feature: speed, target point, and route
        # command are trajectory inputs and must not become a state shortcut.
        self.state_head = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(256, 3),  # DRIVE / STOP / AVOID logits
        )
        self.value_branch_traj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.value_branch_ctrl = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
        )
        self.decoder_ctrl = nn.GRUCell(input_size=256 + 4, hidden_size=256)
        self.output_ctrl = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.dist_mu = nn.Sequential(nn.Linear(256, 2), nn.Softplus())
        self.dist_sigma = nn.Sequential(nn.Linear(256, 2), nn.Softplus())

        self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=256)
        self.output_traj = nn.Linear(256, 2)
        self.init_att = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 29 * 8),
            nn.Softmax(1),
        )
        self.wp_att = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 29 * 8),
            nn.Softmax(1),
        )
        self.merge = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )

    def _perception_forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        network = self.perception
        value = network.conv1(image)
        value = network.bn1(value)
        value = network.relu(value)
        value = network.maxpool(value)
        value = network.layer1(value)
        value = network.layer2(value)
        value = network.layer3(value)
        spatial = network.layer4(value)
        embedding = network.avgpool(spatial)
        embedding = torch.flatten(embedding, 1)
        embedding = network.fc(embedding)
        return embedding, spatial

    def forward_trajectory(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        target_point: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feature_embedding, _spatial = self._perception_forward(image)
        measurement_feature = self.measurements(state)
        action_logits = self.state_head(feature_embedding)
        hidden = self.join_traj(
            torch.cat((feature_embedding, measurement_feature), dim=1)
        )
        waypoint = torch.zeros(
            (hidden.shape[0], 2), device=hidden.device, dtype=hidden.dtype
        )
        predictions = []
        for _ in range(self.config.pred_len):
            hidden = self.decoder_traj(
                torch.cat((waypoint, target_point), dim=1), hidden
            )
            waypoint = waypoint + self.output_traj(hidden)
            predictions.append(waypoint)
        return {
            "waypoints": torch.stack(predictions, dim=1),
            "speed": self.speed_branch(feature_embedding),
            "action_logits": action_logits,
        }

    def forward_original(self, image: torch.Tensor, state: torch.Tensor, target_point: torch.Tensor) -> dict[str, Any]:
        """Run the complete trajectory and multi-step control topology from TCP."""
        feature_embedding, spatial = self._perception_forward(image)
        measurement_feature = self.measurements(state)

        trajectory_feature = self.join_traj(
            torch.cat((feature_embedding, measurement_feature), dim=1)
        )
        waypoint = torch.zeros(
            (trajectory_feature.shape[0], 2),
            device=trajectory_feature.device,
            dtype=trajectory_feature.dtype,
        )
        hidden = trajectory_feature
        waypoints = []
        trajectory_hidden = []
        for _ in range(self.config.pred_len):
            hidden = self.decoder_traj(
                torch.cat((waypoint, target_point), dim=1), hidden
            )
            trajectory_hidden.append(hidden)
            waypoint = waypoint + self.output_traj(hidden)
            waypoints.append(waypoint)
        trajectory_hidden_tensor = torch.stack(trajectory_hidden, dim=1)

        initial_attention = self.init_att(measurement_feature).view(-1, 1, 8, 29)
        attended = torch.sum(spatial * initial_attention, dim=(2, 3))
        control_feature = self.join_ctrl(
            torch.cat((attended, measurement_feature), dim=1)
        )
        policy = self.policy_head(control_feature)
        action_alpha = self.dist_mu(policy)
        action_beta = self.dist_sigma(policy)
        control_hidden = torch.zeros_like(control_feature)
        future_features = []
        future_alpha = []
        future_beta = []
        recurrent_feature = control_feature
        alpha, beta = action_alpha, action_beta
        for index in range(self.config.pred_len):
            control_hidden = self.decoder_ctrl(
                torch.cat((recurrent_feature, alpha, beta), dim=1), control_hidden
            )
            attention = self.wp_att(
                torch.cat((control_hidden, trajectory_hidden_tensor[:, index]), dim=1)
            ).view(-1, 1, 8, 29)
            attended = torch.sum(spatial * attention, dim=(2, 3))
            merged = self.merge(torch.cat((control_hidden, attended), dim=1))
            recurrent_feature = recurrent_feature + self.output_ctrl(merged)
            policy = self.policy_head(recurrent_feature)
            alpha = self.dist_mu(policy)
            beta = self.dist_sigma(policy)
            future_features.append(recurrent_feature)
            future_alpha.append(alpha)
            future_beta.append(beta)

        return {
            "waypoints": torch.stack(waypoints, dim=1),
            "speed": self.speed_branch(feature_embedding),
            "value_traj": self.value_branch_traj(trajectory_feature),
            "value_ctrl": self.value_branch_ctrl(control_feature),
            "feature_traj": trajectory_feature,
            "feature_ctrl": control_feature,
            "action_alpha": action_alpha,
            "action_beta": action_beta,
            "future_features": future_features,
            "future_alpha": future_alpha,
            "future_beta": future_beta,
        }

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        target_point: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.forward_trajectory(image, state, target_point)

    def set_training_phase(self, phase: str) -> None:
        if phase not in {"state_only", "heads", "layer4", "full"}:
            raise ValueError(f"unknown TCP phase: {phase}")
        self.training_phase = phase
        for parameter in self.parameters():
            parameter.requires_grad = False
        if phase == "state_only":
            for parameter in self.state_head.parameters():
                parameter.requires_grad = True
            return
        trajectory_modules: list[nn.Module] = [
            self.measurements,
            self.join_traj,
            self.speed_branch,
            self.state_head,
            self.decoder_traj,
            self.output_traj,
            self.perception.fc,
        ]
        if phase in {"layer4", "full"}:
            trajectory_modules.append(self.perception.layer4)
        if phase == "full":
            trajectory_modules.extend(
                [
                    self.perception.conv1,
                    self.perception.bn1,
                    self.perception.layer1,
                    self.perception.layer2,
                    self.perception.layer3,
                ]
            )
        for module in trajectory_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True) -> TCPMorai:
        super().train(mode)
        if mode:
            # Parameter freezing alone does not freeze BatchNorm running
            # statistics.  State-only training must keep the complete TCP
            # encoder bit-identical to its initialization checkpoint.
            if getattr(self, "training_phase", None) == "state_only":
                self.perception.eval()
            frozen = [
                module
                for module in (
                    self.join_ctrl,
                    self.value_branch_traj,
                    self.value_branch_ctrl,
                    self.policy_head,
                    self.decoder_ctrl,
                    self.output_ctrl,
                    self.dist_mu,
                    self.dist_sigma,
                    self.init_att,
                    self.wp_att,
                    self.merge,
                )
            ]
            for module in frozen:
                if not any(parameter.requires_grad for parameter in module.parameters()):
                    module.eval()
        return self

    def set_original_training_phase(self, phase: str) -> None:
        """Train all original TCP branches while excluding the MORAI-only state head."""
        if phase not in {"heads", "layer4", "full"}:
            raise ValueError(f"unknown original TCP phase: {phase}")
        self.training_phase = f"original_{phase}"
        for parameter in self.parameters():
            parameter.requires_grad = False
        original_modules: list[nn.Module] = [
            self.measurements,
            self.join_traj,
            self.join_ctrl,
            self.speed_branch,
            self.policy_head,
            self.decoder_ctrl,
            self.output_ctrl,
            self.dist_mu,
            self.dist_sigma,
            self.decoder_traj,
            self.output_traj,
            self.init_att,
            self.wp_att,
            self.merge,
            self.perception.fc,
        ]
        if phase in {"layer4", "full"}:
            original_modules.append(self.perception.layer4)
        if phase == "full":
            original_modules.extend(
                [
                    self.perception.conv1,
                    self.perception.bn1,
                    self.perception.layer1,
                    self.perception.layer2,
                    self.perception.layer3,
                ]
            )
        for module in original_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "perception": sum(
                parameter.numel() for parameter in self.perception.parameters()
            ),
            "trajectory_supported": sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if name.startswith(
                    (
                        "perception.",
                        "measurements.",
                        "join_traj.",
                        "speed_branch.",
                        "state_head.",
                        "decoder_traj.",
                        "output_traj.",
                    )
                )
            ),
        }


def load_reproduction_checkpoint(
    model: TCPMorai, checkpoint: Path
) -> dict[str, Any]:
    # The training entry point only accepts the tensor-only artifact extracted
    # from the legacy Lightning checkpoint.  weights_only avoids importing or
    # executing any third-party checkpoint classes during routine fine-tuning.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload.get("model_state", payload))
    if not isinstance(state, dict):
        raise TypeError(f"{checkpoint}: checkpoint state is not a mapping")
    converted: dict[str, torch.Tensor] = {}
    for raw_name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = str(raw_name)
        for prefix in ("model.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        converted[name] = value
    own = model.state_dict()
    compatible = {
        name: value
        for name, value in converted.items()
        if name in own and own[name].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "loaded_tensors": len(compatible),
        "checkpoint_tensors": len(converted),
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "source": "unofficial TCP reproduction shared from upstream issue",
    }


__all__ = ["TCPConfig", "TCPMorai", "load_reproduction_checkpoint"]
