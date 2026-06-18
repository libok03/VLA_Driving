from __future__ import annotations

import argparse

import numpy as np
import torch

from vla_driving.control.pure_pursuit import PurePursuitController
from vla_driving.models.lightweight_transfuser import LightweightTransFuser
from vla_driving.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)

    model = LightweightTransFuser(**cfg["model"])
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    image_h, image_w = cfg["data"]["image_size"]
    image = torch.zeros(1, 3, image_h, image_w)
    lidar = torch.ones(1, cfg["data"]["lidar_size"])
    pose = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    route = torch.zeros(1, cfg["data"]["route_points"], 2)
    route[0, :, 0] = torch.linspace(1.0, float(cfg["data"]["route_points"]), cfg["data"]["route_points"])

    with torch.no_grad():
        waypoints = model(image, lidar, pose, route)[0].numpy()

    controller = PurePursuitController(**cfg["control"])
    steering = controller.steer_from_waypoints(waypoints)
    print("predicted_waypoints:")
    print(np.round(waypoints, 3))
    print(f"steering_rad: {steering:.4f}")


if __name__ == "__main__":
    main()
