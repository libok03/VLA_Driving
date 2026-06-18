# VLA Driving

Lightweight multimodal driving stack for fixed-route driving with:

- camera image
- 2D LiDAR scan
- pose `(x, y, z, yaw)`
- optional route points in ego coordinates

The first baseline is TransFuser-inspired but intentionally small: separate encoders for each modality, compact fusion, waypoint prediction, and a Pure Pursuit controller. Fusion can be either `mlp` or a tiny token-level `transformer`.

## Architecture

```text
Image      -> lightweight CNN ----\
2D LiDAR   -> 1D CNN ---------------> MLP or tiny Transformer -> future waypoints
Pose/Route -> MLP ----------------/

future waypoints -> Pure Pursuit -> steering command
```

## Repository Layout

```text
configs/                 training and model configuration
scripts/                 train and inference entrypoints
src/vla_driving/
  control/               classical controllers
  data/                  dataset loaders and transforms
  models/                lightweight fusion model
  planning/              lap counting and route-state helpers
  utils/                 geometry and config helpers
tests/                   smoke tests
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Expected Dataset Format

Create a metadata JSONL file where each line contains one sample:

```json
{
  "image": "images/000001.jpg",
  "lidar": "lidar/000001.npy",
  "pose": [12.3, 0.0, 0.0, 1.57],
  "lap_progress": 0.42,
  "laps_remaining": 2,
  "route_mode": "main",
  "route": [[1.0, 0.1], [2.0, 0.2], [3.0, 0.2]],
  "future_waypoints": [[1.0, 0.1], [2.0, 0.2], [3.0, 0.2], [4.0, 0.1], [5.0, 0.0]]
}
```

All paths are resolved relative to `data_root`.

## Train

```powershell
python scripts/train.py --config configs/base.yaml
```

## ROS2 Inference

```powershell
python scripts/infer.py --config configs/base.yaml --checkpoint checkpoints/best.pt
```

`scripts/infer.py` is a ROS2 node. By default it subscribes to:

- `/camera/image_raw` (`sensor_msgs/Image`, `rgb8`, `bgr8`, or `mono8`)
- `/scan` (`sensor_msgs/LaserScan`)
- `/odom` (`nav_msgs/Odometry`)
- `/local_route` (`nav_msgs/Path`, route points in ego coordinates)
- `/vla_driving/lap_progress` (`std_msgs/Float32`, `0.0` to `1.0`)
- `/vla_driving/route_mode` (`std_msgs/Int32`, `0` main, `1` shortcut)

It publishes:

- `/vla_driving/steering` (`std_msgs/Float32`)
- `/vla_driving/waypoints` (`std_msgs/Float32MultiArray`, flattened `x, y` pairs)

Topic names and inference rate are configured in `configs/base.yaml`.

## Lap Handling

Three-lap driving is handled outside the neural model with a directed start/finish gate:

- the vehicle must leave the gate area before the counter is armed
- a lap increments only on gate crossing in the configured forward direction
- cooldown and minimum lap progress prevent double counts near the line
- `lap_progress`, `laps_remaining`, and `route_mode` are appended to the model state

Shortcut logic should select the active local route first, then pass those route points to the model. The model predicts waypoints for the selected route rather than deciding the whole race state by itself.

## Route Input

Give the neural model a short local route, not the full track. Keep the full route as world-frame waypoints in a separate route node:

```text
main_route_world:     [[x0, y0], [x1, y1], ...]
shortcut_route_world: [[x0, y0], [x1, y1], ...]
```

At each cycle:

```text
1. choose route_mode: main or shortcut
2. find nearest waypoint from current pose
3. take the next N points, usually 10 to 30
4. transform those points into vehicle/ego coordinates
5. publish them as /local_route
```

The ROS2 inference node expects `/local_route` as `nav_msgs/Path` where each pose position is already in ego coordinates:

```text
x = forward distance from vehicle
y = left/right offset from vehicle
z = unused
```

This keeps the model focused on short-horizon driving. The route node owns global decisions like lap progress, shortcut selection, and merge timing.
