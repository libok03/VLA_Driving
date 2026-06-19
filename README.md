# VLA Driving

Lightweight multimodal driving stack for fixed-route driving with:

- camera image
- 2D LiDAR scan
- pose/state `(x, y, yaw, route_mode)`
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
  "pose": [12.3, 0.0, 1.57],
  "route_mode": "main",
  "route": [[1.0, 0.1], [2.0, 0.2], [3.0, 0.2]],
  "future_waypoints": [[1.0, 0.1, 1.2], [2.0, 0.2, 1.2], [3.0, 0.2, 0.8], [4.0, 0.1, 0.0], [5.0, 0.0, 0.0]]
}
```

All paths are resolved relative to `data_root`.

## Train

```powershell
python scripts/train.py --config configs/base.yaml
```

## Extract Dataset From ROS2 Bag

ROS2 bag files can be converted into the expected dataset layout:

```powershell
python scripts/extract_ros2_bag.py C:\path\to\bag --config configs/base.yaml --output-dir data/ros2_bag --sample-hz 10
```

The extractor reads the ROS2 topic names from `configs/base.yaml` and writes:

```text
data/ros2_bag/
  images/000000.jpg
  lidar/000000.npy
  manifest.jsonl
```

Bag topic meanings:

- `/camera/image_raw` (`sensor_msgs/Image`): front camera image. This is saved as
  `images/000000.jpg` and becomes the model's visual input. Supported encodings are
  `rgb8`, `bgr8`, `mono8`, and `8uc1`.
- `/scan` (`sensor_msgs/LaserScan`): 2D LiDAR range scan. The ranges are saved as
  `lidar/000000.npy` and padded or clipped to `data.lidar_size`.
- `/odom` (`nav_msgs/Odometry`): vehicle world pose. The extractor stores
  `[x, y, yaw]` in the manifest, with yaw computed from the odometry quaternion.
- `/local_route` (`nav_msgs/Path`): optional short route segment in ego coordinates.
  Each pose position should use `x` as forward distance and `y` as lateral offset.
  If missing, `route` is filled with zeros.

`route_mode` is generated from the detected lap count, not read from the bag:

- lap `0` (first lap): `route_mode = 0` (`main`)
- lap `1` and lap `2` (second and third laps): `route_mode = 1` (`shortcut`)

The required extraction topics are `/camera/image_raw`, `/scan`, and `/odom`.
`/local_route` is useful for training, but it is not required for the extractor to
write samples.

If the bag contains expert future waypoint labels as a flattened `std_msgs/Float32MultiArray`
topic, include them for training targets. Each future waypoint is `[x, y, speed]`;
use `speed = 0.0` for stop targets:

```powershell
python scripts/extract_ros2_bag.py C:\path\to\bag --output-dir data/ros2_bag --waypoints-topic /expert/waypoints --require-waypoints
```

If the bag does not contain a label topic, generate `[x, y, speed]` targets from
future `/odom` samples during extraction:

```powershell
python scripts/extract_ros2_bag.py C:\path\to\bag --output-dir data/ros2_bag --generate-waypoints-from-odom --future-step-s 0.2
```

In this mode, the speed label is computed from future odometry displacement over
time. If the vehicle stays still, the generated speed target is `0.0`.

For MCAP bags, add `--storage-id mcap`.

## ROS2 Inference

```powershell
python scripts/infer.py --config configs/base.yaml --checkpoint checkpoints/best.pt
```

`scripts/infer.py` is a ROS2 node. By default it subscribes to:

- `/camera/image_raw` (`sensor_msgs/Image`, `rgb8`, `bgr8`, or `mono8`)
- `/scan` (`sensor_msgs/LaserScan`)
- `/odom` (`nav_msgs/Odometry`)
- `/local_route` (`nav_msgs/Path`, route points in ego coordinates)

It publishes:

- `/vla_driving/steering` (`std_msgs/Float32`)
- `/vla_driving/speed` (`std_msgs/Float32`, target speed in meters per second)
- `/vla_driving/waypoints` (`std_msgs/Float32MultiArray`, flattened `x, y, speed` triples)

Topic names and inference rate are configured in `configs/base.yaml`.

## Lap Handling

Three-lap driving is handled outside the neural model with a directed start/finish gate:

- the vehicle must leave the gate area before the counter is armed
- a lap increments only on gate crossing in the configured forward direction
- cooldown and arming distance prevent double counts near the line
- `route_mode` is appended to the model state
- first lap uses `route_mode = 0`, second and third laps use `route_mode = 1`

Shortcut logic should select the active local route first, then pass those route points to the model. The model predicts waypoints for the selected route rather than deciding the whole race state by itself.

## Route Input

Give the neural model a short local route, not the full track. Keep the full route as world-frame waypoints in a separate route node:

```text
main_route_world:     [[x0, y0], [x1, y1], ...]
shortcut_route_world: [[x0, y0], [x1, y1], ...]
```

At each cycle:

```text
1. choose route_mode from the lap count: first lap main, later laps shortcut
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
