# VLA Driving

Lightweight multimodal driving stack for fixed-route driving with:

- camera-derived lane/traffic-light perception features
- 2D LiDAR scan
- pose/state `(x, y, yaw, lap_index)`
- optional route points in ego coordinates

The first baseline is TransFuser-inspired but intentionally small: separate encoders for each modality, compact fusion, waypoint prediction, and a Pure Pursuit controller. Fusion can be either `mlp` or a tiny token-level `transformer`.

## Architecture

```text
Camera     -> Canny lane edges + traffic-light features -> MLP ----\
2D LiDAR   -> 1D CNN -----------------------------------------------> MLP or tiny Transformer -> future waypoints
Pose/Route -> MLP --------------------------------------------------/

future waypoints [x, y, speed] -> Pure Pursuit + speed command
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
  "perception": [0.0, 0.1, 0.2],
  "lidar": "lidar/000001.npy",
  "pose": [12.3, 0.0, 1.57],
  "lap_index": 1,
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

- `/usb_cam/image_raw/front` (`sensor_msgs/Image`): front camera image. This is saved as
  `images/000000.jpg` for inspection, then converted into a compact `perception`
  vector for the model. The perception vector contains Canny lane-edge summaries and
  traffic-light state features. Supported encodings are `rgb8`, `bgr8`, `mono8`,
  and `8uc1`.
- `/vla_driving/perception_features` (`std_msgs/Float32MultiArray`): optional
  precomputed camera feature vector. If this topic is recorded instead of the raw
  camera image, the extractor uses it directly and does not require image frames in
  the bag.
- `/scan` (`sensor_msgs/LaserScan`): 2D LiDAR range scan. The ranges are saved as
  `lidar/000000.npy` and padded or clipped to `data.lidar_size`.
- `/scan_odom_map` (`nav_msgs/Odometry`): vehicle world pose. The extractor stores
  `[x, y, yaw]` in the manifest, with yaw computed from the odometry quaternion.
- `/local_route` (`nav_msgs/Path`): optional short route segment in ego coordinates.
  Each pose position should use `x` as forward distance and `y` as lateral offset.
  If missing, the extractor can generate `route` from future odometry with
  `--generate-route-from-odom`.

`lap_index` is generated from odometry, not read from the bag:

- lap trigger center: `[-16.886848684798856, 29.194995142160792]`
- lap changes when the vehicle enters within `3.0 m` of that center
- `lap_index = 0` before the first trigger pass
- `lap_index = 1`, `2`, `3` after each trigger pass
- shortcut state is not encoded because shortcut availability is random

The required extraction topics for the current training bag are `/scan`,
`/scan_odom_map`, and either `/usb_cam/image_raw/front` or
`/vla_driving/perception_features`. `/local_route` is not required when route is
generated from the driven future trajectory.

The VLA model does not consume raw RGB images directly. It consumes:

```text
perception: [32]
lidar: [360]
state: [x, y, yaw, lap_index]
route: [10, 2]
```

The default `perception` vector is generated from the camera frame without YOLO:

```text
Canny lane-edge/Hough-line statistics
HSV traffic-light scores: red/yellow/green/total
HSV traffic-light hard state: unknown/red/yellow/green
traffic-light confidence
```

YOLO object detection is optional and disabled by default. Enable it only if the
simple HSV rule is not enough for the track camera and lighting.

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

If the bag does not contain `/local_route`, generate the route input from the same
future odometry trajectory:

```powershell
python scripts/extract_ros2_bag.py C:\path\to\bag --output-dir data/ros2_bag --generate-route-from-odom --route-step-s 0.2
```

For a driving-only bag, both generated labels can be enabled together:

```powershell
python scripts/extract_ros2_bag.py C:\path\to\bag --output-dir data/ros2_bag --generate-route-from-odom --generate-waypoints-from-odom
```

For MCAP bags, add `--storage-id mcap`.

## Camera Feature Publisher

To keep raw camera frames out of the training bag, run the feature publisher while
Unity is publishing `/usb_cam/image_raw/front`:

```bash
PYTHONPATH=src python3 scripts/publish_camera_features.py --config configs/base.yaml
```

Then record the feature topic instead of the camera topic:

```bash
ros2 bag record /vla_driving/perception_features /scan /scan_odom_map
```

## ROS2 Inference

```powershell
python scripts/infer.py --config configs/base.yaml --checkpoint checkpoints/best.pt
```

`scripts/infer.py` is a ROS2 node. By default it subscribes to:

- `/usb_cam/image_raw/front` (`sensor_msgs/Image`, converted to Canny lane + traffic-light features)
- `/scan` (`sensor_msgs/LaserScan`)
- `/scan_odom_map` (`nav_msgs/Odometry`)
- `/local_route` (`nav_msgs/Path`, route points in ego coordinates)

It publishes:

- `/vla_driving/steering` (`std_msgs/Float32`)
- `/vla_driving/speed` (`std_msgs/Float32`, target speed in meters per second)
- `/vla_driving/waypoints` (`std_msgs/Float32MultiArray`, flattened `x, y, speed` triples)

Topic names and inference rate are configured in `configs/base.yaml`.

## Lap Handling

Three-lap driving is handled outside the neural model with a radius trigger:

- the trigger center is configured in `configs/base.yaml`
- the vehicle must first leave the trigger radius before the counter is armed
- a lap increments when the vehicle comes back within the trigger radius
- `lap_index` is appended to the model state as `[x, y, yaw, lap_index]`
- shortcut availability is not encoded because it is random on this track

## Route Input

Give the neural model a short local route, not the full track. During offline
training, `route` can be generated directly from future `/scan_odom_map` poses in
the same bag. It is therefore the expert's driven future path in ego coordinates.

The ROS2 inference node expects `/local_route` as `nav_msgs/Path` where each pose position is already in ego coordinates:

```text
x = forward distance from vehicle
y = left/right offset from vehicle
z = unused
```

This keeps the model focused on short-horizon driving. Lap counting remains a
separate deterministic signal, and random shortcut availability is learned only
through the driven future route/waypoint labels present in the bag.
