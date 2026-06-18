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
  "route": [[1.0, 0.1], [2.0, 0.2], [3.0, 0.2]],
  "future_waypoints": [[1.0, 0.1], [2.0, 0.2], [3.0, 0.2], [4.0, 0.1], [5.0, 0.0]]
}
```

All paths are resolved relative to `data_root`.

## Train

```powershell
python scripts/train.py --config configs/base.yaml
```

## Inference Smoke Test

```powershell
python scripts/infer.py --config configs/base.yaml
```

The inference script creates synthetic inputs and prints predicted waypoints plus a steering command. Replace the synthetic sensor block with ROS2/subscriber inputs when integrating on the vehicle.
