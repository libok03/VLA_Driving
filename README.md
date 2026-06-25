# VLA Driving

카메라 원본 이미지, LiDAR, 현재 위치를 받아서 `/xycar_motor`의 조향각과 속도를 직접 예측하는 ROS2 주행 학습 코드입니다.

현재 기준 입력은 아래 네 가지입니다.

```text
/usb_cam/image_raw/front    카메라 원본 이미지
/scan                       360개 LiDAR 거리값
/scan_odom_map              현재 위치와 yaw
/xycar_motor                사람이 조종한 정답 angle/speed
```

모델 구조는 아래와 같습니다.

```text
image[3,224,224] -> pretrained ResNet18
lidar[360]
pose[4] = relative_x, relative_y, sin(yaw), cos(yaw)

최근 5프레임 -> GRU -> [steering, speed]
```

실시간 실행 시 모델 출력은 `/xycar_motor`로 publish됩니다.

## 파일

```text
configs/motor_control_temporal_camera.yaml
    학습과 추론 설정

scripts/extract_sqlite_motor_bag.py
    ROS2 sqlite bag에서 image, lidar, pose, motor label 추출

scripts/train_motor_control_temporal_camera.py
    ResNet18 + GRU 모델 학습

scripts/infer_motor_control_temporal_camera.py
    ROS2 topic을 받아 실시간으로 /xycar_motor publish

src/vla_driving/data/motor_temporal_image_dataset.py
    이미지 sequence dataset

src/vla_driving/models/motor_temporal_camera.py
    ResNet18 image encoder + GRU 모델
```

## 설치

repo 루트에서 실행합니다.

```bash
pip install -e .
```

ROS2 환경에서 실행할 때는 매 터미널마다 source합니다.

```bash
source /opt/ros/humble/setup.bash
source ~/xycar_ws/install/setup.bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## 1. Bag Topic 확인

bag 안에 필요한 topic이 있는지 먼저 확인합니다.

```bash
sqlite3 BAG.db3 "select name,type from topics;"
```

필요 topic:

```text
/usb_cam/image_raw/front
/scan
/scan_odom_map
/xycar_motor
```

## 2. Bag 추출

단일 bag:

```bash
python scripts/extract_sqlite_motor_bag.py raw_bags/kookmin/driving_data \
  --output-dir data/motor_camera_pose/extracted/driving_data \
  --sample-hz 10 \
  --image-topic /usb_cam/image_raw/front \
  --pose-topic /scan_odom_map
```

여러 bag:

```bash
rm -rf data/motor_camera_pose
mkdir -p data/motor_camera_pose/extracted

for bag in raw_bags/kookmin/driving_data*; do
  [ -d "$bag" ] || continue
  name=$(basename "$bag")
  python scripts/extract_sqlite_motor_bag.py "$bag" \
    --output-dir "data/motor_camera_pose/extracted/$name" \
    --sample-hz 10 \
    --image-topic /usb_cam/image_raw/front \
    --pose-topic /scan_odom_map
done
```

추출 결과:

```text
data/motor_camera_pose/extracted/<bag_name>/manifest.jsonl
data/motor_camera_pose/extracted/<bag_name>/images/*.jpg
data/motor_camera_pose/extracted/<bag_name>/lidar/*.npy
```

## 3. Train/Val Split 생성

```bash
mapfile -t dirs < <(find data/motor_camera_pose/extracted -mindepth 1 -maxdepth 1 -type d | sort -V)

n=${#dirs[@]}
val_count=$(( n / 5 ))
[ "$val_count" -lt 1 ] && val_count=1
train_count=$(( n - val_count ))

python scripts/build_dataset_split.py \
  --output-dir data/motor_camera_pose \
  --train "${dirs[@]:0:$train_count}" \
  --val "${dirs[@]:$train_count}"
```

생성 결과:

```text
data/motor_camera_pose/train.jsonl
data/motor_camera_pose/val.jsonl
```

## 4. 학습

설정 파일에서 dataset 경로와 checkpoint 경로를 맞춥니다.

```yaml
data:
  data_root: data/motor_camera_pose
  train_manifest: data/motor_camera_pose/train.jsonl
  val_manifest: data/motor_camera_pose/val.jsonl

model:
  image_size: [224, 224]
  sequence_length: 5
  camera_pretrained: true

train:
  batch_size: 32

checkpoint_dir: checkpoints/motor_control_temporal_camera
```

학습 실행:

```bash
python scripts/train_motor_control_temporal_camera.py \
  --config configs/motor_control_temporal_camera.yaml
```

결과:

```text
checkpoints/motor_control_temporal_camera/best.pt
```

GPU 메모리가 부족하면 batch size를 낮춥니다.

```bash
sed -i 's/batch_size: 32/batch_size: 16/' configs/motor_control_temporal_camera.yaml
```

pretrained ResNet18은 처음 실행할 때 torchvision weight를 다운로드할 수 있습니다. 서버에 인터넷이 없으면 weight cache를 옮기거나, 임시 확인용으로 `camera_pretrained: false`를 사용할 수 있습니다.

## 5. ROS2 실시간 추론

```bash
source /opt/ros/humble/setup.bash
source ~/xycar_ws/install/setup.bash
export PYTHONPATH=$PWD/src:$PYTHONPATH

python scripts/infer_motor_control_temporal_camera.py \
  --config configs/motor_control_temporal_camera.yaml \
  --checkpoint checkpoints/motor_control_temporal_camera/best.pt
```

구독:

```text
/usb_cam/image_raw/front
/scan
/scan_odom_map
```

발행:

```text
/xycar_motor
/vla_driving/steering
/vla_driving/speed
```

확인:

```bash
ros2 topic echo /xycar_motor --once
ros2 topic info /xycar_motor -v
```

## 메모

위치 입력은 절대 좌표를 그대로 넣지 않고 첫 프레임 기준 상대 위치로 바꿔 사용합니다.

```text
relative_x = x - first_x
relative_y = y - first_y
sin(yaw)
cos(yaw)
```

LiDAR는 5개 요약값이 아니라 `/scan`의 360개 값을 사용합니다.
