# VLA Driving

이 저장소는 두 개의 연구 트랙을 함께 관리합니다.

1. **ROS2 motor-control**: 카메라, 2D LiDAR, pose로 `/xycar_motor`의 조향각과 속도를 예측하는 경량 모델
2. **MORAI V17 planner**: 3대 카메라와 VLP16 BEV, 단일 목표점으로 30 m 공간 경로와 상태를 예측하고 MPC에 전달하는 멀티모달 플래너

raw bag, 변환 NPZ, 학습 출력, checkpoint는 용량 및 데이터 관리 문제로 Git에 포함하지 않습니다. 재현용 소스와 변환·학습 정책은 포함합니다.

## MORAI V17: 현재 선택한 구조

`src/multimodal_planner_v17_spatial30/`는 MORAI용 최종 실험 계보입니다. 입력은 최근 1초의 센서 history와 **30 m goal point**뿐입니다.

```text
Front camera  [B, 5, 3, 360, 640] ─┐
Left camera   [B, 5, 3, 240, 320] ─┼─ Shared pretrained ResNet18 + camera-ID
Right camera  [B, 5, 3, 240, 320] ─┘
VLP16 BEV     [B, 5, 3, 256, 256] ─── Light BEV CNN
                                         ↓
                            8 sensor-conditioned spatial tokens
                                         ↓
                           shared temporal GRU (history=5)
                                         ↓
                     Goal-point cross-attention / fused context
                                         ↓
          ┌──────── DRIVE path: 6 × (x,y) at fixed stations ────────┐
          ├──────── AVOID path: 6 × (x,y) at fixed stations ────────┤
          ├──────── absolute speed: 6 stations (Beta mean) ─────────┤
          └──────── action logits: DRIVE / STOP / AVOID ────────────┘
```

공간 station은 **3, 6, 10, 15, 22, 30 m**이다. 경로는 시간 후 차량 위치가 아니라, ego 기준 route-progress 상의 기하 형상이다. 따라서 planner가 경로 형상을 만들고, 외부 state machine이 상태별 후보를 선택한 뒤 smoothing·0.1 m resampling을 수행해 MPC에 전달한다. STOP은 경로 회귀 대상이 아니라 분류 결과를 통해 목표 속도 0으로 강제한다.

### 왜 이전 버전들이 충분히 안정적이지 않았나

V1~V16에서 의미 있는 실험 결과는 있었지만, 폐루프 주행에 바로 쓰기에는 아래 문제가 남아 있었다.

| 관찰된 문제 | 원인 | V17에서의 변경 |
| --- | --- | --- |
| 경로가 앞에서 꺾이거나 뒤로 갔다가 다시 앞으로 감 | 시간 기반 waypoint를 예측하면서 current speed를 입력에서 제거했다. 시간 후 종방향 위치는 현재 속도 없이 유일하게 결정되지 않아 label이 서로 충돌했다. | 4초/20점 시간 궤적을 폐기하고, 속도와 독립적인 고정 공간 station 6점으로 전환했다. |
| local route를 잘 외우지만 새 구간·곡선에서 불안정 | local route/MGeo/ego state가 모델 입력일 때 route 형상을 복사하는 shortcut이 생겼다. | local route는 **offline label 생성용**으로만 사용하고, 모델 입력에서는 제거했다. 입력은 camera 3대, LiDAR BEV, 30 m goal point다. |
| mode 후보가 frame마다 바뀌어 제어가 흔들릴 수 있음 | 다중 모드 trajectory의 argmax 선택과 경로 회귀를 모델 내부에서 결합했다. | DRIVE·AVOID path와 상태 분류를 분리했다. runtime state machine이 queue/hysteresis 및 안전 조건으로 적용 여부를 결정한다. |
| STOP 데이터가 path 회귀를 훼손 | 정지 상태에는 앞으로의 유효 공간 경로가 없는데 같은 회귀 loss에 넣으면 DRIVE/AVOID target과 충돌한다. | STOP은 action classification과 speed=0 제어에만 사용하고, path loss에서는 제외한다. |
| AVOID가 장애물 위치를 외울 가능성 | 반복된 MORAI 장애물·접근 위치가 적고, AVOID 비율이 DRIVE보다 작다. | camera+LiDAR를 유지하고, 장애물 종류·위치·접근 속도를 다양화한 bag을 추가했다. 성능 평가는 AVOID를 별도 지표로 분리한다. |

즉 V17의 핵심은 “로컬 경로를 그대로 복사하는 residual 모델”이 아니라, **목표점 방향으로 실제 공간 경로를 생성하되 상태 판단은 별도 안전 계층에서 적용**하는 구조다. 이것이 local route 과의존과 time/speed 모순을 동시에 제거하기 위한 변경이다.

### V17 검증 결과

2026-08-20 기준 MORAI validation 결과(고정 3/6/10/15/22/30 m 축)는 아래와 같다. 이는 open-loop 검증이며, 최종 판단은 bag replay와 MPC 폐루프 시험으로 한다.

| 항목 | 결과 |
| --- | ---: |
| 전체 path coordinate MAE | 0.333 m |
| DRIVE ADE / lateral MAE | 0.590 m / 0.568 m |
| AVOID ADE / lateral MAE | 0.535 m / 0.495 m |
| speed MAE | 1.715 m/s (약 6.2 km/h) |
| action accuracy / macro-F1 | 98.23% / 94.77% |
| AVOID precision / recall | 78.63% / 98.92% |

AVOID recall은 높지만 DRIVE를 AVOID로 판정하는 보수적 오탐이 아직 존재한다. 따라서 실제 runtime에서는 raw argmax를 즉시 적용하지 않고 state queue, confidence threshold, TTC 기반 safety monitor를 함께 사용해야 한다. 이 수치는 차선 이탈·충돌이 없다는 보증이 아니다.

### 재현

```bash
pip install -e .

# unit-level fixed-station contract
python -m unittest multimodal_planner_v17_spatial30.test_v17

# dataset manifest와 checkpoint 경로를 준비한 뒤
python -m multimodal_planner_v17_spatial30.train --help
```

V17은 `multimodal_planner_v9`, `v10`, `v13_goal_trajectory`, `v16_30m_candidates`의 공통 encoder·loss·metric 계보를 사용한다. 이 때문에 해당 소스도 `src/`에 함께 포함했다. raw MORAI/Bench2Drive 데이터 변환 결과와 pretrained weights는 별도 저장소 또는 로컬 SSD에서 관리한다.

---

## ROS2 motor-control 트랙

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
