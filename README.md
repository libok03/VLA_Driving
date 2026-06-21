# VLA Driving

ROS2/Unity 기반 Xycar 주행 학습 프로젝트입니다. 현재 가장 잘 동작하는 경로는 **작은 차선 추종 모델**입니다.

```text
camera feature[32] + lidar summary[5]
-> steering 직접 예측
-> /xycar_motor.angle

speed는 현재 고정 또는 teleop label로 별도 수집
```

기존 waypoint VLA 모델도 남아 있지만, 차선 추종 안정화에는 `lane_steering` 파이프라인을 먼저 사용합니다.

## 현재 권장 구조

```text
Unity camera
  -> scripts/publish_camera_features.py
  -> /vla_driving/perception_features

/scan
  -> 5구간 LiDAR 요약

/vla_driving/perception_features + /scan
  -> LaneSteeringMLP
  -> /vla_driving/steering
  -> /xycar_motor.angle
```

현재 모델 입력:

```text
perception_features[32]
lidar_summary[5] = front, front_left, front_right, left, right
```

현재 모델 출력:

```text
steering
```

현재 실시간 주행 속도:

```text
configs/lane_steering.yaml의 fixed_speed 사용
```

## 주요 파일

```text
configs/lane_steering.yaml          현재 권장 차선 추종 config
scripts/publish_camera_features.py  카메라 -> perception feature publisher
scripts/train_lane_steering.py      steering 직접 예측 모델 학습
scripts/infer_lane_steering.py      ROS2 실시간 차선 추종 inference
scripts/teleop_xycar_motor.py       /xycar_motor 직접 조종 및 bag label 수집용
scripts/inspect_lane_steering_data.py
                                    steering label 분포 확인

scripts/extract_ros2_bag.py         ROS2 bag -> dataset 추출
scripts/train.py                    기존 waypoint VLA 학습
scripts/infer.py                    기존 waypoint VLA inference
```

## 설치

Windows에서 학습할 때:

```powershell
cd C:\Users\markp\OneDrive\Desktop\VLA_Driving
pip install -e .
```

WSL/ROS2에서 실행할 때:

```bash
cd ~/VLA_Driving
source /opt/ros/humble/setup.bash
source ~/xycar_ws/install/setup.bash
```

## ROS2 실행 순서

### 1. Unity endpoint

터미널 1:

```bash
source /opt/ros/humble/setup.bash
source ~/xycar_ws/install/setup.bash
ros2 run ros_tcp_endpoint default_server_endpoint
```

그 다음 Unity Play를 한 번만 누릅니다. Play/Stop을 반복하면 `ros_tcp_endpoint`가 `InvalidHandle`로 죽을 수 있습니다.

문제가 생기면:

```bash
pkill -f ros_tcp_endpoint || true
pkill -f default_server_endpoint || true
ros2 daemon stop
ros2 daemon start
```

### 2. 카메라 feature publisher

터미널 2:

```bash
cd ~/VLA_Driving
source /opt/ros/humble/setup.bash
source ~/xycar_ws/install/setup.bash

PYTHONPATH=src:$PYTHONPATH python3 scripts/publish_camera_features.py \
  --config configs/lane_steering.yaml
```

출력 topic:

```text
/vla_driving/perception_features
```

### 3. 차선 추종 inference

터미널 3:

```bash
cd ~/VLA_Driving
source /opt/ros/humble/setup.bash
source ~/xycar_ws/install/setup.bash

PYTHONPATH=src:$PYTHONPATH python3 scripts/infer_lane_steering.py \
  --config configs/lane_steering.yaml \
  --checkpoint checkpoints/lane_steering/best.pt
```

확인:

```bash
ros2 topic echo /xycar_motor --once
ros2 topic echo /vla_driving/steering --once
ros2 topic info /xycar_motor -v
```

정상이라면 `/xycar_motor` publisher가 `vla_lane_steering_inference`로 보입니다.

## ROS teleop으로 expert bag 찍기

Unity 내부 WASD는 `/xycar_motor`로 publish되지 않을 수 있습니다. 모델이 steering/speed를 직접 배우게 하려면 ROS에서 `/xycar_motor`를 publish하면서 운전해야 합니다.

터미널 3 또는 별도 터미널:

```bash
cd ~/VLA_Driving
source /opt/ros/humble/setup.bash
source ~/xycar_ws/install/setup.bash

PYTHONPATH=src:$PYTHONPATH python3 scripts/teleop_xycar_motor.py
```

현재 기본 조작:

```text
w / k   -> speed 10
l       -> speed 20
s/space -> speed 0
a       -> steering 왼쪽으로 80씩 변화
d       -> steering 오른쪽으로 80씩 변화
x       -> steering 0

max steering: +/-100
키를 떼면 steering이 3씩 천천히 0으로 복귀
publish rate: 80 Hz
```

실행 시 값 조절:

```bash
PYTHONPATH=src:$PYTHONPATH python3 scripts/teleop_xycar_motor.py \
  --steer-step 60 \
  --center-step 2 \
  --max-angle 100 \
  --low-speed 10 \
  --high-speed 20
```

## Bag 녹화

### 현재 steering/speed label까지 저장하는 권장 bag

```bash
ros2 bag record \
  /vla_driving/perception_features \
  /scan \
  /scan_odom_map \
  /xycar_motor
```

필수 topic:

```text
/vla_driving/perception_features  모델 input
/scan                             모델 input
/scan_odom_map                    odom 기반 label/검증용
/xycar_motor                      expert steering/speed label
```

녹화 중 `/xycar_motor`가 실제로 찍히는지 확인:

```bash
ros2 topic echo /xycar_motor --once
```

bag 확인:

```bash
source /opt/ros/humble/setup.bash
ros2 bag info ~/rosbag2_YYYY_MM_DD-HH_MM_SS
```

`/xycar_motor`가 없으면 expert steering/speed 직접 학습에는 부족합니다.

## 기존 bag으로 가능한 것

기존 bag이 아래 3개만 가진 경우:

```text
/vla_driving/perception_features
/scan
/scan_odom_map
```

가능:

```text
future odom trajectory에서 steering label 추정
future odom displacement에서 speed label 추정
차선 추종 학습
```

어려움:

```text
실제 expert 조향/속도 직접 학습
정지/출발/보행자/앞차 대응을 명확한 command label로 학습
```

그래서 앞으로는 `/xycar_motor`를 반드시 포함해서 bag을 찍는 것을 권장합니다.

## Dataset 추출

ROS2 bag을 dataset으로 변환:

```bash
cd ~/VLA_Driving
source /opt/ros/humble/setup.bash
source ~/xycar_ws/install/setup.bash

PYTHONPATH=src:$PYTHONPATH python3 scripts/extract_ros2_bag.py \
  ~/rosbag2_YYYY_MM_DD-HH_MM_SS \
  --config configs/lane_steering.yaml \
  --output-dir data/my_bag \
  --sample-hz 10 \
  --generate-waypoints-from-odom
```

기존 waypoint VLA용 route까지 만들 때:

```bash
PYTHONPATH=src:$PYTHONPATH python3 scripts/extract_ros2_bag.py \
  ~/rosbag2_YYYY_MM_DD-HH_MM_SS \
  --config configs/vla_15laps_no_route.yaml \
  --output-dir data/my_bag \
  --sample-hz 10 \
  --generate-route-from-odom \
  --generate-waypoints-from-odom
```

## Train/Val split 만들기

여러 extracted bag을 합쳐서 train/val split 생성:

```bash
python3 scripts/build_dataset_split.py \
  --output-dir data/vla_dataset \
  --train data/bag_1 data/bag_2 data/bag_3 \
  --val data/bag_val
```

생성 결과:

```text
data/vla_dataset/train.jsonl
data/vla_dataset/val.jsonl
```

## Lane steering 학습

학습 전 label 분포 확인:

```powershell
cd C:\Users\markp\OneDrive\Desktop\VLA_Driving
$env:PYTHONPATH="src"
python scripts\inspect_lane_steering_data.py --config configs\lane_steering.yaml
```

100개 overfit 테스트:

```powershell
$env:PYTHONPATH="src"
python scripts\train_lane_steering.py --config configs\lane_steering.yaml --overfit-samples 100
```

100개에서 train loss가 충분히 내려가지 않으면 아래 중 하나를 의심합니다:

```text
sensor와 label timestamp 불일치
steering 부호 반대
label scale 문제
feature가 주행 판단에 부족
```

전체 학습:

```powershell
$env:PYTHONPATH="src"
python scripts\train_lane_steering.py --config configs\lane_steering.yaml
```

결과 checkpoint:

```text
checkpoints/lane_steering/best.pt
```

Windows에서 학습한 checkpoint를 WSL로 복사:

```bash
cd ~/VLA_Driving
mkdir -p checkpoints/lane_steering
cp /mnt/c/Users/markp/OneDrive/Desktop/VLA_Driving/checkpoints/lane_steering/best.pt \
   checkpoints/lane_steering/best.pt
```

## 현재 lane steering config

핵심 설정:

```yaml
model:
  perception_dim: 32
  lidar_summary_dim: 5
  hidden_dim: 64
  output_scale: 50.0

control:
  fixed_speed: 10.0
  steering_output_gain: 50.0
  motor_max_angle: 50.0
```

`fixed_speed`는 `/xycar_motor.speed` command 단위입니다. Unity 화면 km/h와 1:1이 아닐 수 있습니다.

## 기존 waypoint VLA

초기 구조:

```text
perception[32] + lidar[360] + state[x,y,yaw,lap] + route[10,2]
-> future_waypoints[5,3]
-> Pure Pursuit 또는 lateral steering 변환
-> /xycar_motor
```

이 구조는 현재 트랙에서 waypoint가 한쪽으로 치우치거나 평균화되어 차선 추종이 불안정했습니다. 그래서 현재는 `lane_steering`을 기본 경로로 사용합니다.

기존 모델 관련 파일:

```text
configs/base.yaml
configs/vla_15laps.yaml
configs/vla_15laps_no_route.yaml
scripts/train.py
scripts/infer.py
src/vla_driving/models/lightweight_transfuser.py
```

## Troubleshooting

### ros_tcp_endpoint가 `InvalidHandle`로 죽음

증상:

```text
rclpy._rclpy_pybind11.InvalidHandle: cannot use Destroyable because destruction was requested
Publisher already registered for provided node name
Disconnected from 192.168...
```

원인:

```text
Unity가 ROS TCP endpoint에 끊겼다 다시 붙음
Unity Play/Stop 반복
Unity 씬에 ROS publisher/subscriber가 중복 등록
endpoint 중복 실행
```

초기화:

```bash
pkill -f ros_tcp_endpoint || true
pkill -f default_server_endpoint || true
pkill -f teleop_xycar_motor.py || true
pkill -f publish_camera_features.py || true
ros2 daemon stop
ros2 daemon start
```

그 다음 endpoint를 먼저 켜고 Unity Play를 한 번만 누릅니다.

### teleop이 publish되는데 차가 안 움직임

확인:

```bash
ros2 topic echo /xycar_motor --once
ros2 topic info /xycar_motor -v
```

정상 조건:

```text
Publisher count >= 1
Subscription count >= 1
```

`Subscription count`가 0이면 Unity가 `/xycar_motor`를 subscribe하지 않는 상태입니다.

수동 publish 테스트:

```bash
ros2 topic pub --once /xycar_motor xycar_msgs/msg/XycarMotor "{angle: 0.0, speed: 10.0}"
```

이것도 안 움직이면 모델/teleop 문제가 아니라 Unity subscriber 또는 endpoint 문제입니다.

## 추천 개발 순서

1. `scripts/teleop_xycar_motor.py`로 `/xycar_motor` 조종이 되는지 확인
2. `/vla_driving/perception_features`, `/scan`, `/scan_odom_map`, `/xycar_motor` 포함 bag 녹화
3. `ros2 bag info`로 topic count 확인
4. dataset 추출
5. 100개 overfit
6. 전체 학습
7. WSL로 checkpoint 복사
8. `scripts/infer_lane_steering.py`로 실시간 테스트

