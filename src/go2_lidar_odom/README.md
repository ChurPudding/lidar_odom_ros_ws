# go2_lidar_odom

Unitree Go2의 L1 라이다 점군**만**으로 로봇 위치를 구하는 ROS2(Humble) 노드입니다.
IMU와 다리 정보는 쓰지 않습니다.

- 방법: 3D 점-평면 ICP, **3자유도 $(x, y, \psi)$**, 스캔-대-지도
- 출력: 출발점 기준 자세(`nav_msgs/Odometry`, TF), 경로, 정합된 점군, 칸 지도

## 1. 빌드

```bash
sudo apt install python3-numpy python3-scipy      # SciPy 가 없으면 매우 느림
cd ~/go2/lidar_odom_ros_ws
colcon build --symlink-install
source install/setup.bash
```

## 2. 실행

```bash
# L1 이 계속 점군을 내도록 (다른 터미널에 켜 둠)
ros2 topic echo /sportmodestate > /dev/null

ros2 launch go2_lidar_odom lidar_odom.launch.py
ros2 launch go2_lidar_odom lidar_odom.launch.py normal_k:=25          # k 바꾸기
ros2 launch go2_lidar_odom lidar_odom.launch.py params_file:=/내/설정.yaml
```

- **처음 14 프레임(약 0.9초) 동안 로봇을 세워 두세요.** 그 자리가 원점이고, 그때 로봇이 바라보는 방향이 $+x$입니다.
- 처음부터 다시 시작하려면: `ros2 service call /lidar_odom/reset std_srvs/srv/Trigger`
- 파라미터는 **실행할 때만** 정해집니다. 실행 중 `ros2 param set`은 거부됩니다.

## 3. 토픽과 TF

이름은 모두 `config/lidar_odom.yaml`에서 바꿉니다. 빈 문자열(`""`)이면 그 출력은 내지 않습니다.

| 이름 (기본값) | 형식 | 내용 |
|---|---|---|
| `/utlidar/cloud` (받음) | PointCloud2 | L1 점군 |
| `/lidar_odom/odometry` | Odometry | 자세 + 직전 프레임과의 차이로 구한 몸체 속도 |
| TF `lidar_odom` → `lidar_odom_base` | TF | 같은 자세. 로봇의 기존 `odom → base_link`와 겹치지 않게 다른 이름을 씀 |
| `/lidar_odom/path` | Path | 지나온 경로 (1초마다) |
| `/lidar_odom/cloud_registered` | PointCloud2 | 이번 프레임 점을 `lidar_odom` 좌표로 옮긴 것. `cloud_out_mode: all`이면 바닥·낮은 장애물까지 전부, `used`면 위치 계산에 쓴 점만 |
| `/lidar_odom/map` | PointCloud2 | 위치 계산에 쓰는 칸 지도 (1초마다) |

`Odometry`의 공분산은 0으로 비워 둡니다. 필터(EKF 등)에 넣으려면 따로 정해야 합니다.

## 4. 계산 (한 프레임)

1. **점 고르기:** L1 점을 몸체 좌표로 옮긴 뒤, 높이와 거리로 고릅니다.
$$\mathbf p^B = R_{LB}^{\top}\mathbf p^L + \text{LEVER},\qquad r=\lVert\mathbf p^L\rVert > 0.6\ \text{m},\qquad 0 \le z^B \le 2.0\ \text{m}$$
2. **짝짓기:** 직전 자세로 옮긴 점마다 가장 가까운 지도 점 $\mathbf q_i$를 찾고, 0.5 m보다 먼 짝은 버립니다.
3. **법선:** $\mathbf q_i$ 주변 지도 점 $k$개(기본 40)의 공분산에서 가장 작은 고유값의 고유벡터를 씁니다.
$$C_i=\frac1k\sum_{j=1}^{k}(\mathbf q_j-\bar{\mathbf q})(\mathbf q_j-\bar{\mathbf q})^{\top}$$
4. **점-평면 오차와 풀이 (가우스-뉴턴, 최대 100회):**
$$e_i=\mathbf n_i^{\top}\big(R_z(\psi)\,\mathbf p_i+[t_x,\ t_y,\ 0]^{\top}-\mathbf q_i\big),\qquad J_i=\Big[\,n_x,\ \ n_y,\ \ \mathbf n_i^{\top}\tfrac{\partial(R_z\mathbf p_i)}{\partial\psi}\Big]$$
$$\Big(\sum_i J_i^{\top}J_i\Big)\Delta\mathbf x=-\sum_i J_i^{\top}e_i,\qquad \mathbf x\leftarrow\mathbf x+\Delta\mathbf x$$
5. **지도에 쌓기:** 구한 자세로 점을 옮겨 5 cm 칸 지도에 넣습니다. 칸마다 처음 들어온 점 하나만 둡니다.

코어 계산 파일(`go2_lidar_odom/core/`)은 `Lidar_odom_ws/lidar_odom/`의 파일을 그대로 복사한 것입니다.

## 5. 처리 시간 대책

- **받기 큐 깊이 1:** 처리가 밀리면 가장 새 프레임만 처리합니다. 그래서 지연이 쌓이지 않습니다.
- **최근접 트리:** 큰 트리는 새 지도 점이 2000개 넘게 쌓일 때만 새로 만들고, 그 사이 새 점은 작은 트리로 찾습니다. 결과는 매 프레임 새로 만든 것과 같습니다.
- **지도 자르기:** 로봇에서 31 m(L1 최대 거리 30 m + 여유) 밖 지도는 지웁니다.
- 5초마다 로그에 처리 시간 중앙값·p95와 건너뛴 프레임 수가 나옵니다.

## 6. 알고 쓸 것

- **평평한 바닥 가정 (3자유도):** 경사나 턱이 있으면 틀어질 수 있습니다.
- **낮은 장애물:** 몸통 중심보다 낮은 점은 위치 계산에 쓰지 않습니다. `cloud_registered`(all 모드)에는 들어갑니다.
- **움직이는 장애물(사람 등):** 지도에 남아 잘못 짝지어질 수 있습니다. 걸러 내는 기능은 없습니다.
- **작고 둥근 물체(콘, 기둥):** 평면으로 잘 맞지 않습니다.
- 프레임 안 왜곡 보정(deskew)과 루프 닫기는 없습니다.

## 7. 검증 (녹화 재생)

노드가 쓴 자세 CSV를 `Lidar_odom_ws/tools/run_icp3d_bag.py`의 오프라인 결과와 비교합니다.
프레임을 하나도 건너뛰지 않으면 두 결과가 같아야 합니다.

```bash
# 터미널 1: 모든 프레임을 처리하도록 큐를 크게
ros2 launch go2_lidar_odom lidar_odom.launch.py qos_depth:=1000 pose_log_csv:=/tmp/node_poses.csv
# 터미널 2: 절반 속도로 재생
ros2 bag play <녹화 폴더> -r 0.5 --topics /utlidar/cloud
# 끝나면 터미널 1 을 Ctrl+C 로 끄고 비교
python3 src/go2_lidar_odom/tools/compare_poses.py /tmp/node_poses.csv <오프라인 poses.csv>
```

오프라인 결과는 같은 설정으로 만듭니다.

```bash
cd ~/go2/Lidar_odom_ws
python3 tools/run_icp3d_bag.py <녹화 폴더> --dof 3 --structure map --z-min 0 --z-max 2 --r-min 0.6 \
    --bootstrap-frames 14 --voxel 0.05 --normal-k 40 --max-corr-dist 0.5 --no-plot --out /tmp/offline_k40
```
