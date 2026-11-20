#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
node.py — Go2 L1 점군만 3D 점-평면 ICP 오도메트리 ROS2 노드 (lidar_odom_node)

[받는 것]
  cloud_topic (기본 /utlidar/cloud) : sensor_msgs/PointCloud2, L1 좌표 점

[내보내는 것]  (모든 이름은 config/lidar_odom.yaml 에서 바꿈, 빈 문자열이면 내보내지 않음)
  odom_topic              nav_msgs/Odometry   출발점 기준 몸체 자세 + 프레임 차이로 구한 몸체 속도
  TF                      odom_frame → base_frame (publish_tf: true 일 때)
  path_topic              nav_msgs/Path       지나온 경로 (path_publish_period 초마다)
  registered_cloud_topic  sensor_msgs/PointCloud2  이번 프레임 점을 odom_frame 으로 옮긴 것
                          cloud_out_mode: all  = 받은 점 전부 (r > cloud_out_min_range, 바닥·낮은 장애물 포함)
                                          used = 위치 계산에 쓴 점만 (높이·거리로 고른 점)
  map_topic               sensor_msgs/PointCloud2  위치 계산에 쓰는 칸 지도 (map_publish_period 초마다)
  ~/reset                 std_srvs/Trigger    지도·자세를 처음으로 되돌림 (다시 정지 가정부터 시작)

[계산]  pipeline.LidarOdometry.process() 를 그대로 부름 (ROS 없이 도는 부분과 같음)
  출력 시각 = 받은 점군의 header.stamp  (녹화 재생 때도 다른 토픽과 시각이 맞음)

[실시간]
  받기 큐 깊이 qos_depth (기본 1): 처리가 밀리면 오래된 프레임을 버리고 가장 새 것만 처리 → 지연이 쌓이지 않음
  stats_period 초마다 처리 시간·건너뛴 프레임 수를 로그로 남김

[파라미터 변경]
  모든 파라미터는 read_only: 실행할 때 yaml 이나 launch 인자로 정하고, 실행 중에는 바꿀 수 없음
"""
import csv
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from .cloud_io import cloud_to_xyz, stamp_to_sec, xyz_to_cloud
from .core import icp3d as I3
from .pipeline import LidarOdometry, OdomConfig, body_twist, quat_of

# ---------------------------------------------------------------------------- 파라미터 목록
# (이름, 기본값, 설명)  기본값은 config/lidar_odom.yaml 과 같게 둠 (yaml 이 없을 때도 같은 동작)
_R_LB_DEFAULT = [0.523029, -0.838576, 0.15242,
                 -0.810712, -0.544668, -0.214668,
                 0.263034, -0.011292, -0.964721]
PARAMS = [
    # 입력
    ('cloud_topic', '/utlidar/cloud', 'L1 점군 토픽 (sensor_msgs/PointCloud2)'),
    ('qos_reliability', 'best_effort', '받기 QoS: best_effort 또는 reliable'),
    ('qos_depth', 1, '받기 큐 깊이. 1 이면 밀릴 때 가장 새 프레임만 처리'),
    # 출력 이름
    ('odom_frame', 'lidar_odom', '출발점에 고정된 좌표계 이름'),
    ('base_frame', 'lidar_odom_base', '추정한 로봇 몸체 좌표계 이름'),
    ('publish_tf', True, 'odom_frame → base_frame TF 를 낼지'),
    ('odom_topic', '/lidar_odom/odometry', 'nav_msgs/Odometry 토픽 (빈 문자열 = 안 냄)'),
    ('path_topic', '/lidar_odom/path', 'nav_msgs/Path 토픽 (빈 문자열 = 안 냄)'),
    ('path_publish_period', 1.0, '경로를 몇 초마다 낼지'),
    ('registered_cloud_topic', '/lidar_odom/cloud_registered', '이번 프레임 점(odom_frame) 토픽 (빈 문자열 = 안 냄)'),
    ('cloud_out_mode', 'all', 'all = 받은 점 전부, used = 위치 계산에 쓴 점만'),
    ('cloud_out_min_range', 0.6, 'all 모드에서 센서 거리 이 값 이하 점(로봇 몸·다리)은 뺌 [m]'),
    ('map_topic', '/lidar_odom/map', '칸 지도 토픽 (빈 문자열 = 안 냄)'),
    ('map_publish_period', 1.0, '지도를 몇 초마다 낼지 (0 이하 = 안 냄)'),
    # 장착 상수 (go2_calib.py 확정값)
    ('R_LB', _R_LB_DEFAULT, '몸체 → L1 회전 3x3 (행 우선 9개). v_L = R_LB v_B'),
    ('lever', [0.322, 0.005, 0.050], '몸체 좌표에서 본 L1 위치 [m]'),
    # 알고리즘 (pipeline.OdomConfig 와 같은 이름)
    ('dof', 3, '3 = x, y, yaw (평평한 바닥) / 6 = x, y, z, roll, pitch, yaw'),
    ('use_z_min', True, 'true 면 몸체 높이 z_min 아래 점(바닥)을 뺌'),
    ('z_min', 0.0, '몸체 높이 하한 [m] (몸통 중심 기준)'),
    ('use_z_max', True, 'true 면 몸체 높이 z_max 위 점을 뺌'),
    ('z_max', 2.0, '몸체 높이 상한 [m]'),
    ('r_min', 0.6, '센서 거리 하한 [m] (로봇 몸·다리, 발밑 바닥 제거)'),
    ('r_max', 0.0, '센서 거리 상한 [m] (0 이하 = 없음)'),
    ('bootstrap_frames', 14, '처음 몇 프레임을 정지로 볼지 (그동안 로봇을 세워 둘 것)'),
    ('voxel', 0.05, '지도 칸 크기 [m]'),
    ('normal_k', 40, '평면(법선) 이웃 점 수 k'),
    ('max_corr_dist', 0.5, '짝 거리 한계 d_max [m]'),
    ('max_iter', 100, '가우스-뉴턴 최대 반복'),
    ('tol_trans', 1e-4, '수렴 이동 ε_t [m]'),
    ('tol_rot', 1e-5, '수렴 회전 ε_θ [rad]'),
    ('min_corr', 10, '최소 짝 수 (이보다 적으면 그 프레임은 실패로 보고 자세 유지)'),
    ('map_radius', 31.0, '로봇에서 이 수평 거리 밖 지도 칸을 지움 [m] (0 이하 = 안 지움)'),
    ('map_crop_every', 10, '몇 프레임마다 지도를 자를지'),
    ('tree_rebuild_points', 2000, '새 지도 점이 이 수를 넘으면 큰 최근접 트리를 새로 만듦 (0 = 매 프레임)'),
    # 기록
    ('stats_period', 5.0, '몇 초마다 처리 시간 로그를 남길지 (0 이하 = 안 남김)'),
    ('pose_log_csv', '', '프레임별 자세를 적을 CSV 경로 (빈 문자열 = 안 적음)'),
]


class LidarOdomNode(Node):
    def __init__(self):
        super().__init__('lidar_odom')
        # ---- 파라미터 선언 (모두 read_only: 실행 중 변경 불가)
        for name, default, desc in PARAMS:
            self.declare_parameter(name, default, ParameterDescriptor(description=desc, read_only=True))
        p = {name: self.get_parameter(name).value for name, _, _ in PARAMS}
        self.p = p

        # ---- 계산부 (ROS 없이 도는 pipeline)
        cfg = OdomConfig(
            R_LB=np.asarray(p['R_LB'], dtype=float).reshape(3, 3), lever=np.asarray(p['lever'], dtype=float),
            dof=int(p['dof']), z_min=float(p['z_min']) if p['use_z_min'] else None,
            z_max=float(p['z_max']) if p['use_z_max'] else None, r_min=float(p['r_min']), r_max=float(p['r_max']),
            bootstrap_frames=int(p['bootstrap_frames']), voxel=float(p['voxel']), normal_k=int(p['normal_k']),
            max_corr_dist=float(p['max_corr_dist']), max_iter=int(p['max_iter']), tol_trans=float(p['tol_trans']),
            tol_rot=float(p['tol_rot']), min_corr=int(p['min_corr']), map_radius=float(p['map_radius']),
            map_crop_every=int(p['map_crop_every']), tree_rebuild_points=int(p['tree_rebuild_points']))
        self.odom = LidarOdometry(cfg)
        if p['cloud_out_mode'] not in ('all', 'used'):
            raise ValueError(f"cloud_out_mode 는 all 또는 used: {p['cloud_out_mode']}")

        # ---- 받기
        rel = ReliabilityPolicy.BEST_EFFORT if p['qos_reliability'] == 'best_effort' else ReliabilityPolicy.RELIABLE
        qos_in = QoSProfile(reliability=rel, history=HistoryPolicy.KEEP_LAST, depth=int(p['qos_depth']))
        self.create_subscription(PointCloud2, p['cloud_topic'], self.on_cloud, qos_in)

        # ---- 내보내기 (토픽 이름이 빈 문자열이면 만들지 않음)
        self.pub_odom = self.create_publisher(Odometry, p['odom_topic'], 10) if p['odom_topic'] else None
        self.pub_path = self.create_publisher(Path, p['path_topic'], 1) if p['path_topic'] else None
        self.pub_cloud = (self.create_publisher(PointCloud2, p['registered_cloud_topic'], 5)
                          if p['registered_cloud_topic'] else None)
        qos_map = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)   # 늦게 켠 RViz 도 마지막 지도를 받음
        self.pub_map = (self.create_publisher(PointCloud2, p['map_topic'], qos_map)
                        if p['map_topic'] and p['map_publish_period'] > 0 else None)
        self.tf = TransformBroadcaster(self) if p['publish_tf'] else None
        self.create_service(Trigger, '~/reset', self.on_reset)

        # ---- 상태
        self.path = Path()
        self.path.header.frame_id = p['odom_frame']
        self._prev = None                 # (stamp, T) 직전 프레임 — 속도 계산용
        self._last_path_pub = 0.0         # 마지막으로 경로를 낸 벽시계 시각 [s]
        self._last_map_pub = 0.0
        self._gaps = []                   # 최근 점군 시각 간격 [s] (프레임 간격 추정용)
        self._stats_reset()
        self._last_stats = time.monotonic()
        self._csv = None
        if p['pose_log_csv']:
            f = open(p['pose_log_csv'], 'w', newline='', buffering=1)
            self._csv = (f, csv.writer(f))
            self._csv[1].writerow(['k', 'stamp', 'x', 'y', 'z', 'roll_deg', 'pitch_deg', 'yaw_deg', 'status',
                                   'converged', 'iters', 'rmse', 'n_points', 'n_inliers', 'h_eig_min',
                                   'map_size', 't_proc_ms'])

        if not self.odom.scipy_ok:
            self.get_logger().warn('SciPy 가 없어 최근접을 NumPy 전수 계산으로 찾습니다 (매우 느림). '
                                   'sudo apt install python3-scipy 를 권합니다.')
        self.get_logger().info(
            f"시작: {p['cloud_topic']} → {p['odom_topic'] or '(odom 안 냄)'}, TF {p['odom_frame']} → {p['base_frame']}"
            f"{'' if p['publish_tf'] else ' (안 냄)'} | {cfg.dof}자유도, k = {cfg.normal_k}, 칸 {cfg.voxel} m, "
            f"d_max {cfg.max_corr_dist} m, z [{cfg.z_min}, {cfg.z_max}] m, r > {cfg.r_min} m, 지도 반경 {cfg.map_radius} m")
        self.get_logger().info(f'처음 {cfg.bootstrap_frames} 프레임(약 {cfg.bootstrap_frames * 0.065:.1f} 초) 동안은 '
                               '로봇을 세워 두세요. 그 자리가 출발점(원점)입니다.')

    # ------------------------------------------------------------------ 통계
    def _stats_reset(self):
        self._st = dict(n=0, t=[], fail=0, skipped=0, over=0)

    def _frame_interval(self):
        """최근 점군 시각 간격의 중앙값 [s] (아직 모르면 None)"""
        return float(np.median(self._gaps)) if len(self._gaps) >= 5 else None

    def _log_stats(self, res):
        per = self.p['stats_period']
        now = time.monotonic()
        if per <= 0 or now - self._last_stats < per or self._st['n'] == 0:
            return
        t = np.array(self._st['t']) * 1e3
        x, y, yaw = res.T[0, 3], res.T[1, 3], math.degrees(math.atan2(res.T[1, 0], res.T[0, 0]))
        fi = self._frame_interval()
        self.get_logger().info(
            f"{now - self._last_stats:.1f} s 동안 {self._st['n']} 프레임 처리"
            + (f' (건너뛴 프레임 약 {self._st["skipped"]})' if fi else '')
            + (f' | 처리 시간 중앙값 {np.median(t):.1f} ms, p95 {np.percentile(t, 95):.1f} ms, 최대 {t.max():.1f} ms'
               if len(t) else '')
            + (f', 프레임 간격 {fi * 1e3:.0f} ms 넘은 프레임 {self._st["over"]}' if fi else '')
            + f" | 실패 {self._st['fail']} | 지도 {res.map_size} 점 | 자세 x {x:+.2f} y {y:+.2f} m, yaw {yaw:+.1f}°")
        self._last_stats = now
        self._stats_reset()

    # ------------------------------------------------------------------ 콜백
    def on_reset(self, request, response):
        """~/reset 서비스: 지도·자세·경로를 처음으로 되돌림"""
        self.odom.reset()
        self.path.poses = []
        self._prev = None
        response.success = True
        response.message = f'초기화함. 처음 {self.odom.cfg.bootstrap_frames} 프레임 동안 로봇을 세워 두세요.'
        self.get_logger().info(response.message)
        return response

    def on_cloud(self, msg: PointCloud2):
        """점군 한 프레임: 계산 → Odometry·TF·경로·점군·지도 내보내기 → 기록"""
        stamp = stamp_to_sec(msg.header.stamp)
        # ---- 프레임 간격·건너뛴 프레임 추정: 간격이 평소 간격 Δ 의 몇 배인지로 셈
        #      건너뛴 수 ≈ round(gap / Δ) − 1
        if self._prev is not None:
            gap = stamp - self._prev[0]
            fi = self._frame_interval()
            if fi and gap > 1.5 * fi:
                self._st['skipped'] += int(round(gap / fi)) - 1
            if gap > 0:
                self._gaps = (self._gaps + [gap])[-50:]

        P_L = cloud_to_xyz(msg)
        res = self.odom.process(stamp, P_L)
        T = res.T

        # ---- 통계
        self._st['n'] += 1
        if res.status != 'bootstrap':
            self._st['t'].append(res.t_proc)
            fi = self._frame_interval()
            if fi and res.t_proc > fi:
                self._st['over'] += 1
        if res.status == 'fail':
            self._st['fail'] += 1
            self.get_logger().warn(f'프레임 {res.k}: ICP 실패 (짝 {res.n_inliers} < {self.odom.cfg.min_corr}), '
                                   '직전 자세를 유지합니다.', throttle_duration_sec=2.0)

        # ---- 내보내기
        p = self.p
        q = quat_of(T[:3, :3])
        if self.tf is not None:
            tfm = TransformStamped()
            tfm.header.stamp = msg.header.stamp
            tfm.header.frame_id = p['odom_frame']
            tfm.child_frame_id = p['base_frame']
            tfm.transform.translation.x, tfm.transform.translation.y, tfm.transform.translation.z = map(float, T[:3, 3])
            (tfm.transform.rotation.x, tfm.transform.rotation.y,
             tfm.transform.rotation.z, tfm.transform.rotation.w) = map(float, q)
            self.tf.sendTransform(tfm)

        if self.pub_odom is not None:
            od = Odometry()
            od.header.stamp = msg.header.stamp
            od.header.frame_id = p['odom_frame']
            od.child_frame_id = p['base_frame']
            od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = map(float, T[:3, 3])
            (od.pose.pose.orientation.x, od.pose.pose.orientation.y,
             od.pose.pose.orientation.z, od.pose.pose.orientation.w) = map(float, q)
            # 몸체 속도: 직전 프레임과의 자세 차이 / 시간 차이 (정지 가정 구간은 0)
            if self._prev is not None and res.status != 'bootstrap':
                v, w = body_twist(self._prev[1], T, stamp - self._prev[0])
                od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.linear.z = map(float, v)
                od.twist.twist.angular.x, od.twist.twist.angular.y, od.twist.twist.angular.z = map(float, w)
            self.pub_odom.publish(od)

        now = time.monotonic()
        if self.pub_path is not None:
            ps = PoseStamped()
            ps.header.stamp = msg.header.stamp
            ps.header.frame_id = p['odom_frame']
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, T[:3, 3])
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, q)
            self.path.poses.append(ps)
            if now - self._last_path_pub >= p['path_publish_period']:
                self.path.header.stamp = msg.header.stamp
                self.pub_path.publish(self.path)
                self._last_path_pub = now

        hdr = Header(stamp=msg.header.stamp, frame_id=p['odom_frame'])
        if self.pub_cloud is not None:
            if p['cloud_out_mode'] == 'used':
                P_B = res.scan_B                                            # 위치 계산에 쓴 점 (몸체)
            else:
                keep = np.linalg.norm(P_L, axis=1) > p['cloud_out_min_range']
                P_B = self.odom.to_base(P_L[keep])                          # 받은 점 전부 → 몸체
            self.pub_cloud.publish(xyz_to_cloud(I3.transform3(P_B, T), hdr))  # 몸체 → odom_frame

        if self.pub_map is not None and now - self._last_map_pub >= p['map_publish_period']:
            self.pub_map.publish(xyz_to_cloud(self.odom.map.points(), hdr))
            self._last_map_pub = now

        # ---- 기록
        if self._csv is not None:
            rpy = np.rad2deg(I3.R_to_rpy(T[:3, :3]))
            self._csv[1].writerow([res.k, f'{stamp:.9f}'] + [f'{v:.6f}' for v in T[:3, 3]]
                                  + [f'{v:.5f}' for v in rpy]
                                  + [res.status, int(res.converged), res.iterations, f'{res.rmse:.5f}', res.n_points,
                                     res.n_inliers, f'{res.h_eig_min:.4g}', res.map_size, f'{res.t_proc * 1e3:.2f}'])
        self._prev = (stamp, T)
        self._log_stats(res)

    def destroy_node(self):
        if self._csv is not None:
            self._csv[0].close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarOdomNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):     # Ctrl+C 로 끌 때 오류 없이 끝냄
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
