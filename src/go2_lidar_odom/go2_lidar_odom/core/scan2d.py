#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan2d.py — L1 3D 점을 몸체(base_link) 좌표의 2D 점으로 바꿈

[좌표 변환]
  장착 상수 (go2_calib.py)
    R_LB  : 몸체 → L1 회전.  v^L = R_LB v^B
    LEVER : 몸체 좌표에서 본 L1 위치 [m]
  L1 좌표의 점 p^L 을 몸체 좌표로:
    p^B = R_LBᵀ p^L + LEVER                                            ... (8)
  (행 벡터 N 개를 한 번에:  P^B = P^L R_LB + LEVERᵀ)

[점 고르기]
  r = || p^L ||           (센서에서 점까지 거리)
  r_min < r ≤ r_max       (가까운 로봇 다리·바로 앞 바닥 제거, r_max ≤ 0 이면 위 한계 없음)
  z_min ≤ z^B ≤ z_max     (바닥·천장 제거)
  남은 점의 (x^B, y^B) 만 사용 → 3자유도 (x, y, yaw)
"""
import numpy as np


def load_extrinsic(path):
    """YAML 에서 R_LB (3x3), LEVER (3,) 를 읽음"""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    R = np.asarray(cfg['R_LB'], dtype=float)
    lever = np.asarray(cfg['LEVER'], dtype=float)
    if R.shape != (3, 3) or lever.shape != (3,):
        raise ValueError(f'R_LB 는 3x3, LEVER 는 길이 3 이어야 합니다: {R.shape}, {lever.shape}')
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-4):
        raise ValueError('R_LB 가 회전 행렬이 아닙니다 (RᵀR ≠ I 또는 det ≠ 1).')
    return R, lever


def lidar_to_base(P_L, R_LB, lever):
    """식 (8):  p^B = R_LBᵀ p^L + LEVER   (P_L: (N, 3))"""
    return np.asarray(P_L, dtype=float) @ R_LB + lever


def select_2d(P_L, R_LB, lever, z_min=0.0, z_max=2.0, r_min=0.6, r_max=0.0):
    """L1 점 (N, 3) → 몸체 2D 점 (M, 2)"""
    P_L = np.asarray(P_L, dtype=float)
    r = np.linalg.norm(P_L, axis=1)
    P_B = lidar_to_base(P_L, R_LB, lever)
    m = (r > r_min) & (P_B[:, 2] >= z_min) & (P_B[:, 2] <= z_max)
    if r_max > 0:
        m &= r <= r_max
    return P_B[m, :2]
