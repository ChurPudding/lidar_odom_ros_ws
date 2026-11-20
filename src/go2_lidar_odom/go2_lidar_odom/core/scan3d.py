#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan3d.py — L1 3D 점을 몸체(base_link) 좌표의 3D 점으로 바꿈 (2D 로 누르지 않음)

  p^B = R_LBᵀ p^L + LEVER                                   (식 8, scan2d 와 같음)
  r = || p^L ||
  r_min < r ≤ r_max,   z_min ≤ z^B ≤ z_max   (z_min 이 None 이면 아래 한계 없음 → 바닥 포함)
"""
import numpy as np

from .scan2d import lidar_to_base, load_extrinsic   # noqa: F401  (load_extrinsic 은 다시 내보냄)


def select_3d(P_L, R_LB, lever, z_min=None, z_max=2.0, r_min=0.6, r_max=0.0):
    """L1 점 (N, 3) → 몸체 3D 점 (M, 3)"""
    P_L = np.asarray(P_L, dtype=float)
    r = np.linalg.norm(P_L, axis=1)
    P_B = lidar_to_base(P_L, R_LB, lever)
    m = r > r_min
    if z_min is not None:
        m &= P_B[:, 2] >= z_min
    if z_max is not None:
        m &= P_B[:, 2] <= z_max
    if r_max > 0:
        m &= r <= r_max
    return P_B[m]
