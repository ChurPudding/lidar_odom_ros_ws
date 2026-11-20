#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — ROS 없이 도는 한 프레임 처리 (노드는 이 클래스를 부르기만 함)

[한 프레임 k]
  1) 점 고르기 (core.scan3d.select_3d)
       p^B = R_LBᵀ p^L + LEVER,   r = ||p^L|| > r_min,   z_min ≤ z^B ≤ z_max
  2) 처음 N 프레임 (bootstrap_frames) 은 정지로 보고  T^W_k = I,  지도에 그대로 넣음
  3) 그 뒤: 스캔-대-지도 점-평면 ICP (core.icp3d.icp_p2pl3d), 초기값 T^W_{k−1}
       e_i = n_iᵀ ( R p_i + t − q_i ),   H Δx = −g,   x ← x + Δx
       성공하면 T^W_k = ICP 결과,  스캔을 T^W_k 로 옮겨 지도에 넣음
       실패하면 (짝 < min_corr) T^W_k = T^W_{k−1}, 지도에 넣지 않음
  4) map_crop_every 프레임마다, 로봇에서 수평 거리 map_radius 보다 먼 지도 칸을 지움
       || (q_x, q_y) − (x_k, y_k) || > R  →  삭제

[지도: LocalVoxelMap]
  core.voxel_map.VoxelMap 과 규칙이 같음 (칸 크기 v, 칸마다 처음 들어온 점 하나, 넣은 순서 유지)
  다른 점: 점을 NumPy 배열에 이어 붙여 points() 가 매 프레임 배열을 새로 만들지 않음, crop_xy() 추가
  → 자르기로 지워지는 칸이 없으면 오프라인(run_icp3d_bag.py)과 기준 점 배열이 순서까지 같아 결과가 같음
"""
import time
from dataclasses import dataclass

import numpy as np

from .core import icp3d as I3
from .core import scan3d


# ============================================================================ 설정
@dataclass
class OdomConfig:
    R_LB: np.ndarray                 # 몸체 → L1 회전 (3x3),  v^L = R_LB v^B
    lever: np.ndarray                # 몸체 좌표에서 본 L1 위치 [m] (3,)
    dof: int = 3                     # 3 = x, y, yaw / 6 = x, y, z, roll, pitch, yaw
    z_min: object = 0.0              # 몸체 높이 하한 [m] (None = 아래 한계 없음, 바닥 포함)
    z_max: object = 2.0              # 몸체 높이 상한 [m] (None = 위 한계 없음)
    r_min: float = 0.6               # 센서 거리 하한 [m] (로봇 몸·다리, 발밑 바닥 제거)
    r_max: float = 0.0               # 센서 거리 상한 [m] (0 이하 = 없음)
    bootstrap_frames: int = 14       # 처음 N 프레임 정지 가정
    voxel: float = 0.05              # 지도 칸 크기 [m]
    normal_k: int = 40               # 평면(법선) 이웃 점 수 k
    max_corr_dist: float = 0.5       # 짝 거리 한계 d_max [m]
    max_iter: int = 100              # 가우스-뉴턴 최대 반복
    tol_trans: float = 1e-4          # 수렴 ε_t [m]
    tol_rot: float = 1e-5            # 수렴 ε_θ [rad]
    min_corr: int = 10               # 최소 짝 수
    map_radius: float = 31.0         # 지도 자르기 반경 R [m] (0 이하 = 자르지 않음)
    map_crop_every: int = 10         # 몇 프레임마다 자를지
    tree_rebuild_points: int = 2000  # 작은 트리 점이 이 수를 넘으면 큰 트리를 지도 전체로 새로 만듦 (0 = 매 프레임)

    def validate(self):
        R = np.asarray(self.R_LB, dtype=float).reshape(3, 3)
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-4):
            raise ValueError('R_LB 가 회전 행렬이 아닙니다 (RᵀR ≠ I 또는 det ≠ 1).')
        self.R_LB = R
        self.lever = np.asarray(self.lever, dtype=float).reshape(3)
        if self.dof not in (3, 6):
            raise ValueError(f'dof 는 3 또는 6: {self.dof}')
        if self.normal_k < 3:
            raise ValueError(f'normal_k 는 3 이상이어야 평면을 맞출 수 있습니다: {self.normal_k}')
        if not self.voxel > 0:
            raise ValueError(f'voxel 은 0 보다 커야 합니다: {self.voxel}')
        if self.map_crop_every < 1:
            raise ValueError(f'map_crop_every 는 1 이상: {self.map_crop_every}')
        if self.tree_rebuild_points < 0:
            raise ValueError(f'tree_rebuild_points 는 0 이상: {self.tree_rebuild_points}')
        return self


# ============================================================================ 지도
class LocalVoxelMap:
    """칸마다 처음 들어온 점 하나 (core.voxel_map.VoxelMap 과 같은 규칙) + 로봇 주변만 남기는 자르기."""

    def __init__(self, voxel, dim=3, capacity=65536):
        self.voxel = float(voxel)
        self.dim = dim
        self._index = {}                                  # 칸 번호 (tuple) → 행 번호
        self._pts = np.empty((capacity, dim))
        self._keys = np.empty((capacity, dim), dtype=np.int64)
        self._n = 0

    def __len__(self):
        return self._n

    def _grow(self, need):
        cap = len(self._pts)
        if need <= cap:
            return
        while cap < need:
            cap *= 2
        pts = np.empty((cap, self.dim))
        keys = np.empty((cap, self.dim), dtype=np.int64)
        pts[:self._n] = self._pts[:self._n]
        keys[:self._n] = self._keys[:self._n]
        self._pts, self._keys = pts, keys                 # 예전 배열은 그대로 두므로 그 뷰를 쓰던 곳은 안전

    def add(self, points):
        """월드 좌표 점 (N, dim) 을 넣음. 이미 차 있는 칸은 건너뜀. 새로 들어간 점 수를 돌려줌."""
        P = np.asarray(points, dtype=float)
        if P.size == 0:
            return 0
        keys = np.floor(P / self.voxel).astype(np.int64)
        self._grow(self._n + len(P))
        added = 0
        for key, kk, p in zip(map(tuple, keys.tolist()), keys, P):
            if key not in self._index:
                i = self._n
                self._index[key] = i
                self._pts[i] = p
                self._keys[i] = kk
                self._n += 1
                added += 1
        return added

    def points(self):
        """지도 점 (M, dim) — 넣은 순서. 이 배열의 행은 이후 add() 로 바뀌지 않음."""
        return self._pts[:self._n]

    def crop_xy(self, center_xy, radius):
        """수평 거리 radius 보다 먼 칸을 지움 (남은 점의 순서는 유지). 지운 점 수를 돌려줌."""
        if radius <= 0 or self._n == 0:
            return 0
        P = self._pts[:self._n]
        keep = np.hypot(P[:, 0] - center_xy[0], P[:, 1] - center_xy[1]) <= radius
        removed = int(self._n - keep.sum())
        if removed == 0:
            return 0
        pts = np.empty_like(self._pts)
        keys = np.empty_like(self._keys)
        m = int(keep.sum())
        pts[:m] = P[keep]
        keys[:m] = self._keys[:self._n][keep]
        self._pts, self._keys, self._n = pts, keys, m
        self._index = {k: i for i, k in enumerate(map(tuple, keys[:m].tolist()))}
        return removed


# ============================================================================ 최근접 찾기 (큰 트리 + 작은 트리)
class DualTreeNN:
    """core.icp3d.NearestNeighbor 와 같은 모양(target, backend, query, query_k)으로 쓰는 최근접 찾기.

    지도 점 배열 target 을 앞부분 [0, n_big) 과 뒷부분 [n_big, n) 으로 나눔
      큰 트리  : 앞부분. 가끔만 새로 만듦 (LidarOdometry 가 관리)
      작은 트리: 뒷부분 (큰 트리를 만든 뒤 새로 들어온 점). 매 프레임 새로 만듦 (점이 적어 빠름)
    최근접 1 개  : 두 트리에서 각각 가장 가까운 점을 찾고 더 가까운 쪽을 고름
                   min_{q∈A∪B} ||x − q|| = min( min_{q∈A} ||x − q||, min_{q∈B} ||x − q|| )
    최근접 k 개  : 두 트리에서 각각 k 개씩 찾고, 합친 2k 개 중 가까운 k 개를 고름
                   (A∪B 의 가까운 k 개는 반드시 'A 의 가까운 k 개 ∪ B 의 가까운 k 개' 안에 있음)
    → 지도 전체로 트리 하나를 만든 것과 같은 점(같은 행 번호)을 돌려줌. 거리가 완전히 같은 점이 둘일 때만 다를 수 있음.
    """
    backend = 'scipy'

    def __init__(self, target, big_tree, n_big):
        from scipy.spatial import cKDTree
        self.target = target                      # 지도 점 전체 (M, 3). LazyNormals 가 이 행 번호로 이웃을 꺼냄
        self._big = big_tree
        self._n_big = int(n_big)
        n_small = len(target) - self._n_big
        self._small = cKDTree(target[self._n_big:]) if n_small > 0 else None

    def query(self, X):
        """각 점 X 의 최근접 지도 점 1 개: (거리 (N,), 행 번호 (N,))"""
        d, j = self._big.query(X, k=1)
        if self._small is not None:
            d2, j2 = self._small.query(X, k=1)
            closer = d2 < d                        # 작은 트리 쪽이 더 가까우면 그쪽 점으로 바꿈
            d = np.where(closer, d2, d)
            j = np.where(closer, j2 + self._n_big, j)   # 작은 트리 번호 → 지도 전체 행 번호
        return d, j

    def query_k(self, X, k):
        """각 점 X 의 가까운 지도 점 k 개 행 번호 (N, k), 가까운 순 (k 는 지도 점 수를 넘지 않게 자름)"""
        X = np.asarray(X, dtype=float)
        k = int(min(k, len(self.target)))
        kb = min(k, self._n_big)
        db, jb = self._big.query(X, k=kb)
        db, jb = db.reshape(len(X), kb), jb.reshape(len(X), kb)
        if self._small is None:
            return jb[:, :k]
        ks = min(k, len(self.target) - self._n_big)
        ds, js = self._small.query(X, k=ks)
        ds, js = ds.reshape(len(X), ks), js.reshape(len(X), ks) + self._n_big
        d = np.concatenate([db, ds], axis=1)       # 후보 kb + ks 개
        j = np.concatenate([jb, js], axis=1)
        order = np.argsort(d, axis=1, kind='stable')[:, :k]   # 가까운 k 개 (거리가 같으면 큰 트리 쪽 먼저)
        return np.take_along_axis(j, order, axis=1)


# ============================================================================ 한 프레임 결과
@dataclass
class FrameResult:
    k: int                    # 받은 프레임 번호 (0 부터)
    stamp: float              # 점군 header.stamp [s]
    T: np.ndarray             # 4x4, 출발점 기준 몸체 자세 (odom → base)
    status: str               # 'bootstrap' | 'ok' | 'fail'
    converged: bool
    iterations: int
    rmse: float               # 법선 방향 거리 RMS [m]
    n_points: int             # 고른 점 수
    n_inliers: int
    h_eig_min: float          # 마지막 H 의 가장 작은 고유값 (작으면 그 방향이 잘 안 잡힘)
    map_size: int
    map_removed: int          # 이번 프레임에 자르기로 지운 지도 점 수
    t_proc: float             # 이 프레임 처리 시간 [s] (점 고르기 + 트리 + ICP + 지도)
    scan_B: np.ndarray        # 고른 점 (몸체 좌표) — 출력용


# ============================================================================ 오도메트리
class LidarOdometry:
    def __init__(self, cfg: OdomConfig):
        self.cfg = cfg.validate()
        self.reset()

    def reset(self):
        """처음 상태로 되돌림 (지도 비움, 자세 = I, 다음 프레임부터 다시 정지 가정)"""
        self.map = LocalVoxelMap(self.cfg.voxel)
        self.T = np.eye(4)
        self.k = 0
        self._icp_frames = 0
        self._big_tree = None                    # 큰 트리 (지도 앞부분 [0, n_big))
        self._n_big = 0
        self.tree_rebuilds = 0                   # 큰 트리를 새로 만든 횟수 (기록용)

    @property
    def scipy_ok(self):
        """SciPy(KD-트리)가 있는지. 없으면 NumPy 전수 계산이라 매우 느림"""
        try:
            import scipy.spatial  # noqa: F401
            return True
        except ImportError:
            return False

    def _nearest(self, tgt):
        """이번 프레임 ICP 에 쓸 최근접 찾기 객체.
        SciPy 가 있으면 큰 트리 + 작은 트리 (DualTreeNN), 없으면 core 의 NearestNeighbor (NumPy 전수 계산)."""
        if not self.scipy_ok:
            return I3.NearestNeighbor(tgt)
        from scipy.spatial import cKDTree
        n = len(tgt)
        if self._big_tree is None or n - self._n_big > self.cfg.tree_rebuild_points:
            self._big_tree = cKDTree(tgt)        # 지도 전체로 큰 트리를 새로 만듦
            self._n_big = n
            self.tree_rebuilds += 1
        return DualTreeNN(tgt, self._big_tree, self._n_big)

    def _icp_kw(self):
        c = self.cfg
        return dict(dof=c.dof, max_iter=c.max_iter, max_corr_dist=c.max_corr_dist, tol_trans=c.tol_trans,
                    tol_rot=c.tol_rot, min_corr=c.min_corr, normal_k=c.normal_k)

    def select(self, P_L):
        c = self.cfg
        return scan3d.select_3d(P_L, c.R_LB, c.lever, c.z_min, c.z_max, c.r_min, c.r_max)

    def to_base(self, P_L):
        """L1 점 → 몸체 점 (고르지 않고 전부)"""
        return scan3d.lidar_to_base(P_L, self.cfg.R_LB, self.cfg.lever)

    def process(self, stamp, P_L):
        """L1 좌표 점 (N, 3) 한 프레임을 처리하고 FrameResult 를 돌려줌 (NaN/inf 점은 미리 빼서 넣을 것)."""
        c = self.cfg
        t0 = time.perf_counter()
        scan = self.select(P_L)
        nb = max(c.bootstrap_frames, 1)
        res = None
        removed = 0
        if self.k < nb:
            T = np.eye(4)
            self.map.add(scan)
            status = 'bootstrap'
        else:
            tgt = self.map.points()
            if len(scan) >= c.min_corr and len(tgt) >= c.min_corr:
                nn = self._nearest(tgt)
                res = I3.icp_p2pl3d(scan, tgt, init_T=self.T, nn=nn, **self._icp_kw())
            if res is None or res.n_inliers < c.min_corr:
                T = self.T.copy()
                status = 'fail'
            else:
                T = res.T
                self.map.add(I3.transform3(scan, T))
                status = 'ok'
            self._icp_frames += 1
            if c.map_radius > 0 and self._icp_frames % c.map_crop_every == 0:
                removed = self.map.crop_xy(T[:2, 3], c.map_radius)
                if removed:
                    self._big_tree = None        # 행 번호가 바뀌었으므로 다음 프레임에 큰 트리를 새로 만듦
        self.T = T
        out = FrameResult(
            k=self.k, stamp=float(stamp), T=T.copy(), status=status,
            converged=bool(res.converged) if (res is not None and status == 'ok') else status == 'bootstrap',
            iterations=0 if res is None else int(res.iterations),
            rmse=float('nan') if res is None else float(res.rmse),
            n_points=len(scan), n_inliers=0 if res is None else int(res.n_inliers),
            h_eig_min=float(res.h_eig[0]) if (res is not None and res.h_eig is not None) else float('nan'),
            map_size=len(self.map), map_removed=removed, t_proc=time.perf_counter() - t0, scan_B=scan)
        self.k += 1
        return out


# ============================================================================ 자세 도우미
def yaw_of(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


def quat_of(R):
    """회전 행렬 → 쿼터니언 (x, y, z, w)"""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def body_twist(T_prev, T_cur, dt):
    """두 자세의 차이로 몸체 속도 (이전 몸체 좌표):  ΔT = T_prev⁻¹ T_cur,  v = Δt / dt,  ω = log(ΔR) / dt"""
    if dt <= 0:
        return np.zeros(3), np.zeros(3)
    dT = np.linalg.inv(T_prev) @ T_cur
    v = dT[:3, 3] / dt
    R = dT[:3, :3]
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    a = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    phi = 0.5 * w if a < 1e-8 else w * (a / (2.0 * np.sin(a)))
    return v, phi / dt
