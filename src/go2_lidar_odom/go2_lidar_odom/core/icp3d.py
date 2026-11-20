#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
icp3d.py — 3D 점-평면 ICP (3자유도 / 6자유도)

[자세]
  4x4 동차 행렬  T = [[R, t], [0, 1]],   p' = R p + t
  roll φ, pitch θ, yaw ψ :  R = R_z(ψ) R_y(θ) R_x(φ)      (ZYX 순서, ROS tf 와 같음)

[법선]  기준 점 q_i 주변 이웃 k 개의 3x3 공분산
      C_i = (1/k) Σ_j (q_j − q̄)(q_j − q̄)ᵀ ,   n_i = C_i 의 가장 작은 고유값의 고유벡터       ... (15)
  (벽이면 수평 법선, 바닥이면 n ≈ (0, 0, 1))

[오차]  법선 방향 거리만:
      e_i = n_iᵀ ( R p_i + t − q_i )                                                       ... (16)

[3자유도]  x = [t_x, t_y, ψ],   R = R_z(ψ),   t = [t_x, t_y, 0]
      J_i = [ n_x , n_y , n_iᵀ ∂(R_z p_i)/∂ψ ],   ∂(R_z p)/∂ψ = [ −(R_z p)_y , (R_z p)_x , 0 ]ᵀ   ... (17)
      H = Σ J_iᵀ J_i,  g = Σ J_iᵀ e_i,  H Δx = −g,  x ← x + Δx
  바닥 점은 n ≈ (0, 0, 1) 이라 J_i ≈ 0 → x, y, yaw 에 거의 기여하지 않음

[6자유도]  작은 변화 δ = [δt ; δφ] (6개),   R ← Exp(δφ) R,   t ← t + δt
      Exp(δφ) R p ≈ R p + δφ × (R p)
      e_i(δ) ≈ e_i + n_iᵀ δt + n_iᵀ (δφ × R p_i) = e_i + n_iᵀ δt + ((R p_i) × n_i)ᵀ δφ
      J_i = [ n_iᵀ ,  ((R p_i) × n_i)ᵀ ]                                                   ... (18)
      H δ = −g  로 δ 를 구해 위처럼 갱신
  회전 갱신 (로드리게스):
      Exp(φ) = I + (sin α / α) [φ]× + ((1 − cos α) / α²) [φ]×² ,   α = ||φ||                ... (19)
      [φ]× = [[0, −φ_z, φ_y], [φ_z, 0, −φ_x], [−φ_y, φ_x, 0]]
"""
from dataclasses import dataclass, field

import numpy as np


# ============================================================================ 회전·자세
def skew(v):
    """3 벡터 v 를 반대칭(skew) 행렬 [v]× 로 바꿈.  [v]× a = v × a (외적을 행렬 곱으로 표현)"""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(phi):
    """식 (19) 로드리게스. 회전 벡터 φ → 회전 행렬.
    6자유도 ICP 에서 구한 작은 회전 변화 δφ 를 회전 행렬로 바꿔 R 을 갱신할 때 씀.
    ||φ|| 가 0 에 가까우면 sin/cos 나눗셈이 불안정하므로 2차 테일러 근사로 대신함."""
    phi = np.asarray(phi, dtype=float)
    a = np.linalg.norm(phi)
    K = skew(phi)
    if a < 1e-10:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + (np.sin(a) / a) * K + ((1.0 - np.cos(a)) / a ** 2) * K @ K


def rot_z(psi):
    """yaw 각 ψ 만큼 z 축 둘레로 도는 3x3 회전 행렬 R_z(ψ). 3자유도 ICP 의 회전에 씀."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_R(roll, pitch, yaw):
    """오일러 각 (roll, pitch, yaw) → 3x3 회전 행렬.  R = R_z(yaw) R_y(pitch) R_x(roll)  (ZYX, ROS tf 와 같음)"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return rot_z(yaw) @ Ry @ Rx


def R_to_rpy(R):
    """3x3 회전 행렬 → 오일러 각 (roll, pitch, yaw)  (ZYX, rpy_to_R 의 역변환).
    pitch = −asin(R_20), roll = atan2(R_21, R_22), yaw = atan2(R_10, R_00)
    R_20 은 수치 오차로 ±1 을 살짝 넘을 수 있어 asin 전에 잘라냄."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def make_T(R=None, t=None):
    """회전 R(3x3) 과 이동 t(3) 로 4x4 동차 행렬 T 를 만듦. 주지 않은 쪽은 단위 회전 / 0 이동."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def T_to_pose6(T):
    """4x4 동차 행렬 T → 6 원소 자세 벡터 [x, y, z, roll, pitch, yaw] (출력·기록용)"""
    return np.r_[T[:3, 3], R_to_rpy(T[:3, :3])]


def pose6_to_T(p):
    """6 원소 자세 벡터 [x, y, z, roll, pitch, yaw] → 4x4 동차 행렬 T (T_to_pose6 의 역변환)"""
    return make_T(rpy_to_R(p[3], p[4], p[5]), p[:3])


def transform3(points, T):
    """점 묶음 (N, 3) 전체를 T 로 변환해 반환.  p' = R p + t  (행 벡터라 R 대신 Rᵀ 를 오른쪽에 곱함)"""
    P = np.asarray(points, dtype=float)
    return P @ T[:3, :3].T + T[:3, 3]


# ============================================================================ 최근접 (차원 무관)
class NearestNeighbor:
    """기준 점 안에서 최근접 1 개 / k 개.  SciPy 있으면 KD-트리, 없으면 NumPy 전수 계산 (3D 에서는 느림)."""

    def __init__(self, target, backend='auto', chunk=1024):
        """기준 점 (M, d) 을 저장하고 검색 방식을 정함.
        backend='auto' 는 SciPy KD-트리를 먼저 시도하고 없으면 NumPy 로 대체, 'scipy' 는 SciPy 강제.
        NumPy 방식이면 거리 계산에 쓸 ||q||² 를 미리 계산해 둠. chunk 는 NumPy 방식에서 한 번에 처리할 질의 점 수."""
        self.target = np.asarray(target, dtype=float)
        if self.target.ndim != 2 or len(self.target) == 0:
            raise ValueError(f'기준 점은 (M, d) 이고 비어 있지 않아야 합니다: {self.target.shape}')
        if backend not in ('auto', 'scipy', 'numpy'):
            raise ValueError(f'backend 는 auto/scipy/numpy 중 하나: {backend}')
        self.chunk = int(chunk)
        self._tree = None
        if backend in ('auto', 'scipy'):
            try:
                from scipy.spatial import cKDTree
                self._tree = cKDTree(self.target)
            except ImportError:
                if backend == 'scipy':
                    raise
        self.backend = 'scipy' if self._tree is not None else 'numpy'
        self._t_sq = None if self._tree is not None else np.sum(self.target ** 2, axis=1)

    def _d2(self, Xc):
        """(NumPy 방식 전용) 질의 점 묶음 Xc 와 모든 기준 점 사이의 거리 제곱 행렬 (n, M).
        ||x − q||² = ||x||² + ||q||² − 2 xᵀq 로 한 번에 계산 (반올림 때문에 아주 작은 음수가 나올 수 있음)."""
        return np.sum(Xc ** 2, axis=1)[:, None] + self._t_sq[None, :] - 2.0 * Xc @ self.target.T

    def query(self, X):
        """각 질의 점 X 의 최근접 기준 점 1 개를 찾음. 반환: (거리 (N,), 기준 점 인덱스 (N,)).
        ICP 에서 source 점마다 짝이 될 target 점을 고를 때 씀. NumPy 방식은 메모리를 아끼려고 chunk 단위로 나눠 계산."""
        X = np.asarray(X, dtype=float)
        if self._tree is not None:
            return self._tree.query(X, k=1)
        d = np.empty(len(X))
        j = np.empty(len(X), dtype=int)
        for s in range(0, len(X), self.chunk):
            d2 = self._d2(X[s:s + self.chunk])
            jc = np.argmin(d2, axis=1)
            j[s:s + self.chunk] = jc
            d[s:s + self.chunk] = np.sqrt(np.maximum(d2[np.arange(len(jc)), jc], 0.0))
        return d, j

    def query_k(self, X, k):
        """각 질의 점 X 의 가까운 기준 점 k 개 인덱스 (N, k) 를 가까운 순으로 반환 (k 는 기준 점 수를 넘지 않게 자름).
        법선 계산용 이웃을 모을 때 씀. NumPy 방식은 argpartition 으로 k 개만 고른 뒤 그 안에서만 정렬."""
        X = np.asarray(X, dtype=float)
        k = int(min(k, len(self.target)))
        if self._tree is not None:
            _, j = self._tree.query(X, k=k)
            return j.reshape(len(X), k)
        j = np.empty((len(X), k), dtype=int)
        for s in range(0, len(X), self.chunk):
            d2 = self._d2(X[s:s + self.chunk])
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(d2, part, axis=1), axis=1)
            j[s:s + self.chunk] = np.take_along_axis(part, order, axis=1)
        return j


class LazyNormals:
    """식 (15) 법선을 '짝으로 쓰인 기준 점' 에 대해서만 계산하고 저장 (지도 전체를 매번 계산하지 않음)."""

    def __init__(self, nn, k=8):
        """NearestNeighbor(기준 점 검색기) 와 이웃 수 k 를 받고, 법선 저장소를 NaN(아직 계산 안 함) 으로 채워 둠."""
        self.nn = nn
        self.k = int(k)
        d = nn.target.shape[1]
        self.n = np.full((len(nn.target), d), np.nan)

    def get(self, idx):
        """기준 점 인덱스 idx 의 법선 (len(idx), d) 을 반환.
        아직 계산하지 않은 점만 골라 이웃 k 개의 공분산을 만들고, 가장 작은 고유값의 고유벡터를 법선으로 저장 (식 15).
        이미 계산한 점은 저장된 값을 그대로 씀. 법선 부호(앞/뒤)는 정하지 않음 — 식 (16) 오차 제곱에는 영향 없음."""
        idx = np.asarray(idx, dtype=int)
        need = np.unique(idx[np.isnan(self.n[idx, 0])])
        if len(need):
            Q = self.nn.target
            nb = self.nn.query_k(Q[need], self.k)
            X = Q[nb] - Q[nb].mean(axis=1, keepdims=True)
            C = np.einsum('nki,nkj->nij', X, X) / nb.shape[1]
            _, V = np.linalg.eigh(C)
            self.n[need] = V[:, :, 0]
        return self.n[idx]


# ============================================================================ 점-평면 ICP
@dataclass
class ICP3DResult:
    """icp_p2pl3d 의 결과 묶음 (최종 자세, 수렴 여부, 오차, 짝 수, 종료 이유, 반복별 기록)."""
    T: np.ndarray             # 최종 4x4 (source 좌표 → target 좌표)
    converged: bool
    iterations: int
    rmse: float                # 최종 자세에서 e_i (법선 방향 거리) 의 RMS [m]
    n_inliers: int
    n_points: int
    reason: str = ''
    h_eig: object = None       # 마지막 H 의 고유값 (작은 순)
    history: list = field(default_factory=list)


def icp_p2pl3d(source, target, dof=6, init_T=None, max_iter=100, max_corr_dist=1.0, tol_trans=1e-4,
               tol_rot=1e-5, min_corr=10, normal_k=8, nn=None, normals=None, nn_backend='auto'):
    """3D 점-평면 ICP.  dof=3 이면 식 (17), dof=6 이면 식 (18)–(19).
    source 점군을 target 점군(지도)에 맞추는 변환 T (source → target) 를 가우스-뉴턴 반복으로 구함.
    매 반복: 현재 T 로 source 를 옮김 → 최근접 짝 찾기 → 거리 max_corr_dist 이하만 남김
            → 법선 방향 오차·야코비안으로 H Δx = −g 풀기 → T 갱신 → 변화량이 tol 미만이면 수렴.
    nn 을 넘기면 KD-트리를 다시 만들지 않아 같은 지도에 여러 번 정합할 때 빠름.

    source, target : (N, 3), (M, 3)
    init_T         : 4x4 초기값 (dof=3 이면 z·roll·pitch 는 무시하고 x, y, yaw 만 씀)
    normals        : LazyNormals 또는 (M, 3) 배열. None 이면 이웃 normal_k 개로 계산
    """
    if dof not in (3, 6):
        raise ValueError(f'dof 는 3 또는 6: {dof}')
    src = np.asarray(source, dtype=float)
    if nn is None:
        nn = NearestNeighbor(target, backend=nn_backend)
    tgt = nn.target
    if normals is None:
        normals = LazyNormals(nn, normal_k)
    get_n = normals.get if isinstance(normals, LazyNormals) else (lambda j: np.asarray(normals)[j])
    T = np.eye(4) if init_T is None else np.array(init_T, dtype=float)
    if dof == 3:
        _, _, yaw0 = R_to_rpy(T[:3, :3])
        x = np.array([T[0, 3], T[1, 3], yaw0])
        T = make_T(rot_z(x[2]), [x[0], x[1], 0.0])

    def residuals(T_):
        """자세 T_ 에서 source 를 옮겨 짝을 찾고, 거리 조건을 통과한 짝(inlier)의 값을 반환.
        반환: (회전만 적용한 점 R p (야코비안용), 짝 법선 n, 법선 방향 오차 e (식 16), inlier 수)."""
        RP = src @ T_[:3, :3].T
        P = RP + T_[:3, 3]
        d, j = nn.query(P)
        inl = d <= max_corr_dist
        n = get_n(j[inl])
        e = np.sum(n * (P[inl] - tgt[j[inl]]), axis=1)                    # 식 (16)
        return RP[inl], n, e, int(inl.sum())

    # ---- 반복 상태 초기화
    history = []                # 반복마다 {자세 T, rmse, inlier 수} 를 쌓아 두는 기록 (디버깅·시각화용)
    converged = False           # 수렴 조건을 만족해 빠져나왔는지
    reason = 'max_iter'         # 종료 이유. 중간에 break 하지 않으면 '최대 반복 도달' 로 남음
    h_eig = None                # 마지막 H 의 고유값 (작을수록 그 방향이 잘 안 잡힘 → 퇴화 진단용)
    it = 0                      # max_iter=0 이어도 결과에 반복 횟수를 넣을 수 있게 미리 정의

    # ---- 가우스-뉴턴 반복
    for it in range(1, max_iter + 1):
        # 1) 현재 자세 T 로 source 를 옮기고 짝·법선·오차를 구함
        RP, n, e, n_inl = residuals(T)
        # 이번 반복 시작 시점의 자세와 오차 RMS 를 기록 (짝이 하나도 없으면 rmse 는 NaN)
        history.append({'T': T.copy(), 'rmse': float(np.sqrt(np.mean(e ** 2))) if n_inl else np.nan,
                        'n_inliers': n_inl})
        # 2) 짝이 너무 적으면 미지수(dof 개)를 믿을 만하게 풀 수 없으므로 중단
        if n_inl < max(min_corr, dof):
            reason = f'짝 부족 ({n_inl} < {min_corr})'
            break
        # 3) 야코비안 J (n_inl × dof): 각 오차 e_i 가 자세 변화에 얼마나 민감한지
        if dof == 3:
            # 열 = [∂e/∂t_x, ∂e/∂t_y, ∂e/∂ψ].  ∂e/∂ψ = n_x·(−(Rp)_y) + n_y·(Rp)_x
            J = np.column_stack([n[:, 0], n[:, 1], -n[:, 0] * RP[:, 1] + n[:, 1] * RP[:, 0]])   # 식 (17)
        else:
            # 열 = [∂e/∂δt (3개) = n,  ∂e/∂δφ (3개) = (Rp) × n]
            J = np.column_stack([n, np.cross(RP, n)])                                         # 식 (18)
        # 4) 정규방정식 H Δx = −g 구성 (H = JᵀJ : 근사 헤시안, g = Jᵀe : 기울기)
        H = J.T @ J
        g = J.T @ e
        h_eig = np.linalg.eigvalsh(H)   # 대칭 행렬이라 eigvalsh 사용, 작은 순으로 정렬되어 나옴
        # 5) 자세 변화량 Δx 풀기. H 가 특이(예: 평면 하나만 보여 어떤 방향이 안 잡힘)하면 중단
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            reason = 'H 특이 (풀 수 없음)'
            break
        # 6) 자세 갱신 + 이번 반복의 이동량(step_t [m])·회전량(step_r [rad]) 계산
        if dof == 3:
            # x = [t_x, t_y, ψ] 에 그대로 더한 뒤 T 를 다시 만듦 (z, roll, pitch 는 항상 0)
            x = x + dx
            T = make_T(rot_z(x[2]), [x[0], x[1], 0.0])
            step_t, step_r = np.hypot(dx[0], dx[1]), abs(dx[2])
        else:
            # 회전은 왼쪽에 Exp(δφ) 를 곱해 갱신 (R ← Exp(δφ) R), 이동은 더함 (t ← t + δt)
            T = make_T(so3_exp(dx[3:]) @ T[:3, :3], T[:3, 3] + dx[:3])                       # 식 (19)
            step_t, step_r = np.linalg.norm(dx[:3]), np.linalg.norm(dx[3:])
        # 7) 이동·회전 변화가 둘 다 허용치보다 작으면 수렴으로 보고 종료
        if step_t < tol_trans and step_r < tol_rot:
            converged = True
            reason = 'converged'
            break

    # ---- 최종 평가: 마지막으로 갱신된 T 에서 짝·오차를 다시 구해 결과 품질(rmse, inlier 수)을 계산
    #      (반복 안의 rmse 는 갱신 '전' 자세 기준이라 최종 자세 기준으로 한 번 더 계산)
    _, _, e, n_inl = residuals(T)
    rmse = float(np.sqrt(np.mean(e ** 2))) if n_inl else float('nan')
    history.append({'T': T.copy(), 'rmse': rmse, 'n_inliers': n_inl})
    # 결과 묶음 반환 (T: source → target 변환)
    return ICP3DResult(T=T, converged=converged, iterations=it, rmse=rmse, n_inliers=n_inl,
                       n_points=len(src), reason=reason, h_eig=h_eig, history=history)
