#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voxel_map.py — 정합된 스캔을 쌓아 두는 격자 지도 (2D·3D 공용)

[지도]
  칸 크기 v,  칸 번호  c(p) = ( ⌊p_1 / v⌋, ⌊p_2 / v⌋, ... )        (2D 면 2개, 3D 면 3개)
  k 번째 스캔을 추정 자세로 옮겨 넣음:
      M_k = ∪_{j<k} { T^W_j p  |  p ∈ 스캔 j }                        ... (13)
  칸마다 '처음 들어온 점 하나' 만 둠
    → 서 있는 동안 같은 곳을 계속 찍어도 점이 늘지 않고, 처음 자리가 기준으로 고정됨

[스캔-대-지도 정합]
  T^W_k = ICP( 스캔 k → M_k ),  초기값 T^W_{k−1}                        ... (14)
  ICP 결과가 곧 출발점 기준 자세 (곱셈 누적이 필요 없음)
"""
import numpy as np


class VoxelMap:
    def __init__(self, voxel):
        if not voxel > 0:
            raise ValueError(f'칸 크기는 0 보다 커야 합니다: {voxel}')
        self.voxel = float(voxel)
        self.dim = None
        self._cells = {}
        self._cache = None

    def __len__(self):
        return len(self._cells)

    def add(self, points):
        """월드 좌표 점 (N, d) 를 넣음. 이미 차 있는 칸은 건너뜀. 새로 들어간 점 수를 돌려줌."""
        P = np.asarray(points, dtype=float)
        if P.size == 0:
            return 0
        if P.ndim != 2:
            raise ValueError(f'점 배열은 (N, d) 모양이어야 합니다: {P.shape}')
        if self.dim is None:
            self.dim = P.shape[1]
        elif P.shape[1] != self.dim:
            raise ValueError(f'차원이 다릅니다: 지도 {self.dim}D, 넣은 점 {P.shape[1]}D')
        keys = np.floor(P / self.voxel).astype(np.int64)
        added = 0
        for key, p in zip(map(tuple, keys.tolist()), P):
            if key not in self._cells:
                self._cells[key] = p
                added += 1
        if added:
            self._cache = None
        return added

    def points(self):
        """지도 점 (M, d)"""
        if self._cache is None:
            self._cache = (np.array(list(self._cells.values())) if self._cells
                           else np.empty((0, self.dim or 0)))
        return self._cache
