#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_poses.py — 노드가 적은 자세 CSV 와 오프라인 결과(run_icp3d_bag.py 의 poses.csv)를 비교 (검증용, 실행에는 필요 없음)

같은 점군 프레임끼리 header.stamp 로 짝지어 비교:
  위치 차이  Δp_k = || (x, y)_노드 − (x, y)_오프라인 ||
  방향 차이  Δψ_k = | wrap(ψ_노드 − ψ_오프라인) |,   wrap(a) = (a + 180°) mod 360° − 180°
노드가 프레임을 하나도 건너뛰지 않았으면 두 결과는 같아야 함
  (오프라인 CSV 는 x, y 를 소수 다섯째 자리, yaw 를 넷째 자리까지만 적어서 그만큼의 차이(약 1e-5 m, 5e-5°)는 반올림)
노드가 프레임을 건너뛰었으면 그 뒤로는 결과가 달라지는 것이 정상 (초기값·지도가 달라짐)

사용:
  python3 compare_poses.py <노드 CSV> <오프라인 poses.csv>
"""
import csv
import sys

import numpy as np


def load(path, stamp_col, cols):
    """CSV 에서 (시각 배열, 값 배열 (N, len(cols))) 를 읽음"""
    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    st = np.array([float(r[stamp_col]) for r in rows])
    v = np.array([[float(r[c]) for c in cols] for r in rows])
    return st, v


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    s_node, v_node = load(sys.argv[1], 'stamp', ['x', 'y', 'yaw_deg'])
    s_off, v_off = load(sys.argv[2], 'stamp', ['x', 'y', 'yaw_deg'])
    # 시각으로 짝짓기 (1 ms 안이면 같은 프레임)
    j = np.searchsorted(s_off, s_node)
    j = np.clip(j, 1, len(s_off) - 1)
    j = np.where(np.abs(s_off[j - 1] - s_node) < np.abs(s_off[j] - s_node), j - 1, j)
    ok = np.abs(s_off[j] - s_node) < 1e-3
    a, b = v_node[ok], v_off[j[ok]]
    dp = np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])
    dyaw = np.abs((a[:, 2] - b[:, 2] + 180.0) % 360.0 - 180.0)
    print(f'노드 프레임 {len(s_node)}개, 오프라인 프레임 {len(s_off)}개, 짝지은 프레임 {ok.sum()}개 '
          f'(노드가 받지 못한 프레임 {len(s_off) - ok.sum()}개)')
    print(f'위치 차이 최대 {dp.max():.2e} m (중앙값 {np.median(dp):.2e}), 방향 차이 최대 {dyaw.max():.2e}°')
    print(f'끝 자세  노드      x {v_node[-1, 0]:+.3f} y {v_node[-1, 1]:+.3f} m, yaw {v_node[-1, 2]:+.2f}°')
    print(f'         오프라인  x {v_off[-1, 0]:+.3f} y {v_off[-1, 1]:+.3f} m, yaw {v_off[-1, 2]:+.2f}°')
    same = ok.sum() == len(s_off) and dp.max() < 1e-4 and dyaw.max() < 1e-3
    print('판정: 같음 (차이는 CSV 반올림 수준)' if same else
          '판정: 다름 — 노드가 프레임을 건너뛰었는지(위의 받지 못한 프레임 수) 먼저 확인하세요')


if __name__ == '__main__':
    main()
