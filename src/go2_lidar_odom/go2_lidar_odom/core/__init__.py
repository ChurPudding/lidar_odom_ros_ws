"""점군만 3D 점-평면 ICP 코어 (Lidar_odom_ws/lidar_odom 에서 그대로 복사, 수정 없음).

  icp3d.py     점-평면 ICP (3자유도 / 6자유도), 법선, 최근접
  scan3d.py    L1 점 → 몸체 3D 점, 높이·거리로 고르기
  scan2d.py    장착 상수 읽기, L1 → 몸체 좌표 변환 (scan3d 가 씀)
  voxel_map.py 칸 지도 (오프라인 실행과 비교할 때 기준)
"""
