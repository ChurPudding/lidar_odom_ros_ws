#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cloud_io.py — sensor_msgs/PointCloud2 ↔ NumPy 점 배열 (N, 3)

[받기] PointCloud2 의 data 는 점 하나가 point_step 바이트인 바이트 열
  각 필드(x, y, z)는 점 안에서 offset 바이트 위치에 datatype 형으로 들어 있음
  → NumPy 구조체 형(dtype)을 offset·형·point_step 으로 만들어 바이트를 그대로 해석 (복사 없이 빠름)
  행이 여러 줄(height > 1)이고 줄 끝에 빈칸이 있으면(row_step > width × point_step) 줄마다 잘라서 해석
  NaN/inf 가 있는 점은 버림 (Lidar_odom_ws 의 bag_io.read_clouds 와 같은 처리)

[보내기] (N, 3) 점을 x, y, z float32 세 필드의 PointCloud2 로 만듦 (point_step = 12 바이트)
"""
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

# sensor_msgs/PointField datatype 번호 → NumPy 형
_PF_DTYPE = {PointField.INT8: 'i1', PointField.UINT8: 'u1', PointField.INT16: 'i2', PointField.UINT16: 'u2',
             PointField.INT32: 'i4', PointField.UINT32: 'u4', PointField.FLOAT32: 'f4', PointField.FLOAT64: 'f8'}


def cloud_to_xyz(msg: PointCloud2):
    """PointCloud2 → (N, 3) float64 점 배열 (NaN/inf 점 제거)"""
    fields = {f.name: f for f in msg.fields}
    missing = [n for n in ('x', 'y', 'z') if n not in fields]
    if missing:
        raise KeyError(f'PointCloud2 에 필드 {missing} 가 없습니다. 있는 필드: {list(fields)}')
    e = '>' if msg.is_bigendian else '<'
    dt = np.dtype({'names': ['x', 'y', 'z'],
                   'formats': [e + _PF_DTYPE[fields[n].datatype] for n in ('x', 'y', 'z')],
                   'offsets': [fields[n].offset for n in ('x', 'y', 'z')],
                   'itemsize': msg.point_step})
    buf = memoryview(msg.data).cast('B') if not isinstance(msg.data, (bytes, bytearray)) else msg.data
    w, h = int(msg.width), int(msg.height)
    if h <= 1 or msg.row_step == w * msg.point_step:
        arr = np.frombuffer(buf, dtype=dt, count=w * max(h, 1))           # 줄 사이 빈칸 없음 → 한 번에
    else:
        arr = np.concatenate([np.frombuffer(buf, dtype=dt, count=w, offset=r * msg.row_step) for r in range(h)])
    P = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(float)
    return P[np.isfinite(P).all(axis=1)]


def stamp_to_sec(stamp):
    """builtin_interfaces/Time → 초 (float)"""
    return stamp.sec + stamp.nanosec * 1e-9


def xyz_to_cloud(points, header):
    """(N, 3) 점 → PointCloud2 (x, y, z float32). header 는 stamp·frame_id 를 채운 std_msgs/Header"""
    P = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(P)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(('x', 'y', 'z'))]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(P)
    msg.is_dense = True
    msg.data = P.tobytes()
    return msg
