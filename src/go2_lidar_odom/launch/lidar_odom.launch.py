#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lidar_odom.launch.py — 점군만 3D 점-평면 ICP 오도메트리 노드 실행

  ros2 launch go2_lidar_odom lidar_odom.launch.py
  ros2 launch go2_lidar_odom lidar_odom.launch.py normal_k:=25
  ros2 launch go2_lidar_odom lidar_odom.launch.py params_file:=/내/설정.yaml pose_log_csv:=/tmp/poses.csv

인자 (비워 두면 params_file 의 값을 그대로 씀)
  params_file   설정 yaml (기본: 설치된 config/lidar_odom.yaml)
  normal_k      평면 이웃 점 수 k
  dof           3 또는 6
  cloud_topic   L1 점군 토픽
  qos_depth     받기 큐 깊이 (녹화를 천천히 재생해 모든 프레임을 처리할 때 크게)
  pose_log_csv  프레임별 자세 CSV 경로
  use_sim_time  true / false (기본 false)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# (launch 인자 이름, 파라미터로 넘길 때의 형 변환)
_OVERRIDES = [('normal_k', int), ('dof', int), ('cloud_topic', str), ('qos_depth', int), ('pose_log_csv', str)]


def _make_node(context):
    """빈 문자열이 아닌 launch 인자만 골라 yaml 위에 덮어씀"""
    overrides = {}
    for name, conv in _OVERRIDES:
        v = LaunchConfiguration(name).perform(context)
        if v != '':
            overrides[name] = conv(v)
    overrides['use_sim_time'] = LaunchConfiguration('use_sim_time').perform(context).lower() in ('true', '1')
    return [Node(package='go2_lidar_odom', executable='lidar_odom_node', name='lidar_odom', output='screen',
                 parameters=[LaunchConfiguration('params_file').perform(context), overrides])]


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('go2_lidar_odom'), 'config', 'lidar_odom.yaml')
    args = [DeclareLaunchArgument('params_file', default_value=default_params, description='설정 yaml')]
    args += [DeclareLaunchArgument(n, default_value='', description='비우면 yaml 값') for n, _ in _OVERRIDES]
    args += [DeclareLaunchArgument('use_sim_time', default_value='false')]
    return LaunchDescription(args + [OpaqueFunction(function=_make_node)])
