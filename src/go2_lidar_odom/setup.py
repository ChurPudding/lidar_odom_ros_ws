# 패키지 설치 설정 (ament_python)
#   파이썬 모듈: go2_lidar_odom, go2_lidar_odom.core
#   실행 파일:   lidar_odom_node  (ros2 run go2_lidar_odom lidar_odom_node)
#   함께 설치:   config/*.yaml, launch/*.launch.py  → share/go2_lidar_odom/
from glob import glob

from setuptools import setup

package_name = 'go2_lidar_odom'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, f'{package_name}.core'],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml', 'README.md']),
        (f'share/{package_name}/config', glob('config/*.yaml')),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hyosin Lee',
    maintainer_email='hyoshin8325@hknu.ac.kr',
    description='Go2 L1 점군만 3D 점-평면 ICP 오도메트리 (3자유도)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lidar_odom_node = go2_lidar_odom.node:main',
        ],
    },
)
