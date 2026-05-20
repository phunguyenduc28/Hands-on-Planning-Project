import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'online_motion_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'control_tb = online_motion_planning.control_tb:main',
                'rrt_tb = online_motion_planning.rrt_tb:main',
                'frontier_rrt_tb = online_motion_planning.frontier_rrt_tb:main',
                'frontier_birrt_tb = online_motion_planning.frontier_birrt_tb:main',
                'frontier_node = online_motion_planning.frontier_node:main',
                'path_planner_tb = online_motion_planning.path_planner_tb:main',
                'arm_retract_node = online_motion_planning.arm_retract_node:main',
                'map_republisher = online_motion_planning.map_republisher:main',
                'imu_ned_to_enu = online_motion_planning.imu_ned_to_enu:main',
                'path_planner_no_spin = online_motion_planning.path_planner_no_spin:main',
        ],
    },
)
