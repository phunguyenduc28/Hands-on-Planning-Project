import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'dwa_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='huy',
    maintainer_email='dohuy9379@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dwa_service = dwa_planner.control_tb:main',
            'local_costmap = dwa_planner.occupancy_grid_local:main',
            'dwa_goal_node = dwa_planner.dwa_goal_node:main',
            'dwa_old = dwa_planner.control_tb_old:main',
            'local_costmap_old = dwa_planner.occupancy_grid_old:main',
        ],
    },
)
