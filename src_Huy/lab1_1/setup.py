from setuptools import find_packages, setup

package_name = 'lab1_1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ('share/' + package_name + '/launch', ['launch/navigation_pipeline.launch.py']),
        ('share/' + package_name + '/launch', ['launch/planning_and_control.launch.py']),
        ('share/' + package_name + '/launch', ['launch/turtlebot_nav_online_replanning.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='haadi',
    maintainer_email='haadiakhter@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'node = lab1_1.node:main',
            'occupancy_grid_node = lab1_1.occupancy_grid_node:main',
            'occupancy_grid_node_v2 = lab1_1.occupancy_grid_node_v2:main',
            'map_saver = lab1_1.map_saver:main',
            'path_planner_rrt_star = lab1_1.path_planner_rrt_star:main',
            'path_planner_rrt_star_v2 = lab1_1.path_planner_rrt_star_v2:main',
            'node_waypoint_controller = lab1_1.node_waypoint_controller:main',
        ],
    },
)
