import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Resolve paths
    pkg_dir = get_package_share_directory('turtlebot_simulation')
    pkg_turtlebot_desc = get_package_share_directory('turtlebot_description')

    # 2. Define the individual actions
    is_sim_arg = DeclareLaunchArgument(
        'is_sim',
        default_value='false'
    )
    
    map_frame_arg = DeclareLaunchArgument(
        'map_frame',
        default_value='odom'
    )

    # occupancy_grid_node = Node(
    #     package='grid_mapping',
    #     executable='occupancy_grid',
    #     name='occupancy_grid',
    #     output='screen',
    #     parameters=[{
    #         'is_sim': LaunchConfiguration('is_sim'),
    #         'map_frame': LaunchConfiguration('map_frame'),
    #         'base_frame': 'base_footprint',
    #         'laser_frame': 'rplidar'
    #     }]
    # )

    occupancy_grid_node = Node(
        package='grid_mapping',
        executable='occupancy_grid_original',
        name='occupancy_grid',
        output='screen',
        parameters=[{
            'map_frame': LaunchConfiguration('map_frame'),
            'base_frame': 'base_footprint',
            'laser_frame': 'rplidar',
            'inflation_radius': 0.20
        }]
    )

    rrt_planner_node = Node(
        package='online_motion_planning',
        executable='rrt_tb',
        name='rrt_planner',
        output='screen',
        parameters=[{
            'map_frame': LaunchConfiguration('map_frame')
        }]
    )

    rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', PathJoinSubstitution([pkg_turtlebot_desc, 'rviz', 'turtlebot_rrt_costmap.rviz'])]
        )

    # 3. Return the Description
    return LaunchDescription([
        is_sim_arg,
        map_frame_arg,
        occupancy_grid_node, 
        rrt_planner_node
    ])