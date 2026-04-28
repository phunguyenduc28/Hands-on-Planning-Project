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
    
    launch_file = os.path.join(pkg_dir, 'launch', 'turtlebot_basic.launch.py')
    scenario_file = os.path.join(pkg_dir, 'scenarios', 'turtlebot_hoi_circuit1.scn')

    # 2. Define the individual actions
    base_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments={
            'scenario_description': scenario_file
        }.items()
    )

    is_sim_arg = DeclareLaunchArgument(
        'is_sim',
        default_value='true'
    )
    
    map_frame_arg = DeclareLaunchArgument(
        'map_frame',
        default_value='world_enu'
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
    #         'laser_frame': 'turtlebot/rplidar'
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
            'laser_frame': 'turtlebot/rplidar',
            'inflation_radius': 0.28
        }]
    )

    rrt_planner_node = Node(
        package='online_motion_planning',
        executable='frontier_rrt_tb',
        name='rrt_planner',
        output='screen',
        parameters=[{
            'map_frame': LaunchConfiguration('map_frame')
        }]
    )

    delayed_nodes = TimerAction(
        period=5.0,
        actions=[
            occupancy_grid_node,
            rrt_planner_node
        ]
    )

    # 3. Return the Description
    return LaunchDescription([
        is_sim_arg,
        map_frame_arg,
        base_simulation,
        delayed_nodes
    ])