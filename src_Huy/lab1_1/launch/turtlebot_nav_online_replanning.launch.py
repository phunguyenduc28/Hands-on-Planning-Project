#!/usr/bin/env python3
"""
Launch file for autonomous navigation with online replanning.
Brings up:
  - Occupancy Grid Mapping Node
  - RRT* Path Planner with Online Replanning
  - RViz visualization

Assumes simulation is launched separately via turtlebot_basic.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import os


def generate_launch_description():
    # pkg_turtlebot_desc = FindPackageShare('turtlebot_description')
    
    # Launch arguments
    robot_radius_arg = DeclareLaunchArgument(
        'robot_radius',
        default_value=['0.2'],
        description='Robot radius for collision checking'
    )
    
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value=[os.path.expanduser('~/maps/map_latest.yaml')],
        description='Path to saved map YAML file'
    )
    
    rrt_iterations_arg = DeclareLaunchArgument(
        'rrt_max_iterations',
        default_value=['2000'],
        description='Maximum RRT* iterations'
    )
    
    rrt_step_size_arg = DeclareLaunchArgument(
        'rrt_step_size',
        default_value=['0.3'],
        description='RRT* step size'
    )
    
    replan_interval_arg = DeclareLaunchArgument(
        'replan_interval',
        default_value=['1.0'],
        description='Online replanning interval (seconds)'
    )

    # Occupancy Grid Mapping Node
    occupancy_grid_node = Node(
        package='lab1_1',
        executable='occupancy_grid_node',
        name='occupancy_grid_node',
        output='screen',
        parameters=[{
            'grid_size': 20.0,
            'grid_resolution': 0.05,
            'map_frame': 'world_enu',
            'base_frame': 'turtlebot/base_footprint',
            'laser_frame': 'turtlebot/rplidar',
            'p_occ': 0.9
        }]
    )

    # RRT* Path Planner with Online Replanning (v2)
    path_planner_node = Node(
        package='lab1_1',
        executable='path_planner_rrt_star_v2',
        name='path_planner_rrt_star',
        output='screen',
        parameters=[
            {'robot_radius': LaunchConfiguration('robot_radius')},
            {'map_file': LaunchConfiguration('map_file')},
            {'rrt_max_iterations': LaunchConfiguration('rrt_max_iterations')},
            {'rrt_step_size': LaunchConfiguration('rrt_step_size')},
            {'replan_interval': LaunchConfiguration('replan_interval')},
        ]
    )

    # Waypoint Controller for path execution
    waypoint_controller_node = Node(
        package='lab1_1',
        executable='node_waypoint_controller',
        name='waypoint_controller',
        output='screen',
        parameters=[{
            'k_v': 0.5,
            'k_w': 2.0,
            'dist_tolerance': 0.1,
            'angle_tolerance': 0.1,
            'v_max': 0.5,
            'w_max': 1.0,
        }]
    )

    # RViz Visualization
    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     arguments=['-d', PathJoinSubstitution([pkg_turtlebot_desc, 'rviz', 'turtlebot.rviz'])]
    # )

    return LaunchDescription([
        robot_radius_arg,
        map_file_arg,
        rrt_iterations_arg,
        rrt_step_size_arg,
        replan_interval_arg,
        occupancy_grid_node,
        path_planner_node,
        waypoint_controller_node,
    ])
        # rviz_node,
    
