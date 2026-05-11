#!/usr/bin/env python3
"""
Launch file for Path Planning + Waypoint Controller
(Without occupancy grid mapping and map saving)

Use this when you already have a saved occupancy map and just want to:
1. Plan paths
2. Execute waypoints
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    
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
    
    # Nodes
    path_planner_node = Node(
        package='lab1_1',
        executable='path_planner_rrt_star',
        name='path_planner_rrt_star',
        parameters=[
            {'robot_radius': LaunchConfiguration('robot_radius')},
            {'map_file': LaunchConfiguration('map_file')},
            {'rrt_max_iterations': LaunchConfiguration('rrt_max_iterations')},
            {'rrt_step_size': LaunchConfiguration('rrt_step_size')},
        ]
    )
    
    controller_node = Node(
        package='lab1_1',
        executable='node_waypoint_controller',
        name='waypoint_controller',
        parameters=[
            {'k_v': 0.5},
            {'k_w': 2.0},
            {'dist_tolerance': 0.1},
            {'angle_tolerance': 0.1},
            {'v_max': 0.5},
            {'w_max': 1.0},
        ]
    )
    
    return LaunchDescription([
        robot_radius_arg,
        map_file_arg,
        rrt_iterations_arg,
        rrt_step_size_arg,
        path_planner_node,
        controller_node,
    ])
