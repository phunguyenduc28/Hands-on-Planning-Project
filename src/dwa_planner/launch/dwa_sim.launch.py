import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Standalone DWA sim launch.

    Starts the local costmap, DWA service, the goal driver, and RViz2.
    Robot base drivers and odometry must already be running.

    All parameters (map_frame, laser_frame, deadzone, etc.) are read from
    config/dwa_sim_params.yaml — edit that file to change values.

    Usage
    -----
    ros2 launch dwa_planner dwa_sim.launch.py
    # Then click "2D Goal Pose" in RViz2 to send a goal.
    """

    params_file = os.path.join(
        get_package_share_directory('dwa_planner'),
        'config', 'dwa_sim_params.yaml'
    )

    rviz_config = os.path.join(
        get_package_share_directory('turtlebot_description'),
        'rviz', 'turtlebot_dwa_sim.rviz'
    )

    # ── DWA local costmap ────────────────────────────────────────────────────
    dwa_local_costmap_node = Node(
        package='dwa_planner',
        executable='local_costmap',
        name='dwa_local_costmap',
        output='screen',
        parameters=[params_file]
    )

    # ── DWA service ──────────────────────────────────────────────────────────
    dwa_service_node = Node(
        package='dwa_planner',
        executable='dwa_service',
        name='dwa_service_node',
        output='screen',
        parameters=[params_file]
    )

    # ── DWA goal driver ───────────────────────────────────────────────────────
    dwa_goal_node = Node(
        package='dwa_planner',
        executable='dwa_goal_node',
        name='dwa_goal_node',
        output='screen',
        parameters=[params_file]
    )

    # ── RViz2 ─────────────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        dwa_local_costmap_node,
        dwa_service_node,
        dwa_goal_node,
        rviz_node,
    ])