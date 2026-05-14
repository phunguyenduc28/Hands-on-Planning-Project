import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Standalone DWA test launch.

    Starts the local costmap, DWA service, and the goal driver.
    Robot base drivers and odometry must already be running.

    All parameters (map_frame, laser_frame, deadzone, etc.) are read from
    config/dwa_test_params.yaml — edit that file to change values.

    Usage
    -----
    ros2 launch dwa_planner dwa_test.launch.py
    # Then click "2D Goal Pose" in RViz2 to send a goal.
    """

    params_file = os.path.join(
        get_package_share_directory('dwa_planner'),
        'config', 'dwa_test_params.yaml'
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

    return LaunchDescription([
        dwa_local_costmap_node,
        dwa_service_node,
        dwa_goal_node,
    ])
