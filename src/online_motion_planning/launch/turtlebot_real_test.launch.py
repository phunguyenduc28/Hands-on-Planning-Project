import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── YAML parameter file ────────────────────────────────────────────────
    # Real-robot frames: map_frame=odom, laser_frame=rplidar (no turtlebot/ prefix)
    params_file = os.path.join(
        get_package_share_directory('online_motion_planning'),
        'config', 'exploration_real_params.yaml'
    )

    pkg_turtlebot_desc = FindPackageShare('turtlebot_description')

    # ── Global costmap (grid_mapping) ──────────────────────────────────────
    # Subscribes to RTAB-Map's /map → inflates → /inflated_map.
    # RTAB-Map must be running before this node starts.
    global_costmap_node = Node(
        package='grid_mapping',
        executable='occupancy_grid_original',
        name='global_costmap',      # matches YAML section 'global_costmap'
        output='screen',
        parameters=[params_file]
    )

    # ── DWA local costmap (dwa_planner) ───────────────────────────────────
    # Builds a sliding-window costmap from the real RPLidar /turtlebot/scan.
    # Publishes /map_dwa (raw) and /inflated_map_dwa (inflated).
    dwa_local_costmap_node = Node(
        package='dwa_planner',
        executable='local_costmap',
        name='dwa_local_costmap',   # matches YAML section 'dwa_local_costmap'
        output='screen',
        parameters=[params_file]
    )

    # ── DWA service ────────────────────────────────────────────────────────
    # Provides /dwa/compute_velocity.  Dead zone compensation active:
    # linear >= 0.3 m/s, angular >= 0.5 rad/s for any non-zero command.
    # dwa_service_node = Node(
    #     package='dwa_planner',
    #     executable='dwa_service',
    #     name='dwa_service_node',    # matches YAML section 'dwa_service_node'
    #     output='screen',
    #     parameters=[params_file]
    # )

    # ── Frontier detection ─────────────────────────────────────────────────
    frontier_node = Node(
        package='online_motion_planning',
        executable='frontier_node',
        name='frontier_node',
        output='screen',
        parameters=[params_file]
    )

    # ── Path planner (BiRRT* + DWA waypoint execution) ─────────────────────
    # is_sim=false → arm_is_retracted=True from startup; no arm_retract_node needed.
    path_planner_node = Node(
        package='online_motion_planning',
        executable='path_planner_tb',
        name='path_planner_tb',
        output='screen',
        parameters=[params_file]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution(
            [pkg_turtlebot_desc, 'rviz', 'turtlebot_frontier_rrt_costmap_real_robot.rviz'])]
    )

    # ── Timing ────────────────────────────────────────────────────────────
    # Small delay to allow RTAB-Map and robot drivers to publish first TF/scan.
    planning_nodes = TimerAction(
        period=5.0,
        actions=[
            global_costmap_node,
            dwa_local_costmap_node,
            # dwa_service_node,
            frontier_node,
            path_planner_node,
            rviz_node,
        ]
    )

    return LaunchDescription([
        planning_nodes,
    ])
