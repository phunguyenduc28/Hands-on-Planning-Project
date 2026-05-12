import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── YAML parameter file ────────────────────────────────────────────────
    params_file = os.path.join(
        get_package_share_directory('online_motion_planning'),
        'config', 'exploration_params.yaml'
    )

    # ── Launch arguments ───────────────────────────────────────────────────
    # Declare allow overriding key values from the CLI without editing the YAML.
    declare_map_frame = DeclareLaunchArgument(
        'map_frame', default_value='world_enu',
        description='TF frame used as the global map frame')

    declare_scenario = DeclareLaunchArgument(
        'scenario', default_value='turtlebot_hoi_circuit1',
        description='Stonefish scenario name (must match a launch in turtlebot_simulation)')
    
    pkg_turtlebot_desc = FindPackageShare('turtlebot_description')

    # ── Sub-launch: simulation ─────────────────────────────────────────────
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('turtlebot_simulation'),
                'launch', 'turtlebot_hoi_circuit1.launch.py'
            )
        )
    )

    # ── Sub-launch: RTAB-Map ───────────────────────────────────────────────
    # NOTE: realsense_d435i_color.launch.py is designed for the real camera but
    # is also used here for simulation (the camera topics are remapped inside
    # the simulation to match what RTAB-Map expects).  Replace with a
    # simulation-specific launch if you have one.
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rtabmap_examples'),
                'launch', 'realsense_d435i_color.launch.py'
            )
        )
    )

    # ── Image crop node ────────────────────────────────────────────────────
    # Produces:
    #   /turtlebot/camera/depth/image_cropped       → RTAB-Map
    #   /turtlebot/camera/depth/image_scan_cropped  → depthimage_to_laserscan
    image_crop_node = Node(
        package='image_utils',
        executable='image_crop_node',
        name='image_crop_node',
        output='screen',
        parameters=[params_file]
    )

    # NOTE: depthimage_to_laserscan is owned by the rtabmap launch.
    # It subscribes to /turtlebot/camera/depth/image_scan_cropped (produced by
    # image_crop_node below) and publishes to /turtlebot/fake_scan.
    # Do NOT launch a second instance here.

    # ── Arm retract node ───────────────────────────────────────────────────
    # Auto-retracts the SwiftPro arm at startup (is_sim=true in YAML).
    arm_retract_node = Node(
        package='online_motion_planning',
        executable='arm_retract_node',
        name='arm_retract_node',
        output='screen',
        parameters=[params_file]
    )

    # ── Global inflated costmap (grid_mapping) ─────────────────────────────
    # Subscribes to RTAB-Map's /map → inflates → /inflated_map
    # Used by frontier_node (BFS + BiRRT*).
    global_costmap_node = Node(
        package='grid_mapping',
        executable='occupancy_grid_original',
        name='global_costmap',      # matches YAML section 'global_costmap'
        output='screen',
        parameters=[params_file]
    )

    # ── DWA local costmap (dwa_planner) ───────────────────────────────────
    # Builds a small sliding-window costmap from the real laser scan.
    # Publishes /map_dwa (raw) and /inflated_map_dwa (inflated).
    dwa_local_costmap_node = Node(
        package='dwa_planner',
        executable='local_costmap',
        name='dwa_local_costmap',   # matches YAML section 'dwa_local_costmap'
        output='screen',
        parameters=[params_file]
    )

    # ── DWA service ────────────────────────────────────────────────────────
    # Provides /dwa/compute_velocity; uses /inflated_map_dwa.
    dwa_service_node = Node(
        package='dwa_planner',
        executable='dwa_service',
        name='dwa_service_node',    # matches YAML section 'dwa_service_node'
        output='screen',
        parameters=[params_file]
    )

    # ── Frontier detection node ────────────────────────────────────────────
    frontier_node = Node(
        package='online_motion_planning',
        executable='frontier_node',
        name='frontier_node',
        output='screen',
        parameters=[params_file]
    )

    # ── Path planner (BiRRT* + DWA waypoint execution) ─────────────────────
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
            arguments=['-d', PathJoinSubstitution([pkg_turtlebot_desc, 'rviz', 'turtlebot_frontier_rrt_costmap.rviz'])]
        )

    # ── Timing ────────────────────────────────────────────────────────────
    # Nodes that need the simulation camera to be ready (5 s).
    sensor_nodes = TimerAction(
        period=5.0,
        actions=[image_crop_node,
                 arm_retract_node,
                 rviz_node]
    )

    # RTAB-Map needs the simulation running and camera topics available (8 s).
    rtabmap_delayed = TimerAction(period=8.0, actions=[rtabmap_launch])

    # Mapping and planning nodes need enough map data to start (30 s).
    planning_nodes = TimerAction(
        period=5.0,
        actions=[
            global_costmap_node,
            dwa_local_costmap_node,
            dwa_service_node,
            frontier_node,
            path_planner_node,
        ]
    )

    return LaunchDescription([
        declare_map_frame,
        declare_scenario,
        sim_launch,
        sensor_nodes,
        rtabmap_delayed,
        planning_nodes,
    ])
