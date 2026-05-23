import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def xterm(title):
    """Return an xterm prefix that opens a named terminal for the node."""
    return f'xterm -title "{title}" -geometry 120x30 -e'


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
                'launch', 'turtlebot_hoi_circuit2.launch.py'
            )
        )
    )

    # ── Sub-launch: RTAB-Map ───────────────────────────────────────────────
    # Uses ExecuteProcess instead of IncludeLaunchDescription so xterm can be
    # prepended — IncludeLaunchDescription inlines into the parent and has no
    # prefix support.
    rtabmap_launch = ExecuteProcess(
        cmd=[
            'xterm', '-title', 'RTAB-Map', '-geometry', '120x30', '-e',
            'ros2', 'launch', 'rtabmap_examples', 'realsense_d435i_color.launch.py'
        ],
        output='screen'
    )

    # ── IMU NED→ENU converter ─────────────────────────────────────────────
    # Stonefish IMU is in world_NED; negate yaw/omega_z before EKF fusion.
    imu_converter_node = Node(
        package='online_motion_planning',
        executable='imu_ned_to_enu',
        name='imu_ned_to_enu',
        output='screen',
        # prefix=xterm('IMU-Converter'),
    )

    # ── EKF localisation (robot_localization) ─────────────────────────────
    # Fuses /turtlebot/odom + /turtlebot/sensors/imu_enu (converted).
    # Publishes /odometry/filtered.
    ekf_params = os.path.join(
        get_package_share_directory('online_motion_planning'),
        'config', 'ekf_sim_params.yaml'
    )
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_node',
        output='screen',
        # prefix=xterm('EKF-Localisation'),
        parameters=[ekf_params]
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
        # prefix=xterm('ImageCrop'),
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
        # prefix=xterm('ArmRetract'),
        parameters=[params_file]
    )

    # ── Global inflated costmap (grid_mapping) ─────────────────────────────
    # Subscribes to RTAB-Map's /map → inflates → /inflated_map
    # Used by frontier_node (BFS + BiRRT*).
    global_costmap_node = Node(
        package='grid_mapping',
        executable='occupancy_grid_original',
        name='global_costmap',
        output='screen',
        # prefix=xterm('GlobalCostmap'),
        parameters=[params_file]
    )

    dwa_local_costmap_node = Node(
        package='dwa_planner',
        executable='local_costmap',
        name='dwa_local_costmap',
        output='screen',
        # prefix=xterm('DWA-LocalCostmap'),
        parameters=[params_file]
    )

    dwa_service_node = Node(
        package='dwa_planner',
        executable='dwa_service',
        name='dwa_service_node',
        output='screen',
        prefix=xterm('DWA-Service'),
        parameters=[params_file],
        remappings=[('/turtlebot/odom', '/odometry/filtered')]
    )

    frontier_node = Node(
        package='online_motion_planning',
        executable='frontier_node',
        name='frontier_node',
        output='screen',
        prefix=xterm('FrontierExploration'),
        parameters=[params_file]
    )

    path_planner_node = Node(
        package='online_motion_planning',
        executable='path_planner_tb',
        name='path_planner_tb',
        output='screen',
        prefix=xterm('PathPlanner'),
        parameters=[params_file],
        remappings=[('/turtlebot/odom', '/odometry/filtered')]
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
        imu_converter_node,
        ekf_node,
        sensor_nodes,
        rtabmap_delayed,
        planning_nodes,
    ])
