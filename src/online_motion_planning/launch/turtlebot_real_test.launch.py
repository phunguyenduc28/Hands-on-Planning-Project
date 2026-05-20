import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def xterm(title):
    return f'xterm -title "{title}" -geometry 120x30 -e'


def generate_launch_description():

    # ── YAML parameter file ────────────────────────────────────────────────
    # Real-robot frames: map_frame=odom, laser_frame=rplidar (no turtlebot/ prefix)
    params_file = os.path.join(
        get_package_share_directory('online_motion_planning'),
        'config', 'exploration_real_params.yaml'
    )

    # ── Launch arguments ───────────────────────────────────────────────────
    declare_map_frame = DeclareLaunchArgument(
        'map_frame', default_value='odom',
        description='TF frame used as the global map frame')

    pkg_turtlebot_desc = FindPackageShare('turtlebot_description')

    # ── RTAB-Map ───────────────────────────────────────────────────────────
    # Opens in its own xterm window and builds the map from the D435i camera.
    rtabmap_launch = ExecuteProcess(
        cmd=[
            'xterm', '-title', 'RTAB-Map', '-geometry', '120x30', '-e',
            'ros2', 'launch', 'rtabmap_examples', 'realsense_d435i_color_real_robot.launch.py'
        ],
        output='screen'
    )

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
    # NOTE: depthimage_to_laserscan is owned by the rtabmap launch — do NOT
    # launch a second instance here.
    image_crop_node = Node(
        package='image_utils',
        executable='image_crop_node',
        name='image_crop_node',
        output='screen',
        parameters=[params_file],
        remappings=[
                        # ('depth', '/turtlebot/camera/depth/image_depth'),
                        ('/turtlebot/camera/color/image_color', '/turtlebot/camera/color/image_compressed'),
                        ('/turtlebot/camera/depth/image_depth', '/turtlebot/camera/depth/image_rect_raw'),
                        # ('scan', '/turtlebot/fake_scan'),
                        ],
    )

    # ── Map republisher ────────────────────────────────────────────────────
    # RTAB-Map publishes /map infrequently. This node caches the latest map
    # and republishes it at a steady rate so downstream nodes keep running.
    map_republisher_node = Node(
        package='online_motion_planning',
        executable='map_republisher',
        name='map_republisher',
        output='screen',
        prefix=xterm('MapRepublisher'),
        parameters=[{'input_topic': '/map',
                     'output_topic': '/map_fast',
                     'publish_rate': 2.0}]
    )

    # ── Global costmap (grid_mapping) ──────────────────────────────────────
    # Subscribes to /map_fast (republished at steady rate) → inflates → /inflated_map.
    global_costmap_node = Node(
        package='grid_mapping',
        executable='occupancy_grid_original',
        name='global_costmap',
        output='screen',
        # prefix=xterm('GlobalCostmap'),
        parameters=[params_file],
        remappings=[('/map', '/map_fast')]
    )

    # ── DWA local costmap (dwa_planner) ───────────────────────────────────
    # Builds a sliding-window costmap from the real RPLidar /turtlebot/scan.
    # Publishes /map_dwa (raw) and /inflated_map_dwa (inflated).
    dwa_local_costmap_node = Node(
        package='dwa_planner',
        executable='local_costmap',
        name='dwa_local_costmap',
        output='screen',
        # prefix=xterm('DWA-LocalCostmap'),
        parameters=[params_file]
    )

    # ── DWA service ────────────────────────────────────────────────────────
    # Provides /dwa/compute_velocity.  Dead zone compensation active:
    # linear >= 0.3 m/s, angular >= 0.5 rad/s for any non-zero command.
    dwa_service_node = Node(
        package='dwa_planner',
        executable='dwa_service',
        name='dwa_service_node',
        output='screen',
        prefix=xterm('DWA-Service'),
        parameters=[params_file],
        remappings=[('/turtlebot/odom', '/odometry/filtered')]
    )

    # ── Frontier detection ─────────────────────────────────────────────────
    frontier_node = Node(
        package='online_motion_planning',
        executable='frontier_node',
        name='frontier_node',
        output='screen',
        prefix=xterm('FrontierExploration'),
        parameters=[params_file],
        remappings=[('/map', '/map_fast')]
    )

    # ── Path planner (BiRRT* + DWA waypoint execution) ─────────────────────
    # is_sim=false → arm_is_retracted=True from startup; no arm_retract_node needed.
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
        arguments=['-d', PathJoinSubstitution(
            [pkg_turtlebot_desc, 'rviz', 'turtlebot_frontier_rrt_costmap_real_robot.rviz'])]
    )

    # ── Timing ────────────────────────────────────────────────────────────
    # Image crop and RViz start shortly after — they only need the camera driver.
    sensor_nodes = TimerAction(
        period=2.0,
        actions=[image_crop_node, rviz_node]
    )

    # Planning nodes wait for RTAB-Map to have built enough map.
    planning_nodes = TimerAction(
        period=15.0,
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
        map_republisher_node,
        ekf_node,
        rtabmap_launch,
        sensor_nodes,
        planning_nodes,
    ])
