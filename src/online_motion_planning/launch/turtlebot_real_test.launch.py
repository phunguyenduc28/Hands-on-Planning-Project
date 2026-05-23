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

    # ── EKF localisation (DISABLED — replaced by custom localization node) ────
    ekf_params = os.path.join(
        get_package_share_directory('online_motion_planning'),
        'config', 'ekf_sim_params_real.yaml'
    )
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_node',
        output='screen',
        parameters=[ekf_params]
    )

    # ── Custom EKF localisation (wheel encoders + IMU) ─────────────────────
    # Subscribes: /turtlebot/joint_states, /turtlebot/imu
    # Publishes:  /turtlebot/odom_ekf  +  TF odom→base_footprint
    # localization_node = Node(
    #     package='localization',
    #     executable='localization',
    #     name='differential_drive_ekf',
    #     output='screen',
    #     parameters=[{
    #         'odom_frame':             'odom',
    #         'base_frame':             'base_footprint',
    #         'wheel_left_joint_name':  'wheel_left_joint',
    #         'wheel_right_joint_name': 'wheel_right_joint',
    #         # publish_tf=True: broadcasts odom_ekf→odom correction transform.
    #         # No conflict with robot driver's odom→base_footprint.
    #         'publish_tf':             True,
    #     }],
    # )
    # ── Image crop node ────────────────────────────────────────────────────
    # Produces:
    #   /turtlebot/camera/depth/image_cropped        → depthimage_to_laserscan
    #   /turtlebot/camera/depth/camera_info_cropped  → depthimage_to_laserscan
    #   /turtlebot/camera/color/image_cropped        → rtabmap_sync_node
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

    # ── Depth image → fake 2-D laser scan ─────────────────────────────────
    # Converts the bottom-cropped depth image into a LaserScan on
    # /turtlebot/fake_scan used by global_scan_map_node and DWA.
    # Starts after image_crop_node (both in sensor_nodes at t=2 s).
    depth_to_laserscan_node = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depthimage_to_laserscan',
        output='screen',
        remappings=[
            ('depth',            '/turtlebot/camera/depth/image_cropped'),
            ('depth_camera_info','/turtlebot/camera/depth/camera_info_cropped'),
            ('scan',             '/turtlebot/fake_scan'),
        ],
        parameters=[{
            'range_min':    0.28,
            'range_max':    2.0,
            'output_frame': 'camera_link',
        }],
    )

    # ── RTAB-Map sync node ────────────────────────────────────────────────
    # Synchronises color image, depth image, color camera_info, depth camera_info,
    # and fake scan so RTAB-Map receives a coherent time base on all five topics.
    # Must start after image_crop_node produces the cropped topics.
    # rtabmap_sync_node = Node(
    #     package='image_utils',
    #     executable='rtabmap_sync_node',
    #     name='rtabmap_sync_node',
    #     output='screen',
    #     parameters=[{'slop': 0.5, 'queue_size': 50}],
    # )

    # ── Global scan map (fake 2-D lidar → persistent occupancy grid) ─────────
    # Same log-odds ray-casting as the DWA local costmap but with a fixed
    # origin and no clearing, so the map accumulates over the full exploration.
    # Publishes /map_scan (raw) and /inflated_map_scan (inflated).
    global_scan_map_node = Node(
        package='grid_mapping',
        executable='scan_map_global',
        name='global_scan_map_node',
        output='screen',
        prefix=xterm('Global_costmap'),
        parameters=[{
            'map_size':         30.0,
            'map_resolution':   0.05,
            # 'map_frame':        'odom_ekf',
            'map_frame':        'odom',
            'laser_frame':      'camera_link',
            'scan_topic':       '/turtlebot/fake_scan',
            'odom_topic':       '/turtlebot/odom',          # raw wheel odom
            # 'odom_topic':       '/turtlebot/odom_ekf',  # EKF-fused odom
            'p_occ':            0.85,
            'inflation_radius': 0.18,
            'publish_rate':     2.0,
            'range_min':        0.28,
            'range_max':        2.0,
            # 'clear_on_max_range': True,   # creates phantom frontiers with depth-image scan
            'clear_on_max_range': False,
        }],
        # remappings=[('/turtlebot/odom', '/odometry/filtered')]  # old EKF topic
        # remappings=[('/turtlebot/odom', '/turtlebot/odom_ekf')]

    )

    # ── Map republisher (DISABLED — replaced by global_scan_map_node) ────────
    # map_republisher_node = Node(
    #     package='online_motion_planning',
    #     executable='map_republisher',
    #     name='map_republisher',
    #     output='screen',
    #     prefix=xterm('MapRepublisher'),
    #     parameters=[{'input_topic': '/map',
    #                  'output_topic': '/map_fast',
    #                  'publish_rate': 2.0}]
    # )

    # ── Global costmap (DISABLED — replaced by global_scan_map_node) ──────────
    # global_scan_map_node publishes /map_scan (raw) and /inflated_map_scan
    # (inflated) directly at a steady rate, so neither a republisher nor a
    # separate inflator is needed.
    # global_costmap_node = Node(
    #     package='grid_mapping',
    #     executable='occupancy_grid_original',
    #     name='global_costmap',
    #     output='screen',
    #     parameters=[params_file],
    #     remappings=[('/map', '/map_fast')]
    # )

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
        remappings=[
            ('/turtlebot/odom', '/odometry/filtered'),  # old robot_localization EKF
            # ('/turtlebot/odom', '/turtlebot/odom_ekf'),
        ]
    )

    # ── Frontier detection ─────────────────────────────────────────────────
    # /map        → /map_scan          (raw scan map for frontier cell detection)
    # /inflated_map → /inflated_map_scan (inflated scan map for BFS reachability)
    frontier_node = Node(
        package='online_motion_planning',
        executable='frontier_node',
        name='frontier_node',
        output='screen',
        prefix=xterm('FrontierExploration'),
        parameters=[params_file],
        remappings=[
            ('/map',            '/map_scan'),
            ('/inflated_map',   '/inflated_map_scan'),
            ('/turtlebot/odom', '/odometry/filtered'),  # old robot_localization EKF
            # ('/turtlebot/odom', '/turtlebot/odom_ekf'),
        ]
    )

    # ── Path planner (BiRRT* + DWA waypoint execution) ─────────────────────
    # is_sim=false → arm_is_retracted=True from startup; no arm_retract_node needed.
    # /inflated_map → /inflated_map_scan (inflated scan map for BiRRT* binary_map)
    path_planner_node = Node(
        package='online_motion_planning',
        executable='path_planner_tb',
        name='path_planner_tb',
        output='screen',
        prefix=xterm('PathPlanner'),
        parameters=[params_file],
        remappings=[
            ('/turtlebot/odom', '/odometry/filtered'),  # old robot_localization EKF
            # ('/turtlebot/odom', '/turtlebot/odom_ekf'),
            ('/inflated_map',   '/inflated_map_scan'),
        ]
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
        # actions=[image_crop_node, depth_to_laserscan_node, rtabmap_sync_node, global_scan_map_node, rviz_node]
        actions=[image_crop_node, depth_to_laserscan_node, global_scan_map_node, rviz_node]

    )

    # Planning nodes wait for the scan map to accumulate enough data.
    # 8 s is sufficient — scan map starts at t=2 s and builds immediately.
    planning_nodes = TimerAction(
        period=8.0,
        actions=[
            dwa_local_costmap_node,
            dwa_service_node,
            frontier_node,
            path_planner_node,
        ]
    )

    return LaunchDescription([
        declare_map_frame,
        ekf_node,          # DISABLED — replaced by localization_node
        # localization_node,
        rtabmap_launch,
        sensor_nodes,
        planning_nodes,
    ])
