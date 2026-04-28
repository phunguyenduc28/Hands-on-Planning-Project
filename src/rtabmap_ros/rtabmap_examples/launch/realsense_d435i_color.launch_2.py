# Requirements:
#   A realsense D435i
#   Install realsense2 ros2 package (ros-$ROS_DISTRO-realsense2-camera)
# Example:
#   $ ros2 launch rtabmap_examples realsense_d435i_color.launch.py

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# def generate_launch_description():
#     parameters=[{
#           'frame_id':'camera_link',
#           'subscribe_depth':True,
#           'subscribe_odom_info':True,
#           'approx_sync':True,
#           'use_sim_time':True,
#           'wait_imu_to_init':True}]

#     # remappings=[
#     #       ('imu', '/imu/data'),
#     #       ('rgb/image', '/camera/color/image_raw'),
#     #       ('rgb/camera_info', '/camera/color/camera_info'),
#     #       ('depth/image', '/camera/aligned_depth_to_color/image_raw')]
#     remappings=[
#           ('imu', '/imu/data'),
#           ('rgb/image', '/turtlebot/camera/color/image_color'),
#           ('rgb/camera_info', '/turtlebot/camera/color/camera_info'),
#           ('depth/image', '/turtlebot/camera/depth/image_depth')]
def generate_launch_description():
    # Standard parameters for Stonefish simulation
    parameters=[{
        #   'frame_id': 'turtlebot/base_footprint',  # Match your localisation_node
          'frame_id': 'base_footprint',
          'subscribe_depth': False,
          'subscribe_scan': True,           # Enable LiDAR subscription
          'approx_sync': True,              # Essential for sim sensors
        #   'use_sim_time': True,             # Essential for sim clock
          'wait_imu_to_init': True,
          'odom_frame_id': 'world_enu',
        #   'odom_frame_id': 'odom',
          'Reg/Force3DoF': 'true',         # Constrain to 2D plane
          'Reg/Strategy': '0',              # Use Visual (0) not ICP (1) for fake lasers
          'odometry_node_name': '/turtlebot/diff_drive_odometry',
    }]

    # Topic mapping for Stonefish
    remappings=[
          ('imu', '/turtlebot/imu_filtered'),
          ('rgb/image', '/turtlebot/camera/color/image_color'),
        #   ('rgb/image', '/turtlebot/camera/color/image_compressed'),
          ('rgb/camera_info', '/turtlebot/camera/color/camera_info'),
          ('depth/image', '/turtlebot/camera/depth/image_depth'),
        #   ('depth/image', '/turtlebot/camera/depth/image_rect_raw'),
          ('depth/camera_info', '/turtlebot/camera/depth/camera_info'),
          ('scan', '/turtlebot/fake_scan'),
          ('/rtabmap/base_controller/odom', '/turtlebot/odom'),
          ] # The output from depth_to_laserscan

    return LaunchDescription([

        # SetParameter(name='use_sim_time', value=True),
        # Launch arguments
        DeclareLaunchArgument(
            'unite_imu_method', default_value='2',
            description='0-None, 1-copy, 2-linear_interpolation. Use unite_imu_method:="1" if imu topics stop being published.'),

        # Make sure IR emitter is enabled
        SetParameter(name='depth_module.emitter_enabled', value=1),
        
        DeclareLaunchArgument(
            'args', default_value='',
            description='Extra arguments set to rtabmap and odometry nodes.'),
        
        DeclareLaunchArgument(
            'odom_args', default_value='',
            description='Extra arguments just for odometry node. If the same argument is already set in \"args\", it will be overwritten by the one in \"odom_args\".'),


        # Launch camera driver
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([os.path.join(
        #         get_package_share_directory('realsense2_camera'), 'launch'),
        #         '/rs_launch.py']),
        #         launch_arguments={'camera_namespace': '',
        #                           'enable_gyro': 'true',
        #                           'enable_accel': 'true',
        #                           'unite_imu_method': LaunchConfiguration('unite_imu_method'),
        #                           'align_depth.enable': 'true',
        #                           'enable_sync': 'true',
        #                           'rgb_camera.profile': '640x360x30'}.items(),
        # ),
        Node(
            package='depthimage_to_laserscan',
            executable='depthimage_to_laserscan_node',
            name='depthimage_to_laserscan',
            remappings=[('depth', '/turtlebot/camera/depth/image_depth'),
                        ('depth_camera_info', '/turtlebot/camera/depth/camera_info'),
                        ('scan', '/turtlebot/fake_scan'),
                        ],
            parameters=[{'range_max': 10.0, 'output_frame': 'camera_link', 'range_min': 0.28}]
        ),
        # Node(
        #     package='rtabmap_odom', executable='rgbd_odometry', output='screen',
        #     parameters=parameters,
        #     arguments=[LaunchConfiguration    ("args"), LaunchConfiguration("odom_args")],
        #     remappings=remappings),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_link_to_realsense_color',
            # arguments: x y z yaw pitch roll parent_frame child_frame
            # parameters=[{'use_sim_time': True}], # FORCE IT HERE
            arguments=['0', '0', '0', '-1.5708', '0', '-1.5708', 'camera_link', 'turtlebot/realsense_color']
        ),

        Node(
            package='rtabmap_slam', executable='rtabmap', output='screen',
            # prefix=['gnome-terminal -- gdb -ex run --args'],
            parameters=[{
                'frame_id': 'turtlebot/base_footprint', # Match your localisation_node
                # 'frame_id': 'base_footprint', # Match your localisation_node

                'subscribe_depth': True,
                'subscribe_scan': True,                # CRITICAL: Subscribe to the fake scan
                'subscribe_rgbd': False,
                'approx_sync': True,
                'visual_odometry': 'false',
                'odom_topic': '/turtlebot/odom',                 # Connect to your localisation_node
                
                'map_always_update': True,                 # CRITICAL: Keep map updating even if no features are detected
                # 'subscribe_odom': True,
                # 'use_sim_time': True,
                'wait_imu_to_init': True,
                'odom_frame_id': 'world_enu',          # Connect to your localisation_node
                'map_frame_id': 'world_enu',
                # 'odom_frame_id': 'odom',          # Connect to your localisation_node
                # 'map_frame_id': 'odom',
                'publish_tf': False,
                'Optimizer/Strategy': '1',        # Switch to g2o to prevent GTSAM crash
                'Optimizer/GravitySigma':'0', # Disable imu constraints (we are already in 2D)

                # RTAB-Map Specific Tuning
                'Grid/Sensor': '0',                      # 0=LaserScan, 1=Depth Cloud. Set to 0 for fake laser.
                # 2. Define the clearing distance
                'Grid/RangeMax': '10.0',           # FIXED: Clear space up to 5 meters even if nothing is hit
                'Grid/RangeMin': '0.28',            # Optional: Ignore very close readings that may be noisy
                'Grid/RayTracing': 'true',        # Ensure ray tracing is enabled (usually default)
                'Grid/Scan2dUnknownSpaceFilled': 'true',  # CRITICAL: Clears space even if scan is empty
                # 'Grid/FromDepth': 'false',             # Create occupancy grid from laser scan, not depth
                'Reg/Strategy': '0',                   # 0=Visual, 1=ICP. Use 0 because camera FOV is too narrow for ICP
                'Reg/Force3DoF': 'true',               # Force 2D mapping (x, y, yaw)
                'RGBD/ProximityBySpace': 'false',      # Recommended false for narrow FOV setups
                'RGBD/AngularUpdate': '0.01',          # Update map for small rotations
                'RGBD/LinearUpdate': '0.01',           # Update map for small movements
                'RGBD/OptimizeFromGraphEnd': 'false',  # Standard SLAM optimization
                'Vis/MinInliers': '10',                # Minimum features for a valid transformation
                
            }],
            remappings=remappings,
            arguments=['-d', LaunchConfiguration("args"), "--delete_db_on_start", ]),

        Node(
            package='rtabmap_viz', executable='rtabmap_viz', output='screen',
            parameters=parameters,
            remappings=remappings),

        # Compute quaternion of the IMU
        Node(
            package='imu_filter_madgwick', executable='imu_filter_madgwick_node', output='screen',
            parameters=[{'use_mag': False, 
                         'world_frame':'enu', 
                         'fixed_frame': 'world_enu', # FIXED: Aligns with your NED frame
                         'publish_tf':False,
                         'publish_debug_topics': False,
                         'gain': 0.01
                         },
                         ],
            # remappings=[('imu/data_raw', '/camera/imu')]),
            # remappings=[('imu/data_raw', '/turtlebot/sensors/imu_enu'),
            remappings=[('imu/data_raw', '/turtlebot/sensors/imu_data'),
                        ('imu/data', '/turtlebot/imu/filtered')]),
    ])
