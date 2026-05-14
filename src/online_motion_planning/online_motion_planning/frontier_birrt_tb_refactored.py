"""Refactored frontier exploration node using modular components."""

import rclpy
from rclpy.node import Node
import numpy as np
import math
import copy

from geometry_msgs.msg import Pose, Twist, Point
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import MarkerArray, Marker

# Import modular components
from online_motion_planning.visualizer import RVizVisualizer
from online_motion_planning.local_planner import LocalPlanner
from online_motion_planning.path_planner import PathPlanner
from online_motion_planning.frontier_manager import FrontierManager
from online_motion_planning.motion_controller import MotionController
from online_motion_planning.bidirectional_rrt_star import BIRRT_STAR
from online_motion_planning.Point import Point as PointRRT


class SamplingTurtlebotRefactored(Node):
    """Refactored frontier exploration node with modular architecture."""

    def __init__(self):
        super().__init__('sampling_turtlebot_refactored')
        
        # ─── Map State ─────────────────────────────────────────────────────
        self.occupancy_map = None
        self.binary_map = None
        self.origin = None
        self.resolution = None
        self.height = None
        self.width = None
        
        self.rtab_map = None
        self.rtab_origin = None
        self.rtab_resolution = None
        self.rtab_height = None
        self.rtab_width = None
        
        self.inflated_map_msg = None
        self.robot_pose = None
        self.current_yaw = 0.0
        self.current_vel = (0.0, 0.0)
        self.start_world_pos = None
        self.declare_parameter('map_frame', 'world_enu')
        self.binary_map_frame = self.get_parameter('map_frame').value
        
        # ─── Publishers ────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10)
        self.arm_cmd_pub = self.create_publisher(Float64MultiArray, '/turtlebot/swiftpro/joint_velocity_controller/command', 10)
        
        # Visualization publishers (match original exact types and topics)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        self.frontier_all_pub = self.create_publisher(MarkerArray, '/frontier_viz/all_frontiers', 10)
        self.bfs_cells_pub = self.create_publisher(Marker, '/frontier_viz/bfs_cells', 10)
        self.frontier_eval_pub = self.create_publisher(MarkerArray, '/frontier_viz/evaluation', 10)
        self.search_area_pub = self.create_publisher(Marker, '/frontier_viz/search_area', 10)
        self.rrt_tree_a_pub = self.create_publisher(Marker, '/rrt_viz/tree_a', 10)
        self.rrt_tree_b_pub = self.create_publisher(Marker, '/rrt_viz/tree_b', 10)
        self.dwa_traj_pub = self.create_publisher(MarkerArray, '/dwa_trajectories', 10)
        
        # ─── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(Odometry, '/turtlebot/odom', self._odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/inflated_map', self._map_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self._rtab_map_callback, 10)
        self.create_subscription(OccupancyGrid, '/move_base/global_costmap/costmap', self._inflated_costmap_callback, 10)
        self.create_subscription(JointState, '/turtlebot/joint_states', self._joint_state_callback, 10)
        
        # ─── Modular Components ────────────────────────────────────────────
        # Visualizer
        self.visualizer = RVizVisualizer(
            self.frontier_all_pub, self.bfs_cells_pub, self.frontier_eval_pub,
            self.search_area_pub, self.rrt_tree_a_pub, self.rrt_tree_b_pub,
            self.dwa_traj_pub, self.marker_pub, self.get_logger()
        )
        
        # Local planner (DWA)
        self.local_planner = LocalPlanner(
            max_linear_velocity=0.3,
            max_angular_velocity=0.3,
            dwa_max_accel=0.8,
            dwa_max_delta_yaw=1.2,
            dwa_predict_time=2.5,
            dwa_heading_w=8.0,
            dwa_dist_w=6.0,
            dwa_obstacle_w=8.0,
            dwa_velocity_w=0.5,
            dwa_viz_time=4.0,
            logger=self.get_logger()
        )
        
        # Global path planner (BiRRT*)
        self.path_planner = PathPlanner(
            delta_q=4,
            p=0.3,
            max_depth=round(math.log(4, 2)) + 1,  # = 3: max recursion depth for bisection collision checking
            min_dist=5,
            radius=5,
            threshold_path_rewire_dist=5,
            max_iterations_base=4000,
            max_iterations_increment=2000,
            max_iterations_cap=12000,
            max_retry_same_goal=3,
            logger=self.get_logger()
        )
        
        # Frontier manager
        self.frontier_manager = FrontierManager(
            kdist=1,
            karea=2,
            min_frontier_dist_m=0.5,
            visited_frontier_radius_m=0.5,
            local_search_radius=20,
            max_local_search_radius=50,
            frontier_expand_every=3,
            frontier_expand_step=15,
            use_global_search_window=True,
            global_x_min=-3.5,
            global_x_max=3.0,
            global_y_min=-5.0,
            global_y_max=1.0,
            global_search_count_threshold=10,
            local_search_count_threshold=10,
            logger=self.get_logger()
        )
        self.frontier_manager.visualizer = self.visualizer
        
        # Motion controller
        self.motion_controller = MotionController(
            max_linear_velocity=0.3,
            max_angular_velocity=0.3,
            kv=0.5,
            kw=1.0,
            acceptance_radius=0.1,
            scan_distance_threshold=0.4,
            arm_q2_retract=0.040,
            arm_q3_retract=-1.45,
            arm_retract_tol=0.06,
            arm_kp=2.0,
            arm_max_vel=0.3,
            dwa_viz_time=4.0,
            logger=self.get_logger()
        )
        self.motion_controller.cmd_vel_pub = self.cmd_vel_pub
        self.motion_controller.arm_cmd_pub = self.arm_cmd_pub
        self.motion_controller.marker_pub = self.marker_pub
        
        # ─── Timers ────────────────────────────────────────────────────────
        self.control_timer = self.create_timer(0.1, self._control_loop)       # 10 Hz
        self.path_timer = self.create_timer(1, self._planning_loop)           # 1 Hz
        self.viewpoint_timer = self.create_timer(2, self._frontier_loop)      # 0.5 Hz
        
        self.get_logger().info("Refactored frontier exploration node started.")

    # ─── Callbacks ─────────────────────────────────────────────────────────

    def _odom_callback(self, msg):
        """Update robot pose and velocity from odometry."""
        self.robot_pose = msg.pose.pose.position
        # Store current [v, w] so DWA can compute the reachable dynamic window
        self.current_vel = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        # Latch the very first pose as the global search window anchor. Fires exactly
        # once; subsequent messages only update robot_pose and current_yaw.
        if self.start_world_pos is None:
            self.start_world_pos = (self.robot_pose.x, self.robot_pose.y)
            self.get_logger().info(
                f"Global search window anchored at start position "
                f"({self.start_world_pos[0]:.2f}, {self.start_world_pos[1]:.2f})"
            )
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _map_callback(self, msg):
        """Process inflated occupancy map."""
        # Store the raw message so DWA can query cell values via .info and .data
        self.inflated_map_msg = msg
        info = msg.info
        self.resolution = info.resolution
        self.origin = np.array([info.origin.position.x, info.origin.position.y])
        self.height = info.height
        self.width = info.width

        raw = np.array(msg.data, dtype=float).reshape(self.height, self.width)
        raw[raw == -1] = 50.0
        self.occupancy_map = raw
        self.binary_map = np.where(copy.deepcopy(self.occupancy_map) >= 99, 1, 0)

    def _rtab_map_callback(self, msg):
        """Process RTAB-Map occupancy grid."""
        info = msg.info
        self.rtab_resolution = info.resolution
        self.rtab_width = info.width
        self.rtab_height = info.height
        self.rtab_origin = np.array([info.origin.position.x, info.origin.position.y])
        
        raw = np.array(msg.data, dtype=float).reshape(self.rtab_height, self.rtab_width)
        raw[raw == -1] = 50.0
        self.rtab_map = raw
        # self.get_logger().debug(f"[RTAB-Map] Received frontier map {self.rtab_width}x{self.rtab_height}")

    def _inflated_map_callback(self, msg):
        """Store inflated costmap for DWA obstacle checking."""
        if self.inflated_map_msg is None:
            self.get_logger().info("[Inflated Map] Received inflated costmap")
        self.inflated_map_msg = msg

    def _inflated_costmap_callback(self, msg):
        """Alternative callback for global costmap (if published)."""
        # Optional: only use if /move_base/global_costmap/costmap is available
        pass

    def _joint_state_callback(self, msg):
        """Update arm joint states."""
        for name, position in zip(msg.name, msg.position):
            self.motion_controller.update_joint_state(name, position)

    # ─── Main Loops ────────────────────────────────────────────────────────

    def _frontier_loop(self):
        """Frontier detection and selection (0.5 Hz)."""
        if self.motion_controller.rotation_state in ('spinning_360', 'halted'):
            return
        if self.occupancy_map is None or self.rtab_map is None or self.robot_pose is None or self.start_world_pos is None:
            return
        if not self.motion_controller.find_frontier:
            return

        # Wire up references
        self.frontier_manager.occupancy_map = self.occupancy_map
        self.frontier_manager.rtab_map = self.rtab_map
        self.frontier_manager.binary_map = self.binary_map
        self.frontier_manager.robot_pose = self.robot_pose
        self.frontier_manager.start_world_pos = self.start_world_pos
        self.frontier_manager.origin = self.origin
        self.frontier_manager.resolution = self.resolution
        self.frontier_manager.rtab_origin = self.rtab_origin
        self.frontier_manager.rtab_resolution = self.rtab_resolution
        self.frontier_manager.height = self.height
        self.frontier_manager.width = self.width
        self.frontier_manager.rtab_height = self.rtab_height
        self.frontier_manager.rtab_width = self.rtab_width
        self.frontier_manager.visited_frontier_positions = self.motion_controller.visited_frontier_positions
        self.frontier_manager.waypoints = self.motion_controller.waypoints
        # Sync local_search_radius in both directions so count-based expansions
        # (done in motion_controller when a goal is reached) and failure-based
        # expansions (done in frontier_manager when no frontiers are found) both
        # contribute — equivalent to having a single shared variable in the original.
        self.frontier_manager.local_search_radius = max(
            self.frontier_manager.local_search_radius,
            self.motion_controller.local_search_radius
        )
        self.frontier_manager.now = self.get_clock().now().to_msg()
        self.frontier_manager.binary_map_frame = self.binary_map_frame
        self.frontier_manager.visualizer = self.visualizer

        # Clear any stale goal before the search so a previously-reached frontier
        # can never be re-selected on the next tick.
        self.frontier_manager.goal_pose = None

        # Run frontier selection
        self.frontier_manager.select_frontier()

        # Sync search radius back so both objects stay in step
        self.motion_controller.local_search_radius = self.frontier_manager.local_search_radius

        # Sync state back
        self.motion_controller.find_frontier = self.frontier_manager.find_frontier
        if self.frontier_manager.goal_pose is not None:
            self.motion_controller.goal_pose = self.frontier_manager.goal_pose

        # Sync terminal-behaviour signals: following_last_path and spinning_360
        if self.frontier_manager.following_last_path:
            self.motion_controller.following_last_path = True
            self.frontier_manager.following_last_path = False  # consume
        if self.frontier_manager.rotation_state == 'spinning_360':
            self.motion_controller.rotation_state = 'spinning_360'
            self.motion_controller.prev_yaw_for_spin = None
            self.motion_controller.spin_accumulated = 0.0
            self.frontier_manager.rotation_state = 'idle'  # consume

    def _planning_loop(self):
        """Global path planning (1 Hz)."""
        if self.motion_controller.rotation_state in ('spinning_360', 'halted'):
            return
        if self.binary_map is None or self.robot_pose is None:
            return

        # following_last_path mode must be checked before the goal_pose guard:
        # in this mode goal_pose is None but we still need to verify the saved
        # path is still collision-free.
        if self.motion_controller.following_last_path:
            self.path_planner.binary_map = self.binary_map
            self.path_planner.robot_pose = self.robot_pose
            self.path_planner.resolution = self.resolution
            self.path_planner.origin = self.origin
            self.path_planner.cmd_vel_pub = self.cmd_vel_pub
            self.path_planner.waypoints = self.motion_controller.waypoints
            self.path_planner.following_last_path = True
            self.path_planner.goal_pose = None
            self.path_planner.plan(
                find_nearest_free_cell_fn=self.frontier_manager.find_nearest_free_cell,
                viz_cb_setup_fn=self._make_viz_cb
            )
            # If plan() cleared waypoints (path blocked), set terminal spin
            if self.path_planner.waypoints is None:
                self.motion_controller.waypoints = None
                self.motion_controller.rotation_state = 'spinning_360'
                self.motion_controller.prev_yaw_for_spin = None
                self.motion_controller.spin_accumulated = 0.0
            return

        if self.motion_controller.goal_pose is None:
            return

        # Wire up references
        self.path_planner.binary_map = self.binary_map
        self.path_planner.robot_pose = self.robot_pose
        self.path_planner.goal_pose = self.motion_controller.goal_pose
        self.path_planner.resolution = self.resolution
        self.path_planner.origin = self.origin
        self.path_planner.cmd_vel_pub = self.cmd_vel_pub

        # Sync state in
        self.path_planner.waypoints = self.motion_controller.waypoints
        self.path_planner.complete_a_path = self.motion_controller.complete_a_path
        self.path_planner.collide_robot_next_waypoint = self.motion_controller.collide_robot_next_waypoint
        self.path_planner.following_last_path = False

        waypoints = self.path_planner.plan(
            find_nearest_free_cell_fn=self.frontier_manager.find_nearest_free_cell,
            viz_cb_setup_fn=self._make_viz_cb
        )

        # Sync state back
        self.motion_controller.complete_a_path = self.path_planner.complete_a_path
        self.motion_controller.collide_robot_next_waypoint = self.path_planner.collide_robot_next_waypoint

        # Propagate planning failures back to frontier search
        if self.path_planner.find_frontier:
            self.motion_controller.find_frontier = True
            self.motion_controller.goal_pose = None
            self.motion_controller.waypoints = None
            self.path_planner.find_frontier = False  # consume the flag
            return

        if waypoints is not None and len(waypoints) > 0:
            self.motion_controller.waypoints = waypoints
            waypoints_viz = [np.array([self.robot_pose.x, self.robot_pose.y])] + waypoints
            self.motion_controller.publish_waypoints(waypoints_viz, self.visualizer)

    def _make_viz_cb(self, color, z):
        """Helper: create a BiRRT* tree visualisation callback for the given colour/z."""
        pub = self.rrt_tree_a_pub if color == (0.2, 0.6, 1.0) else self.rrt_tree_b_pub
        return self.visualizer.make_rrt_viz_callback(
            pub, self.resolution, self.origin,
            color_rgb=color, z_height=z,
            binary_map_frame=self.binary_map_frame,
            now=self.get_clock().now().to_msg()
        )

    def _control_loop(self):
        """Motion control (10 Hz)."""
        # Wire up references
        self.motion_controller.robot_pose = self.robot_pose
        self.motion_controller.current_yaw = self.current_yaw
        self.motion_controller.current_vel = self.current_vel
        self.motion_controller.now = self.get_clock().now().to_msg()
        self.motion_controller.binary_map_frame = self.binary_map_frame
        self.local_planner.robot_pose = self.robot_pose
        self.local_planner.current_yaw = self.current_yaw
        self.local_planner.current_vel = self.current_vel
        self.local_planner.inflated_map_msg = self.inflated_map_msg
        self.visualizer.local_planner = self.local_planner
        
        # Execute motion control
        self.motion_controller.execute(
            local_planner=self.local_planner,
            visualizer=self.visualizer
        )


def main(args=None):
    rclpy.init(args=args)
    node = SamplingTurtlebotRefactored()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
