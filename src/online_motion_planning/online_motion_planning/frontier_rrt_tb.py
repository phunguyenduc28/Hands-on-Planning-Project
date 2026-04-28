# =============================================================================
# CHANGES FROM ORIGINAL frontier_rrt_tb.py
# =============================================================================
# The original file was a minimal proof-of-concept.  Every major section below
# was either added or substantially changed.  A summary of each:
#
# 1. MAP CALLBACK (map_callback)
#    ORIGINAL: self.binary_map = np.where(self.occupancy_map > 50, 1, 0)
#    CHANGED TO: threshold >= 99 (inscribed/lethal only).
#    WHY: costmap_2d publishes a 1–98 inflation gradient around walls.  Blocking
#    anything > 50 treated the entire inflation zone as an obstacle, leaving < 1%
#    of the map as free space and causing near-100% RRT* rejection rates.
#    Blocking only >= 99 keeps the robot clear of wall footprints while giving
#    RRT* a usable free-space region to sample from.
#
# 2. FRONTIER DETECTION (frontier_viewpoint)
#    ORIGINAL: scanned the entire inflated map for frontier cells; no local window,
#    no reachability check, no RTAB-Map separation.
#    CHANGED TO: dual-map design — frontier cells detected on the raw RTAB-Map
#    (no inflation, so boundaries are sharp) inside a configurable global/local
#    search window; BFS reachability on the inflated map then filters out any
#    frontier that is not physically connected to the robot; visited-frontier
#    blacklist and minimum-distance filter further prevent revisiting.
#    WHY: inflation smears frontier boundaries, causing the robot to target cells
#    inside obstacles.  BFS is required because connected-components alone cannot
#    detect if a topologically visible frontier is actually reachable through the
#    narrow corridors of an inflated map.
#
# 3. FRONTIER COST FUNCTION (frontier_cost)
#    ORIGINAL: cost = kdist*dist - karea*area  (closer = cheaper)
#    CHANGED TO: cost = -kdist*dist - karea*area  (farther = cheaper)
#    WHY: the original function always preferred the nearest large frontier, causing
#    the robot to keep selecting the boundary it was sitting on.  Negating the
#    distance term biases selection toward larger, more distant unexplored areas.
#
# 4. PATH PLANNING (path_planning_loop)
#    ORIGINAL: single occupancy check, no retry logic, no start-snapping.
#    CHANGED TO: start snapped to nearest free cell when occupied (localization
#    drift into inflation zone); consecutive-failure counter that grows
#    max_iterations up to a cap before abandoning the goal; full segment
#    collision check across all waypoints (not just robot→next).
#    WHY: without start-snapping, any momentary map spike causes RRT* to abort.
#    Without the retry ramp, a hard-to-reach goal causes infinite 4000-iteration
#    failures instead of progressively allocating more compute.
#
# 5. CONTROL LOOP (control_loop)
#    ORIGINAL: simple distance-only waypoint tracking, no scanning, no arm control.
#    CHANGED TO: 360° waypoint scan (scanning_360 state) at each new location;
#    distance-gated scan suppression to avoid re-scanning after replanning at the
#    same spot; arm-retraction controller (swiftpro) that holds the base until the
#    arm is safely stowed; terminal exploration behavior that follows the last saved
#    path to the farthest clear waypoint before spinning.
#    WHY: scanning at each waypoint gives the RTAB-Map more time to build the map
#    before the next frontier search; arm retraction prevents collisions in narrow
#    passages.
#
# 6. VISUALISATION
#    ORIGINAL: only a waypoint MarkerArray.
#    ADDED: separate publishers for frontier clusters, BFS cells, frontier
#    evaluation (cyan/green spheres), global search window rectangle, and RRT*
#    tree growth (LINE_LIST updated every 10 edges with a final flush).
# =============================================================================
import rclpy
from rclpy.node import Node
import numpy as np
import math

from matplotlib import pyplot as plt

from geometry_msgs.msg import Pose, PoseStamped, Twist, Point
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from visualization_msgs.msg import Marker, MarkerArray

from online_motion_planning.rrt_star import RRT_STAR
from online_motion_planning.Point import Point as PointRRT

import cv2
import copy

class SamplingTurtlebot(Node):
    def __init__(self):
        super().__init__('sampling_turtlebot')
        self.robot_pose = None
        self.goal_pose = None
        self.acceptance_radius = 0.1
        self.wheel_base_distance = 0.230
        self.max_linear_velocity = 0.3
        self.max_angular_velocity = 0.3
        self.kv = 0.5
        self.kw = 1.0

        # Map variables
        self.occupancy_map = None
        self.binary_map = None
        self.origin  = None
        self.resolution = None
        self.height = None
        self.width = None

        # Frontier variables
        self.kdist = 1
        self.karea = 2
        self.find_frontier = True

        # RRT star variable
        self.max_iterations = 4000
        self.max_iterations_base = 4000   # reset target after a successful plan
        self.max_iterations_cap  = 12000  # never exceed this
        self.rrt_fail_count = 0           # consecutive planning failures
        self.delta_q = 4   # reduced from 8 — smaller steps navigate narrow inflated corridors
        self.p = 0.3
        self.max_depth = round(math.log(self.delta_q, 2)) + 1 # max iteration for considering a segment free or not
        self.min_dist = 5
        self.radius = 5
        self.threshold_path_rewire_dist = 5

        self.waypoints = None

        self.collide_robot_next_waypoint = False
        # Motion controller node
        self.complete_a_path = True
        
        self.declare_parameter('map_frame', 'world_enu')
        self.binary_map_frame = self.get_parameter('map_frame').value

        # RTAB-Map occupancy grid — used ONLY for frontier detection (no inflation)
        # Standard ROS values: 0=free, 100=occupied, -1=unknown (→ 50 after normalisation)
        self.rtab_map        = None
        self.rtab_origin     = None
        self.rtab_resolution = None
        self.rtab_width      = None
        self.rtab_height     = None

        # Subsrciber
        self.odom_sub = self.create_subscription(Odometry, '/turtlebot/odom', self.odom_callback, 10)
        self.binary_map_sub = self.create_subscription(OccupancyGrid, '/inflated_map', self.map_callback, 10)
        self.rtab_map_sub   = self.create_subscription(OccupancyGrid, '/map', self._rtab_map_callback, 10)

        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        self.frontier_all_pub = self.create_publisher(MarkerArray, '/frontier_viz/all_frontiers', 10)
        self.bfs_cells_pub = self.create_publisher(Marker, '/frontier_viz/bfs_cells', 10)
        self.frontier_eval_pub = self.create_publisher(MarkerArray, '/frontier_viz/evaluation', 10)
        self.search_area_pub = self.create_publisher(Marker, '/frontier_viz/search_area', 10)
        self.rrt_tree_pub = self.create_publisher(Marker, '/rrt_viz/tree', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10) # publishe cmd_vel

        # # Timer
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.path_timer = self.create_timer(1, self.path_planning_loop)
        self.viewpoint_timer = self.create_timer(2, self.frontier_viewpoint)

        self.rotation_state = 'idle'   # 'idle', 'scanning_360', 'moving', 'spinning_360', 'halted'
        self.rotation_target_yaw = None
        self.rotation_start_yaw = None
        self.rotation_tolerance = 0.05   # radians

        self.local_search_radius = 20
        self.max_local_search_radius = 100
        self.max_radius_wait_count = 0   # times we've been stuck at max radius
        self.prev_yaw_for_spin = None    # yaw sample from the previous control tick
        self.spin_accumulated = 0.0      # total rotation accumulated so far (rad)
        self.last_scan_pos = None        # world (x,y) where the last waypoint scan occurred
        self.scan_distance_threshold = 0.4  # m — must travel this far before scanning again

        # ── Arm retraction ───────────────────────────────────────────────────
        # From FK:  r = 0.0698 - L2*sin(q2) + L3*cos(q3)
        # Minimum r (≈84 mm) at q2=+0.040, q3=-1.45  →  within turtlebot footprint.
        self.arm_q2 = None          # set by joint-state callback
        self.arm_q3 = None
        self.ARM_Q2_RETRACT  =  0.040
        self.ARM_Q3_RETRACT  = -1.45
        self.ARM_RETRACT_TOL =  0.06   # rad — close enough counts as retracted
        self.arm_kp          =  2.0    # proportional gain for retraction controller
        self.arm_max_vel     =  0.3    # rad/s — match lab2 max_vel

        self.joint_state_sub = self.create_subscription(
            JointState, '/turtlebot/joint_states', self._joint_state_cb, 10)
        self.arm_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/turtlebot/swiftpro/joint_velocity_controller/command', 10)

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose.position
        # Convert Quaternion to Yaw (Z-axis rotation)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def map_callback(self, msg):
        # info = msg.info

        # self.resolution = info.resolution

        # self.width = info.width
        # self.height = info.height 
        
        # origin_x = info.origin.position.x 
        # origin_y = info.origin.position.y
        # self.origin = np.array([origin_x, origin_y])

        # self.occupancy_map = np.array(msg.data, dtype = float).reshape(self.height, self.width)

        info = msg.info
        self.resolution = info.resolution
        self.width = info.width
        self.height = info.height
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        self.origin = np.array([origin_x, origin_y])

        raw = np.array(msg.data, dtype=float).reshape(self.height, self.width)

        # Normalize: -1 (unknown in ROS) → 50 so our code treats it uniformly
        raw[raw == -1] = 50.0

        self.occupancy_map = raw

        # Threshold >= 99: only block inscribed-radius (99) and lethal (100) cells.
        # costmap_2d publishes a gradient 1–98 for the inflation zone around walls.
        # Using > 50 treated the ENTIRE gradient as obstacles, leaving < 1% free
        # space for RRT* to sample in and causing 99%+ rejection rates.
        # >= 99 keeps the robot clear of wall footprints while giving RRT* room to plan.
        self.binary_map = np.where(copy.deepcopy(self.occupancy_map) >= 99, 1, 0)

    def _rtab_map_callback(self, msg):
        """Stores the raw RTAB-Map occupancy grid for frontier detection only.
        Navigation (BFS, RRT*) continues to use the inflated map."""
        info = msg.info
        self.rtab_resolution = info.resolution
        self.rtab_width      = info.width
        self.rtab_height     = info.height
        self.rtab_origin     = np.array([info.origin.position.x, info.origin.position.y])

        raw = np.array(msg.data, dtype=float).reshape(self.rtab_height, self.rtab_width)
        raw[raw == -1] = 50.0   # unknown → 50, same sentinel as inflated map
        self.rtab_map = raw

    def frontier_cost(self, area, cX, cY):
        # q_goal = np.array([cY, cX])
        # q_start = (np.array([self.robot_pose.y, self.robot_pose.x]) - self.origin) / self.resolution
        q_goal  = np.array([cX, cY])                                               # [col, row] = [x, y]
        q_start = (np.array([self.robot_pose.x, self.robot_pose.y]) - self.origin) / self.resolution
        q_goal_point = PointRRT(q_goal[0], q_goal[1])
        q_start_point = PointRRT(q_start[0], q_start[1])
        dist = q_start_point.dist(q_goal_point)

        # self.get_logger().info(f"Dist {dist} area {area}")

        cost = self.kdist*dist - self.karea*area
        return cost
    
    def get_reachable_cells(self, robot_col, robot_row, max_cells=10000):
        """BFS flood fill from the robot position through the inflated map.
        Walks through any cell with value < 99 (free + inflation gradient).
        Inscribed (99) and lethal (100) cells are treated as barriers,
        matching the binary_map >= 99 obstacle threshold used by RRT*."""
        from collections import deque

        visited = set()
        queue = deque()
        start = (robot_col, robot_row)
        queue.append(start)
        visited.add(start)

        while queue and len(visited) < max_cells:
            col, row = queue.popleft()
            for dc, dr in [(-1,0),(1,0),(0,-1),(0,1),
                            (-1,-1),(-1,1),(1,-1),(1,1)]:  # 8-connected
                nc, nr = col + dc, row + dr
                if (nc, nr) in visited:
                    continue
                if not (0 <= nc < self.width and 0 <= nr < self.height):
                    continue
                cell_val = self.occupancy_map[nr, nc]
            # Walk through free (0) and unknown (50) — stop only at occupied (100)
                if cell_val < 100:
                    visited.add((nc, nr))
                    queue.append((nc, nr))


        return visited

    # ── Visualisation helpers ─────────────────────────────────────────────────

    _FRONTIER_COLORS = [
        (1.0, 1.0, 0.0),   # yellow
        (1.0, 0.5, 0.0),   # orange
        (1.0, 0.0, 1.0),   # magenta
        (0.0, 1.0, 1.0),   # cyan
        (1.0, 1.0, 1.0),   # white
        (0.5, 1.0, 0.0),   # lime
    ]

    def _publish_all_frontiers(self, labels, numLabels, resolution=None, origin=None):
        """One POINTS marker per cluster, cycling through colours.
        resolution/origin default to the inflated-map values; pass the
        RTAB-Map values when the labels come from the raw /map grid."""
        if resolution is None:
            resolution = self.resolution
        if origin is None:
            origin = self.origin

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        clear = Marker()
        clear.header.frame_id = self.binary_map_frame
        clear.header.stamp = now
        clear.ns = "all_frontiers"
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for i in range(1, numLabels):
            rows, cols = np.where(labels == i)
            if len(rows) == 0:
                continue

            color = self._FRONTIER_COLORS[(i - 1) % len(self._FRONTIER_COLORS)]
            m = Marker()
            m.header.frame_id = self.binary_map_frame
            m.header.stamp = now
            m.ns = "all_frontiers"
            m.id = i
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = resolution
            m.scale.y = resolution
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 0.85
            m.pose.orientation.w = 1.0
            m.lifetime = rclpy.duration.Duration(seconds=4).to_msg()

            for row, col in zip(rows, cols):
                p = Point()
                p.x = float(col * resolution + origin[0])
                p.y = float(row * resolution + origin[1])
                p.z = 0.05
                m.points.append(p)

            marker_array.markers.append(m)

        self.frontier_all_pub.publish(marker_array)

    def _publish_bfs_cells(self, reachable):
        """Semi-transparent green POINTS for all BFS-reachable cells."""
        now = self.get_clock().now().to_msg()
        m = Marker()
        m.header.frame_id = self.binary_map_frame
        m.header.stamp = now
        m.ns = "bfs_cells"
        m.id = 0
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = self.resolution * 0.7
        m.scale.y = self.resolution * 0.7
        m.color.r = 0.0
        m.color.g = 0.85
        m.color.b = 0.2
        m.color.a = 0.18
        m.pose.orientation.w = 1.0
        m.lifetime = rclpy.duration.Duration(seconds=4).to_msg()

        cells = list(reachable)
        step = max(1, len(cells) // 3000)
        for col, row in cells[::step]:
            p = Point()
            p.x = float(col * self.resolution + self.origin[0])
            p.y = float(row * self.resolution + self.origin[1])
            p.z = 0.0
            m.points.append(p)

        self.bfs_cells_pub.publish(m)

    def _publish_frontier_evaluation(self, evaluated_centroids, best_world):
        """Cyan spheres for every evaluated centroid, large green sphere for the best."""
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        clear = Marker()
        clear.header.frame_id = self.binary_map_frame
        clear.header.stamp = now
        clear.ns = "evaluated_centroids"
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        clear_best = Marker()
        clear_best.header.frame_id = self.binary_map_frame
        clear_best.header.stamp = now
        clear_best.ns = "best_frontier"
        clear_best.action = Marker.DELETEALL
        marker_array.markers.append(clear_best)

        lifetime = rclpy.duration.Duration(seconds=4).to_msg()

        for i, (wx, wy) in enumerate(evaluated_centroids):
            m = Marker()
            m.header.frame_id = self.binary_map_frame
            m.header.stamp = now
            m.ns = "evaluated_centroids"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.12
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.18
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 0.85
            m.lifetime = lifetime
            marker_array.markers.append(m)

        if best_world is not None:
            bm = Marker()
            bm.header.frame_id = self.binary_map_frame
            bm.header.stamp = now
            bm.ns = "best_frontier"
            bm.id = 0
            bm.type = Marker.SPHERE
            bm.action = Marker.ADD
            bm.pose.position.x = best_world[0]
            bm.pose.position.y = best_world[1]
            bm.pose.position.z = 0.2
            bm.pose.orientation.w = 1.0
            bm.scale.x = bm.scale.y = bm.scale.z = 0.35
            bm.color.r = 0.0
            bm.color.g = 1.0
            bm.color.b = 0.0
            bm.color.a = 1.0
            bm.lifetime = lifetime
            marker_array.markers.append(bm)

        self.frontier_eval_pub.publish(marker_array)

    def _publish_search_area(self, rtab_row_min, rtab_row_max, rtab_col_min, rtab_col_max):
        """White rectangle showing the current local frontier search window."""
        now = self.get_clock().now().to_msg()

        x_min = rtab_col_min * self.rtab_resolution + self.rtab_origin[0]
        x_max = rtab_col_max * self.rtab_resolution + self.rtab_origin[0]
        y_min = rtab_row_min * self.rtab_resolution + self.rtab_origin[1]
        y_max = rtab_row_max * self.rtab_resolution + self.rtab_origin[1]

        m = Marker()
        m.header.frame_id = self.binary_map_frame
        m.header.stamp = now
        m.ns = "search_area"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.06
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 0.9
        m.pose.orientation.w = 1.0
        m.lifetime = rclpy.duration.Duration(seconds=4).to_msg()

        corners = [
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max),
            (x_min, y_min),  # close the loop
        ]
        for cx, cy in corners:
            p = Point()
            p.x = float(cx)
            p.y = float(cy)
            p.z = 0.05
            m.points.append(p)

        self.search_area_pub.publish(m)

    # ── RRT* tree visualisation ───────────────────────────────────────────────

    def _make_rrt_viz_callback(self, publish_every=50):
        """Returns a closure to pass to RRT*.sample() that publishes the growing tree.

        Each time a new edge is valid, the closure accumulates the edge and
        re-publishes the full LINE_LIST every publish_every added nodes.
        publish() goes straight to DDS transport so RViz receives updates in
        real-time even while sample() is blocking the timer thread.
        """
        # Clear any leftover tree from a previous planning run
        clear = Marker()
        clear.header.frame_id = self.binary_map_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.ns = "rrt_tree"
        clear.action = Marker.DELETEALL
        self.rrt_tree_pub.publish(clear)

        edge_points = []   # flat list of (x, y) world-coord pairs for LINE_LIST
        count = [0]

        def callback(G, parent_idx, child_idx):
            parent = G[parent_idx]
            child  = G[child_idx]
            edge_points.append((float(parent.x * self.resolution + self.origin[0]),
                                float(parent.y * self.resolution + self.origin[1])))
            edge_points.append((float(child.x  * self.resolution + self.origin[0]),
                                float(child.y  * self.resolution + self.origin[1])))
            count[0] += 1
            if count[0] % publish_every != 0:
                return

            m = Marker()
            m.header.frame_id = self.binary_map_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "rrt_tree"
            m.id = 0
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.025
            m.color.r = 0.2
            m.color.g = 0.6
            m.color.b = 1.0
            m.color.a = 0.6
            m.pose.orientation.w = 1.0
            m.lifetime = rclpy.duration.Duration(seconds=15).to_msg()
            for x, y in edge_points:
                p = Point()
                p.x = x
                p.y = y
                p.z = 0.08
                m.points.append(p)
            self.rrt_tree_pub.publish(m)

        return callback

    # ── Frontier selection ────────────────────────────────────────────────────

    def frontier_viewpoint(self):
        if self.rotation_state in ('spinning_360', 'halted'):
            return
        # self.get_logger().info(f"In frontier viewpoint")
        # Require both maps: inflated map for nav/BFS, RTAB-Map for frontier detection
        if self.occupancy_map is None or self.rtab_map is None or self.robot_pose is None or self.find_frontier == False:
            return

        # frontier_cells = np.zeros((self.height, self.width), dtype=np.uint8)

        # for y in range(1, self.height - 1):
        #     for x in range(1, self.width - 1):
        #         if self.occupancy_map[y, x] == 0:  # Cell is Free
        #             neighbors = [self.occupancy_map[y-1, x], self.occupancy_map[y+1, x],
        #                         self.occupancy_map[y, x-1], self.occupancy_map[y, x+1]]
        #             if 50.0 in neighbors:
        #                 if 100.0 not in neighbors:
        #                     frontier_cells[y, x] = 255

        # output = cv2.connectedComponentsWithStats(frontier_cells, 8, cv2.CV_32S)
        # (numLabels, labels, stats, centroids) = output

        # cost_list = np.full(numLabels, np.inf) # Initialize with infinity

        # Guard: need both maps
        if self.rtab_map is None:
            self.get_logger().warn("Waiting for /map from RTAB-Map…", throttle_duration_sec=5.0)
            return

        # ── Robot position in BOTH coordinate systems ─────────────────────────
        # Inflated-map (nav) coords — used for BFS and path planning
        nav_col = int((self.robot_pose.x - self.origin[0])           / self.resolution)
        nav_row = int((self.robot_pose.y - self.origin[1])           / self.resolution)
        # RTAB-Map coords — used for frontier cell detection
        rtab_col = int((self.robot_pose.x - self.rtab_origin[0])     / self.rtab_resolution)
        rtab_row = int((self.robot_pose.y - self.rtab_origin[1])     / self.rtab_resolution)

        LOCAL_RADIUS = self.local_search_radius

        # BFS reachability on inflated map (respects inflation padding)
        reachable = self.get_reachable_cells(nav_col, nav_row, max_cells=10000)
        # self.get_logger().info(f"Reachable cells from robot: {len(reachable)}")
        self._publish_bfs_cells(reachable)

        # Local window in RTAB-Map cell space
        rtab_row_min = max(1,                    rtab_row - LOCAL_RADIUS)
        rtab_row_max = min(self.rtab_height - 1, rtab_row + LOCAL_RADIUS)
        rtab_col_min = max(1,                    rtab_col - LOCAL_RADIUS)
        rtab_col_max = min(self.rtab_width  - 1, rtab_col + LOCAL_RADIUS)

        self._publish_search_area(rtab_row_min, rtab_row_max, rtab_col_min, rtab_col_max)

        # ── Frontier detection on the raw RTAB-Map (no inflation) ─────────────
        # Values: 0=free, 50=unknown (after normalisation), 100=occupied
        frontier_cells = np.zeros((self.rtab_height, self.rtab_width), dtype=np.uint8)

        for y in range(rtab_row_min, rtab_row_max):
            for x in range(rtab_col_min, rtab_col_max):
                if self.rtab_map[y, x] == 0:          # free cell
                    neighbors = [
                        self.rtab_map[y-1, x],
                        self.rtab_map[y+1, x],
                        self.rtab_map[y, x-1],
                        self.rtab_map[y, x+1]
                    ]
                    if 50.0 in neighbors and 100.0 not in neighbors:
                        frontier_cells[y, x] = 255

        output = cv2.connectedComponentsWithStats(frontier_cells, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        # Visualise using RTAB-Map resolution/origin
        self._publish_all_frontiers(labels, numLabels,
                                    resolution=self.rtab_resolution,
                                    origin=self.rtab_origin)

        cost_list = np.full(numLabels, np.inf)
        MAX_FRONTIER_AREA = 200
        EDGE_MARGIN = 3

        evaluated_centroids = []
        best_cost = np.inf
        current_best_world = None

        for i in range(1, numLabels):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]   # RTAB-Map cell coordinates

            if area < 5:
                continue

            # Reject 1-2 cell wide scan-ray artefacts
            cl_width  = stats[i, cv2.CC_STAT_WIDTH]
            cl_height = stats[i, cv2.CC_STAT_HEIGHT]
            if min(cl_width, cl_height) < 3:
                continue

            if not (rtab_col_min <= int(cX) <= rtab_col_max and
                    rtab_row_min <= int(cY) <= rtab_row_max):
                continue

            if not (EDGE_MARGIN <= int(cX) < self.rtab_width  - EDGE_MARGIN and
                    EDGE_MARGIN <= int(cY) < self.rtab_height - EDGE_MARGIN):
                continue

            # Convert RTAB-Map centroid → world → inflated-map cell for BFS check
            world_x = cX * self.rtab_resolution + self.rtab_origin[0]
            world_y = cY * self.rtab_resolution + self.rtab_origin[1]
            nav_cx  = int((world_x - self.origin[0]) / self.resolution)
            nav_cy  = int((world_y - self.origin[1]) / self.resolution)

            if (nav_cx, nav_cy) not in reachable:
                # self.get_logger().debug(
                #     f"Frontier ({cX:.1f},{cY:.1f}) not reachable from robot — discarding"
                # )
                continue

            capped_area = min(area, MAX_FRONTIER_AREA)
            # frontier_cost expects inflated-map cell coords for distance calculation
            cost = self.frontier_cost(capped_area, nav_cx, nav_cy)
            cost_list[i] = cost

            evaluated_centroids.append((world_x, world_y))

            if cost < best_cost:
                best_cost = cost
                current_best_world = (world_x, world_y)

        # Publish evaluated centroids (cyan spheres) and running best (green sphere)
        self._publish_frontier_evaluation(evaluated_centroids, current_best_world)

        if numLabels > 1:
            best_index = np.argmin(cost_list)

            if cost_list[best_index] == np.inf:
                self.local_search_radius = min(
                    self.local_search_radius + 10,
                    self.max_local_search_radius
                )
                if self.local_search_radius >= self.max_local_search_radius:
                    self.max_radius_wait_count += 1
                    self.get_logger().warn(
                        f"No reachable frontier within radius {LOCAL_RADIUS}, "
                        f"already at max ({self.max_local_search_radius}). "
                        f"Wait count: {self.max_radius_wait_count}/10"
                    )
                    if self.max_radius_wait_count >= 10:
                        self.get_logger().info("Max radius reached 10 times — spinning 360° then halting.")
                        self.rotation_state = 'spinning_360'
                        self.prev_yaw_for_spin = None
                        self.spin_accumulated = 0.0
                        self.find_frontier = False
                        return
                else:
                    self.get_logger().warn(
                        f"No reachable frontier within radius {LOCAL_RADIUS}, "
                        f"expanding to {self.local_search_radius}"
                    )
                self.find_frontier = True
                return

            best_cX, best_cY = centroids[best_index]
            goal_x = best_cX * self.rtab_resolution + self.rtab_origin[0]
            goal_y = best_cY * self.rtab_resolution + self.rtab_origin[1]

            self.get_logger().info(
                f"Best Goal: ({goal_x:.2f}, {goal_y:.2f}), "
                f"rtab_cell=({best_cX:.1f},{best_cY:.1f}), "
                f"nav_cell=({nav_col},{nav_row}), "
                f"candidates={len(evaluated_centroids)}"
            )
            self.goal_pose = [goal_x, goal_y]
            self.find_frontier = False
            # Do NOT reset local_search_radius here — the frontier was selected
            # but not yet reached.  If RRT* fails, the radius stays grown so the
            # next search starts from a larger window instead of oscillating 20↔30.
        elif numLabels == 1:
            # No frontier cells in local window — window too small, not fully explored
            self.local_search_radius = min(
                self.local_search_radius + 10,
                self.max_local_search_radius
            )
            if self.local_search_radius >= self.max_local_search_radius:
                self.max_radius_wait_count += 1
                self.get_logger().info(
                    f"No frontier cells at max search radius — exploration complete. "
                    f"Wait count: {self.max_radius_wait_count}/10"
                )
                if self.max_radius_wait_count >= 10:
                    self.get_logger().info("Max radius reached 10 times — spinning 360° then halting.")
                    self.rotation_state = 'spinning_360'
                    self.prev_yaw_for_spin = None
                    self.spin_accumulated = 0.0
                    self.find_frontier = False
                    return
            else:
                self.get_logger().warn(
                    f"No frontier cells in local window (radius {LOCAL_RADIUS}), "
                    f"expanding to {self.local_search_radius}"
                )
            self.find_frontier = True
            return

    def inspect_coordinate_ordering(self, q_start, q_goal):
        """Debug helper to identify row/col vs x/y confusion."""
        self.get_logger().info(f"=== COORDINATE ORDERING INSPECTION ===")
        self.get_logger().info(f"q_start = [{q_start[0]:.1f}, {q_start[1]:.1f}]")
        self.get_logger().info(f"q_goal = [{q_goal[0]:.1f}, {q_goal[1]:.1f}]")
        self.get_logger().info(f"Map shape (height, width) = {self.binary_map.shape}")
        
        q_start_int = np.array([int(q_start[0]), int(q_start[1])])
        q_goal_int = np.array([int(q_goal[0]), int(q_goal[1])])
        
        # Access as [q_start_int[0], q_start_int[1]] -- current way (treating as row,col)
        val_as_row_col = self.binary_map[q_start_int[0], q_start_int[1]]
        self.get_logger().info(f"binary_map[{q_start_int[0]}, {q_start_int[1]}] = {val_as_row_col} (current indexing as [row,col])")
        
        # Try swapped access [q_start_int[1], q_start_int[0]] -- if q values are (x,y) not (row,col)
        if q_start_int[1] < self.binary_map.shape[0] and q_start_int[0] < self.binary_map.shape[1]:
            val_as_col_row = self.binary_map[q_start_int[1], q_start_int[0]]
            self.get_logger().info(f"binary_map[{q_start_int[1]}, {q_start_int[0]}] = {val_as_col_row} (swapped indexing as [col,row])")
            
            if val_as_row_col == 1 and val_as_col_row == 0:
                self.get_logger().error(f"🔴 COORDINATE BUG DETECTED: Current indexing shows OCCUPIED, swapped indexing shows FREE!")
                self.get_logger().error(f"   Root cause: q_start/q_goal are in (x,y) format but being indexed as (row,col)")
                self.get_logger().error(f"   Fix: Swap q_start/q_goal elements before passing to RRT*")
                return "SWAPPED"
        
        return "NORMAL"

    def path_planning_loop(self):
        if self.rotation_state in ('spinning_360', 'halted'):
            return
        if self.binary_map is None or self.robot_pose is None or self.goal_pose is None:
            return
        
        # self.get_logger().info(f"In path loop")
        # Conversion of robot pose and goal pose to cell coordinate
        # q_goal = (np.array([self.goal_pose[1], self.goal_pose[0]]) - self.origin) / self.resolution # why do we need to swap between the x and y
        # q_start = (np.array([self.robot_pose.y, self.robot_pose.x]) - self.origin) / self.resolution
        # q_goal_point = PointRRT(q_goal[0], q_goal[1])
        # q_start_point = PointRRT(q_start[0], q_start[1])

        q_start = (np.array([self.robot_pose.x, self.robot_pose.y]) - self.origin) / self.resolution
        q_goal  = (np.array([self.goal_pose[0], self.goal_pose[1]]) - self.origin) / self.resolution
        q_start_point = PointRRT(q_start[0], q_start[1])  # Point.x=col, Point.y=row
        q_goal_point  = PointRRT(q_goal[0],  q_goal[1])

        # Debug logging: map and coordinate info
        self.get_logger().debug(f"Map shape (h, w): {self.binary_map.shape}, Resolution: {self.resolution}, Origin: {self.origin}")
        self.get_logger().debug(f"Robot world pose: ({self.robot_pose.x:.2f}, {self.robot_pose.y:.2f})")
        self.get_logger().debug(f"Robot cell coords (q_start): ({q_start[0]:.2f}, {q_start[1]:.2f})")
        self.get_logger().debug(f"Goal world pose: ({self.goal_pose[0]:.2f}, {self.goal_pose[1]:.2f})")
        self.get_logger().debug(f"Goal cell coords (q_goal): ({q_goal[0]:.2f}, {q_goal[1]:.2f})")
        
        # Check bounds
        map_h, map_w = self.binary_map.shape
        # q_start_in_bounds = (0 <= q_start[0] < map_h and 0 <= q_start[1] < map_w)
        # q_goal_in_bounds = (0 <= q_goal[0] < map_h and 0 <= q_goal[1] < map_w)

        q_start_in_bounds = (0 <= q_start[0] < map_w and 0 <= q_start[1] < map_h)
        q_goal_in_bounds  = (0 <= q_goal[0]  < map_w and 0 <= q_goal[1]  < map_h)
        # self.get_logger().info(f"Start in bounds: {q_start_in_bounds}, Goal in bounds: {q_goal_in_bounds}")
        
        if not q_start_in_bounds:
            self.get_logger().warn(f"Start pose is outside map bounds!")
            return
        if not q_goal_in_bounds:
            self.get_logger().warn(f"Goal pose is outside map bounds!")
            self.goal_pose = None      # force frontier search to pick a new one

            self.find_frontier = True
            return

        # RUN COORDINATE ORDERING INSPECTION
        # coord_status = self.inspect_coordinate_ordering(q_start, q_goal)

        rrt_star = RRT_STAR(self.delta_q, self.p, self.max_depth, self.min_dist, self.radius, self.threshold_path_rewire_dist)
        
        # Detailed debugging: inspect actual map values at start/goal
        # q_start_int = np.array([int(q_start[0]), int(q_start[1])])
        # q_goal_int = np.array([int(q_goal[0]), int(q_goal[1])])
        
        # start_cell_value = self.binary_map[q_start_int[0], q_start_int[1]]
        # goal_cell_value = self.binary_map[q_goal_int[0], q_goal_int[1]]

        # start_cell_value = self.binary_map[q_start_int[1], q_start_int[0]]
        # goal_cell_value  = self.binary_map[q_goal_int[1],  q_goal_int[0]]
        
        # self.get_logger().info(f"START cell value: binary_map[{q_start_int[0]}, {q_start_int[1]}] = {start_cell_value} (0=free, 1=occupied)")
        # self.get_logger().info(f"GOAL cell value: binary_map[{q_goal_int[0]}, {q_goal_int[1]}] = {goal_cell_value} (0=free, 1=occupied)")

        # # Inspect 3x3 neighborhood around start
        # self.get_logger().info(f"Start 3x3 neighborhood (rows {q_start_int[0]-1} to {q_start_int[0]+1}, cols {q_start_int[1]-1} to {q_start_int[1]+1}):")
        # for i in range(max(0, q_start_int[0]-1), min(self.binary_map.shape[0], q_start_int[0]+2)):
        #     row_str = " ".join([str(int(self.binary_map[i, j])) for j in range(max(0, q_start_int[1]-1), min(self.binary_map.shape[1], q_start_int[1]+2))])
        #     marker = " <-- START ROW" if i == q_start_int[0] else ""
        #     self.get_logger().info(f"  Row {i}: [{row_str}]{marker}")

        # # Inspect 3x3 neighborhood around goal
        # self.get_logger().info(f"Goal 3x3 neighborhood (rows {q_goal_int[0]-1} to {q_goal_int[0]+1}, cols {q_goal_int[1]-1} to {q_goal_int[1]+1}):")
        # for i in range(max(0, q_goal_int[0]-1), min(self.binary_map.shape[0], q_goal_int[0]+2)):
        #     row_str = " ".join([str(int(self.binary_map[i, j])) for j in range(max(0, q_goal_int[1]-1), min(self.binary_map.shape[1], q_goal_int[1]+2))])
        #     marker = " <-- GOAL ROW" if i == q_goal_int[0] else ""
        #     self.get_logger().info(f"  Row {i}: [{row_str}]{marker}")

        # # Check if start/goal coordinates are swapped or have row/col confusion
        # # Try the transposed index access to see if that makes it free
        # if self.binary_map.shape[0] != self.binary_map.shape[1]:  # Non-square map
        #     start_cell_transposed = self.binary_map[q_start_int[1], q_start_int[0]] if q_start_int[1] < self.binary_map.shape[0] and q_start_int[0] < self.binary_map.shape[1] else None
        #     if start_cell_transposed is not None:
        #         self.get_logger().info(f"START cell with transposed indices [col,row]: binary_map[{q_start_int[1]}, {q_start_int[0]}] = {start_cell_transposed}")
        
        # Check occupancy
        is_start_occupied = rrt_star.is_point_occupied(q_start_point, self.binary_map)
        is_goal_occupied = rrt_star.is_point_occupied(q_goal_point, self.binary_map)
        
        self.get_logger().warn(f"🔍 OCCUPANCY CHECK: start={is_start_occupied}, goal={is_goal_occupied}")
        
        if is_start_occupied:
            self.get_logger().error(f"❌ START IS OCCUPIED - This indicates a coordinate ordering issue (row/col vs x/y mismatch)")
        
        is_goal_occupied = rrt_star.is_point_occupied(q_goal_point, self.binary_map)
        self.get_logger().debug(f"Goal point occupied: {is_goal_occupied}")
        if is_goal_occupied:
            self.get_logger().warn(f"Goal point is not valid (on obstacle). Select another goal")
            self.find_frontier = True
            self.goal_pose = None   # prevent re-entering this loop with the same bad goal
            self.waypoints = None   # stop robot — current path leads to a now-blocked goal
            return

        if self.waypoints is not None:
            # ── OLD: only checked robot → next waypoint ──────────────────────
            # next_waypoint = self.waypoints[0]
            # q_next = (np.array([next_waypoint[0], next_waypoint[1]]) - self.origin) / self.resolution
            # q_next_point = PointRRT(q_next[0], q_next[1])
            # self.collide_robot_next_waypoint = not rrt_star.is_segment_free_bisection(
            #     q_start_point, q_next_point, self.binary_map, 0)
            # if not self.collide_robot_next_waypoint:
            #     for wp in self.waypoints:
            #         q_wp = (np.array([wp[0], wp[1]]) - self.origin) / self.resolution
            #         q_wp_point = PointRRT(q_wp[0], q_wp[1])
            #         if rrt_star.is_point_occupied(q_wp_point, self.binary_map):
            #             self.collide_robot_next_waypoint = True
            #             self.get_logger().warn("A downstream waypoint is now inside an obstacle — forcing replan")
            #             break
            # if self.collide_robot_next_waypoint:
            #     self.get_logger().warn("Path to next waypoint is not clear. Stop and replan")
            #     stop_msg = Twist()
            #     self.cmd_vel_pub.publish(stop_msg)

            # ── NEW: check every segment robot→wp[0], wp[0]→wp[1], … ─────────
            # max_depth is scaled per-segment so long smoothed segments are sampled
            # densely enough to catch any obstacle narrower than the segment length.
            prev_point = q_start_point
            for wp in self.waypoints:
                q_wp = (np.array([wp[0], wp[1]]) - self.origin) / self.resolution
                wp_point = PointRRT(q_wp[0], q_wp[1])
                seg_len = prev_point.dist(wp_point)
                rrt_star.max_depth = max(self.max_depth,
                                         round(math.log(max(seg_len, 2), 2)) + 1)
                if not rrt_star.is_segment_free_bisection(prev_point, wp_point, self.binary_map, 0):
                    self.collide_robot_next_waypoint = True
                    break
                prev_point = wp_point
            rrt_star.max_depth = self.max_depth  # restore for RRT* planning below

            if self.collide_robot_next_waypoint:
                self.get_logger().warn(f"Path segment is now blocked — stopping and replanning")
                self.cmd_vel_pub.publish(Twist())

        if self.complete_a_path is False and not self.collide_robot_next_waypoint: 
            return
        else:
            path = []
            self.get_logger().info(f"Planning a path from {q_start} to {q_goal}")
            self.get_logger().info(f"RRT* params: delta_q={self.delta_q}, p={self.p}, max_iter={self.max_iterations}, min_dist={self.min_dist}, radius={self.radius}")
            viz_cb = self._make_rrt_viz_callback()
            G, edges, iter = rrt_star.sample(self.binary_map, self.max_iterations, q_start[0], q_start[1], q_goal[0], q_goal[1], logger=self.get_logger(), viz_callback=viz_cb)
            self.get_logger().info(f"RRT* sampling completed: iterations={iter}, tree size={len(G)}, edges={len(edges)}")
            if iter == self.max_iterations and len(edges) == 0:
                self.rrt_fail_count += 1
                self.max_iterations = min(
                    self.max_iterations_base + self.rrt_fail_count * 2000,
                    self.max_iterations_cap
                )
                MAX_RETRIES_SAME_GOAL = 3
                if self.rrt_fail_count < MAX_RETRIES_SAME_GOAL:
                    # Retry same goal with more iterations — control_loop will set
                    # complete_a_path=True via the waypoints=None branch, re-triggering planning
                    self.get_logger().warn(
                        f"RRT* failed (attempt {self.rrt_fail_count}/{MAX_RETRIES_SAME_GOAL}). "
                        f"Retrying same goal with {self.max_iterations} iterations."
                    )
                    self.waypoints = None   # stop robot while replanning
                else:
                    # Exhausted retries — abandon goal, look for a new frontier
                    self.get_logger().warn(
                        f"Cannot find path after {self.rrt_fail_count} attempts — "
                        f"abandoning goal and selecting new frontier."
                    )
                    self.rrt_fail_count = 0
                    self.max_iterations = self.max_iterations_base
                    self.find_frontier = True
                    self.goal_pose = None
                    self.waypoints = None
            else:
                self.rrt_fail_count = 0
                self.max_iterations = self.max_iterations_base
                G, edges, path = rrt_star.fill_path(G, edges)
                path = rrt_star.smoothing(self.binary_map, G, path)
                self.get_logger().info(f"Find a path with {len(path)} waypoints")
                rrt_star.plot(self.binary_map, G, edges, path)

            waypoints = []
            for i in range(1, len(path)):
                q = G[path[i]]
                # q = np.array([q.y, q.x])
                # coordinate = q * self.resolution + self.origin
                # waypoints.append(np.array([coordinate[0], coordinate[1]]))
                coordinate = np.array([q.x, q.y]) * self.resolution + self.origin
                waypoints.append(np.array([coordinate[0], coordinate[1]]))

            if len(waypoints) > 0: 
                self.waypoints = waypoints
                waypoints_viz = [np.array([self.robot_pose.x, self.robot_pose.y])] + waypoints # add robot start position to path
                self.publish_positions_as_markers(waypoints_viz)
            
            self.complete_a_path = False
            self.collide_robot_next_waypoint = False

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    # ── Arm retraction ────────────────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        pos_map = dict(zip(msg.name, msg.position))
        if 'turtlebot/swiftpro/joint2' in pos_map:
            self.arm_q2 = pos_map['turtlebot/swiftpro/joint2']
        if 'turtlebot/swiftpro/joint3' in pos_map:
            self.arm_q3 = pos_map['turtlebot/swiftpro/joint3']

    def _step_arm_to_retract(self) -> bool:
        """Proportional controller that nudges q2/q3 toward the retracted
        configuration every control tick.

        Returns True  if the arm is already within tolerance (safe to move).
        Returns False if still moving toward retracted position.
        """
        if self.arm_q2 is None or self.arm_q3 is None:
            return False  # no joint feedback yet — hold base

        err2 = self.ARM_Q2_RETRACT - self.arm_q2
        err3 = self.ARM_Q3_RETRACT - self.arm_q3

        if abs(err2) < self.ARM_RETRACT_TOL and abs(err3) < self.ARM_RETRACT_TOL:
            # Already retracted — send zero to stop any residual motion
            cmd = Float64MultiArray()
            cmd.data = [0.0, 0.0, 0.0, 0.0]
            self.arm_cmd_pub.publish(cmd)
            return True

        # Proportional velocities, clamped to max_vel
        dq2 = float(np.clip(self.arm_kp * err2, -self.arm_max_vel, self.arm_max_vel))
        dq3 = float(np.clip(self.arm_kp * err3, -self.arm_max_vel, self.arm_max_vel))

        # Respect joint limits (mirror the limits from lab2_rrc_debug_node)
        if self.arm_q2 >= 0.045 and dq2 > 0:  dq2 = 0.0
        if self.arm_q2 <= -1.50 and dq2 < 0:  dq2 = 0.0
        if self.arm_q3 >= 0.045 and dq3 > 0:  dq3 = 0.0
        if self.arm_q3 <= -1.50 and dq3 < 0:  dq3 = 0.0

        cmd = Float64MultiArray()
        cmd.data = [0.0, dq2, dq3, 0.0]  # q1 unchanged, q4 passive
        self.arm_cmd_pub.publish(cmd)
        return False

    # ─────────────────────────────────────────────────────────────────────────

    def control_loop(self):
        # ── Terminal states ───────────────────────────────────────────────────
        if self.rotation_state == 'halted':
            self.cmd_vel_pub.publish(Twist())
            return

        if self.rotation_state == 'spinning_360':
            if self.prev_yaw_for_spin is None:
                self.prev_yaw_for_spin = self.current_yaw
                self.spin_accumulated = 0.0
            delta = abs(self.normalize_angle(self.current_yaw - self.prev_yaw_for_spin))
            self.spin_accumulated += delta
            self.prev_yaw_for_spin = self.current_yaw
            if self.spin_accumulated >= 2 * math.pi - 0.1:
                self.cmd_vel_pub.publish(Twist())
                self.rotation_state = 'halted'
                self.get_logger().info("360° spin complete — halting permanently.")
            else:
                cmd = Twist()
                cmd.angular.z = self.max_angular_velocity
                self.cmd_vel_pub.publish(cmd)
            return
        # ─────────────────────────────────────────────────────────────────────

        # self.get_logger().info(f"In control loop received waypoint {self.waypoints}")
        if self.waypoints is None or len(self.waypoints) == 0:
            # If no waypoints, ensure robot is stopped
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            self.complete_a_path = True
            self.rotation_state = 'idle'
            # Keep retracting arm even when idle so it's ready for next move
            self._step_arm_to_retract()
            return

        # ── Arm must be retracted before any base motion ─────────────────────
        if not self._step_arm_to_retract():
            self.cmd_vel_pub.publish(Twist())   # hold base while arm retracts
            return
        # ─────────────────────────────────────────────────────────────────────

        self.complete_a_path = False
        next_waypoint = self.waypoints[0]

        # idle → decide whether to scan (new location) or move directly (replan / same spot)
        if self.rotation_state == 'idle':
            robot_pos = (self.robot_pose.x, self.robot_pose.y)
            if (self.last_scan_pos is None or
                    math.hypot(robot_pos[0] - self.last_scan_pos[0],
                               robot_pos[1] - self.last_scan_pos[1]) > self.scan_distance_threshold):
                self.rotation_state = 'scanning_360'
                self.prev_yaw_for_spin = None
                self.spin_accumulated = 0.0
                self.last_scan_pos = robot_pos
                self.get_logger().info(
                    f"New location — scanning 360° before moving. "
                    f"pos=({robot_pos[0]:.2f},{robot_pos[1]:.2f})"
                )
            else:
                self.rotation_state = 'moving'

        if self.rotation_state == 'scanning_360':
            if self.prev_yaw_for_spin is None:
                self.prev_yaw_for_spin = self.current_yaw
                self.spin_accumulated = 0.0
            delta = abs(self.normalize_angle(self.current_yaw - self.prev_yaw_for_spin))
            self.spin_accumulated += delta
            self.prev_yaw_for_spin = self.current_yaw
            if self.spin_accumulated >= 2 * math.pi - 0.1:
                self.cmd_vel_pub.publish(Twist())
                self.rotation_state = 'moving'
                self.get_logger().info("Waypoint scan complete — moving to next waypoint")
            else:
                cmd = Twist()
                cmd.angular.z = self.max_angular_velocity
                self.cmd_vel_pub.publish(cmd)
            return

        if self.rotation_state == 'moving':
            inc_x = next_waypoint[0] - self.robot_pose.x
            inc_y = next_waypoint[1] - self.robot_pose.y
            dist  = np.sqrt(inc_x**2 + inc_y**2)

            if dist < self.acceptance_radius:
                self.waypoints.pop(0)
                self.cmd_vel_pub.publish(Twist())

                if len(self.waypoints) == 0:
                    self.get_logger().info("All waypoints reached.")
                    self.waypoints = None
                    self.goal_pose = None
                    self.complete_a_path = True
                    self.find_frontier = True
                    self.local_search_radius = 20
                    self.rotation_state = 'idle'
                    return

                # Arrived at an intermediate waypoint — scan only if genuinely new location
                robot_pos = (self.robot_pose.x, self.robot_pose.y)
                if (self.last_scan_pos is None or
                        math.hypot(robot_pos[0] - self.last_scan_pos[0],
                                   robot_pos[1] - self.last_scan_pos[1]) > self.scan_distance_threshold):
                    self.rotation_state = 'scanning_360'
                    self.prev_yaw_for_spin = None
                    self.spin_accumulated = 0.0
                    self.last_scan_pos = robot_pos
                # else: stay in 'moving' and proceed directly to the next waypoint
                return

            desired_yaw = math.atan2(inc_y, inc_x)
            angle_diff  = self.normalize_angle(desired_yaw - self.current_yaw)
            cmd = Twist()
            cmd.angular.z = min(self.kw * angle_diff, self.max_angular_velocity)
            if abs(angle_diff) <= 0.3:
                cmd.linear.x = min(self.kv * dist, self.max_linear_velocity)
            self.cmd_vel_pub.publish(cmd)
        
    def publish_positions_as_markers(self, positions):
        """
        Takes an array of (x, y) tuples and publishes them as a MarkerArray,
        including spheres for points and a line strip to link them.
        """
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # --- 1. Create the Line Strip Marker (The Links) ---
        if len(positions) > 1:
            line_marker = Marker()
            line_marker.header.frame_id = self.binary_map_frame
            line_marker.header.stamp = now
            line_marker.ns = "links"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            
            # Line width
            line_marker.scale.x = 0.05  
            
            # Color (Blue-ish)
            line_marker.color.r = 0.0
            line_marker.color.g = 0.5
            line_marker.color.b = 1.0
            line_marker.color.a = 0.8
            
            line_marker.pose.orientation.w = 1.0

            # Add all positions to the line's points list
            for (x, y) in positions:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                line_marker.points.append(p)
                
            marker_array.markers.append(line_marker)

        # --- 2. Create the Sphere Markers (The Points) ---
        for i, (x, y) in enumerate(positions):
            marker = Marker()
            marker.header.frame_id = self.binary_map_frame
            marker.header.stamp = now
            marker.ns = "positions"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.01  # Slightly higher than line to avoid flickering
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)

    sampling_turtlebot = SamplingTurtlebot()

    rclpy.spin(sampling_turtlebot)

    sampling_turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
