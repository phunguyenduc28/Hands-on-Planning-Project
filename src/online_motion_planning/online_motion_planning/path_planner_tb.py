import rclpy
from rclpy.node import Node
import numpy as np
import math
import copy

from geometry_msgs.msg import Point, PoseStamped, Twist
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

from online_motion_planning.bidirectional_rrt_star import BIRRT_STAR
from online_motion_planning.Point import Point as PointRRT

from dwa_interfaces.srv import ComputeVelocity


class PathPlannerNode(Node):
    """Global path planner (BiRRT*) with DWA-based waypoint execution.

    Receives frontier goals from FrontierNode, plans a collision-free path
    with BiRRT*, then drives through the waypoints by calling the DWA service
    for every control tick.  Arm retraction, 360° waypoint scans, and the
    terminal exploration spin are also managed here.

    Interface
    ---------
    Subscribers
      /frontier_goal (PoseStamped)           — goal from FrontierNode
      /frontier/exploration_complete (Bool)  — terminal exploration signal
      /inflated_map (OccupancyGrid)          — derives binary_map for BiRRT*
      /turtlebot/odom                        — robot pose / yaw
      /turtlebot/joint_states                — arm joint positions

    Publishers
      /cmd_vel                                         — velocity commands
      /visualization_marker_array (MarkerArray)        — planned path
      /rrt_viz/tree_a, /rrt_viz/tree_b (Marker)        — BiRRT* tree viz
      /turtlebot/swiftpro/joint_velocity_controller/command — arm velocity
      /frontier/trigger (Bool)                         — True → find new frontier
      /frontier/goal_reached (PoseStamped)             — reached goal notification

    Service client
      /dwa/compute_velocity (dwa_interfaces/ComputeVelocity)
    """

    def __init__(self):
        super().__init__('path_planner_tb')

        # ── Robot state ──────────────────────────────────────────────────────
        self.robot_pose = None
        self.current_yaw = 0.0

        # ── Motion params ────────────────────────────────────────────────────
        self.declare_parameter('acceptance_radius', 0.1)
        self.acceptance_radius = self.get_parameter('acceptance_radius').value
        self.declare_parameter('scan_distance_threshold', 0.4)
        self.scan_distance_threshold = self.get_parameter('scan_distance_threshold').value
        self.max_linear_velocity = 0.3
        self.max_angular_velocity = 0.3
        self.kv = 0.5     # pure-pursuit fallback gain
        self.kw = 1.0

        # ── Map (inflated → binary_map for BiRRT*) ───────────────────────────
        self.binary_map = None
        self.origin = None
        self.resolution = None
        self.height = None
        self.width = None

        self.declare_parameter('map_frame', 'world_enu')
        self.binary_map_frame = self.get_parameter('map_frame').value

        self.declare_parameter('is_sim', False)
        is_sim = self.get_parameter('is_sim').value
        # arm_is_retracted starts True on real robot (no arm node running),
        # False on sim (arm_retract_node will flip it once the arm is safe).
        self.arm_is_retracted = not is_sim

        # ── Planning state ───────────────────────────────────────────────────
        self.goal_pose = None       # [x, y] in world metres
        self.waypoints = None       # list of np.array([x, y])
        self.complete_a_path = True
        self.collide_robot_next_waypoint = False
        self.rrt_fail_count = 0

        self.declare_parameter('max_iterations', 4000)
        self.max_iterations = self.get_parameter('max_iterations').value
        self.max_iterations_base = self.max_iterations
        self.max_iterations_cap = 12000
        self.max_retry_same_goal = 3
        self.max_iterations_increment = 2000

        self.delta_q = 4
        self.p = 0.3
        self.max_depth = round(math.log(self.delta_q, 2)) + 1
        self.min_dist = 5
        self.radius = 5
        self.threshold_path_rewire_dist = 5

        # ── Terminal exploration ─────────────────────────────────────────────
        # Set to True by /frontier/exploration_complete signal.  Once set, the
        # robot follows its last saved path to the end then spins 360°.
        self.following_last_path = False

        # ── Rotation state machine ───────────────────────────────────────────
        # States: 'idle', 'scanning_360', 'moving', 'spinning_360', 'halted'
        self.rotation_state = 'idle'
        self.prev_yaw_for_spin = None
        self.spin_accumulated = 0.0
        self.last_scan_pos = None
        # scan_distance_threshold is declared above in motion params

        # ── DWA service client ───────────────────────────────────────────────
        self._dwa_future = None
        self._dwa_client = self.create_client(
            ComputeVelocity, '/dwa/compute_velocity')

        # ── Publishers ───────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/turtlebot/cmd_vel', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/visualization_marker_array', 10)
        self.rrt_tree_a_pub = self.create_publisher(
            Marker, '/rrt_viz/tree_a', 10)
        self.rrt_tree_b_pub = self.create_publisher(
            Marker, '/rrt_viz/tree_b', 10)
        self.trigger_pub = self.create_publisher(
            Bool, '/frontier/trigger', 10)
        self.goal_reached_pub = self.create_publisher(
            PoseStamped, '/frontier/goal_reached', 10)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            Odometry, '/turtlebot/odom', self._odom_cb, 10)
        self.create_subscription(
            OccupancyGrid, '/inflated_map', self._map_cb, 10)
        self.create_subscription(
            PoseStamped, '/frontier_goal', self._frontier_goal_cb, 10)
        self.create_subscription(
            Bool, '/frontier/exploration_complete',
            self._exploration_complete_cb, 10)
        self.create_subscription(
            Bool, '/arm/is_retracted', self._arm_retracted_cb, 10)

        # ── Timers ───────────────────────────────────────────────────────────
        self.control_timer = self.create_timer(0.1, self.control_loop)   # 10 Hz
        self.path_timer = self.create_timer(1.0, self.path_planning_loop)  # 1 Hz

        self.get_logger().info('Path planner node started')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        self.robot_pose = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z))

    def _map_cb(self, msg):
        info = msg.info
        self.resolution = info.resolution
        self.width = info.width
        self.height = info.height
        self.origin = np.array([info.origin.position.x, info.origin.position.y])
        raw = np.array(msg.data, dtype=float).reshape(self.height, self.width)
        raw[raw == -1] = 50.0
        # binary_map: 0=passable, 1=lethal (>=99 to match BiRRT* convention)
        self.binary_map = np.where(copy.deepcopy(raw) >= 99, 1, 0)

    def _frontier_goal_cb(self, msg):
        if self.rotation_state in ('spinning_360', 'halted') or self.following_last_path:
            return
        new_goal = [msg.pose.position.x, msg.pose.position.y]
        if self.goal_pose != new_goal:
            self.goal_pose = new_goal
            self.get_logger().info(
                f'New frontier goal: ({new_goal[0]:.2f}, {new_goal[1]:.2f})')

    def _exploration_complete_cb(self, msg):
        if not msg.data or self.following_last_path:
            return
        self.get_logger().info('Exploration complete signal received')
        if self.waypoints is not None and len(self.waypoints) > 0:
            self.following_last_path = True
            self.get_logger().info('Following last path, then spinning 360°')
        else:
            self.rotation_state = 'spinning_360'
            self.prev_yaw_for_spin = None
            self.spin_accumulated = 0.0

    def _arm_retracted_cb(self, msg):
        self.arm_is_retracted = msg.data

    # ── Helper utilities ─────────────────────────────────────────────────────

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _find_nearest_free_cell(self, col, row, max_radius=20):
        """BFS outward to find the closest free (binary_map==0) cell.

        Used to snap the planning start out of an inflation zone when
        localisation drift momentarily places the robot inside an obstacle.
        """
        from collections import deque
        if (0 <= row < self.height and 0 <= col < self.width
                and self.binary_map[row, col] == 0):
            return col, row
        queue = deque([(col, row)])
        visited = {(col, row)}
        while queue:
            c, r = queue.popleft()
            for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nc, nr = c + dc, r + dr
                if (nc, nr) in visited:
                    continue
                if abs(nc - col) > max_radius or abs(nr - row) > max_radius:
                    continue
                if not (0 <= nc < self.width and 0 <= nr < self.height):
                    continue
                visited.add((nc, nr))
                if self.binary_map[nr, nc] == 0:
                    return nc, nr
                queue.append((nc, nr))
        return None, None

    # ── RRT visualisation ─────────────────────────────────────────────────────

    def _make_rrt_viz_callback(self, publisher, publish_every=10,
                               color_rgb=(0.2, 0.6, 1.0), z_height=0.08):
        """Returns (edge_callback, flush) for one BiRRT* tree.

        edge_callback(G, parent_idx, child_idx) — called per new edge.
        flush() — force-publish the final tree state after sample() returns.
        """
        clear = Marker()
        clear.header.frame_id = self.binary_map_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.ns = 'rrt_tree'
        clear.id = 0
        clear.action = Marker.DELETEALL
        publisher.publish(clear)

        edge_points = []
        count = [0]
        r, g, b = color_rgb

        def _pub_marker():
            if not edge_points:
                return
            m = Marker()
            m.header.frame_id = self.binary_map_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'rrt_tree'
            m.id = 0
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.025
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.6
            m.pose.orientation.w = 1.0
            m.lifetime = rclpy.duration.Duration(seconds=15).to_msg()
            for x, y in edge_points:
                p = Point()
                p.x = x
                p.y = y
                p.z = z_height
                m.points.append(p)
            publisher.publish(m)

        def callback(G, pi, ci):
            parent = G[pi]
            child = G[ci]
            edge_points.append((
                float(parent.x * self.resolution + self.origin[0]),
                float(parent.y * self.resolution + self.origin[1])))
            edge_points.append((
                float(child.x * self.resolution + self.origin[0]),
                float(child.y * self.resolution + self.origin[1])))
            count[0] += 1
            if count[0] % publish_every == 0:
                _pub_marker()

        def flush():
            _pub_marker()

        return callback, flush

    # ── Path visualisation ────────────────────────────────────────────────────

    def publish_positions_as_markers(self, positions):
        """Publish waypoints as a LINE_STRIP + SPHERE markers."""
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        if len(positions) > 1:
            line = Marker()
            line.header.frame_id = self.binary_map_frame
            line.header.stamp = now
            line.ns = 'links'
            line.id = 0
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.05
            line.color.r = 0.0
            line.color.g = 0.5
            line.color.b = 1.0
            line.color.a = 0.8
            line.pose.orientation.w = 1.0
            for (x, y) in positions:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                line.points.append(p)
            marker_array.markers.append(line)

        for i, (x, y) in enumerate(positions):
            m = Marker()
            m.header.frame_id = self.binary_map_frame
            m.header.stamp = now
            m.ns = 'positions'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.01
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            m.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

    # ── Frontier communication helpers ────────────────────────────────────────

    def _trigger_frontier_search(self):
        msg = Bool()
        msg.data = True
        self.trigger_pub.publish(msg)

    def _notify_goal_reached(self, goal_x, goal_y):
        msg = PoseStamped()
        msg.header.frame_id = self.binary_map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(goal_x)
        msg.pose.position.y = float(goal_y)
        msg.pose.orientation.w = 1.0
        self.goal_reached_pub.publish(msg)

    # ── Path planning loop (1 Hz) ─────────────────────────────────────────────

    def path_planning_loop(self):
        if self.rotation_state in ('spinning_360', 'halted'):
            return
        if self.binary_map is None or self.robot_pose is None:
            return

        # ── Terminal mode: check last path for new obstacles ──────────────────
        if self.following_last_path:
            if self.waypoints is not None and len(self.waypoints) > 0:
                rrt_check = BIRRT_STAR(
                    self.delta_q, self.p, self.max_depth,
                    self.min_dist, self.radius, self.threshold_path_rewire_dist)
                q_start = (
                    np.array([self.robot_pose.x, self.robot_pose.y]) - self.origin
                ) / self.resolution
                prev_pt = PointRRT(q_start[0], q_start[1])
                blocked = False
                for wp in self.waypoints:
                    q_wp = (np.array([wp[0], wp[1]]) - self.origin) / self.resolution
                    wp_pt = PointRRT(q_wp[0], q_wp[1])
                    seg_len = prev_pt.dist(wp_pt)
                    rrt_check.max_depth = max(
                        self.max_depth, round(math.log(max(seg_len, 2), 2)) + 1)
                    if not rrt_check.is_segment_free_bisection(
                            prev_pt, wp_pt, self.binary_map, 0):
                        blocked = True
                        break
                    prev_pt = wp_pt
                if blocked:
                    self.get_logger().info(
                        'Last path now blocked — spinning 360° at current position')
                    self.cmd_vel_pub.publish(Twist())
                    self.waypoints = None
                    self.rotation_state = 'spinning_360'
                    self.prev_yaw_for_spin = None
                    self.spin_accumulated = 0.0
            return

        if self.goal_pose is None:
            return

        # ── Convert world coordinates to map cells ────────────────────────────
        q_start = (
            np.array([self.robot_pose.x, self.robot_pose.y]) - self.origin
        ) / self.resolution
        q_goal = (
            np.array([self.goal_pose[0], self.goal_pose[1]]) - self.origin
        ) / self.resolution
        q_start_point = PointRRT(q_start[0], q_start[1])
        q_goal_point = PointRRT(q_goal[0], q_goal[1])

        map_h, map_w = self.binary_map.shape
        if not (0 <= q_start[0] < map_w and 0 <= q_start[1] < map_h):
            self.get_logger().warn('Start pose outside map bounds')
            return
        if not (0 <= q_goal[0] < map_w and 0 <= q_goal[1] < map_h):
            self.get_logger().warn(
                'Goal pose outside map bounds — requesting new frontier')
            self.goal_pose = None
            self._trigger_frontier_search()
            return

        rrt_star = BIRRT_STAR(
            self.delta_q, self.p, self.max_depth,
            self.min_dist, self.radius, self.threshold_path_rewire_dist)

        # ── Snap start out of occupied cell if needed ────────────────────────
        if rrt_star.is_point_occupied(q_start_point, self.binary_map):
            free_col, free_row = self._find_nearest_free_cell(
                int(q_start[0]), int(q_start[1]))
            if free_col is None:
                self.get_logger().error(
                    'Start occupied, no free cell nearby — waiting for map update')
                return
            self.get_logger().warn(
                f'Start snapped: ({int(q_start[0])},{int(q_start[1])}) → '
                f'({free_col},{free_row})')
            q_start = np.array([float(free_col), float(free_row)])
            q_start_point = PointRRT(q_start[0], q_start[1])

        if rrt_star.is_point_occupied(q_goal_point, self.binary_map):
            self.get_logger().warn(
                'Goal on obstacle — requesting new frontier')
            self.goal_pose = None
            self.waypoints = None
            self._trigger_frontier_search()
            return

        # ── Collision-check existing path ────────────────────────────────────
        if self.waypoints is not None:
            prev_point = q_start_point
            for wp in self.waypoints:
                q_wp = (np.array([wp[0], wp[1]]) - self.origin) / self.resolution
                wp_point = PointRRT(q_wp[0], q_wp[1])
                seg_len = prev_point.dist(wp_point)
                rrt_star.max_depth = max(
                    self.max_depth, round(math.log(max(seg_len, 2), 2)) + 1)
                if not rrt_star.is_segment_free_bisection(
                        prev_point, wp_point, self.binary_map, 0):
                    self.collide_robot_next_waypoint = True
                    break
                prev_point = wp_point
            rrt_star.max_depth = self.max_depth

            if self.collide_robot_next_waypoint:
                self.get_logger().warn('Path blocked — stopping and replanning')
                self.cmd_vel_pub.publish(Twist())

        if not self.complete_a_path and not self.collide_robot_next_waypoint:
            return  # robot is mid-path, no replan needed

        # ── Run BiRRT* ───────────────────────────────────────────────────────
        self.get_logger().info(
            f'Planning: ({q_start[0]:.1f},{q_start[1]:.1f}) → '
            f'({q_goal[0]:.1f},{q_goal[1]:.1f}), '
            f'max_iter={self.max_iterations}')

        viz_cb_a, flush_a = self._make_rrt_viz_callback(
            self.rrt_tree_a_pub, color_rgb=(0.2, 0.6, 1.0), z_height=0.06)
        viz_cb_b, flush_b = self._make_rrt_viz_callback(
            self.rrt_tree_b_pub, color_rgb=(1.0, 0.45, 0.0), z_height=0.10)

        G, edges, iters = rrt_star.sample(
            self.binary_map, self.max_iterations,
            q_start[0], q_start[1], q_goal[0], q_goal[1],
            logger=self.get_logger(),
            viz_callback_a=viz_cb_a, viz_callback_b=viz_cb_b)
        flush_a()
        flush_b()

        if iters == self.max_iterations and len(edges) == 0:
            # Planning failed — retry with more iterations or abandon goal
            self.rrt_fail_count += 1
            self.max_iterations = min(
                self.max_iterations_base
                + self.rrt_fail_count * self.max_iterations_increment,
                self.max_iterations_cap)
            if self.rrt_fail_count < self.max_retry_same_goal:
                self.get_logger().warn(
                    f'BiRRT* failed (attempt {self.rrt_fail_count}/'
                    f'{self.max_retry_same_goal}) — '
                    f'retrying with {self.max_iterations} iterations')
                self.waypoints = None
            else:
                self.get_logger().warn(
                    'BiRRT* failed after max retries — requesting new frontier')
                self.rrt_fail_count = 0
                self.max_iterations = self.max_iterations_base
                self.goal_pose = None
                self.waypoints = None
                self._trigger_frontier_search()
        else:
            # Planning succeeded
            self.rrt_fail_count = 0
            self.max_iterations = self.max_iterations_base
            G, edges, path = rrt_star.fill_path(G, edges)
            path = rrt_star.smoothing(self.binary_map, G, path)
            self.get_logger().info(f'Path found with {len(path)} waypoints')
            rrt_star.plot(self.binary_map, G, edges, path)

            waypoints = []
            for i in range(1, len(path)):
                q = G[path[i]]
                coordinate = np.array([q.x, q.y]) * self.resolution + self.origin
                waypoints.append(np.array([coordinate[0], coordinate[1]]))

            if waypoints:
                self.waypoints = waypoints
                viz_pts = ([np.array([self.robot_pose.x, self.robot_pose.y])]
                           + waypoints)
                self.publish_positions_as_markers(viz_pts)

        self.complete_a_path = False
        self.collide_robot_next_waypoint = False

    # ── Control loop (10 Hz) ──────────────────────────────────────────────────

    def control_loop(self):
        if self.robot_pose is None:
            return

        # ── Terminal states ───────────────────────────────────────────────────
        if self.rotation_state == 'halted':
            self.cmd_vel_pub.publish(Twist())
            return

        if self.rotation_state == 'spinning_360':
            self._do_spin()
            return

        # ── No waypoints — stop and signal idle ───────────────────────────────
        if self.waypoints is None or len(self.waypoints) == 0:
            self.cmd_vel_pub.publish(Twist())
            self.complete_a_path = True
            self.rotation_state = 'idle'
            return

        # ── Wait for arm_retract_node before base moves (sim only) ────────────
        if not self.arm_is_retracted:
            self.cmd_vel_pub.publish(Twist())
            return

        self.complete_a_path = False
        next_waypoint = self.waypoints[0]

        # ── idle → decide: scan or move directly ──────────────────────────────
        if self.rotation_state == 'idle':
            robot_pos = (self.robot_pose.x, self.robot_pose.y)
            if (self.last_scan_pos is None
                    or math.hypot(robot_pos[0] - self.last_scan_pos[0],
                                  robot_pos[1] - self.last_scan_pos[1])
                    > self.scan_distance_threshold):
                self.rotation_state = 'scanning_360'
                self.prev_yaw_for_spin = None
                self.spin_accumulated = 0.0
                self.last_scan_pos = robot_pos
                self.get_logger().info(
                    f'New location — scanning 360° '
                    f'pos=({robot_pos[0]:.2f},{robot_pos[1]:.2f})')
            else:
                self.rotation_state = 'moving'

        if self.rotation_state == 'scanning_360':
            self._do_scan_360()
            return

        if self.rotation_state == 'moving':
            self._handle_moving(next_waypoint)

    def _do_spin(self):
        """Execute a terminal 360° spin; transition to 'halted' when done."""
        if self.prev_yaw_for_spin is None:
            self.prev_yaw_for_spin = self.current_yaw
            self.spin_accumulated = 0.0
        else:
            delta = self.current_yaw - self.prev_yaw_for_spin
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi
            self.spin_accumulated += abs(delta)
            self.prev_yaw_for_spin = self.current_yaw
        if self.spin_accumulated >= 2 * math.pi - 0.1:
            self.cmd_vel_pub.publish(Twist())
            self.rotation_state = 'halted'
            self.get_logger().info('360° spin complete — halted')
        else:
            cmd = Twist()
            cmd.angular.z = self.max_angular_velocity
            self.cmd_vel_pub.publish(cmd)

    def _do_scan_360(self):
        """Scan 360° at current waypoint; transition to 'moving' when done."""
        if self.prev_yaw_for_spin is None:
            self.prev_yaw_for_spin = self.current_yaw
            self.spin_accumulated = 0.0
        else:
            delta = self.current_yaw - self.prev_yaw_for_spin
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi
            self.spin_accumulated += abs(delta)
            self.prev_yaw_for_spin = self.current_yaw
        if self.spin_accumulated >= 2 * math.pi - 0.1:
            self.cmd_vel_pub.publish(Twist())
            self.rotation_state = 'moving'
            self.get_logger().info('Waypoint scan complete — moving')
        else:
            cmd = Twist()
            cmd.angular.z = self.max_angular_velocity
            self.cmd_vel_pub.publish(cmd)

    def _handle_moving(self, next_waypoint):
        """Execute one control tick toward next_waypoint using DWA or pure pursuit."""
        inc_x = next_waypoint[0] - self.robot_pose.x
        inc_y = next_waypoint[1] - self.robot_pose.y
        dist = math.hypot(inc_x, inc_y)

        # ── Waypoint reached ──────────────────────────────────────────────────
        if dist < self.acceptance_radius:
            self.waypoints.pop(0)
            self.cmd_vel_pub.publish(Twist())

            if len(self.waypoints) == 0:
                reached_goal = self.goal_pose
                self.waypoints = None
                self.goal_pose = None
                self.complete_a_path = True
                self.cmd_vel_pub.publish(Twist())

                if self.following_last_path:
                    self.get_logger().info(
                        'End of last path reached — spinning 360°')
                    self.rotation_state = 'spinning_360'
                    self.prev_yaw_for_spin = None
                    self.spin_accumulated = 0.0
                else:
                    if reached_goal is not None:
                        self._notify_goal_reached(
                            reached_goal[0], reached_goal[1])
                    self.get_logger().info('All waypoints reached')
                    self.rotation_state = 'idle'
                return

            # Intermediate waypoint — scan if at a new location
            robot_pos = (self.robot_pose.x, self.robot_pose.y)
            if (self.last_scan_pos is None
                    or math.hypot(robot_pos[0] - self.last_scan_pos[0],
                                  robot_pos[1] - self.last_scan_pos[1])
                    > self.scan_distance_threshold):
                self.rotation_state = 'scanning_360'
                self.prev_yaw_for_spin = None
                self.spin_accumulated = 0.0
                self.last_scan_pos = robot_pos
            return

        # ── Drive toward waypoint via DWA (or pure-pursuit fallback) ─────────
        self._move_with_dwa_or_pursuit(next_waypoint, inc_x, inc_y)

    def _move_with_dwa_or_pursuit(self, waypoint, inc_x, inc_y):
        """Issue a DWA service request and apply the most recent response.

        Pattern: on each 10 Hz tick we check if the previous async request
        is done and apply it, then immediately issue a new one.  Round-trip
        latency for a local service call is < 10 ms so the response is
        reliably available on the next tick (100 ms later).  Falls back to
        pure-pursuit if the DWA service is not running.
        """
        if self._dwa_client.service_is_ready():
            # Apply the previous response if ready
            if self._dwa_future is not None and self._dwa_future.done():
                try:
                    resp = self._dwa_future.result()
                    if resp.success:
                        cmd = Twist()
                        cmd.linear.x = resp.linear_x
                        cmd.angular.z = resp.angular_z
                        self.cmd_vel_pub.publish(cmd)
                except Exception as e:
                    self.get_logger().error(f'DWA service error: {e}')
                self._dwa_future = None

            # Issue a new request if none is pending
            if self._dwa_future is None:
                req = ComputeVelocity.Request()
                req.goal_x = float(waypoint[0])
                req.goal_y = float(waypoint[1])
                self._dwa_future = self._dwa_client.call_async(req)
        else:
            # Pure-pursuit fallback when DWA service is unavailable
            desired_yaw = math.atan2(inc_y, inc_x)
            angle_diff = self.normalize_angle(desired_yaw - self.current_yaw)
            dist = math.hypot(inc_x, inc_y)
            cmd = Twist()
            cmd.angular.z = min(self.kw * angle_diff, self.max_angular_velocity)
            if abs(angle_diff) <= 0.3:
                cmd.linear.x = min(self.kv * dist, self.max_linear_velocity)
            self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
