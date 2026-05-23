import rclpy
import math
import copy

import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped, Twist
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

from online_motion_planning.bidirectional_rrt_star import BIRRT_STAR
from online_motion_planning.Point import Point as PointRRT

from dwa_interfaces.srv import ComputeVelocity


class PathPlannerNoSpin(Node):
    """BiRRT* path planner with DWA execution — no 360° scanning.

    Differences from PathPlannerNode (path_planner_tb):
      • No scanning_360 / spinning_360 states — robot drives directly.
      • When a goal is unreachable (path blocked, BiRRT* gives up),
        the goal is blacklisted via /frontier/goal_reached so frontier_node
        skips it, then the next best frontier is requested.
      • Consecutive failures are tracked; exploration is declared complete
        after max_consecutive_rejections failures in a row, OR after
        goal_timeout_sec passes with no new goal (frontier area too small
        — frontier_node stopped publishing).

    Parameters
    ----------
    map_frame                   (str,   default 'world_enu')
    is_sim                      (bool,  default False)
    acceptance_radius           (float, default 0.1  m)
    max_iterations              (int,   default 4000)
    max_consecutive_rejections  (int,   default 6)
      Stop after this many consecutive goal rejections in a row.
    goal_timeout_sec            (float, default 30.0 s)
      If no new frontier goal arrives within this time after triggering a
      search, declare exploration complete (no frontiers above the area
      threshold remain).
    """

    def __init__(self):
        super().__init__('path_planner_no_spin')

        # ── Robot state ──────────────────────────────────────────────────────
        self.robot_pose = None
        self.current_yaw = 0.0

        # ── Motion params ────────────────────────────────────────────────────
        self.declare_parameter('acceptance_radius', 0.1)
        self.acceptance_radius = self.get_parameter('acceptance_radius').value

        self.max_linear_velocity = 0.3
        self.max_angular_velocity = 0.3
        self.kv = 0.5
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
        self.arm_is_retracted = not is_sim

        # ── Planning state ───────────────────────────────────────────────────
        self.goal_pose = None
        self.waypoints = None
        self.complete_a_path = True

        self.declare_parameter('max_iterations', 4000)
        self.max_iterations = self.get_parameter('max_iterations').value
        self.max_iterations_base = self.max_iterations
        self.max_iterations_cap = 12000
        self.max_iterations_increment = 2000
        self.max_retry_same_goal = 3
        self.rrt_fail_count = 0

        self.delta_q = 4
        self.p = 0.3
        self.max_depth = round(math.log(self.delta_q, 2)) + 1
        self.min_dist = 5
        self.radius = 5
        self.threshold_path_rewire_dist = 5

        # ── Frontier rejection tracking ──────────────────────────────────────
        self.declare_parameter('max_consecutive_rejections', 6)
        self.max_consecutive_rejections = self.get_parameter(
            'max_consecutive_rejections').value

        self.declare_parameter('goal_timeout_sec', 30.0)
        self.goal_timeout_sec = self.get_parameter('goal_timeout_sec').value

        self.consecutive_rejections = 0
        self.exploration_done = False
        self._last_trigger_time = None    # set when we request a new frontier

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
        self.create_timer(0.1, self._control_loop)    # 10 Hz — drive
        self.create_timer(1.0, self._planning_loop)   # 1 Hz  — plan / check
        self.create_timer(1.0, self._timeout_check)   # 1 Hz  — area timeout

        self.get_logger().info('PathPlannerNoSpin ready')

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
        self.binary_map = np.where(copy.deepcopy(raw) >= 99, 1, 0)

    def _frontier_goal_cb(self, msg):
        if self.exploration_done:
            return
        new_goal = [msg.pose.position.x, msg.pose.position.y]
        if self.goal_pose != new_goal:
            self.goal_pose = new_goal
            self._last_trigger_time = None   # goal arrived — cancel timeout
            self.get_logger().info(
                f'New frontier goal: ({new_goal[0]:.2f}, {new_goal[1]:.2f})')

    def _exploration_complete_cb(self, msg):
        if not msg.data or self.exploration_done:
            return
        self.get_logger().info('Frontier node: no more frontiers — exploration complete')
        self.exploration_done = True
        self.cmd_vel_pub.publish(Twist())

    def _arm_retracted_cb(self, msg):
        self.arm_is_retracted = msg.data

    # ── Frontier helpers ──────────────────────────────────────────────────────

    def _trigger_frontier_search(self):
        msg = Bool()
        msg.data = True
        self.trigger_pub.publish(msg)
        self._last_trigger_time = self.get_clock().now()

    def _notify_goal_reached(self, goal_x, goal_y):
        """Tell frontier_node this position was visited (or rejected)."""
        msg = PoseStamped()
        msg.header.frame_id = self.binary_map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(goal_x)
        msg.pose.position.y = float(goal_y)
        msg.pose.orientation.w = 1.0
        self.goal_reached_pub.publish(msg)

    def _reject_goal(self, reason: str):
        """Blacklist the current goal and request the next best frontier.

        Publishes the failed goal to /frontier/goal_reached so frontier_node
        suppresses it within visited_frontier_radius_m, then triggers a new
        search.  Increments the consecutive rejection counter; halts if the
        limit is exceeded (no more frontiers above the area threshold).
        """
        if self.goal_pose is not None:
            gx, gy = self.goal_pose
            self.get_logger().warn(
                f'Rejecting goal ({gx:.2f},{gy:.2f}) — {reason}')
            self._notify_goal_reached(gx, gy)

        self.goal_pose = None
        self.waypoints = None
        self.complete_a_path = True
        self.rrt_fail_count = 0
        self.max_iterations = self.max_iterations_base
        self._dwa_future = None
        self.cmd_vel_pub.publish(Twist())

        self.consecutive_rejections += 1
        if self.consecutive_rejections >= self.max_consecutive_rejections:
            self.get_logger().info(
                f'{self.consecutive_rejections} consecutive goal rejections — '
                'no valid frontier found above area threshold, exploration complete')
            self.exploration_done = True
            return

        self.get_logger().info(
            f'Requesting next frontier '
            f'(rejection {self.consecutive_rejections}/'
            f'{self.max_consecutive_rejections})')
        self._trigger_frontier_search()

    # ── Goal timeout check (1 Hz) ─────────────────────────────────────────────

    def _timeout_check(self):
        """Declare exploration complete if no frontier arrives after timeout.

        This fires when frontier_node is unable to find any cluster above the
        minimum area threshold — it simply stops publishing goals.
        """
        if self.exploration_done or self._last_trigger_time is None:
            return
        if self.goal_pose is not None:
            return  # goal already arrived
        elapsed = (self.get_clock().now()
                   - self._last_trigger_time).nanoseconds / 1e9
        if elapsed > self.goal_timeout_sec:
            self.get_logger().info(
                f'No frontier goal received in {elapsed:.1f} s — '
                'all remaining frontiers are below the area threshold, '
                'exploration complete')
            self.exploration_done = True
            self._last_trigger_time = None
            self.cmd_vel_pub.publish(Twist())

    # ── Planning loop (1 Hz) ─────────────────────────────────────────────────

    def _planning_loop(self):
        if self.exploration_done:
            return
        if self.binary_map is None or self.robot_pose is None:
            return
        if self.goal_pose is None:
            return

        # ── Convert world → map cells ─────────────────────────────────────
        q_start = (
            np.array([self.robot_pose.x, self.robot_pose.y]) - self.origin
        ) / self.resolution
        q_goal = (
            np.array([self.goal_pose[0], self.goal_pose[1]]) - self.origin
        ) / self.resolution

        map_h, map_w = self.binary_map.shape
        if not (0 <= q_start[0] < map_w and 0 <= q_start[1] < map_h):
            self.get_logger().warn('Start pose outside map — waiting')
            return
        if not (0 <= q_goal[0] < map_w and 0 <= q_goal[1] < map_h):
            self._reject_goal('goal outside map bounds')
            return

        rrt = BIRRT_STAR(
            self.delta_q, self.p, self.max_depth,
            self.min_dist, self.radius, self.threshold_path_rewire_dist)

        q_start_pt = PointRRT(q_start[0], q_start[1])
        q_goal_pt = PointRRT(q_goal[0], q_goal[1])

        # Snap start out of lethal cell if needed
        if rrt.is_point_occupied(q_start_pt, self.binary_map):
            fc, fr = self._find_nearest_free_cell(int(q_start[0]), int(q_start[1]))
            if fc is None:
                self.get_logger().warn('Start in lethal cell, no free cell nearby')
                return
            q_start = np.array([float(fc), float(fr)])
            q_start_pt = PointRRT(q_start[0], q_start[1])

        if rrt.is_point_occupied(q_goal_pt, self.binary_map):
            self._reject_goal('goal on obstacle')
            return

        # ── Collision-check current waypoints ─────────────────────────────
        if self.waypoints is not None:
            blocked = self._path_blocked(rrt, q_start_pt)
            if blocked:
                self.get_logger().warn('Existing path blocked — replanning')
                self.waypoints = None   # force replan below
                self.cmd_vel_pub.publish(Twist())
            elif not self.complete_a_path:
                return   # robot is mid-path and path is still clear

        # ── Run BiRRT* ────────────────────────────────────────────────────
        self.get_logger().info(
            f'Planning ({q_start[0]:.1f},{q_start[1]:.1f}) → '
            f'({q_goal[0]:.1f},{q_goal[1]:.1f}) '
            f'iter={self.max_iterations}')

        viz_a, flush_a = self._make_rrt_viz_callback(
            self.rrt_tree_a_pub, color_rgb=(0.2, 0.6, 1.0), z_height=0.06)
        viz_b, flush_b = self._make_rrt_viz_callback(
            self.rrt_tree_b_pub, color_rgb=(1.0, 0.45, 0.0), z_height=0.10)

        G, edges, iters = rrt.sample(
            self.binary_map, self.max_iterations,
            q_start[0], q_start[1], q_goal[0], q_goal[1],
            logger=self.get_logger(),
            viz_callback_a=viz_a, viz_callback_b=viz_b)
        flush_a()
        flush_b()

        if iters == self.max_iterations and len(edges) == 0:
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
                self._reject_goal(
                    f'BiRRT* failed after {self.max_retry_same_goal} attempts')
        else:
            self.rrt_fail_count = 0
            self.max_iterations = self.max_iterations_base
            G, edges, path = rrt.fill_path(G, edges)
            path = rrt.smoothing(self.binary_map, G, path)
            self.get_logger().info(f'Path found: {len(path)} waypoints')
            rrt.plot(self.binary_map, G, edges, path)

            waypoints = []
            for i in range(1, len(path)):
                q = G[path[i]]
                coord = np.array([q.x, q.y]) * self.resolution + self.origin
                waypoints.append(np.array([coord[0], coord[1]]))

            if waypoints:
                self.waypoints = waypoints
                viz_pts = ([np.array([self.robot_pose.x, self.robot_pose.y])]
                           + waypoints)
                self.publish_positions_as_markers(viz_pts)

        self.complete_a_path = False

    def _path_blocked(self, rrt, q_start_pt) -> bool:
        """Return True if any segment in self.waypoints is colliding."""
        prev = q_start_pt
        for wp in self.waypoints:
            q_wp = (np.array([wp[0], wp[1]]) - self.origin) / self.resolution
            wp_pt = PointRRT(q_wp[0], q_wp[1])
            seg_len = prev.dist(wp_pt)
            rrt.max_depth = max(self.max_depth,
                                round(math.log(max(seg_len, 2), 2)) + 1)
            if not rrt.is_segment_free_bisection(prev, wp_pt, self.binary_map, 0):
                rrt.max_depth = self.max_depth
                return True
            prev = wp_pt
        rrt.max_depth = self.max_depth
        return False

    # ── Control loop (10 Hz) ─────────────────────────────────────────────────

    def _control_loop(self):
        if self.robot_pose is None or self.exploration_done:
            return

        if self.waypoints is None or len(self.waypoints) == 0:
            self.cmd_vel_pub.publish(Twist())
            self.complete_a_path = True
            return

        if not self.arm_is_retracted:
            self.cmd_vel_pub.publish(Twist())
            return

        self.complete_a_path = False
        wp = self.waypoints[0]
        inc_x = wp[0] - self.robot_pose.x
        inc_y = wp[1] - self.robot_pose.y
        dist = math.hypot(inc_x, inc_y)

        if dist < self.acceptance_radius:
            self.waypoints.pop(0)
            self.cmd_vel_pub.publish(Twist())

            if len(self.waypoints) == 0:
                gx, gy = (self.goal_pose if self.goal_pose
                          else (wp[0], wp[1]))
                self._notify_goal_reached(gx, gy)
                self.get_logger().info('Goal reached — requesting next frontier')
                self.goal_pose = None
                self.waypoints = None
                self.complete_a_path = True
                self.consecutive_rejections = 0   # success resets counter
                self._trigger_frontier_search()
            return

        self._move_with_dwa_or_pursuit(wp, inc_x, inc_y)

    def _move_with_dwa_or_pursuit(self, waypoint, inc_x, inc_y):
        if self._dwa_client.service_is_ready():
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

            if self._dwa_future is None:
                req = ComputeVelocity.Request()
                req.goal_x = float(waypoint[0])
                req.goal_y = float(waypoint[1])
                self._dwa_future = self._dwa_client.call_async(req)
        else:
            desired_yaw = math.atan2(inc_y, inc_x)
            angle_diff = math.atan2(
                math.sin(desired_yaw - self.current_yaw),
                math.cos(desired_yaw - self.current_yaw))
            cmd = Twist()
            cmd.angular.z = min(self.kw * angle_diff, self.max_angular_velocity)
            if abs(angle_diff) <= 0.3:
                cmd.linear.x = min(self.kv * math.hypot(inc_x, inc_y),
                                   self.max_linear_velocity)
            self.cmd_vel_pub.publish(cmd)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _find_nearest_free_cell(self, col, row, max_radius=20):
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

    def _make_rrt_viz_callback(self, publisher, publish_every=10,
                               color_rgb=(0.2, 0.6, 1.0), z_height=0.08):
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

        def _pub():
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
            edge_points.append((
                float(G[pi].x * self.resolution + self.origin[0]),
                float(G[pi].y * self.resolution + self.origin[1])))
            edge_points.append((
                float(G[ci].x * self.resolution + self.origin[0]),
                float(G[ci].y * self.resolution + self.origin[1])))
            count[0] += 1
            if count[0] % publish_every == 0:
                _pub()

        return callback, _pub

    def publish_positions_as_markers(self, positions):
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
            for x, y in positions:
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


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNoSpin()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
