import rclpy
from rclpy.node import Node
import numpy as np
import math

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

from dwa_interfaces.srv import ComputeVelocity


class DWAServiceNode(Node):
    """DWA local planner exposed as a ROS 2 service.

    Maintains its own subscriptions to /turtlebot/odom and the inflated map so
    callers only need to supply the current waypoint (goal_x, goal_y).

    Service: /dwa/compute_velocity  (dwa_interfaces/srv/ComputeVelocity)
      Request : float64 goal_x, float64 goal_y
      Response: float64 linear_x, float64 angular_z, bool success
    """

    def __init__(self):
        super().__init__('dwa_service_node')

        # ── Robot state ───────────────────────────────────────────────────────
        self.robot_pose  = None
        self.current_yaw = 0.0
        self.current_vel = [0.0, 0.0]
        self.grid_map    = None

        # ── Kinematic limits ──────────────────────────────────────────────────
        self.max_speed     = 0.26   # TurtleBot3 max linear (m/s)
        self.max_yaw_rate  = 1.5    # TurtleBot3 max angular (rad/s)
        self.max_accel     = 0.8    # m/s²
        self.max_delta_yaw = 2.2    # rad/s²
        self.dt            = 0.1    # s  — must satisfy max_accel*dt > v_step (0.03)
        self.predict_time  = 3.0    # s
        self.viz_time      = 4.0    # s  — longer arcs for RViz clarity

        # ── Cost weights (all terms normalised 0–1 so weights are comparable) ─
        self.heading_cost_weight   = 0.3
        self.dist_cost_weight      = 0.3
        self.obstacle_cost_weight  = 3.0
        self.clearance_cost_weight = 1.5
        self.velocity_cost_weight  = 1.3

        # ── Velocity window tracker ───────────────────────────────────────────
        # The simulator odom always reports (0, 0) so we track the last
        # *commanded* velocity to keep the dynamic window evolving correctly.
        self._last_cmd_v = 0.0
        self._last_cmd_w = 0.0

        # ── Mode lock per waypoint ────────────────────────────────────────────
        # Mode (DWA vs pure pursuit) is decided once when a new waypoint arrives
        # and held for the entire journey to that waypoint.
        self._current_goal     = None   # (goal_x, goal_y) of active waypoint
        self._use_pure_pursuit = False  # locked mode for current waypoint
        self._dwa_aligning     = False  # True while pre-rotating before DWA starts

        # ── Escape sweep state machine ────────────────────────────────────────
        # Triggered when all DWA trajectories are rejected (robot trapped).
        # Sweeps right → centre → left, running a DWA check at each limit.
        self._escape_phase       = 0
        self._escape_ref_yaw     = None
        self._in_escape_sweep    = False
        self._escape_needs_check = False
        self._SWEEP_ANGLE        = math.radians(75)

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('inflated_map_topic', '/inflated_map')
        map_topic = self.get_parameter('inflated_map_topic').value

        self.declare_parameter('vel_deadzone_linear', 0.0)
        self.vel_deadzone_linear = self.get_parameter('vel_deadzone_linear').value
        self.declare_parameter('vel_deadzone_angular', 0.0)
        self.vel_deadzone_angular = self.get_parameter('vel_deadzone_angular').value

        # Below this distance the planner switches from DWA to pure pursuit.
        # Pure pursuit avoids the DWA circling problem when a waypoint is close
        # but behind or slightly to the side of the robot.
        self.declare_parameter('pure_pursuit_threshold', 0.5)
        self.pure_pursuit_threshold = self.get_parameter('pure_pursuit_threshold').value

        # Linear speed used during pure pursuit (m/s).
        self.declare_parameter('pure_pursuit_speed', 0.10)
        self.pure_pursuit_speed = self.get_parameter('pure_pursuit_speed').value

        # Heading error (rad) below which pure pursuit stops rotating and starts
        # driving forward. ~0.2 rad ≈ 11°.
        self.declare_parameter('pure_pursuit_heading_tol', 0.2)
        self.pure_pursuit_heading_tol = self.get_parameter('pure_pursuit_heading_tol').value

        # Heading error (rad) below which DWA pre-alignment stops and DWA begins.
        self.declare_parameter('dwa_heading_tol', 0.2)
        self.dwa_heading_tol = self.get_parameter('dwa_heading_tol').value

        # ── Publishers / subscribers ─────────────────────────────────────────
        self.marker_pub = self.create_publisher(MarkerArray, '/dwa_trajectories', 10)
        self.create_subscription(Odometry, '/turtlebot/odom', self._odom_cb, 10)
        self.create_subscription(OccupancyGrid, map_topic, self._map_cb, 10)
        self.get_logger().info(f'DWA costmap topic: {map_topic}')

        # ── Service server ───────────────────────────────────────────────────
        self.create_service(
            ComputeVelocity, '/dwa/compute_velocity', self._handle_compute)
        self.get_logger().info('DWA service node ready at /dwa/compute_velocity')

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def _odom_cb(self, msg):
        self.robot_pose  = msg.pose.pose.position
        self.current_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2))

    def _map_cb(self, msg):
        self.grid_map = msg

    # =========================================================================
    # SERVICE HANDLER
    # =========================================================================

    def _handle_compute(self, request, response):
        if self.robot_pose is None or self.grid_map is None:
            response.success = False
            return response

        goal_x, goal_y = request.goal_x, request.goal_y

        # ── New waypoint detected ─────────────────────────────────────────────
        new_goal = (goal_x, goal_y)
        if new_goal != self._current_goal:
            self._current_goal    = new_goal
            self._dwa_aligning    = True   # always align yaw before DWA
            self._in_escape_sweep = False
            self._escape_phase    = 0
            self._escape_ref_yaw  = None
            dist_at_arrival = math.hypot(
                goal_x - self.robot_pose.x, goal_y - self.robot_pose.y)
            self.get_logger().info(
                f'[MODE] New waypoint ({goal_x:.2f},{goal_y:.2f})  '
                f'dist={dist_at_arrival:.3f}m  → DWA (aligning first)')

        # ── Pure pursuit mode (disabled — kept for reference) ─────────────────
        # if self._use_pure_pursuit:
        #     dist_to_goal = math.hypot(
        #         goal_x - self.robot_pose.x, goal_y - self.robot_pose.y)
        #     v, w = self._pure_pursuit(goal_x, goal_y, dist_to_goal)
        #     response.linear_x  = float(v)
        #     response.angular_z = float(w)
        #     response.success   = True
        #     return response

        # ── DWA pre-alignment: rotate in place toward waypoint ────────────────
        if self._dwa_aligning:
            goal_angle = math.atan2(
                goal_y - self.robot_pose.y, goal_x - self.robot_pose.x)
            alpha = math.atan2(
                math.sin(goal_angle - self.current_yaw),
                math.cos(goal_angle - self.current_yaw))
            if abs(alpha) > self.dwa_heading_tol:
                w = math.copysign(min(self.max_yaw_rate, 1.5 * abs(alpha)), alpha)
                self.get_logger().info(
                    f'[DWA-ALIGN] α={math.degrees(alpha):.1f}°  w={w:.3f}')
                response.linear_x  = 0.0
                response.angular_z = float(w)
                response.success   = True
                return response
            self._dwa_aligning = False
            # Reset window tracker so DWA starts with a clean velocity window.
            # Without this, stale _last_cmd_w from the previous waypoint biases
            # the dynamic window into a forced turn, causing the robot to circle.
            self._last_cmd_v = 0.0
            self._last_cmd_w = 0.0
            self.get_logger().info('[DWA-ALIGN] aligned — handing off to DWA')

        # ── Escape sweep mode ─────────────────────────────────────────────────
        if self._in_escape_sweep:
            if self._escape_needs_check:
                self._escape_needs_check = False
                v, w, paths, forced = self._compute_dwa(goal_x, goal_y)
                if not forced:
                    self.get_logger().warn(
                        '[ESCAPE] Path found at sweep limit — exiting escape mode')
                    self._in_escape_sweep = False
                    self._escape_phase    = 0
                    self._escape_ref_yaw  = None
                    self._last_cmd_v      = v
                    self._last_cmd_w      = w
                    self._publish_paths(paths, v, w)
                else:
                    self.get_logger().warn(
                        '[ESCAPE] Still blocked at sweep limit — continuing sweep')
                    v, w = 0.0, 0.0
            else:
                v, w = self._escape_sweep_step()

            response.linear_x  = float(v)
            response.angular_z = float(w)
            response.success   = True
            return response

        # ── Normal DWA ────────────────────────────────────────────────────────
        v, w, paths, forced = self._compute_dwa(goal_x, goal_y)

        if forced:
            self._in_escape_sweep = True
            self._escape_ref_yaw  = None
            v, w = self._escape_sweep_step()
        else:
            # Only update window tracker on normal DWA (not escape velocities).
            self._last_cmd_v = v
            self._last_cmd_w = w

        # Dead zone compensation for real robot (no-op when deadzone == 0.0).
        if self.vel_deadzone_linear > 0.0 and 0.0 < v < self.vel_deadzone_linear:
            v = self.vel_deadzone_linear
        if self.vel_deadzone_angular > 0.0 and 0.0 < abs(w) < self.vel_deadzone_angular:
            w = math.copysign(self.vel_deadzone_angular, w)

        self._publish_paths(paths, v, w)

        response.linear_x  = float(v)
        response.angular_z = float(w)
        response.success   = True
        return response

    # =========================================================================
    # GRID HELPERS
    # =========================================================================

    def _world_to_grid(self, x, y):
        info = self.grid_map.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        return gx, gy

    def _get_cell_value(self, x, y):
        if self.grid_map is None:
            return None
        info = self.grid_map.info
        gx, gy = self._world_to_grid(x, y)
        if 0 <= gx < info.width and 0 <= gy < info.height:
            return self.grid_map.data[gy * info.width + gx]
        return None

    # =========================================================================
    # OBSTACLE COST
    # =========================================================================

    def _obstacle_cost(self, traj):
        """Return (avg_penalty, max_val_seen) for the trajectory.

        Hard-reject (inf) only non-start cells with val >= 100 so v=0 rotation
        is never blocked just because the robot is already inside an inflation
        zone.  Soft gradient penalty applies to ALL cells including the start
        cell so the cost baseline is honest — v=0 is not artificially cheaper
        than forward motion.
        """
        if self.grid_map is None:
            return 0.0, 0

        penalty      = 0.0
        max_val_seen = 0
        robot_gx, robot_gy = self._world_to_grid(
            self.robot_pose.x, self.robot_pose.y)

        for x, y, _ in traj:
            gx, gy   = self._world_to_grid(x, y)
            is_start = (gx == robot_gx and gy == robot_gy)

            val = self._get_cell_value(x, y)
            if val is None:
                if not is_start:
                    return float('inf'), 100
                continue

            if val >= 100 and not is_start:
                return float('inf'), 100

            if 0 < val < 100:
                penalty += val / 100.0

            if val > max_val_seen:
                max_val_seen = val

        return penalty / max(1, len(traj)), max_val_seen

    # =========================================================================
    # PURE PURSUIT  (close-range waypoint tracking)
    # =========================================================================

    def _pure_pursuit(self, goal_x, goal_y, dist):
        """Geometric arc controller for close waypoints.

        Two-phase controller:
          Phase 1 — rotate in place until heading error < heading_tol.
          Phase 2 — drive forward with pure-pursuit curvature κ = 2·sin(α)/L.

        Aligning first avoids wide arcs or backward motion when the waypoint
        is nearly behind the robot.
        """
        goal_angle = math.atan2(
            goal_y - self.robot_pose.y,
            goal_x - self.robot_pose.x)
        alpha = math.atan2(
            math.sin(goal_angle - self.current_yaw),
            math.cos(goal_angle - self.current_yaw))

        # Phase 1: heading not aligned — rotate in place.
        if abs(alpha) > self.pure_pursuit_heading_tol:
            w = math.copysign(
                min(self.max_yaw_rate, 1.5 * abs(alpha)),
                alpha)
            self.get_logger().info(
                f'[PURE-PURSUIT] aligning  α={math.degrees(alpha):.1f}°  w={w:.3f}')
            return 0.0, w

        # Phase 2: heading aligned — drive forward on a pure-pursuit arc.
        lookahead = max(dist, 0.01)
        curvature = 2.0 * math.sin(alpha) / lookahead
        v = min(self.pure_pursuit_speed, dist * 0.8)
        w = max(-self.max_yaw_rate, min(self.max_yaw_rate, v * curvature))
        self.get_logger().info(
            f'[PURE-PURSUIT] driving   dist={dist:.3f}m  v={v:.3f}  w={w:.3f}')
        return v, w

    # =========================================================================
    # DWA CORE
    # =========================================================================

    def _dynamic_window(self):
        v = self._last_cmd_v
        w = self._last_cmd_w
        v_min = max(0.0,              v - self.max_accel     * self.dt)
        v_max = min(self.max_speed,   v + self.max_accel     * self.dt)
        w_min = max(-self.max_yaw_rate, w - self.max_delta_yaw * self.dt)
        w_max = min(self.max_yaw_rate,  w + self.max_delta_yaw * self.dt)
        return v_min, v_max, w_min, w_max

    def _simulate(self, v, w, horizon=None):
        if horizon is None:
            horizon = self.predict_time
        x, y, yaw = self.robot_pose.x, self.robot_pose.y, self.current_yaw
        traj = []
        for _ in range(max(1, int(horizon / self.dt))):
            x   += v * math.cos(yaw) * self.dt
            y   += v * math.sin(yaw) * self.dt
            yaw += w * self.dt
            traj.append((x, y, yaw))
        return traj

    def _compute_dwa(self, goal_x, goal_y):
        dist_to_goal = math.hypot(
            goal_x - self.robot_pose.x, goal_y - self.robot_pose.y)
        init_dist = max(dist_to_goal, 0.1)

        # Clip planning horizon so trajectories never overshoot the waypoint
        # into unknown space, which would make every trajectory return obs=inf.
        horizon = max(0.5, min(self.predict_time,
                               dist_to_goal / max(self.max_speed, 0.01)))

        v_min, v_max, w_min, w_max = self._dynamic_window()

        best_v, best_w = 0.0, 0.0
        best_cost      = float('inf')
        best_breakdown = None
        all_paths      = []

        v_samples = list(np.arange(v_min, v_max + 0.01, 0.03))
        if 0.0 not in v_samples:
            v_samples.append(0.0)

        n_total      = 0
        n_rejected   = 0
        per_v_valid    = {}
        per_v_rejected = {}

        for v in v_samples:
            vk = round(v, 3)
            per_v_valid[vk]    = 0
            per_v_rejected[vk] = 0

            for w in np.arange(w_min, w_max + 0.01, 0.06):
                n_total += 1
                traj = self._simulate(v, w, horizon)
                obs_cost, max_val = self._obstacle_cost(traj)

                path_data = {
                    'v': v, 'w': w, 'traj': traj,
                    'cost': float('inf'), 'feasible': False,
                }

                if obs_cost == float('inf'):
                    n_rejected += 1
                    per_v_rejected[vk] += 1
                else:
                    per_v_valid[vk] += 1
                    path_data['feasible'] = True

                lx, ly, lyaw = traj[-1]
                goal_angle = math.atan2(goal_y - ly, goal_x - lx)
                heading_err = abs(math.atan2(
                    math.sin(goal_angle - lyaw),
                    math.cos(goal_angle - lyaw)))

                heading_err_norm = heading_err / math.pi
                dist_norm        = math.hypot(goal_x - lx, goal_y - ly) / init_dist
                clearance_cost   = max_val / 100.0
                velocity_cost    = (self.max_speed - v) / self.max_speed

                cost = (
                    self.heading_cost_weight   * heading_err_norm +
                    self.dist_cost_weight      * dist_norm +
                    self.obstacle_cost_weight  * obs_cost +
                    self.clearance_cost_weight * clearance_cost +
                    self.velocity_cost_weight  * velocity_cost
                )
                path_data['cost'] = cost
                all_paths.append(path_data)

                if cost < best_cost:
                    best_cost = cost
                    best_v, best_w = v, w
                    best_breakdown = {
                        'v': v, 'w': w, 'total': cost,
                        'heading': self.heading_cost_weight   * heading_err_norm,
                        'dist':    self.dist_cost_weight      * dist_norm,
                        'obs':     self.obstacle_cost_weight  * obs_cost,
                        'clear':   self.clearance_cost_weight * clearance_cost,
                        'vel':     self.velocity_cost_weight  * velocity_cost,
                    }

        # ── Logging ───────────────────────────────────────────────────────────
        self.get_logger().info(
            f'[DWA] window v=[{v_min:.2f},{v_max:.2f}] w=[{w_min:.2f},{w_max:.2f}]'
            f'  total={n_total} valid={n_total - n_rejected} rejected={n_rejected}'
            f'  dist={init_dist:.2f}m')
        for vk in sorted(per_v_valid.keys()):
            status = 'BLOCKED' if per_v_valid[vk] == 0 and vk > 0 else 'ok'
            self.get_logger().info(
                f'[DWA]   v={vk:.3f}: valid={per_v_valid[vk]}'
                f'  rejected={per_v_rejected[vk]}  [{status}]')

        if best_cost == float('inf'):
            self.get_logger().warn(
                f'[DWA] ALL trajectories rejected — entering escape sweep.'
                f'  dist={init_dist:.2f}m  yaw={math.degrees(self.current_yaw):.1f}°')
            return 0.0, 0.0, all_paths, True

        bd = best_breakdown
        self.get_logger().info(
            f'[DWA] best v={bd["v"]:.3f} w={bd["w"]:.3f}  cost={bd["total"]:.3f}'
            f'  [head={bd["heading"]:.3f} dist={bd["dist"]:.3f}'
            f'  obs={bd["obs"]:.3f} clr={bd["clear"]:.3f} vel={bd["vel"]:.3f}]')
        return best_v, best_w, all_paths, False

    # =========================================================================
    # ESCAPE SWEEP
    # =========================================================================

    def _escape_sweep_step(self):
        """One control tick of the right → centre → left escape sweep."""
        ROTATE_RATE = self.max_yaw_rate * 0.6
        TOL         = 0.08   # ~4.6° deadband

        if self._escape_ref_yaw is None:
            self._escape_ref_yaw = self.current_yaw
            self._escape_phase   = 1
            self.get_logger().warn(
                f'[ESCAPE] Sweep started — ref={math.degrees(self._escape_ref_yaw):.1f}°'
                f'  right_target={math.degrees(self._escape_ref_yaw - self._SWEEP_ANGLE):.1f}°'
                f'  left_target={math.degrees(self._escape_ref_yaw + self._SWEEP_ANGLE):.1f}°')

        ref = self._escape_ref_yaw
        err = math.atan2(
            math.sin(self.current_yaw - ref),
            math.cos(self.current_yaw - ref))

        if self._escape_phase == 1:          # sweep right (CW, w < 0)
            if err <= -(self._SWEEP_ANGLE - TOL):
                self._escape_phase       = 2
                self._escape_needs_check = True
                self.get_logger().warn(
                    f'[ESCAPE] Phase 1: right limit {math.degrees(self.current_yaw):.1f}°'
                    f' — checking for path before returning to centre')
            else:
                self.get_logger().info(
                    f'[ESCAPE] Phase 1 (sweep RIGHT): err={math.degrees(err):.1f}°'
                    f' / target={math.degrees(-self._SWEEP_ANGLE):.1f}°')
            return 0.0, -ROTATE_RATE

        elif self._escape_phase == 2:        # return to centre (CCW, w > 0)
            if abs(err) < TOL:
                self._escape_phase = 3
                self.get_logger().warn(
                    f'[ESCAPE] Phase 2→3: centre reached at {math.degrees(self.current_yaw):.1f}°'
                    f' — sweeping left to {math.degrees(ref + self._SWEEP_ANGLE):.1f}°')
            else:
                self.get_logger().info(
                    f'[ESCAPE] Phase 2 (return CENTRE): err={math.degrees(err):.1f}°')
            return 0.0, ROTATE_RATE

        else:                                # sweep left (CCW, w > 0), phase 3
            if err >= (self._SWEEP_ANGLE - TOL):
                self._escape_phase       = 1
                self._escape_ref_yaw     = self.current_yaw
                self._escape_needs_check = True
                self.get_logger().warn(
                    f'[ESCAPE] Phase 3: left limit {math.degrees(self.current_yaw):.1f}°'
                    f' — checking for path before restarting sweep')
            else:
                self.get_logger().info(
                    f'[ESCAPE] Phase 3 (sweep LEFT): err={math.degrees(err):.1f}°'
                    f' / target={math.degrees(self._SWEEP_ANGLE):.1f}°')
            return 0.0, ROTATE_RATE

    # =========================================================================
    # VISUALISATION
    # =========================================================================

    @staticmethod
    def _line_marker(frame, stamp, ns, mid, scale, color, lifetime):
        m = Marker()
        m.header.frame_id    = frame
        m.header.stamp       = stamp
        m.ns                 = ns
        m.id                 = mid
        m.type               = Marker.LINE_STRIP
        m.action             = Marker.ADD
        m.scale.x            = scale
        m.color              = color
        m.pose.orientation.w = 1.0
        m.lifetime           = lifetime
        return m

    def _publish_paths(self, paths, bv, bw):
        if self.grid_map is None:
            return

        marker_array = MarkerArray()
        now      = self.get_clock().now().to_msg()
        frame    = self.grid_map.header.frame_id
        lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        feasible   = [p for p in paths
                      if p['feasible']
                      and not (abs(p['v'] - bv) < 1e-3 and abs(p['w'] - bw) < 1e-3)]
        infeasible = [p for p in paths if not p['feasible']]

        cost_min   = min((p['cost'] for p in feasible), default=0.0)
        cost_range = max(
            max((p['cost'] for p in feasible), default=0.0) - cost_min, 1e-6)

        cid = 0

        # All candidate trajectories — solid purple, thin
        for i, p in enumerate(feasible):
            m = self._line_marker(frame, now, 'dwa_all', i + 1, 0.01,
                                  ColorRGBA(r=0.6, g=0.0, b=1.0, a=1.0), lifetime)
            for x, y, _ in p['traj']:
                pt = Point(); pt.x = float(x); pt.y = float(y); pt.z = 0.05
                m.points.append(pt)
            marker_array.markers.append(m)

            # Small cost label at trajectory midpoint
            if p['traj']:
                mx, my, _ = p['traj'][len(p['traj']) // 2]
                lm = Marker()
                lm.header.frame_id = frame
                lm.header.stamp    = now
                lm.ns              = 'dwa_costs'
                lm.id              = i + 10001
                lm.type            = Marker.TEXT_VIEW_FACING
                lm.action          = Marker.ADD
                lm.lifetime        = lifetime
                lm.pose.position.x = float(mx)
                lm.pose.position.y = float(my)
                lm.pose.position.z = 0.15
                lm.scale.z         = 0.03
                lm.color           = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
                lm.text            = f'{p["cost"]:.3f}'
                marker_array.markers.append(lm)

        # Best trajectory — solid green, thick
        best_path = next(
            (p for p in paths if abs(p['v'] - bv) < 1e-3 and abs(p['w'] - bw) < 1e-3),
            None)
        if best_path is not None:
            m = self._line_marker(frame, now, 'dwa_best', 9999, 0.04,
                                  ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), lifetime)
            for x, y, _ in best_path['traj']:
                pt = Point(); pt.x = float(x); pt.y = float(y); pt.z = 0.05
                m.points.append(pt)
            marker_array.markers.append(m)

            # Best cost label — white, slightly larger
            bx, by, _ = best_path['traj'][len(best_path['traj']) // 2]
            lm = Marker()
            lm.header.frame_id = frame
            lm.header.stamp    = now
            lm.ns              = 'dwa_costs'
            lm.id              = 19999
            lm.type            = Marker.TEXT_VIEW_FACING
            lm.action          = Marker.ADD
            lm.lifetime        = lifetime
            lm.pose.position.x = float(bx)
            lm.pose.position.y = float(by)
            lm.pose.position.z = 0.25
            lm.scale.z         = 0.04
            lm.color           = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            lm.text            = f'*{best_path["cost"]:.3f}'
            marker_array.markers.append(lm)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = DWAServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
