import rclpy
from rclpy.node import Node
import numpy as np
import math

from geometry_msgs.msg import PoseStamped, Twist, Point
from std_msgs.msg import Float64MultiArray, ColorRGBA
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray


class DWATurtlebot(Node):
    def __init__(self):
        super().__init__('dwa_grid_controller')

        # --------------------------------------------------
        # Robot State & Map
        # --------------------------------------------------
        self.robot_pose  = None
        self.current_yaw = 0.0
        self.goal_pose   = None
        self.current_vel = [0.0, 0.0]
        self.grid_map    = None  # OccupancyGrid

        # --------------------------------------------------
        # Robot Limits
        # --------------------------------------------------
        self.max_speed     = 0.35   # m/s
        self.max_yaw_rate  = 1.8    # rad/s
        self.max_accel     = 0.8    # m/s²
        self.max_delta_yaw = 1.2    # rad/s²
        self.dt            = 0.1    # s
        self.predict_time  = 2.5    # s

        # --------------------------------------------------
        # Robot Size
        # --------------------------------------------------
        self.robot_radius  = 0.0   # m — hard collision check
        self.safety_margin = 0.0   # m — soft penalty

        # --------------------------------------------------
        # Cost Weights
        # --------------------------------------------------
        self.heading_cost_weight  = 8.0
        self.dist_cost_weight     = 6.0
        self.obstacle_cost_weight = 8.0   
        self.velocity_cost_weight = 0.5

        # --------------------------------------------------
        # Stuck Detection
        # --------------------------------------------------
        self.last_goal_dist    = None
        self.no_progress_count = 0
        self.stuck_threshold   = 40   
        self.recovery_steps    = 0
        self.recovery_duration = 15   

        # --------------------------------------------------
        # Publishers & Subscribers
        # --------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist,       '/turtlebot/cmd_vel',  10)
        self.marker_pub  = self.create_publisher(MarkerArray, '/dwa_trajectories',   10)
        self.arm_pub = self.create_publisher(
            Float64MultiArray,
            '/turtlebot/swiftpro/joint_velocity_controller/command',
            10)

        self.create_subscription(Odometry,      '/turtlebot/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped,   '/goal_pose',      self.goal_callback, 10)
        self.create_subscription(OccupancyGrid, '/inflated_map',   self.map_callback,  10)

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("Grid-Based DWA Planner Started")

    # ======================================================
    # CALLBACKS
    # ======================================================

    def map_callback(self, msg: OccupancyGrid):
        self.grid_map = msg

    def odom_callback(self, msg: Odometry):
        self.robot_pose  = msg.pose.pose.position
        self.current_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        )

    def goal_callback(self, msg: PoseStamped):
        self.goal_pose         = msg.pose.position

        # Fold robot arm
        arm = Float64MultiArray()
        arm.data = [0.0, 0.0, -1.0, 0.0]
        self.arm_pub.publish(arm)

        self.last_goal_dist    = None
        self.no_progress_count = 0
        self.recovery_steps    = 0
        self.get_logger().info(
            f"New goal received: ({self.goal_pose.x:.2f}, {self.goal_pose.y:.2f})"
        )

    # ======================================================
    # GRID HELPERS
    # ======================================================

    def _world_to_grid(self, x: float, y: float):
        """Convert world coordinates (m) → grid cell indices (gx, gy)."""
        info = self.grid_map.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        return gx, gy

    def _get_cell_value(self, x: float, y: float):
        """
        Return cell value at world position (x, y).
        - 0..100 : occupancy probability
        - -1     : unknown
        - None   : out of bounds
        """
        if self.grid_map is None:
            return None
        info = self.grid_map.info
        gx, gy = self._world_to_grid(x, y)
        if 0 <= gx < info.width and 0 <= gy < info.height:
            return self.grid_map.data[gy * info.width + gx]
        return None  # out of bounds

    def is_collision(self, x: float, y: float) -> bool:
        """
        True if cell is occupied (val > 50).
        Out-of-bounds → collision.
        Unknown (-1) → NOT hard collision, only soft penalty.
        """
        val = self._get_cell_value(x, y)
        if val is None:
            return True    # out of bounds → collision
        return val > 50

    def is_near_obstacle(self, x: float, y: float) -> bool:
        """
        True if cell is occupied (>50) or unknown (-1).
        Used for proximity soft penalty only.
        """
        val = self._get_cell_value(x, y)
        if val is None:
            return True    # out of bounds → treat as near obstacle for safety
        if val == -1:
            return True    # unknown → treat as near obstacle for safety
        return val > 50 

    # ======================================================
    # OBSTACLE COST
    # ======================================================

    def obstacle_cost(self, traj: list) -> float:
        """
        Check trajectory against grid:
        1. Hard collision at robot center → inf
        2. Hard collision at footprint (8 points on robot_radius circle) → inf
        3. Soft penalty if safety_margin touches obstacle/unknown → 0.0–1.0
        """
        if self.grid_map is None:
            return 0.0

        angles_8 = [i * (2.0 * math.pi / 8) for i in range(8)]
        penalty  = 0.0
        sampled  = traj[::3]  # sample every 3 steps for performance

        for x, y, yaw in sampled:
            # ── 1. Robot center ───────────────────────────
            if self.is_collision(x, y):
                return float('inf')

            # ── 2. Footprint (robot_radius) ───────────────
            for a in angles_8:
                fx = x + self.robot_radius * math.cos(a)
                fy = y + self.robot_radius * math.sin(a)
                if self.is_collision(fx, fy):
                    return float('inf')

            # ── 3. Safety margin (soft) ───────────────────
            # normalize per-step hit count to 0.0–1.0
            hit_count = 0
            for a in angles_8:
                sx = x + self.safety_margin * math.cos(a)
                sy = y + self.safety_margin * math.sin(a)
                if self.is_near_obstacle(sx, sy):
                    hit_count += 1
            penalty += hit_count / 8.0  # 0.0–1.0 per step

        n = len(sampled)
        return penalty / n if n > 0 else 0.0  # final range: 0.0–1.0

    # ======================================================
    # DWA CORE
    # ======================================================

    def dynamic_window(self):
        """Compute feasible velocity window based on acceleration limits."""
        v, w = self.current_vel
        v_min = max(0.0,               v - self.max_accel     * self.dt)
        v_max = min(self.max_speed,    v + self.max_accel     * self.dt)
        w_min = max(-self.max_yaw_rate, w - self.max_delta_yaw * self.dt)
        w_max = min( self.max_yaw_rate, w + self.max_delta_yaw * self.dt)
        return v_min, v_max, w_min, w_max

    def simulate_trajectory(self, v: float, w: float) -> list:
        """Simulate trajectory for velocity pair (v, w) over predict_time seconds."""
        x, y, yaw = self.robot_pose.x, self.robot_pose.y, self.current_yaw
        traj  = []
        steps = int(self.predict_time / self.dt)
        for _ in range(steps):
            x   += v * math.cos(yaw) * self.dt
            y   += v * math.sin(yaw) * self.dt
            yaw += w * self.dt
            traj.append((x, y, yaw))
        return traj

    def compute_dwa(self):
        """
        Sample velocity space within dynamic window,
        compute cost for each trajectory, return best (v, w).
        Returns (best_v, best_w, all_paths).
        """
        v_min, v_max, w_min, w_max = self.dynamic_window()
        best_v, best_w, best_cost  = 0.0, 0.0, float('inf')
        all_paths = []

        for v in np.arange(v_min, v_max + 0.01, 0.05):
            for w in np.arange(w_min, w_max + 0.01, 0.1):
                traj     = self.simulate_trajectory(v, w)
                obs_cost = self.obstacle_cost(traj)

                # Reject immediately on hard collision
                if obs_cost == float('inf'):
                    continue

                # ── Goal costs ────────────────────────────
                lx, ly, lyaw = traj[-1]
                goal_angle   = math.atan2(
                    self.goal_pose.y - ly,
                    self.goal_pose.x - lx
                )
                heading_err = abs(math.atan2(
                    math.sin(goal_angle - lyaw),
                    math.cos(goal_angle - lyaw)
                ))
                dist_end = math.hypot(
                    self.goal_pose.x - lx,
                    self.goal_pose.y - ly
                )

                cost = (
                    self.heading_cost_weight  * heading_err +
                    self.dist_cost_weight     * dist_end    +
                    self.obstacle_cost_weight * obs_cost    +
                    self.velocity_cost_weight * (self.max_speed - v)
                )

                all_paths.append({
                    'v': v, 'w': w, 'traj': traj, 'cost': cost
                })

                if cost < best_cost:
                    best_cost, best_v, best_w = cost, v, w

        return best_v, best_w, all_paths

    # ======================================================
    # CONTROL LOOP
    # ======================================================

    def control_loop(self):
        # Wait for required data
        if self.robot_pose is None or self.goal_pose is None or self.grid_map is None:
            return

        # ── Check if goal reached ─────────────────────────
        dist = math.hypot(
            self.goal_pose.x - self.robot_pose.x,
            self.goal_pose.y - self.robot_pose.y
        )
        if dist < 0.25:
            self.cmd_vel_pub.publish(Twist())
            self.goal_pose = None
            self.get_logger().info("✓ Goal Reached!")
            return

        # ── Stuck detection ───────────────────────────────
        if self.last_goal_dist is not None:
            progress = self.last_goal_dist - dist
            if progress < 0.01:   # less than 1 cm per step
                self.no_progress_count += 1
            else:
                self.no_progress_count = 0
        self.last_goal_dist = dist

        # ── Recovery mode ─────────────────────────────────
        if self.recovery_steps > 0:
            cmd = Twist()
            cmd.angular.z = 0.8   # rotate in place
            self.cmd_vel_pub.publish(cmd)
            self.recovery_steps -= 1
            self.get_logger().warn(
                f"Recovery mode: {self.recovery_steps} steps remaining"
            )
            return

        if self.no_progress_count >= self.stuck_threshold:
            self.get_logger().warn("Robot stuck! Starting recovery rotation.")
            self.no_progress_count = 0
            self.recovery_steps    = self.recovery_duration
            return

        # ── DWA Calculation ───────────────────────────────
        bv, bw, paths = self.compute_dwa()
        cmd = Twist()

        if not paths:
            self.get_logger().warn("No valid path found! Rotating in place.")
            cmd.angular.z = 0.5
        else:
            cmd.linear.x  = float(bv)
            cmd.angular.z = float(bw)

        self.cmd_vel_pub.publish(cmd)
        self.publish_paths(paths, bv, bw)

    # ======================================================
    # VISUALIZATION
    # ======================================================

    def publish_paths(self, paths: list, bv: float, bw: float):
        """Publish all trajectories to RViz; best path in green."""
        arr = MarkerArray()

        for i, p in enumerate(paths[::5]):   # subsample to avoid spamming RViz
            m                 = Marker()
            m.header.frame_id = self.grid_map.header.frame_id
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns     = "dwa_paths"
            m.id     = i
            m.type   = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.02

            is_best = (
                abs(p['v'] - bv) < 1e-3 and
                abs(p['w'] - bw) < 1e-3
            )
            if is_best:
                m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)   # green
            else:
                m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.2)   # dim yellow

            for x, y, _ in p['traj']:
                m.points.append(Point(x=float(x), y=float(y), z=0.05))

            arr.markers.append(m)

        self.marker_pub.publish(arr)


# ======================================================
# MAIN
# ======================================================

def main():
    rclpy.init()
    node = DWATurtlebot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()