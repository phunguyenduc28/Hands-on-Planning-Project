import rclpy
from rclpy.node import Node
import numpy as np
import math

from geometry_msgs.msg import PoseStamped, Twist, Point
from std_msgs.msg import Float64MultiArray, ColorRGBA
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


class DWATurtlebot(Node):

    """
    DWA (Dynamic Window Approach) Local Planner Node.
    This node simulates multiple possible velocity trajectories to find the 
    optimal path that avoids obstacles while moving toward a goal.
    """

    def __init__(self):
        super().__init__('dwa_turtlebot_controller')

        # --------------------------------------------------
        # Robot State
        # --------------------------------------------------
        self.robot_pose = None             # Current robot position from odometry
        self.current_yaw = 0.0             # Current heading angle (rad)
        self.goal_pose = None              # Goal position from RViz
        self.current_vel = [0.0, 0.0]      # [linear velocity, angular velocity]
        self._obs_cache = []               # Cached obstacle points from LiDAR

        # --------------------------------------------------
        # Robot Limits
        # --------------------------------------------------
        self.max_speed = 0.35              # Maximum forward speed (m/s)
        self.max_yaw_rate = 1.8            # Maximum turning speed (rad/s)
        self.max_accel = 0.8               # Maximum linear acceleration (m/s^2)
        self.max_delta_yaw = 1.2           # Maximum angular acceleration (rad/s^2)

        self.dt = 0.1                      # Control loop period (10 Hz)
        self.predict_time = 2.5            # Trajectory prediction horizon (s)

        # --------------------------------------------------
        # Robot Size
        # --------------------------------------------------
        self.robot_radius = 0.42           # Collision radius of robot (m)
        self.safety_margin = 0.85          # Safe distance threshold (m)

        # --------------------------------------------------
        # Cost Weights
        # --------------------------------------------------
        self.heading_cost_weight = 8.0    # Prefer facing goal direction
        self.dist_cost_weight = 6.0       # Prefer ending closer to goal
        self.obstacle_cost_weight = 40.0  # Prefer safer paths from obstacles
        self.velocity_cost_weight = 0.5   # Prefer higher speed movement

        # --------------------------------------------------
        # Recovery / Stuck Detection
        # --------------------------------------------------
        self.last_goal_dist = None        # Previous distance to goal
        self.no_progress_count = 0        # Consecutive cycles without progress
        self.stuck_threshold = 15         # Trigger recovery after 15 loops

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/turtlebot/cmd_vel', 10)

        self.marker_pub = self.create_publisher(
            MarkerArray, '/dwa_trajectories', 10)

        self.arm_pub = self.create_publisher(
            Float64MultiArray,
            '/turtlebot/swiftpro/joint_velocity_controller/command',
            10)

        # --------------------------------------------------
        # Subscribers
        # --------------------------------------------------
        self.create_subscription(
            Odometry, '/turtlebot/odom',
            self.odom_callback, 10)

        self.create_subscription(
            PoseStamped, '/goal_pose',
            self.goal_callback, 10)

        self.create_subscription(
            LaserScan, '/scan',
            self.scan_callback, 10)

        # Timer
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Improved DWA Planner Started")

    # ======================================================
    # CALLBACKS
    # ======================================================

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose.position

        self.current_vel = [
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        ]

        q = msg.pose.pose.orientation

        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        )

    def goal_callback(self, msg):
        self.goal_pose = msg.pose.position

        # Fold robot arm
        arm = Float64MultiArray()
        arm.data = [0.0, 0.0, -1.0, 0.0]
        self.arm_pub.publish(arm)

        self.last_goal_dist = None
        self.no_progress_count = 0

        self.get_logger().info(
            f"New Goal: ({self.goal_pose.x:.2f}, {self.goal_pose.y:.2f})")

    def scan_callback(self, msg):
        if self.robot_pose is None:
            return

        obs = []

        rx = self.robot_pose.x
        ry = self.robot_pose.y
        yaw = self.current_yaw

        for i, r in enumerate(msg.ranges):

            if msg.range_min < r < 3.5:

                angle = msg.angle_min + i * msg.angle_increment + yaw

                ox = rx + r * math.cos(angle)
                oy = ry + r * math.sin(angle)

                obs.append((ox, oy))

        self._obs_cache = obs

    # ======================================================
    # DYNAMIC WINDOW
    # ======================================================

    def dynamic_window(self):

        v = self.current_vel[0]
        w = self.current_vel[1]

        v_min = max(0.0, v - self.max_accel * self.dt)
        v_max = min(self.max_speed, v + self.max_accel * self.dt)

        w_min = max(-self.max_yaw_rate,
                    w - self.max_delta_yaw * self.dt)

        w_max = min(self.max_yaw_rate,
                    w + self.max_delta_yaw * self.dt)

        return v_min, v_max, w_min, w_max

    # ======================================================
    # TRAJECTORY SIMULATION
    # ======================================================

    def simulate_trajectory(self, v, w):

        x = self.robot_pose.x
        y = self.robot_pose.y
        yaw = self.current_yaw

        traj = []

        steps = int(self.predict_time / self.dt)

        for _ in range(steps):

            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += w * self.dt

            traj.append((x, y, yaw))

        return traj

    # ======================================================
    # OBSTACLE COST
    # ======================================================

    def obstacle_cost(self, traj, v):

        if len(self._obs_cache) == 0:
            return 0.0

        obs = np.array(self._obs_cache)

        min_global = float('inf')
        total_penalty = 0.0

        for x, y, _ in traj:

            diff = obs - np.array([x, y])
            dist = np.hypot(diff[:, 0], diff[:, 1])

            min_d = np.min(dist)

            # collision
            if min_d < self.robot_radius:
                return float('inf')

            # braking distance
            braking = (v * v) / (2.0 * self.max_accel + 1e-6)

            if min_d < braking + self.robot_radius:
                return float('inf')

            min_global = min(min_global, min_d)

            if min_d < self.safety_margin:
                total_penalty += (self.safety_margin - min_d) ** 2

        cost = total_penalty / len(traj)

        if min_global < self.safety_margin:
            cost += (self.safety_margin - min_global) * 10.0

        return cost

    # ======================================================
    # MAIN DWA COMPUTATION
    # ======================================================

    def compute_dwa(self):

        v_min, v_max, w_min, w_max = self.dynamic_window()

        best_v = 0.0
        best_w = 0.0
        best_cost = float('inf')
        all_paths = []

        gx = self.goal_pose.x
        gy = self.goal_pose.y

        rx = self.robot_pose.x
        ry = self.robot_pose.y

        current_goal_dist = math.hypot(gx-rx, gy-ry)

        for v in np.arange(v_min, v_max + 0.001, 0.02):
            for w in np.arange(w_min, w_max + 0.001, 0.04):

                traj = self.simulate_trajectory(v, w)

                obs_cost = self.obstacle_cost(traj, v)

                if obs_cost == float('inf'):
                    continue

                lx, ly, lyaw = traj[-1]

                # heading error
                goal_angle = math.atan2(gy-ly, gx-lx)

                heading_error = abs(math.atan2(
                    math.sin(goal_angle - lyaw),
                    math.cos(goal_angle - lyaw)
                ))

                # distance to goal
                dist_end = math.hypot(gx-lx, gy-ly)

                # final cost
                cost = (
                    self.heading_cost_weight * heading_error +
                    self.dist_cost_weight * dist_end +
                    self.obstacle_cost_weight * obs_cost +
                    self.velocity_cost_weight * (self.max_speed - v)
                )

                all_paths.append({
                    'v': v,
                    'w': w,
                    'traj': traj,
                    'cost': cost
                })

                if cost < best_cost:
                    best_cost = cost
                    best_v = v
                    best_w = w

        return best_v, best_w, all_paths

    # ======================================================
    # RECOVERY TURN
    # ======================================================

    def smart_recovery(self):

        left = 0
        right = 0

        for ox, oy in self._obs_cache:

            dx = ox - self.robot_pose.x
            dy = oy - self.robot_pose.y

            angle = math.atan2(dy, dx) - self.current_yaw

            angle = math.atan2(math.sin(angle), math.cos(angle))

            if angle > 0:
                left += 1
            else:
                right += 1

        cmd = Twist()

        if left > right:
            cmd.angular.z = -0.7
        else:
            cmd.angular.z = 0.7

        return cmd

    # ======================================================
    # CONTROL LOOP
    # ======================================================

    def control_loop(self):

        if self.robot_pose is None or self.goal_pose is None:
            return

        gx = self.goal_pose.x
        gy = self.goal_pose.y

        rx = self.robot_pose.x
        ry = self.robot_pose.y

        goal_dist = math.hypot(gx-rx, gy-ry)

        # reached goal
        if goal_dist < 0.25:
            self.cmd_vel_pub.publish(Twist())
            self.goal_pose = None
            self.get_logger().info("Goal Reached")
            return

        # progress monitor
        if self.last_goal_dist is not None:

            if goal_dist > self.last_goal_dist - 0.01:
                self.no_progress_count += 1
            else:
                self.no_progress_count = 0

        self.last_goal_dist = goal_dist

        # stuck recovery
        if self.no_progress_count > self.stuck_threshold:
            self.get_logger().warn("Robot stuck -> Recovery")
            cmd = self.smart_recovery()
            self.cmd_vel_pub.publish(cmd)
            return

        # compute dwa
        bv, bw, paths = self.compute_dwa()

        cmd = Twist()

        if len(paths) == 0:
            self.get_logger().warn("No valid path -> Recovery")
            cmd = self.smart_recovery()
        else:
            cmd.linear.x = float(bv)
            cmd.angular.z = float(bw)

        self.cmd_vel_pub.publish(cmd)

        self.publish_paths(paths, bv, bw)

    # ======================================================
    # RVIZ VISUALIZATION
    # ======================================================

    def publish_paths(self, paths, bv, bw):

        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, p in enumerate(paths[::3]):

            m = Marker()

            m.header.frame_id = "world_enu"
            m.header.stamp = now
            m.ns = "dwa_paths"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD

            m.scale.x = 0.02

            best = (
                abs(p['v'] - bv) < 1e-3 and
                abs(p['w'] - bw) < 1e-3
            )

            if best:
                m.color = ColorRGBA(
                    r=0.0, g=1.0, b=0.0, a=1.0)
                m.scale.x = 0.05
            else:
                m.color = ColorRGBA(
                    r=1.0, g=1.0, b=0.0, a=0.15)

            for x, y, _ in p['traj']:
                pt = Point()
                pt.x = float(x)
                pt.y = float(y)
                pt.z = 0.03
                m.points.append(pt)

            arr.markers.append(m)

        self.marker_pub.publish(arr)


# ==========================================================
# MAIN
# ==========================================================

def main(args=None):

    rclpy.init(args=args)

    node = DWATurtlebot()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.cmd_vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
