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
        self.robot_pose  = None
        self.current_yaw = 0.0
        self.goal_pose   = None
        self.current_vel = [0.0, 0.0]
        self.grid_map    = None

        # --------------------------------------------------
        self.max_speed     = 0.26   # TurtleBot3 max linear speed (m/s)
        self.max_yaw_rate  = 1.8    # TurtleBot3 max angular speed (rad/s)
        self.max_accel     = 0.8
        self.max_delta_yaw = 1.2
        self.dt            = 0.1
        self.predict_time  = 5.0

       
        # --------------------------------------------------
        self.heading_cost_weight   = 0.3
        self.dist_cost_weight      = 0.3
        self.obstacle_cost_weight  = 5.0    
        self.velocity_cost_weight  = 0.3
        self.clearance_cost_weight = 5.5     

        # --------------------------------------------------
        self._last_cmd_v = 0.0
        self._last_cmd_w = 0.0

        self.cmd_vel_pub = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10)
        self.marker_pub  = self.create_publisher(MarkerArray, '/dwa_trajectories', 10)

        self.arm_pub = self.create_publisher(
            Float64MultiArray,
            '/turtlebot/swiftpro/joint_velocity_controller/command',
            10
        )

        self.create_subscription(Odometry, '/turtlebot/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(OccupancyGrid, '/inflated_map', self.map_callback, 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("DWA with Visualization (Inflated Map Only) Started")

    # ======================================================
    # CALLBACKS
    # ======================================================

    def map_callback(self, msg):
        self.grid_map = msg

    def odom_callback(self, msg):
        self.robot_pose  = msg.pose.pose.position
        self.current_vel = [
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        ]

        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        )

    def goal_callback(self, msg):
        self.goal_pose = msg.pose.position

        arm = Float64MultiArray()
        arm.data = [0.0, 0.0, -1.0, 0.0]
        self.arm_pub.publish(arm)

    # ======================================================
    # GRID
    # ======================================================

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

    def is_collision(self, x, y):
        val = self._get_cell_value(x, y)
        if val is None:
            return True
        return val > 50

    # ======================================================
    # OBSTACLE COST 
    # ======================================================

    def obstacle_cost(self, traj):
        """
        Returns (penalty, min_clearance_value).
        penalty: 0 if completely safe, inf if collision, otherwise scaled.
        min_clearance_value: lowest cell cost seen (used for clearance term).
        """
        if self.grid_map is None:
            return 0.0, 0

        penalty = 0.0
        max_val_seen = 0  # how close did we get to an obstacle

        # The robot already occupies its starting cell — skip it so that
        # v=0 rotation trajectories are never rejected just because the
        # robot is inside an inflation zone.
        robot_gx, robot_gy = self._world_to_grid(
            self.robot_pose.x, self.robot_pose.y)

        for x, y, _ in traj:
            gx, gy = self._world_to_grid(x, y)
            if gx == robot_gx and gy == robot_gy:
                continue

            val = self._get_cell_value(x, y)

            if val is None:
                return float('inf'), 100
            if val >= 100:
                return float('inf'), 100

            # Scale soft costs much more aggressively than before.
            # In an inflated map, cells 1..50 mean "near obstacle".
            # Treat them with quadratic falloff so trajectories grazing
            # obstacles get punished significantly.
            # val=50 is unobserved space — with clear_on_max_range=True, any cell
            # the LIDAR ray passes through is explicitly cleared to 0, so val=50
            # means genuinely unseen (behind a wall).  Penalising it at 1.0/step
            # was blocking every trajectory that entered open-but-unscanned space.
            # Only penalise cells that are actually in the inflation soft zone (1-49).
            # if 0 < val < 50:
            #     normalized = val / 50.0
            #     penalty += normalized

            if val is not None and val > max_val_seen:
                max_val_seen = val

        # Average over trajectory length so longer paths aren't unfairly punished
        return penalty / max(1, len(traj)), max_val_seen

    # ======================================================
    # DWA
    # ======================================================

    def dynamic_window(self):
        # odom always reports vel=(0,0) in this simulator, so using current_vel
        # freezes the window at w=[-0.12,0.12] every tick regardless of what was
        # commanded.  Use the last issued command so the window evolves correctly.
        v = self._last_cmd_v
        w = self._last_cmd_w

        v_min = max(0.0, v - self.max_accel * self.dt)
        v_max = min(self.max_speed, v + self.max_accel * self.dt)

        w_min = max(-self.max_yaw_rate, w - self.max_delta_yaw * self.dt)
        w_max = min(self.max_yaw_rate, w + self.max_delta_yaw * self.dt)

        return v_min, v_max, w_min, w_max

    def simulate_trajectory(self, v, w):
        x, y, yaw = self.robot_pose.x, self.robot_pose.y, self.current_yaw

        traj = []
        steps = int(self.predict_time / self.dt)

        for _ in range(steps):
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += w * self.dt
            traj.append((x, y, yaw))

        return traj

    def compute_dwa(self):
        v_min, v_max, w_min, w_max = self.dynamic_window()

        init_dist = math.hypot(
            self.goal_pose.x - self.robot_pose.x,
            self.goal_pose.y - self.robot_pose.y
        )
        init_dist = max(init_dist, 0.1)

        best_v, best_w = 0.0, 0.0
        best_cost = float('inf')
        all_paths = []
        best_breakdown = None

        v_samples = list(np.arange(v_min, v_max + 0.01, 0.03))
        if 0.0 not in v_samples:
            v_samples.append(0.0)

        n_total    = 0
        n_rejected = 0
        per_v_valid    = {}
        per_v_rejected = {}

        for v in v_samples:
            vk = round(v, 3)
            per_v_valid[vk]    = 0
            per_v_rejected[vk] = 0

            for w in np.arange(w_min, w_max + 0.01, 0.03):
                n_total += 1
                traj = self.simulate_trajectory(v, w)
                obs_cost, max_val = self.obstacle_cost(traj)
                
                path_data = {'v': v, 'w': w, 'traj': traj, 'cost': float('inf'), 'is_rejected': False}
                if obs_cost == float('inf'):
                    n_rejected += 1
                    per_v_rejected[vk] += 1
                    # continue
                else:
                    per_v_valid[vk] += 1
                
                lx, ly, lyaw = traj[-1]

                goal_angle = math.atan2(
                    self.goal_pose.y - ly,
                    self.goal_pose.x - lx
                )
                heading_err = abs(math.atan2(
                    math.sin(goal_angle - lyaw),
                    math.cos(goal_angle - lyaw)
                ))
                heading_err_norm = heading_err / math.pi

                dist      = math.hypot(self.goal_pose.x - lx, self.goal_pose.y - ly)
                dist_norm = dist / init_dist

                # max_val is 0 (all free) or 50 (grazed unknown space).
                # val=100 already causes inf rejection so it never reaches here.
                # Normalise against 100 so val=50 → 0.5 penalty.
                clearance_cost = max_val / 100.0
                velocity_cost  = (self.max_speed - v) / self.max_speed

                cost = (
                    self.heading_cost_weight   * heading_err_norm +
                    self.dist_cost_weight      * dist_norm +
                    self.obstacle_cost_weight  * obs_cost +
                    self.clearance_cost_weight * clearance_cost +
                    self.velocity_cost_weight  * velocity_cost
                )
                path_data['cost'] = cost
                all_paths.append(path_data) # SAVE VALID PATH
                # all_paths.append({'v': v, 'w': w, 'traj': traj, 'cost': cost})

                if cost < best_cost:
                    best_cost = cost
                    best_v    = v
                    best_w    = w
                    best_breakdown = {
                        'v': v, 'w': w, 'total': cost,
                        'heading': self.heading_cost_weight  * heading_err_norm,
                        'dist':    self.dist_cost_weight     * dist_norm,
                        'obs':     self.obstacle_cost_weight * obs_cost,
                        'clear':   self.clearance_cost_weight * clearance_cost,
                        'vel':     self.velocity_cost_weight  * velocity_cost,
                    }

        # ---- logging ----
        self.get_logger().info(
            f"[DWA] window v=[{v_min:.2f},{v_max:.2f}] w=[{w_min:.2f},{w_max:.2f}]"
            f"  total={n_total} valid={n_total-n_rejected} rejected={n_rejected}"
            f"  dist={init_dist:.2f}m"
        )

        for vk in sorted(per_v_valid.keys()):
            status = "BLOCKED" if per_v_valid[vk] == 0 and vk > 0 else "ok"
            self.get_logger().info(
                f"[DWA]   v={vk:.3f}: valid={per_v_valid[vk]}"
                f"  rejected={per_v_rejected[vk]}  [{status}]"
            )
        
        if best_cost == float('inf'):
            self.get_logger().warn(
                f"[DWA] ALL {n_total} trajectories rejected — rotating to escape."
                f"  dist={init_dist:.2f}m  yaw={math.degrees(self.current_yaw):.1f}°"
            )
            
            # --- NEW LOGIC STARTS HERE ---
            # 1. Calculate the angle to the goal
            goal_angle = math.atan2(
                self.goal_pose.y - self.robot_pose.y,
                self.goal_pose.x - self.robot_pose.x
            )
            
            # 2. Find the shortest difference between current yaw and goal angle
            yaw_error = math.atan2(
                math.sin(goal_angle - self.current_yaw),
                math.cos(goal_angle - self.current_yaw)
            )
            
            # 3. Rotate in the direction of the goal
            best_v = 0.0
            if yaw_error > 0:
                best_w = self.max_yaw_rate * 0.6  # Rotate Left
            else:
                best_w = -self.max_yaw_rate * 0.6 # Rotate Right
            # --- NEW LOGIC ENDS HERE ---

            return best_v, best_w, all_paths, True   # forced=True
        else:
            bd = best_breakdown
            self.get_logger().info(
                f"[DWA] best v={bd['v']:.3f} w={bd['w']:.3f}  cost={bd['total']:.3f}"
                f"  [head={bd['heading']:.3f} dist={bd['dist']:.3f}"
                f"  obs={bd['obs']:.3f} clr={bd['clear']:.3f} vel={bd['vel']:.3f}]"
            )

            # all_fwd_blocked = all(
            #     per_v_valid.get(round(v, 3), 0) == 0
            #     for v in v_samples if v > 0.001
            # )
            # if all_fwd_blocked:
            #     # All forward paths blocked but v=0 survives.
            #     # Force a rotation to sweep the robot past the wall.
            #     # Use forced_rotation=True so the control loop does NOT feed
            #     # this w back into _last_cmd_w — that would lock the dynamic
            #     # window at high angular velocity and cause the robot to circle.
            #     best_v = 0.0
            #     best_w = self.max_yaw_rate * 0.5   # 0.9 rad/s
            #     self.get_logger().warn(
            #         f"[DWA] All forward trajectories blocked — forcing rotation"
            #         f" w={best_w:.2f} (window will NOT be updated to avoid circling)."
            #         f"  dist={init_dist:.2f}m  yaw={math.degrees(self.current_yaw):.1f}°"
            #     )
            #     return best_v, best_w, all_paths, True   # forced=True

        return best_v, best_w, all_paths, False

    # ======================================================
    # VISUALIZATION
    # ======================================================

    def publish_paths(self, paths, bv, bw):
        marker_array = MarkerArray()
        now   = self.get_clock().now().to_msg()
        frame = self.grid_map.header.frame_id

        clear = Marker()
        clear.action          = Marker.DELETEALL
        clear.header.frame_id = frame
        clear.header.stamp    = now
        marker_array.markers.append(clear)

        best_path   = None
        other_paths = []
        for p in paths:
            if abs(p['v'] - bv) < 1e-3 and abs(p['w'] - bw) < 1e-3:
                best_path = p
            else:
                other_paths.append(p)

        # All candidate trajectories — solid purple, thin thread
        for i, p in enumerate(other_paths[::3]):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp    = now
            m.ns              = "dwa_all"
            m.id              = i + 1
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.01
            m.color           = ColorRGBA(r=0.6, g=0.0, b=1.0, a=1.0)
            for x, y, _ in p['traj']:
                pt = Point(); pt.x = float(x); pt.y = float(y); pt.z = 0.05
                m.points.append(pt)
            marker_array.markers.append(m)

        # Chosen trajectory — solid green, always drawn outside subsampling
        if best_path is not None:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp    = now
            m.ns              = "dwa_best"
            m.id              = 9999
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.04
            m.color           = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            for x, y, _ in best_path['traj']:
                pt = Point(); pt.x = float(x); pt.y = float(y); pt.z = 0.05
                m.points.append(pt)
            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

    # ======================================================
    # CONTROL LOOP
    # ======================================================

    def control_loop(self):
        if self.robot_pose is None or self.goal_pose is None or self.grid_map is None:
            return

        dist = math.hypot(
            self.goal_pose.x - self.robot_pose.x,
            self.goal_pose.y - self.robot_pose.y
        )

        if dist < 0.25:
            self.cmd_vel_pub.publish(Twist())
            self.goal_pose = None
            return

        v, w, paths, forced = self.compute_dwa()

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)

        self.cmd_vel_pub.publish(cmd)

        # Only update the window tracker from normal DWA choices.
        # When the rotation is forced (wall escape), feeding w back in
        # would lock the window at high angular velocity and cause the
        # robot to circle instead of escaping cleanly.
        if not forced:
            self._last_cmd_v = v
            self._last_cmd_w = w
        self.publish_paths(paths, v, w)


# ======================================================
# MAIN
# ======================================================

def main():
    rclpy.init()
    node = DWATurtlebot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()