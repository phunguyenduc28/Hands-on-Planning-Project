import rclpy
from rclpy.node import Node
import numpy as np
import math
import matplotlib.pyplot as plt

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
        self.max_speed     = 0.35
        self.max_yaw_rate  = 2.5
        self.max_accel     = 0.5
        self.max_delta_yaw = 1.2
        self.dt            = 0.1
        self.predict_time  = 2.5

        # --------------------------------------------------
        self.heading_cost_weight  = 3.0
        self.dist_cost_weight     = 6.0
        self.obstacle_cost_weight = 20.0
        self.velocity_cost_weight = 0.5

        # --------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10)
        self.marker_pub  = self.create_publisher(MarkerArray, '/dwa_trajectories', 10)

        self.arm_pub = self.create_publisher(
            Float64MultiArray,
            '/turtlebot/swiftpro/joint_velocity_controller/command',
            10
        )

        self.create_subscription(Odometry, '/turtlebot/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(OccupancyGrid, '/inflated_map_dwa', self.map_callback, 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("DWA with Visualization (Inflated Map Only) Started")

        self.dwa_viz_count = 0

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
            grid_data = np.array(self.grid_map.data).reshape((info.height, info.width))
            return grid_data[gx, gy]
            # return self.grid_map.data[gy * info.width + gx]
        return None

    def is_collision(self, x, y):
        val = self._get_cell_value(x, y)
        if val is None:
            return True
        return val > 50

    # ======================================================
    # OBSTACLE COST (clean)
    # ======================================================

    def obstacle_cost(self, traj, sampled_points_for_viz):
        if self.grid_map is None:
            return 0.0

        penalty = 0.0
        sampled = traj[::3]

        for x, y, _ in sampled:
            sampled_points_for_viz.append((x, y)) # Add point for visualization

            val = self._get_cell_value(x, y)

            if val is None:
                return float('inf')
            if val > 50:
                return float('inf')

            if val == -1:
                penalty += 0.2
            elif val > 30:
                penalty += 0.5
            elif val > 10:
                penalty += 0.2

        return penalty / max(1, len(sampled))

    # ======================================================
    # DWA
    # ======================================================

    def dynamic_window(self):
        v, w = self.current_vel

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

        best_v, best_w = 0.0, 0.0
        best_cost = float('inf')
        all_paths = []
        
        # List to store all sampled points from all trajectories for visualization
        all_sampled_points_for_viz = []

        i = 0

        # info = self.grid_map.info
        # grid_data = np.array(self.grid_map.data).reshape((info.height, info.width))
        # self.get_logger().info(f"grid data {grid_data}")

        for v in np.arange(v_min, v_max + 0.01, 0.03):
            for w in np.arange(w_min, w_max + 0.01, 0.06):

                i += 1
                sampled_points_for_current_traj = []
                traj = self.simulate_trajectory(v, w)
                obs_cost = self.obstacle_cost(traj, sampled_points_for_current_traj)
                all_sampled_points_for_viz.extend(sampled_points_for_current_traj)


                if obs_cost == float('inf'):
                    continue

                lx, ly, lyaw = traj[-1]

                goal_angle = math.atan2(
                    self.goal_pose.y - ly,
                    self.goal_pose.x - lx
                )

                heading_err = abs(math.atan2(
                    math.sin(goal_angle - lyaw),
                    math.cos(goal_angle - lyaw)
                ))

                dist = math.hypot(
                    self.goal_pose.x - lx,
                    self.goal_pose.y - ly
                )

                cost = (
                    self.heading_cost_weight * heading_err +
                    self.dist_cost_weight * dist +
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
                
                self.get_logger().info(f"{(v, w)} cost {cost} best cost {cost} heading {heading_err} dist {dist} obstacle {obs_cost}")
        
        # After computing DWA for all paths, visualize
        self.visualize_dwa_results(all_paths, best_v, best_w, all_sampled_points_for_viz)

        return best_v, best_w, all_paths

    # ======================================================
    # VISUALIZATION (FIXED)
    # ======================================================

    def publish_paths(self, paths, bv, bw):
        marker_array = MarkerArray()

        for i, p in enumerate(paths[::5]):

            m = Marker()
            m.header.frame_id = self.grid_map.header.frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "dwa"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.02

            is_best = abs(p['v'] - bv) < 1e-3 and abs(p['w'] - bw) < 1e-3

            if is_best:
                m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            else:
                m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.2)

            for x, y, _ in p['traj']:
                pt = Point()
                pt.x = float(x)
                pt.y = float(y)
                pt.z = 0.05
                m.points.append(pt)

            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

    def visualize_dwa_results(self, paths, best_v, best_w, sampled_points):
        if self.grid_map is None or self.robot_pose is None or self.goal_pose is None:
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the Occupancy Grid
        info = self.grid_map.info
        grid_data = np.array(self.grid_map.data).reshape((info.height, info.width))
        
        # Rotate and flip the grid data to match typical ROS visualization (origin bottom-left)
        # grid_data = np.flipud(grid_data) # Flip vertically
        
        # Replace -1 (unknown) with a specific value for visualization, e.g., 50
        grid_data[grid_data == -1] = 50

        ax.imshow(
            grid_data,
            cmap='gray',
            origin='lower', # Set origin to lower-left for correct orientation
            extent=[
                info.origin.position.x, info.origin.position.x + info.width * info.resolution,
                info.origin.position.y, info.origin.position.y + info.height * info.resolution
            ]
        )

        # Plot robot's current position
        ax.plot(self.robot_pose.x, self.robot_pose.y, 'ro', markersize=10, label='Robot Start')
        ax.arrow(
            self.robot_pose.x, self.robot_pose.y,
            0.2 * math.cos(self.current_yaw), 0.2 * math.sin(self.current_yaw),
            head_width=0.1, head_length=0.1, fc='r', ec='r'
        )

        # Plot goal position
        ax.plot(self.goal_pose.x, self.goal_pose.y, 'g*', markersize=15, label='Goal')

        # Plot all sampled points from obstacle cost calculation
        if sampled_points:
            sample_xs = [p[0] for p in sampled_points]
            sample_ys = [p[1] for p in sampled_points]
            ax.plot(sample_xs, sample_ys, 'cx', markersize=3, alpha=0.5, label='Sampled Points')


        # Plot all trajectories
        for p in paths:
            traj_x = [pt[0] for pt in p['traj']]
            traj_y = [pt[1] for pt in p['traj']]

            is_best = abs(p['v'] - best_v) < 1e-3 and abs(p['w'] - best_w) < 1e-3
            if is_best:
                ax.plot(traj_x, traj_y, 'g-', linewidth=2, alpha=0.8, label='Best Trajectory' if 'Best Trajectory' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                ax.plot(traj_x, traj_y, 'y--', linewidth=0.8, alpha=0.3)
        
        self.dwa_viz_count += 1
        if self.dwa_viz_count % 10 == 0:
            ax.set_title("DWA Trajectories and Obstacle Cost Samples")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.legend()
            ax.set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.savefig(f"./dwa_viz/dwa_visualization_{self.dwa_viz_count}.png")
            plt.close(fig)
            self.get_logger().info("DWA visualization saved to dwa_visualization.png")

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

        v, w, paths = self.compute_dwa()

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)

        self.cmd_vel_pub.publish(cmd)

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