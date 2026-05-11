import rclpy
from rclpy.node import Node
import numpy as np
import math

# ROS 2 Standard message imports
from geometry_msgs.msg import PoseStamped, Twist, Point
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class DWATurtlebot(Node):
    """
    DWA (Dynamic Window Approach) Local Planner Node.
    This node simulates multiple possible velocity trajectories to find the 
    optimal path that avoids obstacles while moving toward a goal.
    """

    def __init__(self):
        super().__init__('dwa_turtlebot_controller')
        
        # --- Robot State Variables ---
        self.robot_pose = None      # Current position (x, y)
        self.current_yaw = 0.0      # Orientation in radians
        self.goal_pose = None       # Targeted navigation goal
        self.current_vel = [0.0, 0.0]  # Current [linear_v, angular_w]
        self.scan_data = None       # Latest raw Lidar sensor data

        # --- DWA Kinematic Constraints ---
        # These define the physical limits of the robot's movement
        self.max_speed = 0.35            # Max linear velocity (m/s)
        self.max_yaw_rate = 1.8          # Max angular velocity (rad/s)
        self.max_accel = 0.8             # Max acceleration
        self.dt = 0.1                    # Control loop period (s)
        self.predict_time = 3.5          # Prediction horizon (seconds to look ahead)
        
        # --- Cost Function Tuning Weights ---
        # Adjust these weights to change the robot's driving behavior
        self.heading_cost_weight = 10.0   # Weight for aligning with goal direction
        self.velocity_cost_weight = 0.1   # Weight for maintaining high speed
        self.obstacle_cost_weight = 120.0 # High weight for safety/obstacle avoidance
        self.robot_radius = 0.42          # Physical footprint (collision zone)
        self.safety_margin = 0.9          # Inflation zone (starts avoiding objects here)

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/dwa_trajectories', 10)
        self.joint_arm_pub = self.create_publisher(Float64MultiArray, '/turtlebot/swiftpro/joint_velocity_controller/command', 10)
        
        # --- Subscribers ---
        self.odom_sub = self.create_subscription(Odometry, '/turtlebot/odom', self.odom_callback, 10)
        self.goal_pose_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Main timer to trigger the control loop every 100ms (0.1s)
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("DWA Controller: System Online. Documentation: Detailed English.")

    def odom_callback(self, msg):
        """Update robot state from Odometry data."""
        self.robot_pose = msg.pose.pose.position
        self.current_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        
        # Convert Quaternion orientation to Euler Yaw
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 
                                      1 - 2 * (q.y * q.y + q.z * q.z))

    def scan_callback(self, msg):
        """Buffer incoming Lidar scans."""
        self.scan_data = msg

    def goal_pose_callback(self, msg):
        """Update the navigation target from Rviz2 or external planner."""
        self.goal_pose = msg.pose.position 
        joint_command = Float64MultiArray()
        joint_command.data = [0.0, 0.0, -1.0, 0.0]  # Example: Move all joints at 0.5 rad/s
        self.joint_arm_pub.publish(joint_command)   
        self.get_logger().info(f"Target Updated: ({self.goal_pose.x:.2f}, {self.goal_pose.y:.2f})")

    def compute_dwa(self):
        """
        Core DWA Logic:
        1. Sample the velocity space (v, w).
        2. Predict future trajectories for each sample.
        3. Score trajectories based on goal alignment, speed, and obstacle distance.
        """
        # Define the search space [min_v, max_v, min_w, max_w]
        dw = [0.0, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        best_v, best_w = 0.0, 0.0
        min_cost = float('inf')
        valid_paths = []

        # Iterate through linear and angular velocity samples
        for v in np.arange(dw[0], dw[1] + 0.01, 0.05):
            for w in np.arange(dw[2], dw[3] + 0.01, 0.05):
                
                # Predict the future path (Trajectory Simulation)
                traj = []
                x, y, yaw = self.robot_pose.x, self.robot_pose.y, self.current_yaw
                sim_steps = 15 
                step_size = self.predict_time / sim_steps

                for _ in range(sim_steps):
                    x += v * math.cos(yaw) * step_size
                    y += v * math.sin(yaw) * step_size
                    yaw += w * step_size
                    traj.append((x, y, yaw))

                # Evaluate collision risk
                obs_cost = self.calc_obstacle_cost(traj)
                if obs_cost == float('inf'):
                    continue # Discard trajectories that lead to a crash

                # End-point analysis for goal alignment
                lx, ly, lyaw = traj[-1]
                goal_angle = math.atan2(self.goal_pose.y - ly, self.goal_pose.x - lx)
                heading_err = abs(math.atan2(math.sin(goal_angle - lyaw), math.cos(goal_angle - lyaw)))
                dist_to_goal = math.sqrt((self.goal_pose.x - lx)**2 + (self.goal_pose.y - ly)**2)
                
                # Aggregate Cost Calculation (Objective Function)
                cost = (self.heading_cost_weight * heading_err) + \
                       (4.0 * dist_to_goal) + \
                       (self.obstacle_cost_weight * obs_cost) + \
                       (self.velocity_cost_weight * (self.max_speed - v))
                
                valid_paths.append({'v': v, 'w': w, 'path': traj, 'cost': cost})
                
                if cost < min_cost:
                    min_cost, best_v, best_w = cost, v, w

        return best_v, best_w, valid_paths

    def calc_obstacle_cost(self, trajectory):
        """Calculate penalty based on proximity to Lidar-detected obstacles."""
        if not self.scan_data:
            return 0.0
        
        # Convert raw Lidar ranges to Cartesian points in the World Frame
        obs_points = []
        for i, r in enumerate(self.scan_data.ranges):
            if self.scan_data.range_min < r < 3.5: # Ignore points beyond 3.5m to save CPU
                angle = self.scan_data.angle_min + i * self.scan_data.angle_increment + self.current_yaw
                obs_points.append((self.robot_pose.x + r * math.cos(angle), 
                                   self.robot_pose.y + r * math.sin(angle)))

        min_dist_global = float('inf')

        # Check every simulated point on the path against every obstacle point
        for tx, ty, _ in trajectory:
            for ox, oy in obs_points:
                dist = math.sqrt((tx - ox)**2 + (ty - oy)**2)
                
                # Immediate return if within physical collision radius
                if dist < self.robot_radius:
                    return float('inf') 
                
                if dist < min_dist_global:
                    min_dist_global = dist

        # Apply a quadratic penalty for entering the Safety Inflation Zone
        if min_dist_global < self.safety_margin:
            return (self.safety_margin - min_dist_global) ** 2 * 200.0
        
        return 0.0

    def control_loop(self):
        """Executive loop: Determines the command and publishes to /cmd_vel."""
        if not self.robot_pose or not self.goal_pose:
            return
        
        # Calculate Euclidean distance to goal
        dist = math.sqrt((self.goal_pose.x - self.robot_pose.x)**2 + 
                         (self.goal_pose.y - self.robot_pose.y)**2)
        
        # Stop condition: Goal reached
        if dist < 0.25:
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info("Goal Reached.")
            self.goal_pose = None
            return

        # Execute DWA to find best velocity
        bv, bw, paths = self.compute_dwa()
        cmd = Twist()
        
        # Emergency Check: Ensure the chosen 'best' path doesn't crash in the first few steps
        is_immediate_collision = False
        if bv > 0 and paths:
            best_path = next((p['path'] for p in paths if p['v'] == bv and p['w'] == bw), [])
            for i in range(min(len(best_path), 5)): # Check first 5 steps (approx 0.5s)
                tx, ty, _ = best_path[i]
                if self.check_immediate_collision(tx, ty):
                    is_immediate_collision = True
                    break

        # Fallback: If trapped or immediate danger, stop linear motion and spin to find an opening
        if not paths or is_immediate_collision:
            self.get_logger().warn("High collision risk! Recovery spin initiated.")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.6 
        else:
            cmd.linear.x = float(bv)
            cmd.angular.z = float(bw)
        
        self.cmd_vel_pub.publish(cmd)
        self.publish_trajectories(paths, bv, bw)

    def check_immediate_collision(self, x, y):
        """Utility to check if a specific world coordinate intersects an obstacle."""
        if not self.scan_data:
            return False
        for i, r in enumerate(self.scan_data.ranges):
            if self.scan_data.range_min < r < 2.0:
                angle = self.scan_data.angle_min + i * self.scan_data.angle_increment + self.current_yaw
                ox = self.robot_pose.x + r * math.cos(angle)
                oy = self.robot_pose.y + r * math.sin(angle)
                if math.sqrt((x - ox)**2 + (y - oy)**2) < self.robot_radius:
                    return True
        return False

    def publish_trajectories(self, paths, bv, bw):
        """Visualizer: Render simulated paths in RViz for debugging."""
        msg = MarkerArray()
        now = self.get_clock().now().to_msg()
        # Downsample to every 3rd path to reduce visualization overhead
        for i, p in enumerate(paths[::3]): 
            m = Marker()
            m.header.frame_id, m.header.stamp, m.id = "world_enu", now, i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            
            # The chosen trajectory is Green; candidate trajectories are faint Yellow
            is_best = (abs(p['v'] - bv) < 0.001 and abs(p['w'] - bw) < 0.001)
            if is_best:
                m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
                m.scale.x = 0.05
            else:
                m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.1)
                m.scale.x = 0.01

            for tx, ty, _ in p['path']:
                m.points.append(Point(x=float(tx), y=float(ty), z=0.05))
            msg.markers.append(m)
        self.marker_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DWATurtlebot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown Safety: Ensure robot stops before exiting
        node.cmd_vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()