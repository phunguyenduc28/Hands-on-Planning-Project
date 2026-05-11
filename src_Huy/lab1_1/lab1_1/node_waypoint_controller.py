#!/usr/bin/env python3
"""
Waypoint-based Controller Node
Receives waypoint lists from path planner and executes them sequentially.
"""
import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def euler_from_quaternion(x, y, z, w):
    """Convert quaternion to yaw angle."""
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


class WaypointController(Node):
    """Controller for executing waypoint sequences."""
    
    def __init__(self):
        super().__init__('waypoint_controller')
        
        # Control Gains
        self.declare_parameter('k_v', 0.5)  # Linear velocity gain
        self.declare_parameter('k_w', 2.0)  # Angular velocity gain
        self.declare_parameter('dist_tolerance', 0.1)  # Position tolerance (meters)
        self.declare_parameter('angle_tolerance', 0.1)  # Angle tolerance (radians)
        self.declare_parameter('v_max', 0.5)  # Max linear velocity
        self.declare_parameter('w_max', 1.0)  # Max angular velocity
        
        # Getting values for parameters from launch files
        self.k_v = self.get_parameter('k_v').value
        self.k_w = self.get_parameter('k_w').value
        self.dist_tolerance = self.get_parameter('dist_tolerance').value
        self.angle_tolerance = self.get_parameter('angle_tolerance').value
        self.v_max = self.get_parameter('v_max').value
        self.w_max = self.get_parameter('w_max').value
        
        # State Variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.odometry_received = False  # Flag to track if odometry has been received
        self.last_logged_pose = None  # Track last logged pose to detect movement
        
        # Waypoint tracking
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.executing_path = False # flag to indicate if currently executing a path
        
        # Subscriptions
        self.create_subscription(Odometry, '/turtlebot/odom', self.odometry_callback, 10)
        self.create_subscription(Path, '/plan', self.plan_callback, 10)
        
        # Publisher for velocity commands
        self.velocity_publisher = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10)
        
        # Publisher for status messages for debugging purposes
        self.status_pub = self.create_publisher(String, '/controller_status', 10)
        
        # Control loop timer at 10Hz
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self._debug_counter = 0  # Counter for controlling debug log frequency
        self.get_logger().info("Waypoint Controller initialized")
    
    def odometry_callback(self, msg):
        """Update robot pose from odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        o = msg.pose.pose.orientation
        # no need to invert the current_theta since I am using my localaisation node which already gives me the correct orientation in the world_enu frame
        self.current_theta = euler_from_quaternion(o.x, o.y, o.z, o.w)
        
        # Set flag on first odometry message
        if not self.odometry_received:
            self.odometry_received = True
            self.last_logged_pose = (self.current_x, self.current_y)
            # to avoud spamming logs, only log the first odometry message and then log subsequent messages only when the robot has moved significantly (e.g., 0.1m)
            self.get_logger().info(f"[ODOM] First odometry received at: ({self.current_x:.3f}, {self.current_y:.3f})")
        
        # Log only when robot moves significantly (0.1m threshold)
        if self.last_logged_pose is not None:
            dist_moved = math.sqrt((self.current_x - self.last_logged_pose[0])**2 + 
                                 (self.current_y - self.last_logged_pose[1])**2)
            if dist_moved > 0.1:  # Only log when moved > 0.1m
                self.get_logger().info(f"[ODOM] Robot moved to: ({self.current_x:.3f}, {self.current_y:.3f}, theta={self.current_theta:.3f})")
                self.last_logged_pose = (self.current_x, self.current_y)
    
    def plan_callback(self, msg):
        """Receive planned path (nav_msgs::Path) and convert to waypoints."""
        # Initialize waypoints list from the received path message
        self.waypoints = []
        for pose in msg.poses:
            self.waypoints.append((pose.pose.position.x, pose.pose.position.y))
        
        self.current_waypoint_idx = 0
        # Set executing_path to True when a new path is received, so that control loop starts executing the waypoints
        self.executing_path = True
        self.has_single_goal = False
        
        self.get_logger().info(f"Path received with {len(self.waypoints)} waypoints (frame: {msg.header.frame_id})")
        for i, (x, y) in enumerate(self.waypoints):
            # The following was just for debugging purposes to log the waypoints received from the planner. You can comment it out if you find it too verbose.
            self.get_logger().info(f"  Waypoint {i}: ({x:.3f}, {y:.3f})")
        self.publish_status(f"Executing path with {len(self.waypoints)} waypoints")
    
    def control_loop(self):
        """Main control loop executing waypoint sequence from planner."""
        # Wait for first odometry message
        if not self.odometry_received:
            self.stop_robot()
            return
        
        if self.executing_path:
            self.execute_waypoint_sequence()
        else:
            self.stop_robot()
    
    
    def execute_waypoint_sequence(self):
        """Execute a sequence of waypoints."""
        # same controller as in first part of lab, now instead of single goal pose, we are using the current waypoint from the list of waypoints received from the planner
        if self.current_waypoint_idx >= len(self.waypoints):
            # All waypoints completed
            self.stop_robot()
            self.executing_path = False
            self.get_logger().info("All waypoints completed!")
            self.publish_status("All waypoints completed!")
            return
        
        # Get current waypoint
        target = self.waypoints[self.current_waypoint_idx]
        inc_x = target[0] - self.current_x
        inc_y = target[1] - self.current_y
        distance = math.sqrt(inc_x**2 + inc_y**2)
        
        # Debug: Log state every 10 iterations
        # value is saved in _debug_counter attribute of the class, which is initialized to 0 and incremented every time this control loop runs.
        static_counter = getattr(self, '_debug_counter', 0)
        if static_counter % 10 == 0:
            self.get_logger().info(
                f"[WP {self.current_waypoint_idx}/{len(self.waypoints)}] "
                f"Robot: ({self.current_x:.3f}, {self.current_y:.3f}) -> "
                f"Target: ({target[0]:.3f}, {target[1]:.3f}), distance={distance:.3f}m"
            )
        self._debug_counter = static_counter + 1
        
        # Check if reached waypoint
        if distance < self.dist_tolerance:
            self.current_waypoint_idx += 1
            wp_num = self.current_waypoint_idx
            total_wp = len(self.waypoints)
            self.get_logger().info(f"Waypoint {wp_num}/{total_wp} reached")
            self.publish_status(f"Waypoint {wp_num}/{total_wp} reached")
            return
        
        # Compute desired angle
        target_angle = math.atan2(inc_y, inc_x)
        angle_error = target_angle - self.current_theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        # Compute control commands
        cmd = Twist()
        # clamping used to limit the maximum velocity commands to prevent the robot from trying to move too fast, 
        # which can cause instability or overshooting, especially when far from the waypoint. 
        cmd.angular.z = self.clamp(self.k_w * angle_error, -self.w_max, self.w_max)
        
        # Move linearly only if aligned
        if abs(angle_error) < 0.2:
            cmd.linear.x = self.clamp(self.k_v * distance, -self.v_max, self.v_max)
        else:
            cmd.linear.x = 0.0
        
        self.velocity_publisher.publish(cmd)
    
    def stop_robot(self):
        """Send zero velocity command."""
        stop_msg = Twist() # All fields default to zero
        self.velocity_publisher.publish(stop_msg)
    
    def clamp(self, value, min_val, max_val):
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))
    
    def publish_status(self, message):
        """Publish status message."""
        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)


def main():
    rclpy.init()
    node = WaypointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
