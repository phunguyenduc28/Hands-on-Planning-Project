#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from tutorial_interfaces.srv import Pose

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')

        # Control Gains (Adjust these to make the robot faster/slower)
        self.k_v = 0.5  # Linear gain
        self.k_w = 2.0  # Angular gain
        self.dist_tolerance = 0.1 # Stop when within 10cm

        # State Variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.has_goal = False

        # Subscriptions
        self.create_subscription(Odometry, '/turtlebot/odom', self.odometry_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 10)

        # Publisher
        self.velocity_publisher_ = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10)

        # Service
        self.srv = self.create_service(Pose, '/goto', self.service_callback)

        # Timer: Runs the control logic at 10Hz
        self.timer = self.create_timer(0.1, self.control_loop)

    def odometry_callback(self, msg):
        # Correctly access orientation components
        o = msg.pose.pose.orientation
        _, _, self.current_theta = euler_from_quaternion(o.x, o.y, o.z, o.w)
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def goal_pose_callback(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.has_goal = True
        self.get_logger().info(f'Topic Goal: {self.goal_x}, {self.goal_y}')

    def service_callback(self, request, response):
        self.goal_x = request.x
        self.goal_y = request.y
        self.has_goal = True
        self.get_logger().info(f'Service Goal: {self.goal_x}, {self.goal_y}')
        # response.success = True # Uncomment if srv has a 'success' field
        return response

    def control_loop(self):
        if not self.has_goal:
            return

        # 1. Compute position error
        inc_x = self.goal_x - self.current_x
        inc_y = self.goal_y - self.current_y
        distance = math.sqrt(inc_x**2 + inc_y**2)

        # Stop if we reached the goal
        if distance < self.dist_tolerance:
            self.stop_robot()
            self.has_goal = False
            self.get_logger().info("Goal Reached!")
            return

        # Angular velocity (The fixed formula)
        target_angle = math.atan2(inc_y, inc_x)
        angle_error = target_angle - self.current_theta
        
        # Normalize angle to [-pi, pi] to avoid spinning the long way
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        cmd = Twist()
        cmd.angular.z = self.k_w * angle_error

        # Linear velocity logic from your screenshot
        # Only move forward if we are facing the right way (e.g., error < 0.2 rad)
        if abs(angle_error) < 0.2:
            cmd.linear.x = self.k_v * distance
        else:
            cmd.linear.x = 0.0

        self.velocity_publisher_.publish(cmd)

    def stop_robot(self):
        stop_msg = Twist()
        self.velocity_publisher_.publish(stop_msg)

def main():
    rclpy.init()
    node = MinimalService()
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