import rclpy
from rclpy.node import Node
import numpy as np
import math

from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry


class ControlTurtlebot(Node):

    def __init__(self):
        super().__init__('control_turtlebot')
        self.robot_pose = None
        self.goal_pose = None
        self.acceptance_radius = 0.1
        self.max_linear_velocity = 1.0
        self.kv = 2.0
        self.kw = 2.0
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10) # publishe cmd_vel
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/turtlebot/odom',
            self.odom_callback,
            10
        )
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10)

        self.timer = self.create_timer(0.1, self.control_loop)

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose.position
        # Convert Quaternion to Yaw (Z-axis rotation)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
    def goal_pose_callback(self, msg):
        self.goal_pose = msg.pose.position
        self.get_logger().info(f"Recive goal pose {self.goal_pose.x, self.goal_pose.y}")
        
    def control_loop(self):
        if self.robot_pose is None or self.goal_pose is None:
            return

        cmd_vel = Twist()
        inc_x = self.goal_pose.x - self.robot_pose.x
        inc_y = self.goal_pose.y - self.robot_pose.y
        dist = np.sqrt((inc_x)**2 + (inc_y)**2)

        if dist < self.acceptance_radius:
            self.get_logger().info(f"Reach goal!")
            self.goal_pose = None
            self.cmd_vel_pub.publish(cmd_vel)
            return

        w = self.kw * (math.atan2(inc_y, inc_x) - self.current_yaw)
        if abs(w) <= 0.5:
            v = min(self.kv * np.sqrt(inc_x**2 + inc_y**2), self.max_linear_velocity) # clipped the maximum linear velocity
        else:
            # prioritize turning to align the heading
            v = 0.0
        
        cmd_vel.linear.x = v
        cmd_vel.angular.z = w
        
        self.get_logger().info(f"Linear {v}, Angular  {w}")
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)

    control_turtlebot = ControlTurtlebot()

    rclpy.spin(control_turtlebot)

    control_turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
