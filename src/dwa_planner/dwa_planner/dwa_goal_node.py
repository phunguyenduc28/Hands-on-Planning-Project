#!/usr/bin/env python3
"""
dwa_goal_node.py

Standalone DWA test driver.

Subscribes to /goal_pose (geometry_msgs/PoseStamped) — click "2D Goal Pose"
in RViz2 to set a target.  The node calls /dwa/compute_velocity at 10 Hz and
forwards the result to /turtlebot/cmd_vel until the robot is within
acceptance_radius of the goal.

Compatible with the existing dwa_service and dwa_local_costmap nodes; the
path_planner_tb is NOT required.

Parameters
----------
acceptance_radius     (float, default 0.15 m)
map_frame             (str, default 'odom')   — frame for status logging

Subscribers
-----------
/goal_pose            geometry_msgs/PoseStamped   — from RViz2 2D Goal Pose tool
/turtlebot/odom       nav_msgs/Odometry

Publishers
----------
/turtlebot/cmd_vel    geometry_msgs/Twist

Service client
--------------
/dwa/compute_velocity dwa_interfaces/srv/ComputeVelocity
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry

from dwa_interfaces.srv import ComputeVelocity


class DWAGoalNode(Node):

    def __init__(self):
        super().__init__('dwa_goal_node')

        self.declare_parameter('acceptance_radius', 0.15)
        self.acceptance_radius = self.get_parameter('acceptance_radius').value

        self.declare_parameter('map_frame', 'odom')
        self.map_frame = self.get_parameter('map_frame').value

        self.goal = None          # (goal_x, goal_y)
        self.robot_pose = None
        self._dwa_future = None

        self.cmd_vel_pub = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10)

        self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_cb, 10)
        self.create_subscription(
            Odometry, '/turtlebot/odom', self._odom_cb, 10)

        self._dwa_client = self.create_client(
            ComputeVelocity, '/dwa/compute_velocity')

        self.create_timer(0.1, self._control_loop)   # 10 Hz

        self.get_logger().info(
            f'DWA goal node ready — publish a /goal_pose to start driving '
            f'(acceptance_radius={self.acceptance_radius} m)')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _goal_cb(self, msg: PoseStamped):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self._dwa_future = None
        self.get_logger().info(
            f'New goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f}) '
            f'[frame={msg.header.frame_id}]')

    def _odom_cb(self, msg: Odometry):
        self.robot_pose = msg.pose.pose.position

    # ── Control loop (10 Hz) ─────────────────────────────────────────────────

    def _control_loop(self):
        if self.goal is None or self.robot_pose is None:
            return

        goal_x, goal_y = self.goal
        dist = math.hypot(goal_x - self.robot_pose.x,
                          goal_y - self.robot_pose.y)

        # Goal reached — stop and clear
        if dist < self.acceptance_radius:
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info(
                f'Goal reached ({goal_x:.2f}, {goal_y:.2f}) — stopped')
            self.goal = None
            self._dwa_future = None
            return

        if not self._dwa_client.service_is_ready():
            self.get_logger().warn(
                'Waiting for /dwa/compute_velocity service…', throttle_duration_sec=2.0)
            return

        # Apply the previous async response if it has arrived
        if self._dwa_future is not None and self._dwa_future.done():
            try:
                resp = self._dwa_future.result()
                if resp.success:
                    cmd = Twist()
                    cmd.linear.x = resp.linear_x
                    cmd.angular.z = resp.angular_z
                    self.cmd_vel_pub.publish(cmd)
            except Exception as e:
                self.get_logger().error(f'DWA service call failed: {e}')
            self._dwa_future = None

        # Issue a new request if none is in-flight
        if self._dwa_future is None:
            req = ComputeVelocity.Request()
            req.goal_x = float(goal_x)
            req.goal_y = float(goal_y)
            self._dwa_future = self._dwa_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = DWAGoalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_vel_pub.publish(Twist())  # stop on exit
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
