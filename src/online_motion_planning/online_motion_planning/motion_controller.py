"""Motion controller for base and arm control."""

import math
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


class MotionController:
    """Manages robot motion control including arm retraction and base movement."""

    def __init__(self, max_linear_velocity, max_angular_velocity, kv, kw,
                 acceptance_radius, scan_distance_threshold, arm_q2_retract,
                 arm_q3_retract, arm_retract_tol, arm_kp, arm_max_vel, dwa_viz_time, logger):
        """Initialize motion controller.
        
        Args:
            max_linear_velocity: Max forward speed (m/s)
            max_angular_velocity: Max angular speed (rad/s)
            kv: Linear velocity gain
            kw: Angular velocity gain
            acceptance_radius: Distance to consider waypoint reached (m)
            scan_distance_threshold: Distance to trigger new scan (m)
            arm_q2_retract: Target arm joint 2 position (rad)
            arm_q3_retract: Target arm joint 3 position (rad)
            arm_retract_tol: Tolerance for retraction (rad)
            arm_kp: Arm proportional gain
            arm_max_vel: Max arm velocity (rad/s)
            dwa_viz_time: DWA visualization time horizon (s)
            logger: ROS logger instance
        """
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.kv = kv
        self.kw = kw
        self.acceptance_radius = acceptance_radius
        self.scan_distance_threshold = scan_distance_threshold
        self.arm_q2_retract = arm_q2_retract
        self.arm_q3_retract = arm_q3_retract
        self.arm_retract_tol = arm_retract_tol
        self.arm_kp = arm_kp
        self.arm_max_vel = arm_max_vel
        self.dwa_viz_time = dwa_viz_time
        self.logger = logger
        
        # Joint state
        self.arm_q2 = None
        self.arm_q3 = None
        
        # References set externally
        self.robot_pose = None
        self.current_yaw = 0.0
        self.current_vel = (0.0, 0.0)
        self.cmd_vel_pub = None
        self.arm_cmd_pub = None
        self.marker_pub = None
        self.now = None
        
        # State variables
        self.waypoints = None
        self.goal_pose = None
        self.complete_a_path = True
        self.following_last_path = False
        self.collide_robot_next_waypoint = False
        self.rotation_state = 'idle'
        self.last_scan_pos = None
        self.prev_yaw_for_spin = None
        self.spin_accumulated = 0.0
        self.frontiers_explored_count = 0
        self.visited_frontier_positions = []
        self.find_frontier = True
        self.local_search_radius = 20
        self.max_local_search_radius = 50
        self.frontier_expand_every = 3
        self.frontier_expand_step = 15
        self.binary_map_frame = "map"

    def update_joint_state(self, joint_name, position):
        """Update arm joint state."""
        if joint_name == 'turtlebot/swiftpro/joint2':
            self.arm_q2 = position
        elif joint_name == 'turtlebot/swiftpro/joint3':
            self.arm_q3 = position

    def _step_arm_to_retract(self) -> bool:
        """Retract arm towards safe position."""
        if self.arm_q2 is None or self.arm_q3 is None:
            return False

        err2 = self.arm_q2_retract - self.arm_q2
        err3 = self.arm_q3_retract - self.arm_q3

        if abs(err2) < self.arm_retract_tol and abs(err3) < self.arm_retract_tol:
            if self.arm_cmd_pub:
                cmd = Float64MultiArray()
                cmd.data = [0.0, 0.0, 0.0, 0.0]
                self.arm_cmd_pub.publish(cmd)
            return True

        dq2 = float(np.clip(self.arm_kp * err2, -self.arm_max_vel, self.arm_max_vel))
        dq3 = float(np.clip(self.arm_kp * err3, -self.arm_max_vel, self.arm_max_vel))

        if self.arm_q2 >= 0.045 and dq2 > 0:
            dq2 = 0.0
        if self.arm_q2 <= -1.50 and dq2 < 0:
            dq2 = 0.0
        if self.arm_q3 >= 0.045 and dq3 > 0:
            dq3 = 0.0
        if self.arm_q3 <= -1.50 and dq3 < 0:
            dq3 = 0.0

        if self.arm_cmd_pub:
            cmd = Float64MultiArray()
            cmd.data = [0.0, dq2, dq3, 0.0]
            self.arm_cmd_pub.publish(cmd)
        return False

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def execute(self, local_planner=None, visualizer=None):
        """Main control loop execution."""
        # Terminal states
        if self.rotation_state == 'halted':
            if self.cmd_vel_pub:
                self.cmd_vel_pub.publish(Twist())
            return

        if self.rotation_state == 'spinning_360':
            if self.prev_yaw_for_spin is None:
                self.prev_yaw_for_spin = self.current_yaw
                self.spin_accumulated = 0.0
            else:
                delta = self.current_yaw - self.prev_yaw_for_spin
                if delta > math.pi:
                    delta -= 2 * math.pi
                elif delta < -math.pi:
                    delta += 2 * math.pi
                self.spin_accumulated += abs(delta)
                self.prev_yaw_for_spin = self.current_yaw

            if self.spin_accumulated >= 2 * math.pi - 0.1:
                if self.cmd_vel_pub:
                    self.cmd_vel_pub.publish(Twist())
                self.rotation_state = 'halted'
                self.logger.info("360° spin complete — halting permanently.")
            else:
                cmd = Twist()
                cmd.angular.z = self.max_angular_velocity
                if self.cmd_vel_pub:
                    self.cmd_vel_pub.publish(cmd)
            return

        # No waypoints
        if self.waypoints is None or len(self.waypoints) == 0:
            if self.cmd_vel_pub:
                self.cmd_vel_pub.publish(Twist())
            self.complete_a_path = True
            self.rotation_state = 'idle'
            # Keep retracting arm even when idle so it's ready for next move
            self._step_arm_to_retract()
            return

        # Arm must be retracted before any base motion
        if not self._step_arm_to_retract():
            if self.cmd_vel_pub:
                self.cmd_vel_pub.publish(Twist())
            return

        self.complete_a_path = False
        next_waypoint = self.waypoints[0]

        # Idle → decide scan vs move
        if self.rotation_state == 'idle':
            robot_pos = (self.robot_pose.x, self.robot_pose.y)
            if (self.last_scan_pos is None or
                math.hypot(robot_pos[0] - self.last_scan_pos[0],
                          robot_pos[1] - self.last_scan_pos[1]) > self.scan_distance_threshold):
                self.rotation_state = 'scanning_360'
                self.prev_yaw_for_spin = None
                self.spin_accumulated = 0.0
                self.last_scan_pos = robot_pos
                self.logger.info(
                    f"New location — scanning 360° before moving. "
                    f"pos=({robot_pos[0]:.2f},{robot_pos[1]:.2f})"
                )
            else:
                self.rotation_state = 'moving'

        # Scanning
        if self.rotation_state == 'scanning_360':
            if self.prev_yaw_for_spin is None:
                self.prev_yaw_for_spin = self.current_yaw
                self.spin_accumulated = 0.0
            else:
                delta = self.current_yaw - self.prev_yaw_for_spin
                if delta > math.pi:
                    delta -= 2 * math.pi
                elif delta < -math.pi:
                    delta += 2 * math.pi
                self.spin_accumulated += abs(delta)
                self.prev_yaw_for_spin = self.current_yaw

            if self.spin_accumulated >= 2 * math.pi - 0.1:
                if self.cmd_vel_pub:
                    self.cmd_vel_pub.publish(Twist())
                self.rotation_state = 'moving'
                self.logger.info("Waypoint scan complete — moving to next waypoint")
            else:
                cmd = Twist()
                cmd.angular.z = self.max_angular_velocity
                if self.cmd_vel_pub:
                    self.cmd_vel_pub.publish(cmd)
            return

        # Moving
        if self.rotation_state == 'moving':
            inc_x = next_waypoint[0] - self.robot_pose.x
            inc_y = next_waypoint[1] - self.robot_pose.y
            dist = np.sqrt(inc_x ** 2 + inc_y ** 2)

            if dist < self.acceptance_radius:
                self.waypoints.pop(0)
                if self.cmd_vel_pub:
                    self.cmd_vel_pub.publish(Twist())

                if len(self.waypoints) == 0:
                    reached_goal = self.goal_pose
                    self.waypoints = None
                    self.goal_pose = None
                    self.complete_a_path = True
                    if self.cmd_vel_pub:
                        self.cmd_vel_pub.publish(Twist())

                    if self.following_last_path:
                        self.logger.info(
                            "End of last path reached — spinning 360° then halting."
                        )
                        self.rotation_state = 'spinning_360'
                        self.prev_yaw_for_spin = None
                        self.spin_accumulated = 0.0
                    else:
                        if reached_goal is not None:
                            self.visited_frontier_positions.append(
                                (float(reached_goal[0]), float(reached_goal[1]))
                            )
                            self.frontiers_explored_count += 1
                            if self.frontiers_explored_count % self.frontier_expand_every == 0:
                                self.local_search_radius = min(
                                    self.local_search_radius + self.frontier_expand_step,
                                    self.max_local_search_radius
                                )
                                self.logger.info(
                                    f"Explored {self.frontiers_explored_count} frontiers — "
                                    f"global search window expanded to radius {self.local_search_radius} cells"
                                )
                        self.logger.info(
                            f"All waypoints reached. "
                            f"Frontiers explored: {self.frontiers_explored_count}, "
                            f"search radius: {self.local_search_radius} cells"
                        )
                        self.find_frontier = True
                        self.rotation_state = 'idle'
                    return

                # Check if intermediate waypoint is truly new
                robot_pos = (self.robot_pose.x, self.robot_pose.y)
                if (self.last_scan_pos is None or
                    math.hypot(robot_pos[0] - self.last_scan_pos[0],
                              robot_pos[1] - self.last_scan_pos[1]) > self.scan_distance_threshold):
                    self.rotation_state = 'scanning_360'
                    self.prev_yaw_for_spin = None
                    self.spin_accumulated = 0.0
                    self.last_scan_pos = robot_pos
                return

            # Move towards waypoint using DWA or pure pursuit.
            # Falls back to pure pursuit if the inflated map hasn't arrived yet
            # (same guard as the original: inflated_map_msg must not be None).
            if local_planner is not None and local_planner.inflated_map_msg is not None:
                local_planner.robot_pose = self.robot_pose
                local_planner.current_vel = self.current_vel
                local_planner.current_yaw = self.current_yaw
                v, w, paths = local_planner.compute(next_waypoint[0], next_waypoint[1])
                cmd = Twist()
                cmd.linear.x  = float(v)
                cmd.angular.z = float(w)
                if self.cmd_vel_pub:
                    self.cmd_vel_pub.publish(cmd)
                if visualizer:
                    visualizer.publish_dwa_paths(paths, v, w, self.dwa_viz_time, self.binary_map_frame, self.now)
            else:
                self.logger.warn(
                    "[DWA] Inflated map not yet received — falling back to pure pursuit",
                    throttle_duration_sec=2.0
                )
                desired_yaw = math.atan2(inc_y, inc_x)
                angle_diff  = self.normalize_angle(desired_yaw - self.current_yaw)
                cmd = Twist()
                cmd.angular.z = min(self.kw * angle_diff, self.max_angular_velocity)
                if abs(angle_diff) <= 0.3:
                    cmd.linear.x = min(self.kv * dist, self.max_linear_velocity)
                if self.cmd_vel_pub:
                    self.cmd_vel_pub.publish(cmd)

    def publish_waypoints(self, positions, visualizer=None):
        """Publish waypoint path for visualization."""
        if visualizer:
            visualizer.publish_waypoints(positions, self.binary_map_frame, self.now)
