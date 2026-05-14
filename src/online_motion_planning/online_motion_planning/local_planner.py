"""Local path planner using Dynamic Window Approach (DWA)."""

import math
import numpy as np


class LocalPlanner:
    """DWA-based local planner for obstacle-aware trajectory selection."""

    def __init__(self, max_linear_velocity, max_angular_velocity, dwa_max_accel,
                 dwa_max_delta_yaw, dwa_predict_time, dwa_heading_w, dwa_dist_w,
                 dwa_obstacle_w, dwa_velocity_w, dwa_viz_time, logger):
        """Initialize local planner with DWA parameters.
        
        Args:
            max_linear_velocity: Max forward speed (m/s)
            max_angular_velocity: Max angular speed (rad/s)
            dwa_max_accel: Max acceleration (m/s²)
            dwa_max_delta_yaw: Max yaw rate change (rad/s²)
            dwa_predict_time: Prediction horizon (s)
            dwa_heading_w: Heading error weight
            dwa_dist_w: Distance-to-goal weight
            dwa_obstacle_w: Obstacle avoidance weight
            dwa_velocity_w: Velocity smoothness weight
            dwa_viz_time: Visualization time horizon (s)
            logger: ROS logger instance
        """
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.dwa_max_accel = dwa_max_accel
        self.dwa_max_delta_yaw = dwa_max_delta_yaw
        self.dwa_predict_time = dwa_predict_time
        self.dwa_heading_w = dwa_heading_w
        self.dwa_dist_w = dwa_dist_w
        self.dwa_obstacle_w = dwa_obstacle_w
        self.dwa_velocity_w = dwa_velocity_w
        self.dwa_viz_time = dwa_viz_time
        self.logger = logger
        
        # References set externally
        self.inflated_map_msg = None
        self.robot_pose = None
        self.current_vel = (0.0, 0.0)
        self.current_yaw = 0.0

    def get_cell_value(self, x, y):
        """Look up the inflated-map cost at world position (x, y)."""
        if self.inflated_map_msg is None:
            return None
        info = self.inflated_map_msg.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        if 0 <= gx < info.width and 0 <= gy < info.height:
            return self.inflated_map_msg.data[gy * info.width + gx]
        return None

    def obstacle_cost(self, traj):
        """Compute soft obstacle penalty along trajectory.
        
        Lethal threshold is >= 99 (inscribed radius / wall).
        Soft penalties for cells 10-98 encourage staying away from walls
        even inside the passable inflation zone.
        """
        if self.inflated_map_msg is None:
            return 0.0
        penalty = 0.0
        sampled = traj[::3]   # sample every 3rd point for speed
        for x, y, _ in sampled:
            val = self.get_cell_value(x, y)
            if val is None or val >= 99:   # lethal
                return float('inf')
            if val > 50:
                penalty += 1.0   # heavy penalty
            elif val > 30:
                penalty += 0.5
            elif val > 10:
                penalty += 0.2
        return penalty

    def dynamic_window(self):
        """Compute reachable (v, w) range given current velocity and acceleration limits."""
        v, w = self.current_vel
        dt = 0.1
        v_min = max(0.0, v - self.dwa_max_accel * dt)
        v_max = min(self.max_linear_velocity, v + self.dwa_max_accel * dt)
        w_min = max(-self.max_angular_velocity, w - self.dwa_max_delta_yaw * dt)
        w_max = min(self.max_angular_velocity, w + self.dwa_max_delta_yaw * dt)
        return v_min, v_max, w_min, w_max

    def simulate_trajectory(self, v, w, predict_time=None):
        """Forward-simulate robot pose over predict_time seconds."""
        if predict_time is None:
            predict_time = self.dwa_predict_time
        x, y, yaw = self.robot_pose.x, self.robot_pose.y, self.current_yaw
        dt = 0.1
        traj = []
        for _ in range(max(1, int(predict_time / dt))):
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += w * dt
            traj.append((x, y, yaw))
        return traj

    def compute(self, goal_x, goal_y):
        """Evaluate all (v, w) combinations and return best command + all paths.
        
        Args:
            goal_x, goal_y: Waypoint coordinates (world frame)
            
        Returns:
            best_v, best_w: Optimal command
            all_paths: List of all evaluated trajectories with costs
        """
        if self.robot_pose is None:
            return 0.0, 0.0, []

        dist_to_goal = math.hypot(goal_x - self.robot_pose.x,
                                  goal_y - self.robot_pose.y)
        # Adaptive prediction horizon based on distance to goal
        horizon = max(0.5, min(self.dwa_predict_time,
                              dist_to_goal / max(self.max_linear_velocity, 0.01)))

        v_min, v_max, w_min, w_max = self.dynamic_window()
        best_v, best_w = 0.0, 0.0
        best_cost = float('inf')
        all_paths = []

        for v in np.arange(v_min, v_max + 0.01, 0.03):
            for w in np.arange(w_min, w_max + 0.01, 0.06):
                traj = self.simulate_trajectory(v, w, predict_time=horizon)
                obs_cost = self.obstacle_cost(traj)
                if obs_cost == float('inf'):
                    continue

                lx, ly, lyaw = traj[-1]
                goal_angle = math.atan2(goal_y - ly, goal_x - lx)
                heading_err = abs(math.atan2(
                    math.sin(goal_angle - lyaw),
                    math.cos(goal_angle - lyaw)
                ))
                dist = math.hypot(goal_x - lx, goal_y - ly)

                cost = (self.dwa_heading_w * heading_err +
                        self.dwa_dist_w * dist +
                        self.dwa_obstacle_w * obs_cost +
                        self.dwa_velocity_w * (self.max_linear_velocity - v))

                all_paths.append({'v': v, 'w': w, 'traj': traj, 'cost': cost})

                if cost < best_cost:
                    best_cost = cost
                    best_v, best_w = v, w

        return best_v, best_w, all_paths
