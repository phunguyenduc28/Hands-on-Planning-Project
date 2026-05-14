"""Global path planner using BiRRT*."""

import math
import numpy as np
from geometry_msgs.msg import Twist
from online_motion_planning.bidirectional_rrt_star import BIRRT_STAR
from online_motion_planning.Point import Point as PointRRT


class PathPlanner:
    """Global path planning using BiRRT* (Bidirectional Rapidly-exploring Random Tree)."""

    def __init__(self, delta_q, p, max_depth, min_dist, radius, threshold_path_rewire_dist,
                 max_iterations_base, max_iterations_increment, max_iterations_cap,
                 max_retry_same_goal, logger):
        self.delta_q = delta_q
        self.p = p
        self.max_depth = max_depth
        self.min_dist = min_dist
        self.radius = radius
        self.threshold_path_rewire_dist = threshold_path_rewire_dist
        self.max_iterations_base = max_iterations_base
        self.max_iterations_increment = max_iterations_increment
        self.max_iterations_cap = max_iterations_cap
        self.max_retry_same_goal = max_retry_same_goal
        self.logger = logger

        # References set externally
        self.binary_map = None
        self.robot_pose = None
        self.goal_pose = None
        self.resolution = None
        self.origin = None
        self.cmd_vel_pub = None

        # State variables
        self.max_iterations = max_iterations_base
        self.rrt_fail_count = 0
        self.waypoints = None
        self.complete_a_path = True
        self.collide_robot_next_waypoint = False
        self.following_last_path = False

        # Signal flag: set to True when planning determines a new frontier should be searched.
        # The node checks this after plan() returns and propagates it to motion_controller.
        self.find_frontier = False

    def plan(self, find_nearest_free_cell_fn, viz_cb_setup_fn):
        """Execute global path planning using BiRRT*.

        Returns waypoints list on success, None otherwise.
        Sets self.find_frontier = True when the caller should trigger a new frontier search.
        """
        if self.binary_map is None or self.robot_pose is None:
            return None

        # ── following_last_path mode ──────────────────────────────────────────
        # Check every segment for obstacles; if blocked set waypoints=None so
        # the caller knows to start the terminal 360° spin.
        if self.following_last_path:
            if self.waypoints is not None and len(self.waypoints) > 0:
                rrt_check = BIRRT_STAR(self.delta_q, self.p, self.max_depth,
                                       self.min_dist, self.radius, self.threshold_path_rewire_dist)
                q_start = (np.array([self.robot_pose.x, self.robot_pose.y]) - self.origin) / self.resolution
                prev_pt = PointRRT(q_start[0], q_start[1])
                blocked = False
                for wp in self.waypoints:
                    q_wp   = (np.array([wp[0], wp[1]]) - self.origin) / self.resolution
                    wp_pt  = PointRRT(q_wp[0], q_wp[1])
                    seg_len = prev_pt.dist(wp_pt)
                    rrt_check.max_depth = max(self.max_depth,
                                              round(math.log(max(seg_len, 2), 2)) + 1)
                    if not rrt_check.is_segment_free_bisection(prev_pt, wp_pt, self.binary_map, 0):
                        blocked = True
                        break
                    prev_pt = wp_pt
                if blocked:
                    self.logger.info(
                        "Last path now blocked — spinning 360° at current position."
                    )
                    if self.cmd_vel_pub:
                        self.cmd_vel_pub.publish(Twist())
                    self.waypoints = None
            return None   # never replan in this mode

        if self.goal_pose is None:
            return None

        q_start = (np.array([self.robot_pose.x, self.robot_pose.y]) - self.origin) / self.resolution
        q_goal  = (np.array([self.goal_pose[0], self.goal_pose[1]]) - self.origin) / self.resolution
        q_start_point = PointRRT(q_start[0], q_start[1])
        q_goal_point  = PointRRT(q_goal[0],  q_goal[1])

        self.logger.debug(f"Map shape (h, w): {self.binary_map.shape}, Resolution: {self.resolution}, Origin: {self.origin}")
        self.logger.debug(f"Robot world pose: ({self.robot_pose.x:.2f}, {self.robot_pose.y:.2f})")
        self.logger.debug(f"Robot cell coords (q_start): ({q_start[0]:.2f}, {q_start[1]:.2f})")
        self.logger.debug(f"Goal world pose: ({self.goal_pose[0]:.2f}, {self.goal_pose[1]:.2f})")
        self.logger.debug(f"Goal cell coords (q_goal): ({q_goal[0]:.2f}, {q_goal[1]:.2f})")

        # ── Bounds check ─────────────────────────────────────────────────────
        map_h, map_w = self.binary_map.shape
        q_start_in_bounds = (0 <= q_start[0] < map_w and 0 <= q_start[1] < map_h)
        q_goal_in_bounds  = (0 <= q_goal[0]  < map_w and 0 <= q_goal[1]  < map_h)

        if not q_start_in_bounds:
            self.logger.warn(f"Start pose is outside map bounds!")
            return None
        if not q_goal_in_bounds:
            self.logger.warn(f"Goal pose is outside map bounds!")
            self.find_frontier = True   # signal caller to pick a new goal
            return None

        rrt_star = BIRRT_STAR(self.delta_q, self.p, self.max_depth, self.min_dist,
                               self.radius, self.threshold_path_rewire_dist)

        # ── Start snapping ───────────────────────────────────────────────────
        is_start_occupied = rrt_star.is_point_occupied(q_start_point, self.binary_map)
        is_goal_occupied  = rrt_star.is_point_occupied(q_goal_point,  self.binary_map)

        if is_start_occupied:
            free_col, free_row = find_nearest_free_cell_fn(int(q_start[0]), int(q_start[1]))
            if free_col is None:
                self.logger.error(
                    "Start is occupied and no free cell found within 20 cells — waiting for map update."
                )
                return None
            self.logger.warn(
                f"Start cell ({int(q_start[0])},{int(q_start[1])}) is occupied — "
                f"snapping to nearest free cell ({free_col},{free_row})."
            )
            q_start       = np.array([float(free_col), float(free_row)])
            q_start_point = PointRRT(q_start[0], q_start[1])

        if is_goal_occupied:
            self.logger.warn(f"Goal point is not valid (on obstacle). Select another goal")
            self.find_frontier = True   # signal caller to pick a new goal
            self.waypoints = None       # stop robot — current path leads to a blocked goal
            return None

        # ── Existing path collision check ────────────────────────────────────
        if self.waypoints is not None:
            prev_point = q_start_point
            for wp in self.waypoints:
                q_wp = (np.array([wp[0], wp[1]]) - self.origin) / self.resolution
                wp_point = PointRRT(q_wp[0], q_wp[1])
                seg_len = prev_point.dist(wp_point)
                rrt_star.max_depth = max(self.max_depth,
                                         round(math.log(max(seg_len, 2), 2)) + 1)
                if not rrt_star.is_segment_free_bisection(prev_point, wp_point, self.binary_map, 0):
                    self.collide_robot_next_waypoint = True
                    break
                prev_point = wp_point
            rrt_star.max_depth = self.max_depth  # restore for RRT* planning below

            if self.collide_robot_next_waypoint:
                self.logger.warn(f"Path segment is now blocked — stopping and replanning")
                if self.cmd_vel_pub:
                    self.cmd_vel_pub.publish(Twist())

        if self.complete_a_path is False and not self.collide_robot_next_waypoint:
            return None  # still executing current path, no replan needed

        # ── Plan new path ────────────────────────────────────────────────────
        self.logger.info(f"Planning a path from {q_start} to {q_goal}")
        self.logger.info(f"RRT* params: delta_q={self.delta_q}, p={self.p}, max_iter={self.max_iterations}, min_dist={self.min_dist}, radius={self.radius}")

        viz_cb_a, flush_a = viz_cb_setup_fn((0.2, 0.6, 1.0), 0.06)
        viz_cb_b, flush_b = viz_cb_setup_fn((1.0, 0.45, 0.0), 0.10)

        G, edges, iter = rrt_star.sample(
            self.binary_map, self.max_iterations,
            q_start[0], q_start[1], q_goal[0], q_goal[1],
            logger=self.logger,
            viz_callback_a=viz_cb_a, viz_callback_b=viz_cb_b,
        )
        flush_a()
        flush_b()
        self.logger.info(f"RRT* sampling completed: iterations={iter}, tree size={len(G)}, edges={len(edges)}")

        if iter == self.max_iterations and len(edges) == 0:
            self.rrt_fail_count += 1
            self.max_iterations = min(
                self.max_iterations_base + self.rrt_fail_count * self.max_iterations_increment,
                self.max_iterations_cap
            )
            if self.rrt_fail_count < self.max_retry_same_goal:
                self.logger.warn(
                    f"RRT* failed (attempt {self.rrt_fail_count}/{self.max_retry_same_goal}). "
                    f"Retrying same goal with {self.max_iterations} iterations."
                )
                self.waypoints = None
            else:
                self.logger.warn(
                    f"Cannot find path after {self.rrt_fail_count} attempts — "
                    f"abandoning goal and selecting new frontier."
                )
                self.rrt_fail_count = 0
                self.max_iterations = self.max_iterations_base
                self.find_frontier = True   # signal caller to pick a new goal
                self.waypoints = None
            return None

        self.rrt_fail_count = 0
        self.max_iterations = self.max_iterations_base
        G, edges, path = rrt_star.fill_path(G, edges)
        path = rrt_star.smoothing(self.binary_map, G, path)
        self.logger.info(f"Find a path with {len(path)} waypoints")
        rrt_star.plot(self.binary_map, G, edges, path)

        waypoints = []
        for i in range(1, len(path)):
            q = G[path[i]]
            coordinate = np.array([q.x, q.y]) * self.resolution + self.origin
            waypoints.append(np.array([coordinate[0], coordinate[1]]))

        self.complete_a_path = False
        self.collide_robot_next_waypoint = False
        return waypoints if len(waypoints) > 0 else None
