"""Frontier detection and selection manager."""

import math
import numpy as np
import cv2
from collections import deque
from online_motion_planning.Point import Point as PointRRT


class FrontierManager:
    """Manages frontier detection, evaluation, and selection."""

    def __init__(self, kdist, karea, min_frontier_dist_m, visited_frontier_radius_m,
                 local_search_radius, max_local_search_radius, frontier_expand_every,
                 frontier_expand_step, use_global_search_window, global_x_min, global_x_max,
                 global_y_min, global_y_max, global_search_count_threshold,
                 local_search_count_threshold, logger):
        self.kdist = kdist
        self.karea = karea
        self.min_frontier_dist_m = min_frontier_dist_m
        self.visited_frontier_radius_m = visited_frontier_radius_m
        self.local_search_radius = local_search_radius
        self.max_local_search_radius = max_local_search_radius
        self.frontier_expand_every = frontier_expand_every
        self.frontier_expand_step = frontier_expand_step
        self.use_global_search_window = use_global_search_window
        self.global_x_min = global_x_min
        self.global_x_max = global_x_max
        self.global_y_min = global_y_min
        self.global_y_max = global_y_max
        self.global_search_count_threshold = global_search_count_threshold
        self.local_search_count_threshold = local_search_count_threshold
        self.logger = logger

        # References set externally
        self.occupancy_map = None
        self.rtab_map = None
        self.binary_map = None
        self.robot_pose = None
        self.start_world_pos = None
        self.origin = None
        self.resolution = None
        self.rtab_origin = None
        self.rtab_resolution = None
        self.height = None
        self.width = None
        self.rtab_height = None
        self.rtab_width = None
        self.waypoints = None   # used by terminal behaviour to decide follow-last vs spin

        # State variables
        self.visited_frontier_positions = []
        self.frontiers_explored_count = 0
        self.max_radius_wait_count = 0
        self.find_frontier = True
        self.goal_pose = None

        # Terminal-behaviour output signals — read by the node after select_frontier()
        # and consumed (reset) immediately after syncing to motion_controller.
        self.following_last_path = False
        self.rotation_state = 'idle'   # 'idle' | 'spinning_360'

        # Visualization references
        self.visualizer = None
        self.now = None
        self.binary_map_frame = "world_enu"

    # ─── Cost & BFS helpers ───────────────────────────────────────────────────

    def frontier_cost(self, area, cX, cY):
        """Compute cost for a frontier cluster (negative: lower = better)."""
        q_goal  = np.array([cX, cY])
        q_start = (np.array([self.robot_pose.x, self.robot_pose.y]) - self.origin) / self.resolution
        dist = PointRRT(q_start[0], q_start[1]).dist(PointRRT(q_goal[0], q_goal[1]))
        return -self.kdist * dist - self.karea * area

    def get_reachable_cells(self, robot_col, robot_row, max_cells=10000):
        """BFS flood fill from the robot position through the inflated map."""
        visited = set()
        queue = deque()
        start = (robot_col, robot_row)
        queue.append(start)
        visited.add(start)

        while queue and len(visited) < max_cells:
            col, row = queue.popleft()
            for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nc, nr = col + dc, row + dr
                if (nc, nr) in visited:
                    continue
                if not (0 <= nc < self.width and 0 <= nr < self.height):
                    continue
                cell_val = self.occupancy_map[nr, nc]
                if cell_val < 100:
                    visited.add((nc, nr))
                    queue.append((nc, nr))

        return visited

    def find_nearest_free_cell(self, col, row, max_radius=20):
        """BFS outward from (col, row) to find the closest cell where binary_map == 0."""
        if (0 <= row < self.height and 0 <= col < self.width
                and self.binary_map[row, col] == 0):
            return col, row

        queue = deque([(col, row)])
        visited = {(col, row)}
        while queue:
            c, r = queue.popleft()
            for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nc, nr = c + dc, r + dr
                if (nc, nr) in visited:
                    continue
                if abs(nc - col) > max_radius or abs(nr - row) > max_radius:
                    continue
                if not (0 <= nc < self.width and 0 <= nr < self.height):
                    continue
                visited.add((nc, nr))
                if self.binary_map[nr, nc] == 0:
                    return nc, nr
                queue.append((nc, nr))

        return None, None

    # ─── Main entry point ─────────────────────────────────────────────────────

    def select_frontier(self):
        """Frontier detection and selection — identical logic to frontier_viewpoint()."""
        if self.rotation_state in ('spinning_360', 'halted'):
            return

        if (self.occupancy_map is None or self.rtab_map is None or
                self.robot_pose is None or self.start_world_pos is None):
            return

        nav_col = int((self.robot_pose.x - self.origin[0]) / self.resolution)
        nav_row = int((self.robot_pose.y - self.origin[1]) / self.resolution)

        # ── Search window anchor ───────────────────────────────────────────
        if self.use_global_search_window:
            anchor_x, anchor_y = self.start_world_pos
        else:
            anchor_x, anchor_y = self.robot_pose.x, self.robot_pose.y

        anchor_rtab_col = int((anchor_x - self.rtab_origin[0]) / self.rtab_resolution)
        anchor_rtab_row = int((anchor_y - self.rtab_origin[1]) / self.rtab_resolution)

        SEARCH_RADIUS = self.local_search_radius

        # ── BFS reachability ───────────────────────────────────────────────
        reachable = self.get_reachable_cells(nav_col, nav_row, max_cells=10000)
        if self.visualizer:
            self.visualizer.publish_bfs_cells(reachable, self.binary_map_frame, self.now,
                                              self.resolution, self.origin)

        # ── Local window bounds ────────────────────────────────────────────
        rtab_row_min = max(1,                    anchor_rtab_row - SEARCH_RADIUS)
        rtab_row_max = min(self.rtab_height - 1, anchor_rtab_row + SEARCH_RADIUS)
        rtab_col_min = max(1,                    anchor_rtab_col - SEARCH_RADIUS)
        rtab_col_max = min(self.rtab_width  - 1, anchor_rtab_col + SEARCH_RADIUS)

        # ── Global mode override ───────────────────────────────────────────
        if self.use_global_search_window:
            rtab_col_min = max(1,
                int((self.global_x_min - self.rtab_origin[0]) / self.rtab_resolution))
            rtab_col_max = min(self.rtab_width  - 1,
                int((self.global_x_max - self.rtab_origin[0]) / self.rtab_resolution))
            rtab_row_min = max(1,
                int((self.global_y_min - self.rtab_origin[1]) / self.rtab_resolution))
            rtab_row_max = min(self.rtab_height - 1,
                int((self.global_y_max - self.rtab_origin[1]) / self.rtab_resolution))

        # ── Publish search area & log ──────────────────────────────────────
        now = self.now
        if self.use_global_search_window:
            if self.visualizer:
                self.visualizer.publish_search_area_world(
                    self.global_x_min, self.global_x_max,
                    self.global_y_min, self.global_y_max,
                    self.binary_map_frame, now)
            eff_x_min = rtab_col_min * self.rtab_resolution + self.rtab_origin[0]
            eff_x_max = rtab_col_max * self.rtab_resolution + self.rtab_origin[0]
            eff_y_min = rtab_row_min * self.rtab_resolution + self.rtab_origin[1]
            eff_y_max = rtab_row_max * self.rtab_resolution + self.rtab_origin[1]
            self.logger.info(
                f"[SEARCH] GLOBAL window — "
                f"defined: x=[{self.global_x_min:.1f},{self.global_x_max:.1f}] "
                f"y=[{self.global_y_min:.1f},{self.global_y_max:.1f}] | "
                f"effective (clipped to RTAB map): x=[{eff_x_min:.1f},{eff_x_max:.1f}] "
                f"y=[{eff_y_min:.1f},{eff_y_max:.1f}]"
            )
        else:
            if self.visualizer:
                self.visualizer.publish_search_area(
                    rtab_row_min, rtab_row_max, rtab_col_min, rtab_col_max,
                    self.binary_map_frame, now, self.rtab_resolution, self.rtab_origin)
            eff_x_min = rtab_col_min * self.rtab_resolution + self.rtab_origin[0]
            eff_x_max = rtab_col_max * self.rtab_resolution + self.rtab_origin[0]
            eff_y_min = rtab_row_min * self.rtab_resolution + self.rtab_origin[1]
            eff_y_max = rtab_row_max * self.rtab_resolution + self.rtab_origin[1]
            self.logger.info(
                f"[SEARCH] LOCAL window — "
                f"radius={self.local_search_radius} cells | "
                f"bounds: x=[{eff_x_min:.1f},{eff_x_max:.1f}] "
                f"y=[{eff_y_min:.1f},{eff_y_max:.1f}]"
            )

        # ── Frontier detection on raw RTAB-Map ────────────────────────────
        frontier_cells = np.zeros((self.rtab_height, self.rtab_width), dtype=np.uint8)

        for y in range(rtab_row_min, rtab_row_max):
            for x in range(rtab_col_min, rtab_col_max):
                if self.rtab_map[y, x] == 0:
                    neighbors = [
                        self.rtab_map[y - 1, x],
                        self.rtab_map[y + 1, x],
                        self.rtab_map[y, x - 1],
                        self.rtab_map[y, x + 1]
                    ]
                    if 50.0 in neighbors and 100.0 not in neighbors:
                        frontier_cells[y, x] = 255

        output = cv2.connectedComponentsWithStats(frontier_cells, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        if self.visualizer:
            self.visualizer.publish_all_frontiers(
                labels, numLabels, self.binary_map_frame, now,
                resolution=self.rtab_resolution, origin=self.rtab_origin)

        # ── Evaluate candidates ────────────────────────────────────────────
        cost_list = np.full(numLabels, np.inf)
        MAX_FRONTIER_AREA = 200
        EDGE_MARGIN = 3

        evaluated_centroids = []
        best_cost = np.inf
        current_best_world = None

        for i in range(1, numLabels):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]

            if area < 5:
                continue

            cl_width  = stats[i, cv2.CC_STAT_WIDTH]
            cl_height = stats[i, cv2.CC_STAT_HEIGHT]
            if min(cl_width, cl_height) < 3:
                continue

            if not (rtab_col_min <= int(cX) <= rtab_col_max and
                    rtab_row_min <= int(cY) <= rtab_row_max):
                continue

            if not (EDGE_MARGIN <= int(cX) < self.rtab_width  - EDGE_MARGIN and
                    EDGE_MARGIN <= int(cY) < self.rtab_height - EDGE_MARGIN):
                continue

            world_x = cX * self.rtab_resolution + self.rtab_origin[0]
            world_y = cY * self.rtab_resolution + self.rtab_origin[1]
            nav_cx  = int((world_x - self.origin[0]) / self.resolution)
            nav_cy  = int((world_y - self.origin[1]) / self.resolution)

            if (nav_cx, nav_cy) not in reachable:
                continue

            world_dist = math.hypot(world_x - self.robot_pose.x,
                                    world_y - self.robot_pose.y)
            if world_dist < self.min_frontier_dist_m:
                continue

            if any(math.hypot(world_x - vx, world_y - vy) < self.visited_frontier_radius_m
                   for vx, vy in self.visited_frontier_positions):
                continue

            capped_area = min(area, MAX_FRONTIER_AREA)
            cost = self.frontier_cost(capped_area, nav_cx, nav_cy)
            cost_list[i] = cost
            evaluated_centroids.append((world_x, world_y))

            if cost < best_cost:
                best_cost = cost
                current_best_world = (world_x, world_y)

        if self.visualizer:
            self.visualizer.publish_frontier_evaluation(
                evaluated_centroids, current_best_world, self.binary_map_frame, now)

        # ── Select best or handle failure ──────────────────────────────────
        if numLabels > 1:
            best_index = np.argmin(cost_list)
            if cost_list[best_index] == np.inf:
                self._handle_no_reachable_frontier()
                return

            best_cX, best_cY = centroids[best_index]
            goal_x = best_cX * self.rtab_resolution + self.rtab_origin[0]
            goal_y = best_cY * self.rtab_resolution + self.rtab_origin[1]
            # nav_col/nav_row already computed above for BFS; recompute for the best centroid
            best_nav_cx = int((goal_x - self.origin[0]) / self.resolution)
            best_nav_cy = int((goal_y - self.origin[1]) / self.resolution)

            self.logger.info(
                f"Best Goal: ({goal_x:.2f}, {goal_y:.2f}), "
                f"rtab_cell=({best_cX:.1f},{best_cY:.1f}), "
                f"nav_cell=({nav_col},{nav_row}), "
                f"candidates={len(evaluated_centroids)}"
            )
            self.goal_pose = [goal_x, goal_y]
            self.find_frontier = False

        elif numLabels == 1:
            self._handle_no_frontiers()

    # ─── Terminal-behaviour helpers ───────────────────────────────────────────

    def _trigger_terminal(self):
        """Decide whether to follow the last path or spin immediately."""
        self.find_frontier = False
        if self.waypoints is not None and len(self.waypoints) > 0:
            self.following_last_path = True  # node will read and consume this
        else:
            self.rotation_state = 'spinning_360'  # node will read and consume this

    def _handle_no_reachable_frontier(self):
        """Handle case when frontier clusters exist but none are reachable."""
        if self.use_global_search_window:
            self.max_radius_wait_count += 1
            self.logger.warn(
                f"[GLOBAL SEARCH] No reachable frontier inside window "
                f"x=[{self.global_x_min:.1f},{self.global_x_max:.1f}] "
                f"y=[{self.global_y_min:.1f},{self.global_y_max:.1f}] — "
                f"wait count: {self.max_radius_wait_count}/{self.global_search_count_threshold}"
            )
            if self.max_radius_wait_count >= self.global_search_count_threshold:
                if self.waypoints is not None and len(self.waypoints) > 0:
                    self.logger.info(
                        f"Global window exhausted {self.global_search_count_threshold} times — following last path "
                        f"({len(self.waypoints)} waypoints), then spinning 360°."
                    )
                else:
                    self.logger.info(
                        f"Global window exhausted {self.global_search_count_threshold} times — no saved path, "
                        "spinning 360° in place."
                    )
                self._trigger_terminal()
                return
            self.find_frontier = True
            return

        prev_radius = self.local_search_radius
        self.local_search_radius = min(
            self.local_search_radius + 10,
            self.max_local_search_radius
        )
        if self.local_search_radius >= self.max_local_search_radius:
            self.max_radius_wait_count += 1
            self.logger.warn(
                f"[LOCAL SEARCH] No reachable frontier — radius already at max "
                f"({self.max_local_search_radius} cells). "
                f"Wait count: {self.max_radius_wait_count}/{self.local_search_count_threshold}"
            )
            if self.max_radius_wait_count >= self.local_search_count_threshold:
                if self.waypoints is not None and len(self.waypoints) > 0:
                    self.logger.info(
                        f"Max radius reached {self.local_search_count_threshold} times — following last path to farthest "
                        f"clear point ({len(self.waypoints)} waypoints), then spinning 360°."
                    )
                else:
                    self.logger.info(
                        f"Max radius reached {self.local_search_count_threshold} times — no saved path, spinning 360° in place."
                    )
                self._trigger_terminal()
                return
        else:
            self.logger.warn(
                f"[LOCAL SEARCH] No reachable frontier — radius GREW: "
                f"{prev_radius} -> {self.local_search_radius} cells "
                f"(max={self.max_local_search_radius})"
            )
        self.find_frontier = True

    def _handle_no_frontiers(self):
        """Handle case when no frontier cells exist at all in the search window."""
        if self.use_global_search_window:
            self.max_radius_wait_count += 1
            self.logger.warn(
                f"[GLOBAL SEARCH] No frontier cells inside window "
                f"x=[{self.global_x_min:.1f},{self.global_x_max:.1f}] "
                f"y=[{self.global_y_min:.1f},{self.global_y_max:.1f}] — "
                f"wait count: {self.max_radius_wait_count}/{self.global_search_count_threshold}"
            )
            if self.max_radius_wait_count >= self.global_search_count_threshold:
                if self.waypoints is not None and len(self.waypoints) > 0:
                    self.logger.info(
                        f"Global window exhausted {self.global_search_count_threshold} times — following last path "
                        f"({len(self.waypoints)} waypoints), then spinning 360°."
                    )
                else:
                    self.logger.info(
                        f"Global window exhausted {self.global_search_count_threshold} times — no saved path, "
                        "spinning 360° in place."
                    )
                self._trigger_terminal()
                return
            self.find_frontier = True
            return

        prev_radius = self.local_search_radius
        self.local_search_radius = min(
            self.local_search_radius + 10,
            self.max_local_search_radius
        )
        if self.local_search_radius >= self.max_local_search_radius:
            self.max_radius_wait_count += 1
            self.logger.info(
                f"[LOCAL SEARCH] No frontier cells — radius already at max "
                f"({self.max_local_search_radius} cells). "
                f"Wait count: {self.max_radius_wait_count}/{self.local_search_count_threshold}"
            )
            if self.max_radius_wait_count >= self.local_search_count_threshold:
                if self.waypoints is not None and len(self.waypoints) > 0:
                    self.logger.info(
                        f"Max radius reached {self.local_search_count_threshold} times — following last path to farthest "
                        f"clear point ({len(self.waypoints)} waypoints), then spinning 360°."
                    )
                else:
                    self.logger.info(
                        f"Max radius reached {self.local_search_count_threshold} times — no saved path, spinning 360° in place."
                    )
                self._trigger_terminal()
                return
        else:
            self.logger.warn(
                f"[LOCAL SEARCH] No frontier cells — radius GREW: "
                f"{prev_radius} → {self.local_search_radius} cells "
                f"(max={self.max_local_search_radius})"
            )
        self.find_frontier = True
