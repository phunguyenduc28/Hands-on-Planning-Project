import rclpy
from rclpy.node import Node
import numpy as np
import math
import cv2

from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray


class FrontierNode(Node):
    """Frontier detection and selection node.

    Detects unexplored boundary cells on the raw RTAB-Map (/map), clusters
    them with connected-components, scores each cluster, and publishes the
    best one to /frontier_goal (PoseStamped).

    Interface
    ---------
    Subscribers
      /turtlebot/odom               — robot position
      /inflated_map (OccupancyGrid) — used for BFS reachability check
      /map          (OccupancyGrid) — raw RTAB-Map for frontier cell detection
      /frontier/trigger (Bool)      — True  → enable next search cycle
      /frontier/goal_reached (PoseStamped) → path planner reached a frontier;
                                             update visited list and trigger search

    Publishers
      /frontier_goal (PoseStamped)              — selected frontier goal
      /frontier/exploration_complete (Bool)     — True when no frontiers remain
      /frontier_viz/all_frontiers (MarkerArray) — coloured POINTS per cluster
      /frontier_viz/bfs_cells     (Marker)      — semi-transparent BFS overlay
      /frontier_viz/evaluation    (MarkerArray) — cyan/green spheres for candidates
      /frontier_viz/search_area   (Marker)      — white/cyan search window rect
    """

    def __init__(self):
        super().__init__('frontier_node')

        # ── Robot state ──────────────────────────────────────────────────────
        self.robot_pose = None
        self.current_yaw = 0.0
        self.start_world_pos = None

        # ── Inflated map (BFS reachability + nav-cell conversion) ────────────
        self.occupancy_map = None
        self.origin = None
        self.resolution = None
        self.height = None
        self.width = None

        # ── Raw RTAB-Map (frontier detection only) ───────────────────────────
        self.rtab_map = None
        self.rtab_origin = None
        self.rtab_resolution = None
        self.rtab_width = None
        self.rtab_height = None

        # ── Frontier scoring weights ─────────────────────────────────────────
        self.declare_parameter('kdist', 1.0)
        self.kdist = self.get_parameter('kdist').value
        self.declare_parameter('karea', 2.0)
        self.karea = self.get_parameter('karea').value

        # ── Search gate: True → run detection on next timer tick ─────────────
        self.find_frontier = True

        # True once the first frontier goal has been published.  The terminal
        # exploration countdown (max_radius_wait_count) is suppressed until this
        # flag is set so that the "exploration complete" signal is never fired
        # during the map-build startup phase (when the RTAB map has no free
        # cells yet and every de tection tick returns numLabels==1).
        self.frontiers_ever_found = False

        # ── Rejection filters ────────────────────────────────────────────────
        self.declare_parameter('min_frontier_dist_m', 0.3)
        self.min_frontier_dist_m = self.get_parameter('min_frontier_dist_m').value
        self.visited_frontier_positions = []
        self.declare_parameter('visited_frontier_radius_m', 0.3)
        self.visited_frontier_radius_m = self.get_parameter('visited_frontier_radius_m').value
        self.declare_parameter('min_frontier_area', 1)
        self.min_frontier_area = self.get_parameter('min_frontier_area').value
        self.declare_parameter('min_frontier_dim', 1)
        self.min_frontier_dim = self.get_parameter('min_frontier_dim').value
        self.declare_parameter('max_frontier_area', 200)
        self.max_frontier_area = self.get_parameter('max_frontier_area').value
        self.declare_parameter('frontier_edge_margin', 3)
        self.frontier_edge_margin = self.get_parameter('frontier_edge_margin').value

        # ── Count-based search-window expansion ──────────────────────────────
        self.frontiers_explored_count = 0
        self.declare_parameter('frontier_expand_every', 3)
        self.frontier_expand_every = self.get_parameter('frontier_expand_every').value
        self.declare_parameter('frontier_expand_step', 15)
        self.frontier_expand_step = self.get_parameter('frontier_expand_step').value

        # ── Search window ────────────────────────────────────────────────────
        # use_global_search_window=True: fixed world_enu rectangle (tune bounds below)
        # use_global_search_window=False: robot-centred growing square
        self.declare_parameter('use_global_search_window', True)
        self.use_global_search_window = self.get_parameter('use_global_search_window').value
        self.declare_parameter('global_x_min', -1.5)
        self.global_x_min = self.get_parameter('global_x_min').value
        self.declare_parameter('global_x_max', 1.0)
        self.global_x_max = self.get_parameter('global_x_max').value
        self.declare_parameter('global_y_min', -20.0)
        self.global_y_min = self.get_parameter('global_y_min').value
        self.declare_parameter('global_y_max', 1.0)
        self.global_y_max = self.get_parameter('global_y_max').value
        self.declare_parameter('global_search_count_threshold', 10)
        self.global_search_count_threshold = self.get_parameter('global_search_count_threshold').value

        self.declare_parameter('local_search_radius', 20)
        self.local_search_radius = self.get_parameter('local_search_radius').value
        self.declare_parameter('max_local_search_radius', 50)
        self.max_local_search_radius = self.get_parameter('max_local_search_radius').value
        self.declare_parameter('local_search_count_threshold', 10)
        self.local_search_count_threshold = self.get_parameter('local_search_count_threshold').value
        self.max_radius_wait_count = 0

        self.declare_parameter('bfs_max_cells', 10000)
        self.bfs_max_cells = self.get_parameter('bfs_max_cells').value

        # ── Map frame ────────────────────────────────────────────────────────
        self.declare_parameter('map_frame', 'world_enu')
        self.binary_map_frame = self.get_parameter('map_frame').value

        # ── Publishers ───────────────────────────────────────────────────────
        self.frontier_goal_pub = self.create_publisher(
            PoseStamped, '/frontier_goal', 10)
        self.exploration_complete_pub = self.create_publisher(
            Bool, '/frontier/exploration_complete', 10)
        self.frontier_all_pub = self.create_publisher(
            MarkerArray, '/frontier_viz/all_frontiers', 10)
        self.bfs_cells_pub = self.create_publisher(
            Marker, '/frontier_viz/bfs_cells', 10)
        self.frontier_eval_pub = self.create_publisher(
            MarkerArray, '/frontier_viz/evaluation', 10)
        self.search_area_pub = self.create_publisher(
            Marker, '/frontier_viz/search_area', 10)
        self.frontier_labels_pub = self.create_publisher(
            MarkerArray, '/frontier_viz/labels', 10)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            Odometry, '/turtlebot/odom', self._odom_cb, 10)
        self.create_subscription(
            OccupancyGrid, '/inflated_map', self._inflated_map_cb, 10)
        self.create_subscription(
            OccupancyGrid, '/map', self._rtab_map_cb, 10)
        self.create_subscription(
            Bool, '/frontier/trigger', self._trigger_cb, 10)
        self.create_subscription(
            PoseStamped, '/frontier/goal_reached', self._goal_reached_cb, 10)

        self.viewpoint_timer = self.create_timer(2.0, self.frontier_viewpoint)

        self.get_logger().info('Frontier node started')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        self.robot_pose = msg.pose.pose.position
        if self.start_world_pos is None:
            self.start_world_pos = (self.robot_pose.x, self.robot_pose.y)
            self.get_logger().info(
                f'Global search window anchored at '
                f'({self.start_world_pos[0]:.2f}, {self.start_world_pos[1]:.2f})')
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z))

    def _inflated_map_cb(self, msg):
        info = msg.info
        self.resolution = info.resolution
        self.width = info.width
        self.height = info.height
        self.origin = np.array([info.origin.position.x, info.origin.position.y])
        raw = np.array(msg.data, dtype=float).reshape(self.height, self.width)
        raw[raw == -1] = 50.0
        self.occupancy_map = raw

    def _rtab_map_cb(self, msg):
        info = msg.info
        self.rtab_resolution = info.resolution
        self.rtab_width = info.width
        self.rtab_height = info.height
        self.rtab_origin = np.array(
            [info.origin.position.x, info.origin.position.y])
        raw = np.array(msg.data, dtype=float).reshape(
            self.rtab_height, self.rtab_width)
        raw[raw == -1] = 50.0
        self.rtab_map = raw

    def _trigger_cb(self, msg):
        if msg.data:
            self.find_frontier = True

    def _goal_reached_cb(self, msg):
        """Called by the path planner when a frontier goal is successfully reached."""
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.visited_frontier_positions.append((gx, gy))
        self.frontiers_explored_count += 1

        if self.frontiers_explored_count % self.frontier_expand_every == 0:
            self.local_search_radius = min(
                self.local_search_radius + self.frontier_expand_step,
                self.max_local_search_radius)
            self.get_logger().info(
                f'Explored {self.frontiers_explored_count} frontiers — '
                f'search radius expanded to {self.local_search_radius} cells')

        self.find_frontier = True

    # ── BFS reachability ─────────────────────────────────────────────────────

    def frontier_cost(self, area, cX, cY):
        """Cost function: prefer large frontiers that are far from the robot.
        Negative so np.argmin selects the best candidate.
        """
        q_goal = np.array([cX, cY])
        q_start = (
            np.array([self.robot_pose.x, self.robot_pose.y]) - self.origin
        ) / self.resolution
        dist = float(np.linalg.norm(q_goal - q_start))
        return -self.kdist * dist - self.karea * area

    def get_reachable_cells(self, robot_col, robot_row, max_cells=10000):
        """BFS flood-fill on the inflated map from the robot position.

        Cells with value < 100 (free + inflation gradient + unknown=50) are
        walkable; inscribed (99) and lethal (100) cells are barriers, matching
        the binary_map >= 99 threshold used by BiRRT*.
        """
        from collections import deque
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
                if self.occupancy_map[nr, nc] < 100:
                    visited.add((nc, nr))
                    queue.append((nc, nr))
        return visited

    # ── Visualisation helpers ─────────────────────────────────────────────────

    _FRONTIER_COLORS = [
        (1.0, 1.0, 0.0),  # yellow
        (1.0, 0.5, 0.0),  # orange
        (1.0, 0.0, 1.0),  # magenta
        (0.0, 1.0, 1.0),  # cyan
        (1.0, 1.0, 1.0),  # white
        (0.5, 1.0, 0.0),  # lime
    ]

    def _publish_all_frontiers(self, labels, numLabels,
                               resolution=None, origin=None):
        if resolution is None:
            resolution = self.resolution
        if origin is None:
            origin = self.origin

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        clear = Marker()
        clear.header.frame_id = self.binary_map_frame
        clear.header.stamp = now
        clear.ns = 'all_frontiers'
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for i in range(1, numLabels):
            rows, cols = np.where(labels == i)
            if len(rows) == 0:
                continue
            color = self._FRONTIER_COLORS[(i - 1) % len(self._FRONTIER_COLORS)]
            m = Marker()
            m.header.frame_id = self.binary_map_frame
            m.header.stamp = now
            m.ns = 'all_frontiers'
            m.id = i
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = resolution
            m.scale.y = resolution
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 0.85
            m.pose.orientation.w = 1.0
            m.lifetime = rclpy.duration.Duration(seconds=4).to_msg()
            for row, col in zip(rows, cols):
                p = Point()
                p.x = float(col * resolution + origin[0])
                p.y = float(row * resolution + origin[1])
                p.z = 0.05
                m.points.append(p)
            marker_array.markers.append(m)

        self.frontier_all_pub.publish(marker_array)

    def _publish_bfs_cells(self, reachable):
        now = self.get_clock().now().to_msg()
        m = Marker()
        m.header.frame_id = self.binary_map_frame
        m.header.stamp = now
        m.ns = 'bfs_cells'
        m.id = 0
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = self.resolution * 0.7
        m.scale.y = self.resolution * 0.7
        m.color.r = 0.0
        m.color.g = 0.85
        m.color.b = 0.2
        m.color.a = 0.18
        m.pose.orientation.w = 1.0
        m.lifetime = rclpy.duration.Duration(seconds=4).to_msg()
        cells = list(reachable)
        step = max(1, len(cells) // 3000)
        for col, row in cells[::step]:
            p = Point()
            p.x = float(col * self.resolution + self.origin[0])
            p.y = float(row * self.resolution + self.origin[1])
            p.z = 0.0
            m.points.append(p)
        self.bfs_cells_pub.publish(m)

    def _publish_frontier_evaluation(self, evaluated_centroids, best_world):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        lifetime = rclpy.duration.Duration(seconds=4).to_msg()

        for ns in ('evaluated_centroids', 'best_frontier'):
            clr = Marker()
            clr.header.frame_id = self.binary_map_frame
            clr.header.stamp = now
            clr.ns = ns
            clr.action = Marker.DELETEALL
            marker_array.markers.append(clr)

        for i, (wx, wy) in enumerate(evaluated_centroids):
            m = Marker()
            m.header.frame_id = self.binary_map_frame
            m.header.stamp = now
            m.ns = 'evaluated_centroids'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.12
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.18
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 0.85
            m.lifetime = lifetime
            marker_array.markers.append(m)

        if best_world is not None:
            bm = Marker()
            bm.header.frame_id = self.binary_map_frame
            bm.header.stamp = now
            bm.ns = 'best_frontier'
            bm.id = 0
            bm.type = Marker.SPHERE
            bm.action = Marker.ADD
            bm.pose.position.x = best_world[0]
            bm.pose.position.y = best_world[1]
            bm.pose.position.z = 0.2
            bm.pose.orientation.w = 1.0
            bm.scale.x = bm.scale.y = bm.scale.z = 0.35
            bm.color.r = 0.0
            bm.color.g = 1.0
            bm.color.b = 0.0
            bm.color.a = 1.0
            bm.lifetime = lifetime
            marker_array.markers.append(bm)

        self.frontier_eval_pub.publish(marker_array)

    def _publish_frontier_labels(self, numLabels, centroids,
                                 resolution=None, origin=None):
        """White TEXT_VIEW_FACING markers showing each cluster's index."""
        if resolution is None:
            resolution = self.rtab_resolution
        if origin is None:
            origin = self.rtab_origin

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        clear = Marker()
        clear.header.frame_id = self.binary_map_frame
        clear.header.stamp = now
        clear.ns = 'frontier_labels'
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for i in range(1, numLabels):
            cX, cY = centroids[i]
            m = Marker()
            m.header.frame_id = self.binary_map_frame
            m.header.stamp = now
            m.ns = 'frontier_labels'
            m.id = i
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = float(cX * resolution + origin[0])
            m.pose.position.y = float(cY * resolution + origin[1])
            m.pose.position.z = 0.35
            m.pose.orientation.w = 1.0
            m.scale.z = 0.18
            m.color.r = m.color.g = m.color.b = 1.0
            m.color.a = 1.0
            m.text = str(i)
            m.lifetime = rclpy.duration.Duration(seconds=4).to_msg()
            marker_array.markers.append(m)

        self.frontier_labels_pub.publish(marker_array)

    def _publish_search_area(self, rtab_row_min, rtab_row_max,
                             rtab_col_min, rtab_col_max):
        """White LINE_STRIP rectangle for the local search window."""
        now = self.get_clock().now().to_msg()
        x_min = rtab_col_min * self.rtab_resolution + self.rtab_origin[0]
        x_max = rtab_col_max * self.rtab_resolution + self.rtab_origin[0]
        y_min = rtab_row_min * self.rtab_resolution + self.rtab_origin[1]
        y_max = rtab_row_max * self.rtab_resolution + self.rtab_origin[1]
        m = Marker()
        m.header.frame_id = self.binary_map_frame
        m.header.stamp = now
        m.ns = 'search_area'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.06
        m.color.r = m.color.g = m.color.b = 1.0
        m.color.a = 0.9
        m.pose.orientation.w = 1.0
        m.lifetime = rclpy.duration.Duration(seconds=4).to_msg()
        for cx, cy in [(x_min, y_min), (x_max, y_min),
                       (x_max, y_max), (x_min, y_max), (x_min, y_min)]:
            p = Point()
            p.x = float(cx)
            p.y = float(cy)
            p.z = 0.05
            m.points.append(p)
        self.search_area_pub.publish(m)

    def _publish_search_area_world(self, x_min, x_max, y_min, y_max):
        """Cyan LINE_STRIP rectangle drawn directly in world-enu metres."""
        now = self.get_clock().now().to_msg()
        m = Marker()
        m.header.frame_id = self.binary_map_frame
        m.header.stamp = now
        m.ns = 'search_area'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.06
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 0.9
        m.pose.orientation.w = 1.0
        m.lifetime = rclpy.duration.Duration(seconds=4).to_msg()
        for cx, cy in [(x_min, y_min), (x_max, y_min),
                       (x_max, y_max), (x_min, y_max), (x_min, y_min)]:
            p = Point()
            p.x = float(cx)
            p.y = float(cy)
            p.z = 0.05
            m.points.append(p)
        self.search_area_pub.publish(m)

    # ── Exploration-complete signal ───────────────────────────────────────────

    def _signal_exploration_complete(self):
        self.find_frontier = False
        msg = Bool()
        msg.data = True
        self.exploration_complete_pub.publish(msg)
        self.get_logger().info(
            'Exploration complete — signalling path planner')

    # ── Main frontier search (0.5 Hz timer) ──────────────────────────────────

    def frontier_viewpoint(self):
        if not self.find_frontier:
            return
        if (self.occupancy_map is None or self.rtab_map is None
                or self.robot_pose is None or self.start_world_pos is None):
            return

        # ── Robot cell on the inflated map ────────────────────────────────────
        nav_col = int((self.robot_pose.x - self.origin[0]) / self.resolution)
        nav_row = int((self.robot_pose.y - self.origin[1]) / self.resolution)

        # ── Search-window anchor ──────────────────────────────────────────────
        if self.use_global_search_window:
            anchor_x, anchor_y = self.start_world_pos
        else:
            anchor_x, anchor_y = self.robot_pose.x, self.robot_pose.y

        anchor_rtab_col = int(
            (anchor_x - self.rtab_origin[0]) / self.rtab_resolution)
        anchor_rtab_row = int(
            (anchor_y - self.rtab_origin[1]) / self.rtab_resolution)

        SEARCH_RADIUS = self.local_search_radius
        reachable = self.get_reachable_cells(nav_col, nav_row, max_cells=self.bfs_max_cells)
        self._publish_bfs_cells(reachable)

        # ── Window bounds in RTAB-Map cells ──────────────────────────────────
        rtab_row_min = max(1, anchor_rtab_row - SEARCH_RADIUS)
        rtab_row_max = min(self.rtab_height - 1, anchor_rtab_row + SEARCH_RADIUS)
        rtab_col_min = max(1, anchor_rtab_col - SEARCH_RADIUS)
        rtab_col_max = min(self.rtab_width - 1, anchor_rtab_col + SEARCH_RADIUS)

        if self.use_global_search_window:
            rtab_col_min = max(1, int(
                (self.global_x_min - self.rtab_origin[0]) / self.rtab_resolution))
            rtab_col_max = min(self.rtab_width - 1, int(
                (self.global_x_max - self.rtab_origin[0]) / self.rtab_resolution))
            rtab_row_min = max(1, int(
                (self.global_y_min - self.rtab_origin[1]) / self.rtab_resolution))
            rtab_row_max = min(self.rtab_height - 1, int(
                (self.global_y_max - self.rtab_origin[1]) / self.rtab_resolution))
            self._publish_search_area_world(
                self.global_x_min, self.global_x_max,
                self.global_y_min, self.global_y_max)
            self.get_logger().info(
                f'[SEARCH] GLOBAL x=[{self.global_x_min:.1f},{self.global_x_max:.1f}] '
                f'y=[{self.global_y_min:.1f},{self.global_y_max:.1f}]',
                throttle_duration_sec=5.0)
        else:
            self._publish_search_area(
                rtab_row_min, rtab_row_max, rtab_col_min, rtab_col_max)
            self.get_logger().info(
                f'[SEARCH] LOCAL radius={self.local_search_radius} cells',
                throttle_duration_sec=5.0)

        # ── Frontier cell detection on raw RTAB-Map ───────────────────────────
        # Pre-process: fill isolated/scattered unknown cells (50) that sit
        # within the free region.  Morphological closing on the free mask
        # absorbs small unknown gaps so they don't generate interior frontier
        # cells that chain the whole explored area into one giant cluster.
        # Kernel size controls how large a gap gets filled — tune as needed.
        free_mask = (self.rtab_map == 0).astype(np.uint8) * 255
        close_kernel = np.ones((5, 5), np.uint8)
        free_mask_closed = cv2.morphologyEx(
            free_mask, cv2.MORPH_CLOSE, close_kernel)

        # A free cell adjacent to an unknown cell (after gap-filling) with no
        # occupied neighbour is a frontier: the robot can see into unexplored space.
        frontier_cells = np.zeros(
            (self.rtab_height, self.rtab_width), dtype=np.uint8)
        for y in range(rtab_row_min, rtab_row_max):
            for x in range(rtab_col_min, rtab_col_max):
                if free_mask_closed[y, x] == 255:   # effectively free after closing
                    neighbors_raw = [
                        self.rtab_map[y - 1, x], self.rtab_map[y + 1, x],
                        self.rtab_map[y, x - 1], self.rtab_map[y, x + 1],
                    ]
                    neighbors_closed = [
                        free_mask_closed[y - 1, x], free_mask_closed[y + 1, x],
                        free_mask_closed[y, x - 1], free_mask_closed[y, x + 1],
                    ]
                    # Frontier only if a neighbour is unknown in the CLOSED map
                    # (i.e. a real boundary, not just a scattered interior gap)
                    # and no occupied cell is adjacent in the raw map.
                    if 0 in neighbors_closed and 100.0 not in neighbors_raw:
                        frontier_cells[y, x] = 255

        # If 4-connectivity still produces overly large clusters (thin bridges
        # chaining separate regions), uncomment the erosion below to break them:
        # kernel = np.ones((3, 3), np.uint8)
        # frontier_cells = cv2.erode(frontier_cells, kernel, iterations=1)

        output = cv2.connectedComponentsWithStats(
            frontier_cells, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        # ── Option 3: max cluster size + K-means subdivision ─────────────────
        # If a cluster is larger than max_cluster_cells pixels, split it into
        # sub-clusters using K-means.  Useful when the map is sparse and large
        # connected frontier regions should be treated as separate targets.
        # To enable: uncomment the block below and set max_cluster_cells.
        #
        # max_cluster_cells = 200   # pixels — tune to your map resolution
        # new_labels   = labels.copy()
        # new_centroids = list(centroids)
        # new_stats     = list(stats)
        # next_label    = numLabels
        # for lbl in range(1, numLabels):
        #     area = stats[lbl, cv2.CC_STAT_AREA]
        #     if area <= max_cluster_cells:
        #         continue
        #     # Extract pixel coordinates of this cluster
        #     pts = np.column_stack(np.where(labels == lbl)).astype(np.float32)
        #     k   = max(2, int(area // max_cluster_cells))
        #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        #                 10, 1.0)
        #     _, km_labels, km_centers = cv2.kmeans(
        #         pts, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        #     # Re-assign pixels to new sub-cluster labels
        #     new_labels[labels == lbl] = 0   # clear original label
        #     for sub in range(k):
        #         mask = (km_labels.flatten() == sub)
        #         sub_pts = pts[mask].astype(int)
        #         for r, c in sub_pts:
        #             new_labels[r, c] = next_label
        #         new_centroids.append(km_centers[sub][::-1])  # (col, row)
        #         new_stats.append(stats[lbl])                 # reuse parent stats
        #         next_label += 1
        # labels    = new_labels
        # centroids = np.array(new_centroids)
        # numLabels = next_label

        # ── Diagnostics: log per-tick stats so failures are easy to trace ─────
        rtab_free_cells = int(np.sum(self.rtab_map == 0))
        self.get_logger().info(
            f'[FRONTIER_DBG] rtab_free={rtab_free_cells}  '
            f'raw_clusters={numLabels - 1}  '
            f'bfs_reachable={len(reachable)}  '
            f'rtab_size={self.rtab_width}x{self.rtab_height}  '
            f'inflated_size={self.width}x{self.height}  '
            f'robot_navcell=({nav_col},{nav_row})',
            throttle_duration_sec=2.0)

        self._publish_all_frontiers(
            labels, numLabels,
            resolution=self.rtab_resolution, origin=self.rtab_origin)
        self._publish_frontier_labels(
            numLabels, centroids,
            resolution=self.rtab_resolution, origin=self.rtab_origin)

        # ── Score each cluster ────────────────────────────────────────────────
        cost_list = np.full(numLabels, np.inf)
        evaluated_centroids = []
        best_cost = np.inf
        current_best_world = None

        reject_area = reject_shape = reject_window = 0
        reject_edge = reject_bfs = reject_dist = reject_visited = 0

        # Candidates rejected only by distance filter — used as last-resort
        # fallback when nothing else passes (e.g. at startup with a tiny map).
        dist_fallback = []

        for i in range(1, numLabels):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]

            if area < self.min_frontier_area:
                reject_area += 1
                self.get_logger().info(
                    f'[F#{i}] REJECT  reason=too_small  '
                    f'area={area} < {self.min_frontier_area}')
                continue
            if min(stats[i, cv2.CC_STAT_WIDTH],
                   stats[i, cv2.CC_STAT_HEIGHT]) < self.min_frontier_dim:
                reject_shape += 1
                self.get_logger().info(
                    f'[F#{i}] REJECT  reason=thin_shape  '
                    f'w={stats[i, cv2.CC_STAT_WIDTH]} '
                    f'h={stats[i, cv2.CC_STAT_HEIGHT]} '
                    f'< {self.min_frontier_dim}')
                continue
            if not (rtab_col_min <= int(cX) <= rtab_col_max
                    and rtab_row_min <= int(cY) <= rtab_row_max):
                reject_window += 1
                self.get_logger().info(
                    f'[F#{i}] REJECT  reason=outside_search_window  '
                    f'centroid=({cX:.1f},{cY:.1f}) '
                    f'window_col=[{rtab_col_min},{rtab_col_max}] '
                    f'window_row=[{rtab_row_min},{rtab_row_max}]')
                continue
            if not (self.frontier_edge_margin <= int(cX) < self.rtab_width - self.frontier_edge_margin
                    and self.frontier_edge_margin <= int(cY) < self.rtab_height - self.frontier_edge_margin):
                reject_edge += 1
                self.get_logger().info(
                    f'[F#{i}] REJECT  reason=map_edge  '
                    f'centroid=({cX:.1f},{cY:.1f}) margin={self.frontier_edge_margin}')
                continue

            world_x = cX * self.rtab_resolution + self.rtab_origin[0]
            world_y = cY * self.rtab_resolution + self.rtab_origin[1]
            nav_cx = int((world_x - self.origin[0]) / self.resolution)
            nav_cy = int((world_y - self.origin[1]) / self.resolution)

            if (nav_cx, nav_cy) not in reachable:
                reject_bfs += 1
                self.get_logger().info(
                    f'[F#{i}] REJECT  reason=bfs_unreachable  '
                    f'world=({world_x:.2f},{world_y:.2f})  '
                    f'nav_cell=({nav_cx},{nav_cy})')
                continue

            capped_area = min(area, self.max_frontier_area)
            cost = self.frontier_cost(capped_area, nav_cx, nav_cy)
            robot_dist = math.hypot(
                world_x - self.robot_pose.x, world_y - self.robot_pose.y)

            is_visited = any(math.hypot(world_x - vx, world_y - vy)
                             < self.visited_frontier_radius_m
                             for vx, vy in self.visited_frontier_positions)

            if robot_dist < self.min_frontier_dist_m:
                reject_dist += 1
                if not is_visited:
                    dist_fallback.append((world_x, world_y, cost))
                    self.get_logger().info(
                        f'[F#{i}] REJECT  reason=too_close  '
                        f'dist={robot_dist:.2f}m < {self.min_frontier_dist_m}m  '
                        f'world=({world_x:.2f},{world_y:.2f})  added to fallback')
                else:
                    self.get_logger().info(
                        f'[F#{i}] REJECT  reason=too_close+visited  '
                        f'dist={robot_dist:.2f}m  world=({world_x:.2f},{world_y:.2f})  '
                        f'skipped from fallback')
                continue
            if is_visited:
                reject_visited += 1
                self.get_logger().info(
                    f'[F#{i}] REJECT  reason=already_visited  '
                    f'world=({world_x:.2f},{world_y:.2f})')
                continue

            cost_list[i] = cost
            evaluated_centroids.append((world_x, world_y))
            self.get_logger().info(
                f'[F#{i}] CANDIDATE  cost={cost:.2f}  area={area}  '
                f'dist={robot_dist:.2f}m  world=({world_x:.2f},{world_y:.2f})')
            if cost < best_cost:
                best_cost = cost
                current_best_world = (world_x, world_y)

        # If nothing passed the full filter but distance-only rejects exist,
        # use the best-scoring one so the robot does not get stuck at startup.
        if not evaluated_centroids and dist_fallback:
            dist_fallback.sort(key=lambda t: t[2])
            fbx, fby, _ = dist_fallback[0]
            evaluated_centroids.append((fbx, fby))
            current_best_world = (fbx, fby)
            best_cost = dist_fallback[0][2]
            self.get_logger().warn(
                f'[FRONTIER] All clusters within min_dist — '
                f'using nearest distance-relaxed fallback '
                f'({fbx:.2f}, {fby:.2f})')

        if numLabels > 1:
            self.get_logger().info(
                f'[FRONTIER_DBG] cluster rejections — '
                f'area:{reject_area} shape:{reject_shape} '
                f'window:{reject_window} edge:{reject_edge} '
                f'bfs:{reject_bfs} dist:{reject_dist} '
                f'visited:{reject_visited}  '
                f'passed:{len(evaluated_centroids)}',
                throttle_duration_sec=2.0)

        self._publish_frontier_evaluation(evaluated_centroids, current_best_world)

        # ── Handle no-frontier cases ──────────────────────────────────────────
        def _handle_no_frontier():
            # During the startup / map-build phase the RTAB map may have zero
            # free cells and detection always returns numLabels==1.  Do NOT
            # count those ticks toward the terminal exploration limit — only
            # start counting once the map is populated enough for exploration
            # (signalled by at least one frontier having been published).
            if not self.frontiers_ever_found:
                self.get_logger().info(
                    '[FRONTIER] Map still building at startup — '
                    'skipping terminal countdown until first frontier found.',
                    throttle_duration_sec=4.0)
                self.find_frontier = True
                return

            if self.use_global_search_window:
                self.max_radius_wait_count += 1
                self.get_logger().warn(
                    f'[GLOBAL] No reachable frontier — '
                    f'wait count {self.max_radius_wait_count}/'
                    f'{self.global_search_count_threshold}')
                if self.max_radius_wait_count >= self.global_search_count_threshold:
                    self._signal_exploration_complete()
                    return
                self.find_frontier = True
            else:
                prev = self.local_search_radius
                self.local_search_radius = min(
                    self.local_search_radius + 10, self.max_local_search_radius)
                if self.local_search_radius >= self.max_local_search_radius:
                    self.max_radius_wait_count += 1
                    self.get_logger().warn(
                        f'[LOCAL] Radius at max — '
                        f'wait count {self.max_radius_wait_count}/'
                        f'{self.local_search_count_threshold}')
                    if self.max_radius_wait_count >= self.local_search_count_threshold:
                        self._signal_exploration_complete()
                        return
                else:
                    self.get_logger().warn(
                        f'[LOCAL] Radius grew: {prev} → {self.local_search_radius}')
                self.find_frontier = True

        if numLabels <= 1:
            _handle_no_frontier()
            return

        # Use current_best_world if the fallback already selected a goal;
        # otherwise find the best from cost_list.
        if current_best_world is None:
            best_index = np.argmin(cost_list)
            if cost_list[best_index] == np.inf:
                _handle_no_frontier()
                return
            best_cX, best_cY = centroids[best_index]
            goal_x = best_cX * self.rtab_resolution + self.rtab_origin[0]
            goal_y = best_cY * self.rtab_resolution + self.rtab_origin[1]
            self.get_logger().info(
                f'[F#{best_index}] SELECTED  cost={cost_list[best_index]:.2f}  '
                f'world=({goal_x:.2f},{goal_y:.2f})')
        else:
            goal_x, goal_y = current_best_world
            self.get_logger().info(
                f'[SELECTED] world=({goal_x:.2f},{goal_y:.2f})  '
                f'cost={best_cost:.2f}')

        # ── Publish selected frontier goal ────────────────────────────────────
        self.get_logger().info(
            f'Frontier goal: ({goal_x:.2f}, {goal_y:.2f}), '
            f'candidates={len(evaluated_centroids)}')

        msg = PoseStamped()
        msg.header.frame_id = self.binary_map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = goal_x
        msg.pose.position.y = goal_y
        msg.pose.orientation.w = 1.0
        self.frontier_goal_pub.publish(msg)
        self.find_frontier = False
        self.frontiers_ever_found = True   # map is populated — start terminal countdown


def main(args=None):
    rclpy.init(args=args)
    node = FrontierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
