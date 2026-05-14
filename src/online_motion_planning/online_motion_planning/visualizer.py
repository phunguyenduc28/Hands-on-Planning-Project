"""RViz visualization helper for frontier exploration and path planning."""

import rclpy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class RVizVisualizer:
    """Encapsulates all RViz marker publishing for frontiers, paths, and trees."""

    _FRONTIER_COLORS = [
        (1.0, 1.0, 0.0),   # yellow
        (1.0, 0.5, 0.0),   # orange
        (1.0, 0.0, 1.0),   # magenta
        (0.0, 1.0, 1.0),   # cyan
        (1.0, 1.0, 1.0),   # white
        (0.5, 1.0, 0.0),   # lime
    ]

    def __init__(self, frontier_all_pub, bfs_cells_pub, frontier_eval_pub,
                 search_area_pub, rrt_tree_a_pub, rrt_tree_b_pub, dwa_traj_pub,
                 marker_pub, logger):
        """Initialize visualizer with publisher references.
        
        Args:
            frontier_all_pub: Publisher for frontier clusters
            bfs_cells_pub: Publisher for BFS reachable cells
            frontier_eval_pub: Publisher for evaluated centroids
            search_area_pub: Publisher for search window rectangles
            rrt_tree_a_pub: Publisher for BiRRT* tree A
            rrt_tree_b_pub: Publisher for BiRRT* tree B
            dwa_traj_pub: Publisher for DWA trajectories
            marker_pub: Publisher for waypoint paths
            logger: ROS logger instance
        """
        self.frontier_all_pub = frontier_all_pub
        self.bfs_cells_pub = bfs_cells_pub
        self.frontier_eval_pub = frontier_eval_pub
        self.search_area_pub = search_area_pub
        self.rrt_tree_a_pub = rrt_tree_a_pub
        self.rrt_tree_b_pub = rrt_tree_b_pub
        self.dwa_traj_pub = dwa_traj_pub
        self.marker_pub = marker_pub
        self.logger = logger

    def publish_all_frontiers(self, labels, numLabels, binary_map_frame, now,
                              resolution=None, origin=None, rtab_resolution=None,
                              rtab_origin=None):
        """Publish frontier clusters as colored POINTS markers.
        Each cluster is one marker with all its frontier cells colored distinctly."""
        if resolution is None and rtab_resolution is not None:
            resolution = rtab_resolution
        if origin is None and rtab_origin is not None:
            origin = rtab_origin

        import numpy as np
        
        marker_array = MarkerArray()

        clear = Marker()
        clear.header.frame_id = binary_map_frame
        clear.header.stamp = now
        clear.ns = "all_frontiers"
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        # Extract frontier cells for each cluster
        for i in range(1, numLabels):
            rows, cols = np.where(labels == i)
            if len(rows) == 0:
                continue

            color_idx = (i - 1) % len(self._FRONTIER_COLORS)
            r, g, b = self._FRONTIER_COLORS[color_idx]
            
            m = Marker()
            m.header.frame_id = binary_map_frame
            m.header.stamp = now
            m.ns = "all_frontiers"
            m.id = i
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = resolution
            m.scale.y = resolution
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.85

            m.pose.orientation.w = 1.0
            m.lifetime = rclpy.duration.Duration(seconds=4).to_msg()

            # Add all frontier cells from this cluster
            for row, col in zip(rows, cols):
                p = Point()
                p.x = float(col * resolution + origin[0])
                p.y = float(row * resolution + origin[1])
                p.z = 0.05
                m.points.append(p)

            marker_array.markers.append(m)

        self.frontier_all_pub.publish(marker_array)

    def publish_bfs_cells(self, reachable, binary_map_frame, now, resolution, origin):
        """Publish BFS-reachable cells as semi-transparent green POINTS."""
        m = Marker()
        m.header.frame_id = binary_map_frame
        m.header.stamp = now
        m.ns = "bfs_cells"
        m.id = 0
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = resolution * 0.7
        m.scale.y = resolution * 0.7
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
            p.x = float(col * resolution + origin[0])
            p.y = float(row * resolution + origin[1])
            p.z = 0.0
            m.points.append(p)

        self.bfs_cells_pub.publish(m)

    def publish_frontier_evaluation(self, evaluated_centroids, best_world,
                                   binary_map_frame, now):
        """Publish evaluated centroids (cyan) and best frontier (green)."""
        marker_array = MarkerArray()

        clear = Marker()
        clear.header.frame_id = binary_map_frame
        clear.header.stamp = now
        clear.ns = "evaluated_centroids"
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        clear_best = Marker()
        clear_best.header.frame_id = binary_map_frame
        clear_best.header.stamp = now
        clear_best.ns = "best_frontier"
        clear_best.action = Marker.DELETEALL
        marker_array.markers.append(clear_best)

        lifetime = rclpy.duration.Duration(seconds=4).to_msg()

        for i, (wx, wy) in enumerate(evaluated_centroids):
            m = Marker()
            m.header.frame_id = binary_map_frame
            m.header.stamp = now
            m.ns = "evaluated_centroids"
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
            bm.header.frame_id = binary_map_frame
            bm.header.stamp = now
            bm.ns = "best_frontier"
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

    def publish_search_area(self, rtab_row_min, rtab_row_max, rtab_col_min, rtab_col_max,
                           binary_map_frame, now, rtab_resolution, rtab_origin):
        """Publish local search window as white LINE_STRIP rectangle."""
        x_min = rtab_col_min * rtab_resolution + rtab_origin[0]
        x_max = rtab_col_max * rtab_resolution + rtab_origin[0]
        y_min = rtab_row_min * rtab_resolution + rtab_origin[1]
        y_max = rtab_row_max * rtab_resolution + rtab_origin[1]

        m = Marker()
        m.header.frame_id = binary_map_frame
        m.header.stamp = now
        m.ns = "search_area"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.06
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 0.9
        m.pose.orientation.w = 1.0
        m.lifetime = rclpy.duration.Duration(seconds=4).to_msg()

        corners = [
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max),
            (x_min, y_min),
        ]
        for cx, cy in corners:
            p = Point()
            p.x = float(cx)
            p.y = float(cy)
            p.z = 0.05
            m.points.append(p)

        self.search_area_pub.publish(m)

    def publish_search_area_world(self, x_min, x_max, y_min, y_max,
                                 binary_map_frame, now):
        """Publish global search window as cyan LINE_STRIP rectangle."""
        m = Marker()
        m.header.frame_id = binary_map_frame
        m.header.stamp = now
        m.ns = "search_area"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.06
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 1.0  # cyan
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

    def make_rrt_viz_callback(self, publisher, resolution, origin,
                             publish_every=10, color_rgb=(0.2, 0.6, 1.0),
                             z_height=0.08, binary_map_frame="map", now=None):
        """Create a visualization callback for BiRRT* tree growth."""
        clear = Marker()
        clear.header.frame_id = binary_map_frame
        clear.header.stamp = now if now else self.logger.get_clock().now().to_msg() if hasattr(self.logger, 'get_clock') else None
        clear.ns = "rrt_tree"
        clear.id = 0
        clear.action = Marker.DELETEALL
        publisher.publish(clear)

        edge_points = []
        count = [0]
        r, g, b = color_rgb

        def _publish_marker():
            if not edge_points:
                return
            m = Marker()
            m.header.frame_id = binary_map_frame
            m.header.stamp = now if now else self.logger.get_clock().now().to_msg() if hasattr(self.logger, 'get_clock') else None
            m.ns = "rrt_tree"
            m.id = 0
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.025
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.6
            m.pose.orientation.w = 1.0
            m.lifetime = rclpy.duration.Duration(seconds=15).to_msg()
            for x, y in edge_points:
                pt = Point()
                pt.x = float(x)
                pt.y = float(y)
                pt.z = z_height
                m.points.append(pt)
            publisher.publish(m)

        def callback(G, parent_idx, child_idx):
            parent = G[parent_idx]
            child = G[child_idx]
            edge_points.append((float(parent.x * resolution + origin[0]),
                               float(parent.y * resolution + origin[1])))
            edge_points.append((float(child.x * resolution + origin[0]),
                               float(child.y * resolution + origin[1])))
            count[0] += 1
            if count[0] % publish_every == 0:
                _publish_marker()

        def flush():
            _publish_marker()

        return callback, flush

    def publish_dwa_paths(self, paths, best_v, best_w, dwa_viz_time, binary_map_frame, now):
        """Publish DWA candidate and best trajectories.

        All trajectories (including the best) are re-simulated with dwa_viz_time
        so the visualised paths are longer than the planning horizon — making them
        easy to see in RViz regardless of how close the next waypoint is.

        Visual scheme:
          Best trajectory  — thick BRIGHT YELLOW (0.1 wide, z=0.3, fully opaque)
          Candidate paths  — thin  LIGHT GREY    (0.02 wide, z=0.05, a=0.35)
        """
        if not hasattr(self, 'local_planner') or self.local_planner is None:
            return
        if self.local_planner.robot_pose is None:
            return

        marker_array = MarkerArray()
        lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()

        # First draw all candidates (low z), then the best (high z) so it's always on top
        candidate_id = 0
        for p in paths[::3]:                           # downsample candidates
            is_best = abs(p['v'] - best_v) < 1e-3 and abs(p['w'] - best_w) < 1e-3
            if is_best:
                continue                                # drawn separately below
            viz_traj = self.local_planner.simulate_trajectory(p['v'], p['w'],
                                                              predict_time=dwa_viz_time)
            m = Marker()
            m.header.frame_id = binary_map_frame
            m.header.stamp    = now
            m.ns       = "dwa_candidates"
            m.id       = candidate_id
            m.type     = Marker.LINE_STRIP
            m.action   = Marker.ADD
            m.scale.x  = 0.02
            m.color    = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.35)   # light grey
            m.pose.orientation.w = 1.0
            m.lifetime = lifetime
            for x, y, _ in viz_traj:
                pt = Point(); pt.x = float(x); pt.y = float(y); pt.z = 0.05
                m.points.append(pt)
            marker_array.markers.append(m)
            candidate_id += 1

        # Best trajectory — drawn last so it renders on top.
        # Always re-simulate using best_v/best_w directly (never search paths[::3])
        # so the yellow marker is published on every tick regardless of downsampling.
        m = Marker()
        m.header.frame_id = binary_map_frame
        m.header.stamp    = now
        m.ns       = "dwa_best"
        m.id       = 0                       # single marker, always replaces previous
        m.type     = Marker.LINE_STRIP
        m.action   = Marker.ADD
        m.scale.x  = 0.10                   # thick so it stands out clearly
        m.color    = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)   # bright yellow
        m.pose.orientation.w = 1.0
        m.lifetime = lifetime
        viz_traj = self.local_planner.simulate_trajectory(best_v, best_w,
                                                          predict_time=dwa_viz_time)
        for x, y, _ in viz_traj:
            pt = Point(); pt.x = float(x); pt.y = float(y); pt.z = 0.30
            m.points.append(pt)
        marker_array.markers.append(m)

        self.dwa_traj_pub.publish(marker_array)

    def publish_waypoints(self, positions, binary_map_frame, now):
        """Publish waypoint path as line strip + spheres."""
        marker_array = MarkerArray()

        if len(positions) > 1:
            line_marker = Marker()
            line_marker.header.frame_id = binary_map_frame
            line_marker.header.stamp = now
            line_marker.ns = "links"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.05
            line_marker.color.r = 0.0
            line_marker.color.g = 0.5
            line_marker.color.b = 1.0
            line_marker.color.a = 0.8
            line_marker.pose.orientation.w = 1.0

            for (x, y) in positions:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                line_marker.points.append(p)

            marker_array.markers.append(line_marker)

        for i, (x, y) in enumerate(positions):
            marker = Marker()
            marker.header.frame_id = binary_map_frame
            marker.header.stamp = now
            marker.ns = "positions"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.01
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)
