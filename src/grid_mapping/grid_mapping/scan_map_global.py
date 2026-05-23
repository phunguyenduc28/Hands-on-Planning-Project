import rclpy
from rclpy.node import Node
import numpy as np
import math

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
import tf2_ros


class GridMap:
    LMAX = 6.91
    LMIN = -6.91

    def __init__(self, center, cell_size=0.05, map_size=30):
        self.cell_size = cell_size
        self.grid = np.zeros((int(map_size / cell_size), int(map_size / cell_size)))
        self.origin = np.array(center) - np.array([map_size, map_size]) / 2
        self.height, self.width = self.grid.shape

    def position_to_cell(self, position):
        x_pos, y_pos = position
        x_cell = int(np.floor((x_pos - self.origin[0]) / self.cell_size))
        y_cell = int(np.floor((y_pos - self.origin[1]) / self.cell_size))
        return (x_cell, y_cell)

    def update_cell(self, uv, p):
        if p <= 0.0 or p >= 1.0:
            return
        x_cell, y_cell = uv
        if x_cell < 0 or x_cell >= self.width or y_cell < 0 or y_cell >= self.height:
            return
        l = np.log(p / (1.0 - p))
        self.grid[y_cell, x_cell] += l
        self.grid[y_cell, x_cell] = np.clip(
            self.grid[y_cell, x_cell], self.LMIN, self.LMAX)

    def add_ray(self, ray_init_position, ray_angle, ray_range, p_occ,
                mark_occupied=True):
        x_init, y_init = ray_init_position
        x_final = x_init + ray_range * np.cos(ray_angle)
        y_final = y_init + ray_range * np.sin(ray_angle)

        x_init_cell, y_init_cell = self.position_to_cell(ray_init_position)
        x_final_cell, y_final_cell = self.position_to_cell((x_final, y_final))

        points = list(self.bresenham(
            x_init_cell, y_init_cell, x_final_cell, y_final_cell))
        if len(points) == 0:
            return

        p_free = 0.35
        for pt in points[:-1]:
            self.update_cell(pt, p_free)
        if mark_occupied:
            self.update_cell(points[-1], p_occ)

    @staticmethod
    def bresenham(x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points

    def get_occupancy_grid_array(self):
        """Vectorised conversion. Unobserved cells → -1 (ROS unknown)."""
        result = np.full(self.grid.shape, -1, dtype=np.int8)
        observed = np.abs(self.grid) >= 0.01
        prob = 1.0 / (1.0 + np.exp(-self.grid))
        val = (prob * 100).astype(int)
        result[observed & (val > 60)] = 100
        result[observed & (val < 40)] = 0
        return result

    def get_inflated_grid(self, inflation_radius_m):
        grid_data = self.get_occupancy_grid_array()
        radius_cells = int(math.ceil(inflation_radius_m / self.cell_size))
        if radius_cells <= 0:
            return grid_data

        y, x = np.ogrid[-radius_cells:radius_cells + 1,
                        -radius_cells:radius_cells + 1]
        mask = x ** 2 + y ** 2 <= radius_cells ** 2

        inflated = grid_data.copy()
        rows, cols = np.where(grid_data == 100)
        for r, c in zip(rows, cols):
            r0 = max(0, r - radius_cells)
            r1 = min(self.height, r + radius_cells + 1)
            c0 = max(0, c - radius_cells)
            c1 = min(self.width, c + radius_cells + 1)
            mr0 = radius_cells - (r - r0)
            mr1 = mr0 + (r1 - r0)
            mc0 = radius_cells - (c - c0)
            mc1 = mc0 + (c1 - c0)
            inflated[r0:r1, c0:c1][mask[mr0:mr1, mc0:mc1]] = 100
        return inflated.astype(np.int8)

    def get_origin(self):
        return self.origin


class GlobalScanMapNode(Node):
    """Global occupancy grid built from the fake 2-D laser scan.

    Uses the same log-odds ray-casting and real-robot sign convention as the
    DWA local costmap (occupancy_grid_local.py), but the grid origin is fixed
    on the robot's first known pose and the log-odds are never cleared —
    every scan accumulates into a persistent global map.

    Parameters
    ----------
    map_size            total side length (m) of the square grid   [30.0]
    map_resolution      cell size (m)                               [0.05]
    map_frame           TF frame the map lives in                   [odom]
    laser_frame         TF frame of the fake scan source            [camera_link]
    scan_topic          LaserScan input topic                       [/turtlebot/fake_scan]
    odom_topic          Odometry for anchoring the grid origin      [/turtlebot/odom]
    p_occ               hit probability                             [0.85]
    inflation_radius    obstacle inflation radius (m)               [0.2]
    publish_rate        map publish rate (Hz)                       [2.0]
    range_min           ignore ranges below this (m)                [0.28]
    range_max           ignore ranges at or above this (m)          [2.0]
    clear_on_max_range  cast a free-space ray for max-range beams   [false]
    """

    def __init__(self):
        super().__init__('global_scan_map_node')

        self.declare_parameter('map_size',          30.0)
        self.declare_parameter('map_resolution',    0.05)
        self.declare_parameter('map_frame',         'odom')
        self.declare_parameter('laser_frame',       'camera_link')
        self.declare_parameter('scan_topic',        '/turtlebot/fake_scan')
        self.declare_parameter('odom_topic',        '/turtlebot/odom')
        self.declare_parameter('p_occ',             0.85)
        self.declare_parameter('inflation_radius',  0.2)
        self.declare_parameter('publish_rate',      2.0)
        self.declare_parameter('range_min',         0.28)
        self.declare_parameter('range_max',         2.0)
        self.declare_parameter('clear_on_max_range', True)

        self._map_size         = self.get_parameter('map_size').value
        self._resolution       = self.get_parameter('map_resolution').value
        self._map_frame        = self.get_parameter('map_frame').value
        self._laser_frame      = self.get_parameter('laser_frame').value
        self._scan_topic       = self.get_parameter('scan_topic').value
        self._odom_topic       = self.get_parameter('odom_topic').value
        self._p_occ            = self.get_parameter('p_occ').value
        self._inflation_radius = self.get_parameter('inflation_radius').value
        self._range_min        = self.get_parameter('range_min').value
        self._range_max        = self.get_parameter('range_max').value
        self._clear_max        = self.get_parameter('clear_on_max_range').value

        # Grid is created once the first odom arrives so it is centred on
        # the robot's starting position. Until then it is None.
        self._grid_map: GridMap | None = None
        self._origin_anchored = False

        self._latest_scan: LaserScan | None = None
        self._first_scan_logged = False
        self._first_tf_logged   = False
        self._scan_count        = 0

        self._tf_buffer   = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=30))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(Odometry, self._odom_topic,
                                 self._odom_cb, 10)
        self.create_subscription(LaserScan, self._scan_topic,
                                 self._scan_cb, 10)

        self._map_pub      = self.create_publisher(
            OccupancyGrid, '/map_scan',          10)
        self._inflated_pub = self.create_publisher(
            OccupancyGrid, '/inflated_map_scan', 10)

        publish_rate = self.get_parameter('publish_rate').value
        self.create_timer(1.0 / publish_rate, self._publish_cb)

        self.get_logger().info(
            f'GlobalScanMapNode ready  '
            f'map={self._map_size}m  res={self._resolution}m  '
            f'frame={self._map_frame}  laser={self._laser_frame}  '
            f'scan={self._scan_topic}  '
            f'range=[{self._range_min},{self._range_max}]m  '
            f'p_occ={self._p_occ}  inflation={self._inflation_radius}m')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        if self._origin_anchored:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self._grid_map = GridMap(
            center=[x, y],
            cell_size=self._resolution,
            map_size=self._map_size)
        self._origin_anchored = True
        self.get_logger().info(
            f'Grid anchored at ({x:.2f},{y:.2f})  '
            f'cells={self._grid_map.width}×{self._grid_map.height}')

    def _scan_cb(self, msg: LaserScan):
        if not self._origin_anchored:
            self.get_logger().warn(
                f'Scan arrived on {self._scan_topic} but grid origin not yet '
                'anchored — waiting for first odom message.',
                throttle_duration_sec=5.0)
            return
        if not self._first_scan_logged:
            self.get_logger().info(
                f'[SCAN] First scan received  '
                f'frame={msg.header.frame_id}  '
                f'beams={len(msg.ranges)}  '
                f'angle=[{math.degrees(msg.angle_min):.1f}°,'
                f'{math.degrees(msg.angle_max):.1f}°]  '
                f'range=[{msg.range_min:.2f},{msg.range_max:.2f}]m')
            self._first_scan_logged = True
        self._latest_scan = msg
        self._update_from_scan(msg)

    # ── Map update ────────────────────────────────────────────────────────────

    def _update_from_scan(self, scan: LaserScan):
        try:
            scan_time = rclpy.time.Time.from_msg(scan.header.stamp)
            tf_used_latest = False
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._map_frame, self._laser_frame, scan_time,
                    timeout=rclpy.duration.Duration(seconds=0.3))
            except tf2_ros.TransformException:
                tf = self._tf_buffer.lookup_transform(
                    self._map_frame, self._laser_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.3))
                tf_used_latest = True

            if tf_used_latest:
                self.get_logger().warn(
                    f'[TF] Exact timestamp lookup failed for '
                    f'{self._laser_frame}→{self._map_frame} — '
                    'using latest available transform (may be stale)',
                    throttle_duration_sec=5.0)

            tx  = tf.transform.translation.x
            ty  = tf.transform.translation.y
            q   = tf.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            if not self._first_tf_logged:
                self.get_logger().info(
                    f'[TF] First successful lookup  '
                    f'{self._laser_frame}→{self._map_frame}  '
                    f'pos=({tx:.3f},{ty:.3f})  yaw={math.degrees(yaw):.1f}°')
                self._first_tf_logged = True

            ranges = np.array(scan.ranges)
            angles = np.arange(
                scan.angle_min,
                scan.angle_max + scan.angle_increment,
                scan.angle_increment)[:len(ranges)]

            n_occupied  = 0
            n_free_only = 0
            n_skipped   = 0

            for r, a in zip(ranges, angles):
                is_max = not np.isfinite(r) or r >= self._range_max
                if is_max and not self._clear_max:
                    n_skipped += 1
                    continue
                eff_r = (scan.range_max * 0.9) if is_max else r
                if eff_r < self._range_min:
                    n_skipped += 1
                    continue
                # Real-robot sign convention (same as occupancy_grid_local.py)
                beam_angle = yaw + a
                self._grid_map.add_ray(
                    (tx, ty), beam_angle, eff_r, self._p_occ,
                    mark_occupied=not is_max)
                if is_max:
                    n_free_only += 1
                else:
                    n_occupied += 1

            self._scan_count += 1
            self.get_logger().info(
                f'[SCAN #{self._scan_count}] '
                f'occ={n_occupied}  free_only={n_free_only}  skipped={n_skipped}  '
                f'laser=({tx:.2f},{ty:.2f})  yaw={math.degrees(yaw):.1f}°',
                throttle_duration_sec=3.0)

        except tf2_ros.TransformException as e:
            self.get_logger().warn(
                f'[TF] Lookup failed ({self._laser_frame}→{self._map_frame}): {e}  '
                f'— check that the TF tree is being published',
                throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(
                f'Scan update error ({type(e).__name__}): {e}',
                throttle_duration_sec=5.0)

    # ── Publish ───────────────────────────────────────────────────────────────

    def _publish_cb(self):
        if not self._origin_anchored:
            self.get_logger().info(
                'Waiting for first odom to anchor grid origin...',
                throttle_duration_sec=5.0)
            return

        raw      = self._grid_map.get_occupancy_grid_array()
        inflated = self._grid_map.get_inflated_grid(self._inflation_radius)
        # stamp=0 tells RViz to use the latest available TF rather than a specific
        # timestamp — avoids clock-mismatch failures when robot and laptop clocks differ.
        now      = self.get_clock().now().to_msg()   # laptop clock — mismatches robot TF
        # now      = rclpy.time.Time(seconds=0).to_msg()
        origin   = self._grid_map.get_origin()

        total = raw.size
        n_occ  = int(np.sum(raw == 100))
        n_free = int(np.sum(raw == 0))
        n_unk  = int(np.sum(raw == -1))
        n_inf  = int(np.sum(inflated == 100)) - n_occ
        self.get_logger().info(
            f'[MAP] scans={self._scan_count}  '
            f'free={n_free} ({100.0 * n_free / total:.1f}%)  '
            f'occ={n_occ}  inflated_extra={n_inf}  unknown={n_unk}  '
            f'origin=({origin[0]:.2f},{origin[1]:.2f})',
            throttle_duration_sec=10.0)

        def _make_msg(data: np.ndarray) -> OccupancyGrid:
            m = OccupancyGrid()
            m.header.stamp              = now
            m.header.frame_id           = self._map_frame
            m.info.resolution           = self._resolution
            m.info.width                = self._grid_map.width
            m.info.height               = self._grid_map.height
            m.info.origin.position.x    = float(origin[0])
            m.info.origin.position.y    = float(origin[1])
            m.info.origin.orientation.w = 1.0
            m.data = data.flatten().tolist()
            return m

        self._map_pub.publish(_make_msg(raw))
        self._inflated_pub.publish(_make_msg(inflated))


def main(args=None):
    rclpy.init(args=args)
    node = GlobalScanMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
