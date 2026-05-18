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
    
    def __init__(self, center, cell_size=0.1, map_size=20):
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
            self.grid[y_cell, x_cell], self.LMIN, self.LMAX
        )
    
    def add_ray(self, ray_init_position, ray_angle, ray_range, p_occ, mark_occupied=True):
        x_init, y_init = ray_init_position
        x_final = x_init + ray_range * np.cos(ray_angle)
        y_final = y_init + ray_range * np.sin(ray_angle)
        
        x_init_cell, y_init_cell = self.position_to_cell(ray_init_position)
        x_final_cell, y_final_cell = self.position_to_cell((x_final, y_final))
        
        points = list(self.bresenham(x_init_cell, y_init_cell, x_final_cell, y_final_cell))
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
    
    def log_odds_to_probability(self, log_odds):
        if abs(log_odds) < 0.01:
            return 50
        
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        val = int(prob * 100)
        
        if val > 60: return 100
        if val < 40: return 0
        return 50
    
    def get_occupancy_grid_array(self):
        grid_prob = np.zeros_like(self.grid)
        for i in range(self.height):
            for j in range(self.width):
                grid_prob[i, j] = self.log_odds_to_probability(self.grid[i, j])
        return grid_prob.astype(np.int8)
    
    def get_origin(self):
        return self.origin

    def get_inflated_grid(self, inflation_radius_m, cost_scaling_factor=1.0):
        grid_data = self.get_occupancy_grid_array()

        radius_cells = int(math.ceil(inflation_radius_m / self.cell_size))
        if radius_cells <= 0:
            return grid_data

        # Pure gradient inflation: actual obstacle stays at 100 (hard reject).
        # Cells within inflation_radius get a cost decreasing from 99 (closest)
        # to 1 (at the boundary), giving DWA a smooth repulsion field.
        # The gradient is steep enough near robot_radius that DWA strongly avoids
        # those cells while still allowing passage through tight gaps if necessary.
        # cost_scaling_factor > 1 keeps costs high further from the wall.
        inflated_grid = grid_data.copy().astype(np.int16)
        rows, cols = np.where(grid_data == 100)

        if len(rows) == 0:
            return inflated_grid.astype(np.int8)

        # Iterate outer → inner so inner rings overwrite outer rings.
        for d in range(radius_cells, 0, -1):
            linear = 1.0 - float(d) / radius_cells
            cost = max(1, round(99 * (linear ** (1.0 / cost_scaling_factor))))

            y, x = np.ogrid[-d:d + 1, -d:d + 1]
            mask = x**2 + y**2 <= d**2

            for r, c in zip(rows, cols):
                r_start = max(0, r - d)
                r_end   = min(self.height, r + d + 1)
                c_start = max(0, c - d)
                c_end   = min(self.width,  c + d + 1)

                m_r_start = d - (r - r_start)
                m_r_end   = m_r_start + (r_end - r_start)
                m_c_start = d - (c - c_start)
                m_c_end   = m_c_start + (c_end - c_start)

                slc    = mask[m_r_start:m_r_end, m_c_start:m_c_end]
                region = inflated_grid[r_start:r_end, c_start:c_end]
                region[slc & (region < cost)] = cost

        return np.clip(inflated_grid, -128, 127).astype(np.int8)


class OccupancyGridNode(Node):
    def __init__(self):
        super().__init__('occupancy_grid_node')
        
        self.declare_parameter('grid_size', 3.0)
        self.declare_parameter('grid_resolution', 0.05)
        self.declare_parameter('map_frame', 'world_enu')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('laser_frame', 'turtlebot/rplidar')
        self.declare_parameter('p_occ', 0.9)
        self.declare_parameter('inflation_radius', 0.25)
        self.declare_parameter('clear_on_max_range', True)
        # cost_scaling_factor controls how aggressively the DWA is repelled from walls.
        # cost = 99 * (1 - d/radius)^(1/factor)
        #   factor=1.0 → linear (default)
        #   factor>1.0 → costs stay HIGH further from the wall (more rejection)
        #   factor<1.0 → costs drop quickly near wall, gentle at boundary
        self.declare_parameter('cost_scaling_factor', 3.0)

        grid_size             = self.get_parameter('grid_size').value
        self.grid_resolution  = self.get_parameter('grid_resolution').value
        self.map_frame        = self.get_parameter('map_frame').value
        self.base_frame       = self.get_parameter('base_frame').value
        self.laser_frame      = self.get_parameter('laser_frame').value
        self.p_occ            = self.get_parameter('p_occ').value
        self.inflation_radius = self.get_parameter('inflation_radius').value
        self.clear_on_max_range    = self.get_parameter('clear_on_max_range').value
        self.cost_scaling_factor   = self.get_parameter('cost_scaling_factor').value

        self.grid_map  = GridMap(center=[0.0, 0.0], cell_size=self.grid_resolution, map_size=grid_size)
        self.grid_size = grid_size
        
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0
        self.latest_scan = None
        self.tf_ok       = False   # True only when TF is fresh and valid
        
        self.tf_buffer   = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.create_subscription(Odometry,   '/turtlebot/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan,  '/turtlebot/scan', self.scan_callback, 10)
        
        self.map_pub          = self.create_publisher(OccupancyGrid, '/map',          10)
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, '/inflated_map', 10)
        
        self.timer = self.create_timer(0.1, self.timer_callback)  # 5 Hz

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        t3 = 2.0 * (qw * qz + qx * qy)
        t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(t3, t4)

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_theta = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def scan_callback(self, msg):
        # Store only — do NOT process here.
        # Processing (grid clear + ray tracing + gradient inflation) is slow in
        # Python.  If we process in scan_callback the executor queues up multiple
        # scans and falls further behind each cycle, causing 1-2 s stale scans.
        # The timer_callback processes the latest stored scan at a fixed rate and
        # discards anything older than 200 ms.
        self.latest_scan = msg

    def timer_callback(self):
        if self.latest_scan is None:
            return

        scan_age_ms = (
            self.get_clock().now() -
            rclpy.time.Time.from_msg(self.latest_scan.header.stamp)
        ).nanoseconds / 1e6

        if scan_age_ms > 200.0:
            self.get_logger().warn(
                f'[MAP] Discarding scan that is {scan_age_ms:.0f}ms old — '
                f'gradient inflation is too slow for this rate. '
                f'Consider reducing inflation_radius or grid_size.'
            )
            self.latest_scan = None
            return

        self.grid_map.origin = np.array([
            self.robot_x - self.grid_size / 2,
            self.robot_y - self.grid_size / 2
        ])
        self.grid_map.grid.fill(0.0)
        self.update_grid_from_scan(self.latest_scan)
        self.publish_occupancy_grid()
        self.latest_scan = None

    def update_grid_from_scan(self, scan_msg):
        try:
            scan_time = rclpy.time.Time.from_msg(scan_msg.header.stamp)
            now_time  = self.get_clock().now()
            scan_age_ms = (now_time - scan_time).nanoseconds / 1e6

            tf_source = 'scan_timestamp'
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.laser_frame,
                    scan_time,
                    timeout=rclpy.duration.Duration(seconds=0.5)  # was 0.5 — long block
                )
            except tf2_ros.TransformException as e:
                tf_source = 'current_time_FALLBACK'
                self.get_logger().warn(
                    f'[MAP] TF at scan_time failed ({e}), falling back to current time.'
                    f'  scan_age={scan_age_ms:.1f}ms'
                )
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.laser_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )

            robot_x_map = transform.transform.translation.x
            robot_y_map = transform.transform.translation.y

            q = transform.transform.rotation
            laser_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

            # Log any significant mismatch between TF position and odom position.
            # A large delta means the map origin and scan projection are inconsistent,
            # which causes the visual "snap" between frames.
            odom_tf_dx = robot_x_map - self.robot_x
            odom_tf_dy = robot_y_map - self.robot_y
            odom_tf_dist = math.hypot(odom_tf_dx, odom_tf_dy)
            if odom_tf_dist > 0.05:
                self.get_logger().warn(
                    f'[MAP] TF/odom mismatch: tf=({robot_x_map:.3f},{robot_y_map:.3f})'
                    f'  odom=({self.robot_x:.3f},{self.robot_y:.3f})'
                    f'  delta={odom_tf_dist:.3f}m'
                    f'  tf_source={tf_source}'
                    f'  scan_age={scan_age_ms:.1f}ms'
                )

            min_range = 0.15
            max_range = 12.0

            ranges = np.array(scan_msg.ranges)

            angles_laser = np.arange(
                scan_msg.angle_min,
                scan_msg.angle_max + scan_msg.angle_increment,
                scan_msg.angle_increment
            )

            for range_val, angle_laser in zip(ranges, angles_laser):

                is_max_range = not np.isfinite(range_val) or range_val >= max_range

                if is_max_range and not self.clear_on_max_range:
                    continue

                effective_range = scan_msg.range_max if is_max_range else range_val

                if effective_range < min_range:
                    continue

            
                beam_angle = laser_yaw - angle_laser

                if is_max_range:
                    effective_range *= 0.9

                self.grid_map.add_ray(
                    (robot_x_map, robot_y_map),
                    beam_angle,
                    effective_range,
                    self.p_occ,
                    mark_occupied=not is_max_range
                )

        except Exception as ex:
            self.get_logger().error(str(ex))

    def publish_occupancy_grid(self):
        raw_data      = self.grid_map.get_occupancy_grid_array()
        inflated_data = self.grid_map.get_inflated_grid(self.inflation_radius, self.cost_scaling_factor)

        now    = self.get_clock().now().to_msg()
        origin = self.grid_map.get_origin()

        def create_msg(data_array):
            m = OccupancyGrid()
            m.header.stamp    = now
            m.header.frame_id = self.map_frame
            m.info.resolution = self.grid_map.cell_size
            m.info.width      = self.grid_map.width
            m.info.height     = self.grid_map.height
            m.info.origin.position.x   = origin[0]
            m.info.origin.position.y   = origin[1]
            m.info.origin.orientation.w = 1.0
            m.data = data_array.flatten().tolist()
            return m

        self.map_pub.publish(create_msg(raw_data))
        self.inflated_map_pub.publish(create_msg(inflated_data))


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()