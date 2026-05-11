#!/usr/bin/env python3
"""
Occupancy Grid Mapping Node
Subscribes to /odom and /scan, builds probabilistic occupancy grid,
publishes nav_msgs/OccupancyGrid for visualization in RViz.
Supports both live ROS topics and offline CSV file playback.
"""
import rclpy
from rclpy.node import Node
import numpy as np
import math

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
import tf2_ros

from lab1_1.load_csv_files import wrap_angle


class GridMap:
    """
    Probabilistic occupancy grid map using log-odds representation.
    
    Log-odds: positive values = likely occupied, negative = likely free
    """
    
    LMAX = 6.91   # Log-odds max (equivalent to p=0.999)
    LMIN = -6.91  # Log-odds min (equivalent to p=0.001)
    
    def __init__(self, center, cell_size=0.1, map_size=20):
        """
        Initialize grid map.
        
        Args:
            center: [x, y] center position
            cell_size: resolution in meters
            map_size: total size (square) in meters
        """
        self.cell_size = cell_size
        self.grid = np.zeros((int(map_size / cell_size), int(map_size / cell_size)))
        self.origin = np.array(center) - np.array([map_size, map_size]) / 2
        self.height, self.width = self.grid.shape
    
    def position_to_cell(self, position):
        """Convert world coordinates to grid cell indices."""
        x_pos, y_pos = position
        x_cell = int(np.floor((x_pos - self.origin[0]) / self.cell_size))
        y_cell = int(np.floor((y_pos - self.origin[1]) / self.cell_size))
        return (x_cell, y_cell)
    
    def update_cell(self, uv, p):
        """Update single cell with occupancy probability p."""
        if p <= 0.0 or p >= 1.0:
            return
        
        x_cell, y_cell = uv
        if x_cell < 0 or x_cell >= self.width or y_cell < 0 or y_cell >= self.height:
            return
        
        # Log-odds update
        l = np.log(p / (1.0 - p))
        self.grid[y_cell, x_cell] += l
        
        # Saturate
        self.grid[y_cell, x_cell] = np.clip(
            self.grid[y_cell, x_cell], self.LMIN, self.LMAX
        )
    
    def add_ray(self, ray_init_position, ray_angle, ray_range, p_occ, mark_occupied=True):
        """
        Update grid cells along a laser ray.
        Marks free space along ray, occupied at endpoint.
        """
        x_init, y_init = ray_init_position
        x_final = x_init + ray_range * np.cos(ray_angle)
        y_final = y_init + ray_range * np.sin(ray_angle)
        
        x_init_cell, y_init_cell = self.position_to_cell(ray_init_position)
        x_final_cell, y_final_cell = self.position_to_cell((x_final, y_final))
        
        # Bresenham line
        points = list(self.bresenham(x_init_cell, y_init_cell, x_final_cell, y_final_cell))
        if len(points) == 0:
            return
        
        # Mark cells along ray as free
        for pt in points[:-1]:
            self.update_cell(pt, 1.0 - p_occ)
        
        # Mark endpoint as occupied
        if mark_occupied:
            self.update_cell(points[-1], p_occ)
        else:
            # If it was max range, the last point is also free
            self.update_cell(points[-1], 1 - p_occ)
    
    @staticmethod
    def bresenham(x0, y0, x1, y1):
        """Bresenham's line algorithm."""
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
        """Convert log-odds to occupancy probability [0-100] or -1 for unknown."""
        # Convert log-odds to probability using sigmoid
        # p = 1 - (1 / (1 + exp(L)))
        prob = 1.0 - (1.0 / (1.0 + np.exp(log_odds)))
        
        # Clamp probability to 0-100 range becauase ROS OccupancyGrid uses int8 with -1 for unknown, 0 for free, and 100 for occupied
        return int(np.clip(prob * 100, 0, 100))
    
    def get_occupancy_grid_array(self):
        """Get occupancy grid as array for OccupancyGrid message."""
        grid_prob = np.zeros_like(self.grid)
        for i in range(self.height):
            for j in range(self.width):
                grid_prob[i, j] = self.log_odds_to_probability(self.grid[i, j])
        return grid_prob.astype(np.int8)
    
    def get_origin(self):
        """Get grid origin."""
        return self.origin


class OccupancyGridNode(Node):
    def __init__(self):
        super().__init__('occupancy_grid_node')
        
        # Declare parameters
        self.declare_parameter('grid_size', 20.0)
        self.declare_parameter('grid_resolution', 0.05)
        # self.declare_parameter('map_frame', 'world_ned')
        # self.declare_parameter('map_frame', 'odom') #<--- For rosbag file
        self.declare_parameter('map_frame', 'world_enu')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('laser_frame', 'turtlebot/rplidar')
        # self.declare_parameter('laser_frame', 'rplidar') #<--- For rosbag file

        self.declare_parameter('p_occ', 0.9)
        
        # Get parameters
        grid_size = self.get_parameter('grid_size').value
        self.grid_resolution = self.get_parameter('grid_resolution').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.laser_frame = self.get_parameter('laser_frame').value
        p_occ = self.get_parameter('p_occ').value
        self.p_occ = p_occ
        
        # Initialize GridMap (centered at origin)
        self.grid_map = GridMap(center=[0.0, 0.0], cell_size=self.grid_resolution, map_size=grid_size)
        self.grid_size = grid_size
        
        # State
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.latest_scan = None
        self.scan_received = False
        
        # TF buffer and listener
        # Increased cache time to 30 seconds for bag file playback
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/turtlebot/odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/turtlebot/scan', self.scan_callback, 10
        )
        
        # Publisher for occupancy grid
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        
        # Timer to publish map periodically
        self.timer = self.create_timer(0.5, self.timer_callback)
        
        self.get_logger().info(
            f"Occupancy Grid Node Started. Grid: {grid_size}x{grid_size}m @ {self.grid_resolution}m/cell"
        )
        self.get_logger().info(
            f"Frame configuration: map_frame='{self.map_frame}', base_frame='{self.base_frame}', laser_frame='{self.laser_frame}'"
        )
    
    def quaternion_to_yaw(self, qx, qy, qz, qw):
        """Convert quaternion to yaw angle."""
        t3 = 2.0 * (qw * qz + qx * qy)
        t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(t3, t4)
    
    def odom_callback(self, msg):
        """Update robot pose from odometry."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.robot_theta = self.quaternion_to_yaw(qx, qy, qz, qw)
    
    def scan_callback(self, msg):
        """Store latest scan for processing."""
        self.latest_scan = msg
        self.scan_received = True
        # self.update_grid_from_scan(msg)
        self.get_logger().debug(f"Scan received: {len(msg.ranges)} ranges from frame '{msg.header.frame_id}'")
    
    def timer_callback(self):
        """Periodically update and publish occupancy grid."""
        # Update grid from latest scan if available
        if self.scan_received and self.latest_scan is not None:
            self.scan_received = False
            self.update_grid_from_scan(self.latest_scan)
            # Log grid statistics
            grid_min = np.min(self.grid_map.grid)
            grid_max = np.max(self.grid_map.grid)
            grid_mean = np.mean(self.grid_map.grid)
            self.get_logger().info(f"Grid updated. Stats - Min: {grid_min:.2f}, Max: {grid_max:.2f}, Mean: {grid_mean:.2f}")
        
        # Publish occupancy grid
        self.publish_occupancy_grid()
    
    def update_grid_from_scan(self, scan_msg):
        """Update grid map from laser scan using ray-casting."""
        try:
            # Get transform from laser frame to map frame
            # Use scan message timestamp (important for bag file playback)
            scan_time = rclpy.time.Time.from_msg(scan_msg.header.stamp)
            self.get_logger().debug(f"Looking up transform from '{self.laser_frame}' to '{self.map_frame}' at time {scan_time.nanoseconds}")
            
            try:
                # Try exact timestamp first (for precise synchronization)
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.laser_frame,
                    scan_time,
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
            except tf2_ros.TransformException:
                # Fallback: use latest available transform (helps with timing issues)
                self.get_logger().debug(f"Exact timestamp lookup failed, using latest available transform")
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.laser_frame,
                    rclpy.time.Time(),  # Latest time
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
            
            # Robot position (same as laser in 2D environment) in map frame
            robot_x_map = transform.transform.translation.x
            robot_y_map = transform.transform.translation.y
            
            # Laser frame's yaw relative to map
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            laser_yaw = self.quaternion_to_yaw(qx, qy, qz, qw)
            
            self.get_logger().debug(f"Laser at ({robot_x_map:.2f}, {robot_y_map:.2f}), yaw={laser_yaw:.2f}")
            
            # Process each beam
            min_range = 0.15
            max_range = 12.0
            
            ranges = np.array(scan_msg.ranges)

            # No array of angles is provided in the LaserScan message, so we need to compute it based on angle_min, angle_max, and angle_increment
            angles_laser = np.arange(
                scan_msg.angle_min,
                scan_msg.angle_max + scan_msg.angle_increment,
                scan_msg.angle_increment
            )
            
            valid_rays = 0
            for i, (range_val, angle_laser) in enumerate(zip(ranges, angles_laser)):
                # If range is infinite, we still want to clear the space along that ray
                is_max_range = not np.isfinite(range_val) or range_val >= max_range
                
                
                if is_max_range:
                    effective_range = scan_msg.range_max # Or a slightly smaller value
                else:
                    effective_range = range_val

                if effective_range < min_range:
                    continue
                
                # beam_angle = laser_yaw + angle_laser

                # if map frame is world_enu
                beam_angle = laser_yaw - angle_laser
                
                # If it's a max-range hit, only clear the path, don't mark an obstacle
                # if is_max_range:
                    # pass 
                # else:
                self.grid_map.add_ray(
                    (robot_x_map, robot_y_map),
                    beam_angle,
                    effective_range,
                    self.p_occ,
                    mark_occupied=not is_max_range
                )
            
            self.get_logger().info(f"Processed {valid_rays}/{len(ranges)} valid rays")
        
        except tf2_ros.TransformException as ex:
            self.get_logger().error(f"CRITICAL: Transform lookup failed. Frames: '{self.laser_frame}' -> '{self.map_frame}'. Error: {ex}")
        except Exception as ex:
            self.get_logger().error(f"Error during grid update: {type(ex).__name__}: {ex}")
    
    def publish_occupancy_grid(self):
        """Publish the current occupancy grid as OccupancyGrid message."""
        # Get occupancy grid data
        grid_data = self.grid_map.get_occupancy_grid_array()
        
        # Create OccupancyGrid message
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        
        # Map info
        msg.info.map_load_time = msg.header.stamp
        msg.info.resolution = self.grid_map.cell_size
        msg.info.width = self.grid_map.width
        msg.info.height = self.grid_map.height
        
        # Origin
        origin = self.grid_map.get_origin()
        msg.info.origin.position.x = origin[0]
        msg.info.origin.position.y = origin[1]
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # Convert grid to flat array in row-major order
        # ROS 2 expects a Python list, not a numpy array!
        msg.data = grid_data.flatten().tolist()
        
        # Count data for logging purposes as I was encoutering some errors
        occupied_count = sum(1 for v in msg.data if v > 50)
        free_count = sum(1 for v in msg.data if 0 <= v <= 50)
        unknown_count = sum(1 for v in msg.data if v == -1)
        
        self.get_logger().debug(f"Publishing map: frame_id='{msg.header.frame_id}', size={msg.info.width}x{msg.info.height}, "
                               f"occupied={occupied_count}, free={free_count}, unknown={unknown_count}")
        
        self.map_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
