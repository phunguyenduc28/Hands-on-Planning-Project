#!/usr/bin/env python3
"""
RRT* Path Planning Node
Loads occupancy map, subscribes to /goal_pose topic, performs RRT* path planning,
and publishes waypoints for visualization and execution.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist, Quaternion, Pose
# from lab1_1_interfaces.msg import WaypointList, Waypoint  # Custom message
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
import random
from scipy.spatial.distance import cdist
import yaml
import os
import tf2_ros
import tf2_geometry_msgs
from rclpy.time import Duration


class RRTStarPlanner:
    """RRT* path planning algorithm with circular footprint collision checking."""
    
    def __init__(self, grid, resolution, origin, robot_radius=0.2, max_iterations=2000, step_size=0.3, logger=None):
        """
        Initialize RRT* planner.
        
        Args:
            grid: Occupancy grid (0-100, -1=unknown)
            resolution: Grid cell size in meters
            origin: [x, y] of grid origin
            robot_radius: Radius for circular footprint collision checking
            max_iterations: Maximum RRT* iterations
            step_size: Step size for tree expansion
            logger: Optional ROS logger for debug output
        """
        self.grid = grid
        self.resolution = resolution
        self.origin = np.array(origin)
        self.robot_radius = robot_radius
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.logger = logger
        
        self.height, self.width = grid.shape
        
        # RRT* parameters
        self.rewire_radius = 1.5  # Radius for rewiring neighbors
        self.goal_sample_rate = 0.15  # Probability of sampling goal
        
        # Store last tree for visualization
        self.last_tree = None

        
    def cell_to_world(self, cell):
        """Convert grid cell to world coordinates."""
        x = self.origin[0] + cell[0] * self.resolution
        y = self.origin[1] + cell[1] * self.resolution
        return (x, y)
    
    def world_to_cell(self, world_pos):
        """Convert world coordinates to grid cell."""
        x, y = world_pos
        cell_x = int(np.floor((x - self.origin[0]) / self.resolution))
        cell_y = int(np.floor((y - self.origin[1]) / self.resolution))
        return (cell_x, cell_y)
    
    def is_collision_free(self, pos1, pos2, num_checks=50):
        """
        Check if path segment from pos1 to pos2 is collision-free.
        Uses circular footprint for collision checking.
        Increased checks for strict collision detection.
        """
        # Check many points along the line (increased to 50 for strict checking)
        # more number of checks means more points are checked along the line segment, which can help 
        # catch collisions that might be missed with fewer checks, especially for larger step sizes or 
        # narrow passages. However, it also increases computation time, so it's a trade-off between
        #  accuracy and performance.
        for i in range(num_checks + 1):
            t = i / num_checks # how much to interpolate between pos1 and pos2
            check_x = pos1[0] + t * (pos2[0] - pos1[0])
            check_y = pos1[1] + t * (pos2[1] - pos1[1])
            
            if not self.is_point_free(check_x, check_y):
                return False
        return True
    
    def is_point_free(self, x, y):
        """Check if point with robot radius is free from obstacles."""
        cell_x, cell_y = self.world_to_cell((x, y))
        
        # Check cells within robot radius with minimal buffer
        # Use only actual robot radius, no extra buffer
        # Calculate how many grid cells span the robot's radius. 
        # This determines how many cells around the center point we need to check for occupancy.
        radius_cells = int(np.ceil(self.robot_radius / self.resolution))
        for dx in range(-radius_cells, radius_cells + 1):
            # Loop through x-direction offsets: -2, -1, 0, 1, 2 (if radius_cells=2)
            for dy in range(-radius_cells, radius_cells + 1):
                # Nested loop through y-direction offsets: -2, -1, 0, 1, 2 (if radius_cells=2)
                cx = cell_x + dx
                cy = cell_y + dy
                
                # Check bounds - out of bounds is still treated as occupied
                if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                    return False
                
                # Only treat occupied cells as obstacles, not unknown
                # Unknown cells are allowed (unmapped areas)
                occupancy = self.grid[cy, cx]
                if occupancy > 50:  # Only check for occupied, not unknown
                    return False
        
        return True
    
    def plan(self, start, goal):
        """
        Plan path from start to goal using RRT*.
        
        Args:
            start: (x, y) start position
            goal: (x, y) goal position
            
        Returns:
            List of waypoints if successful, None otherwise
        """
        # Check if start and goal are valid
        # I added this log because I was getting many errors about transforms and 
        # it was not clear if the issue was with the planner or with the transforms. 
        # This log helps confirm that the start and goal positions are valid and 
        # free before planning begins.
        if self.logger:
            self.logger.info(f"[RRT* PLANNING] START: {start}, GOAL: {goal}")
        if not self.is_point_free(start[0], start[1]):
            if self.logger:
                self.logger.error(f"[RRT*] Start position is not free!")
            return None
        if not self.is_point_free(goal[0], goal[1]):
            if self.logger:
                self.logger.error(f"[RRT*] Goal position is not free!")
            return None
        
        # Tree structure: {node_id: (position, parent_id, cost_to_root)}
        # used for the purpose of visualisation
        tree = {0: (start, -1, 0.0)}
        
        if self.logger:
            self.logger.info(f"[RRT*] Starting planning loop, max_iterations={self.max_iterations}")
        
        try:
            for iteration in range(self.max_iterations):
                # Progress logging (every 100 iterations)
                if iteration % 100 == 0 and self.logger:
                    self.logger.info(f"[RRT*] Iteration {iteration}/{self.max_iterations}, Tree size: {len(tree)}")
                
                # Sample random point (with goal biasing)
                if random.random() < self.goal_sample_rate:
                    random_point = goal
                else:
                    # Random point within bounds\
                    # origin maybe at different place
                    x = self.origin[0] + random.uniform(0, self.width * self.resolution)
                    y = self.origin[1] + random.uniform(0, self.height * self.resolution)
                    random_point = (x, y)
                
                # Find nearest node in tree
                # nearest_id = min(tree.keys(), 
                #                key=lambda n: np.linalg.norm(np.array(tree[n][0]) - np.array(random_point)))
                
                # nearest_pos = tree[nearest_id][0]
                nearest_id = 0
                min_distance = float('inf')

                for node_id in tree.keys():
                    node_pos = tree[node_id][0]  # Get position of this node
                    distance = np.linalg.norm(np.array(node_pos) - np.array(random_point))
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_id = node_id

                nearest_pos = tree[nearest_id][0]
                # Extend towards random point
                direction = np.array(random_point) - np.array(nearest_pos)
                dist = np.linalg.norm(direction)
                
                if dist < 1e-6:
                    continue
                
                direction = direction / dist
                new_pos = tuple(np.array(nearest_pos) + direction * min(self.step_size, dist))
                
                # Check collision
                if not self.is_collision_free(nearest_pos, new_pos):
                    continue
                
                # Find nearby nodes for rewiring
                new_node_id = len(tree)
                nearest_cost = tree[nearest_id][2] + np.linalg.norm(np.array(new_pos) - np.array(nearest_pos))
                parent_id = nearest_id
                
                # Rewire: find neighbors within rewire_radius
                for node_id, (node_pos, _, node_cost) in tree.items():
                    dist_to_new = np.linalg.norm(np.array(new_pos) - np.array(node_pos))
                    if dist_to_new < self.rewire_radius and self.is_collision_free(node_pos, new_pos):
                        cost_through_new = node_cost + dist_to_new
                        if cost_through_new < nearest_cost:
                            parent_id = node_id
                            nearest_cost = cost_through_new
                
                # Add new node to tree
                tree[new_node_id] = (new_pos, parent_id, nearest_cost)
                
                # Check if goal is reached
                if np.linalg.norm(np.array(new_pos) - np.array(goal)) < self.step_size:
                    tree[new_node_id + 1] = (goal, new_node_id, nearest_cost + 
                                            np.linalg.norm(np.array(goal) - np.array(new_pos)))
                    # Extract path
                    path = self.extract_path(tree, new_node_id + 1)
                    # Store tree for visualization
                    self.last_tree = tree
                    # Smooth path (disabled to see actual RRT* path structure)
                    # path = self.smooth_path(path)
                    return path
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"[RRT*] Exception in planning loop: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
            return None
        
        # Log why planning failed
        if self.logger:
            self.logger.error(f"[RRT*] Failed after {self.max_iterations} iterations. Tree size: {len(tree)}")
            closest_node_id = min(tree.keys(), 
                                key=lambda n: np.linalg.norm(np.array(tree[n][0]) - np.array(goal)))
            closest_dist = np.linalg.norm(np.array(tree[closest_node_id][0]) - np.array(goal))
            self.logger.error(f"[RRT*] Closest node distance to goal: {closest_dist:.3f}m (step_size={self.step_size}m)")
        
        # Store tree for visualization even on failure
        self.last_tree = tree
        return None  # No path found
    
    def extract_path(self, tree, node_id):
        """Extract path from tree by backtracking from node_id to root."""
        path = []
        current_id = node_id
        
        while current_id != -1:
            path.append(tree[current_id][0])
            current_id = tree[current_id][1]
        
        return list(reversed(path))
    
    def smooth_path(self, path, iterations=10):
        """Smooth path by removing unnecessary waypoints (reduced iterations to keep waypoints)."""
        if len(path) <= 2:
            return path
        
        smoothed = list(path)
        initial_len = len(smoothed)
        
        # Only smooth if path is long enough
        for _ in range(iterations):
            if len(smoothed) <= 3:  # Keep at least start, one intermediate, goal
                break
            
            # Try to skip intermediate points
            i = random.randint(0, len(smoothed) - 3)
            if self.is_collision_free(smoothed[i], smoothed[i + 2]):
                smoothed.pop(i + 1)
        
        # Debug: only log if significant change
        if len(smoothed) < initial_len:
            pass  # Silently skip logging minor smoothing
        
        return smoothed
    
    def get_tree_visualization_markers(self):
        """Generate visualization markers for the RRT* tree."""
        from geometry_msgs.msg import Point
        markers = MarkerArray()
        
        if self.last_tree is None or len(self.last_tree) == 0:
            return markers
        
        tree = self.last_tree
        marker_id = 0
        
        # Tree edges (lines from parent to child)
        edge_marker = Marker()
        edge_marker.header.frame_id = 'world_enu'
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.id = marker_id
        edge_marker.color.r = 0.5
        edge_marker.color.g = 0.5
        edge_marker.color.b = 0.5
        edge_marker.color.a = 0.6
        edge_marker.scale.x = 0.02  # Line width
        
        # Tree nodes (spheres)
        node_marker = Marker()
        node_marker.header.frame_id = 'world_enu'
        node_marker.type = Marker.LINE_STRIP
        node_marker.action = Marker.ADD
        node_marker.id = marker_id + 1
        node_marker.color.r = 0.0
        node_marker.color.g = 0.8
        node_marker.color.b = 0.0
        node_marker.color.a = 0.8
        node_marker.scale.x = 0.05  # Points size
        
        # Add tree edges and nodes
        for node_id, (pos, parent_id, cost) in tree.items():
            # Add node
            p = Point()
            p.x = float(pos[0])
            p.y = float(pos[1])
            p.z = 0.0
            node_marker.points.append(p)
            
            # Add edge to parent
            if parent_id != -1 and parent_id in tree:
                parent_pos = tree[parent_id][0]
                # Start point (parent)
                p_start = Point()
                p_start.x = float(parent_pos[0])
                p_start.y = float(parent_pos[1])
                p_start.z = 0.0
                edge_marker.points.append(p_start)
                
                # End point (child)
                p_end = Point()
                p_end.x = float(pos[0])
                p_end.y = float(pos[1])
                p_end.z = 0.0
                edge_marker.points.append(p_end)
        
        markers.markers.append(edge_marker)
        markers.markers.append(node_marker)
        
        return markers


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_rrt_star')
        
        # Parameters
        self.declare_parameter('robot_radius', 0.2)
        self.declare_parameter('map_file', os.path.expanduser('~/maps/map_latest.yaml'))
        self.declare_parameter('rrt_max_iterations', 2000)
        self.declare_parameter('rrt_step_size', 0.3)
        
        self.robot_radius = self.get_parameter('robot_radius').value
        self.map_file = self.get_parameter('map_file').value
        self.max_iterations = self.get_parameter('rrt_max_iterations').value
        self.step_size = self.get_parameter('rrt_step_size').value
        
        # State
        self.planner = None
        self.current_pose = None
        self.last_logged_pose = None
        self.loaded_map = None
        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_origin = None
        self.map_grid_data = None
        
        # TF2 for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.waypoint_pub = self.create_publisher(Path, '/plan', 10)
        # self.marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)
        self.tree_viz_pub = self.create_publisher(MarkerArray, '/rrt_tree_visualization', 10)
        
        # Load map and publish it
        self.load_map_from_file()
        
        # Timer to continuously publish map (for RViz to pick it up)
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)
        
        # Subscriptions
        self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_pose_callback, 10
        )
        self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        # The above subscriber is necessary because I initialise the planner 
        # after loading the map, so I need to listen to the /map topic to know 
        # when the planner is ready.
        self.create_subscription(
            Odometry, '/turtlebot/odom', self.current_pose_callback, 10
        )
        
        self.get_logger().info("=" * 60)
        if self.planner is not None:
            self.get_logger().info("✓ Planner created from MAP FILE")
            self.get_logger().info(f"  Grid shape: {self.planner.grid.shape}")
            self.get_logger().info(f"  Unique values: {np.unique(self.planner.grid)}")
        else:
            self.get_logger().info("⚠ Planner not created from file. Waiting for /map topic...")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Path Planner Node initialized")
    
    def load_map_from_file(self):
        """Load occupancy map from YAML + PGM files and publish it."""
        # Expand home directory if needed
        map_file = os.path.expanduser(self.map_file)
        
        if not os.path.exists(map_file):
            self.get_logger().error(f"Map file not found: {map_file}")
            self.get_logger().error(f"Attempted path: {self.map_file}")
            return
        
        self.get_logger().info(f"Loading map from: {map_file}")
        
        try:
            # Load YAML metadata
            with open(map_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            self.get_logger().info(f"YAML loaded: {yaml_data}")
            
            resolution = yaml_data['resolution']
            origin = yaml_data['origin'][:2]
            free_thresh = yaml_data.get('free_thresh', 0.196)
            occupied_thresh = yaml_data.get('occupied_thresh', 0.65)
            negate = yaml_data.get('negate', 0)
            
            self.get_logger().info(f"Thresholds - Free: {free_thresh}, Occupied: {occupied_thresh}, Negate: {negate}")
            
            # Load PGM image
            pgm_file = os.path.join(os.path.dirname(map_file), yaml_data['image'])
            self.get_logger().info(f"Loading PGM from: {pgm_file}")
            
            if not os.path.exists(pgm_file):
                self.get_logger().error(f"PGM file not found: {pgm_file}")
                return
            
            from PIL import Image
            img = Image.open(pgm_file)
            self.get_logger().info(f"PGM image loaded: {img.mode} {img.size}")
            
            grid_data = np.array(img, dtype=np.uint8)
            
            # Convert PGM directly to ROS 0-100 scale, then apply thresholds
            # PGM: 0=occupied, 254=free, 127=unknown
            # ROS: 0=free, 100=occupied, -1=unknown
            # 1. Convert PGM to ROS format
            # White (254) -> 0 (Free)
            # Black (2)   -> 100 (Occupied)
            # Grey (127)  -> -1 (Unknown)
            # reference - https://wiki.ros.org/map_server#map_saver
            grid_ros = np.zeros_like(grid_data, dtype=np.int8)

            for i in range(grid_data.shape[0]):
                for j in range(grid_data.shape[1]):
                    pgm_val = grid_data[i, j]
                    
                    if pgm_val >= 250:        # White/Free
                        grid_ros[i, j] = 0
                    elif pgm_val <= 10:       # Black/Walls
                        grid_ros[i, j] = 100
                    else:                     # 127/Grey/Unexplored
                        grid_ros[i, j] = -1
                    
            # Store map metadata for publishing
            self.map_height, self.map_width = grid_ros.shape
            self.map_resolution = resolution
            self.map_origin = origin
            self.map_grid_data = grid_ros
            
            self.get_logger().info(f"Map loaded: width={self.map_width}, height={self.map_height}, resolution={self.map_resolution}")
            self.get_logger().info(f"Map origin: {self.map_origin}")
            self.get_logger().info(f"Grid data min/max: {np.min(grid_ros)}/{np.max(grid_ros)}")
            self.get_logger().info(f"Grid unique values: {np.unique(grid_ros)}")
            
            # Check a few sample cells
            sample_cells = [(0, 0), (self.map_height//2, self.map_width//2), (self.map_height-1, self.map_width-1)]
            for cy, cx in sample_cells:
                if 0 <= cx < self.map_width and 0 <= cy < self.map_height:
                    self.get_logger().info(f"Sample cell [{cx}, {cy}] = {grid_ros[cy, cx]}")
            
            # Publish the map
            self.publish_map()
            
            self.planner = RRTStarPlanner(
                grid_ros, resolution, origin,
                robot_radius=self.robot_radius,
                max_iterations=self.max_iterations,
                step_size=self.step_size,
                logger=self.get_logger()
            )
            
            self.get_logger().info(f"Loaded map from {map_file}")
            self.get_logger().info(f"Map published to /map topic (width={self.map_width}, height={self.map_height})")
            
        except Exception as e:
            self.get_logger().error(f"Error loading map: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def map_callback(self, msg):
        """Subscribe to map updates and create planner if not already done.
        Created for the purpose of debugging and ensuring that the planner is initialized with the correct map data"""
        if self.planner is not None:
            self.get_logger().info("[MAP CALLBACK] Planner already initialized, ignoring /map topic")
            return
        
        self.get_logger().info("[MAP CALLBACK] Creating planner from /map topic")
        
        # Convert OccupancyGrid message to numpy array
        width = msg.info.width
        height = msg.info.height
        grid_data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        
        self.get_logger().info(f"[MAP CALLBACK] Grid from /map: width={width}, height={height}")
        self.get_logger().info(f"[MAP CALLBACK] Grid unique values: {np.unique(grid_data)}")
        
        # Create planner
        self.planner = RRTStarPlanner(
            grid_data,
            msg.info.resolution,
            [msg.info.origin.position.x, msg.info.origin.position.y],
            robot_radius=self.robot_radius,
            max_iterations=self.max_iterations,
            step_size=self.step_size,
            logger=self.get_logger()
        )
        
        self.get_logger().info("Planner initialized from /map topic")
    
    def current_pose_callback(self, msg):
        """Update current robot pose from ground truth odometry."""
        # Odometry is already in world_ned frame, use it directly
        new_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        if self.current_pose != new_pose:
            self.current_pose = new_pose
            
            # Only log if robot has moved >0.1m from last logged pose
            should_log = False
            if self.last_logged_pose is None:
                should_log = True
            else:
                dist_moved = ((self.current_pose[0] - self.last_logged_pose[0])**2 + 
                            (self.current_pose[1] - self.last_logged_pose[1])**2) ** 0.5
                if dist_moved > 0.1:
                    should_log = True
            
            if should_log:
                # Convert to cell coordinates and check occupancy
                cell = self.planner.world_to_cell(self.current_pose)
                cell_x, cell_y = cell
                
                # Check if cell is within bounds
                if 0 <= cell_x < self.planner.width and 0 <= cell_y < self.planner.height:
                    occupancy = self.planner.grid[cell_y, cell_x]
                    if occupancy == -1:
                        status = "UNKNOWN"
                    elif occupancy > 50:
                        status = "OCCUPIED"
                    else:
                        status = "FREE"
                    self.get_logger().info(f"[POSE UPDATE] Robot at ({self.current_pose[0]:.3f}, {self.current_pose[1]:.3f}) | Cell [{cell_x}, {cell_y}] = {occupancy} ({status})")
                else:
                    self.get_logger().info(f"[POSE UPDATE] Robot at ({self.current_pose[0]:.3f}, {self.current_pose[1]:.3f}) | Cell [{cell_x}, {cell_y}] = OUT OF BOUNDS")
                
                self.last_logged_pose = self.current_pose
    
    def publish_map(self):
        """Publish the loaded occupancy map to /map topic."""
        if self.map_grid_data is None:
            self.get_logger().warn("No map data to publish")
            return
        
        # Create OccupancyGrid message
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'world_enu'
        # map_msg.header.frame_id = 'odom'  # for rosbag
        
        # Set map info
        map_msg.info.resolution = float(self.map_resolution)
        map_msg.info.width = int(self.map_width)
        map_msg.info.height = int(self.map_height)
        
        # Set origin
        map_msg.info.origin = Pose()
        map_msg.info.origin.position.x = float(self.map_origin[0])
        map_msg.info.origin.position.y = float(self.map_origin[1])
        map_msg.info.origin.position.z = 0.0
        
        # Set data - flatten the grid
        flat_data = self.map_grid_data.flatten().tolist()
        map_msg.data = flat_data
        
        # Debug log
        if self.map_height > 0 and self.map_width > 0:
            self.get_logger().debug(f"Publishing map: {self.map_width}x{self.map_height}, data length: {len(map_msg.data)}")
        
        # Publish
        self.map_pub.publish(map_msg)
    def goal_pose_callback(self, msg):
        """Plan path when goal pose is received."""
        # Transform goal using TF2 when there is a mismatch of frames between world_ned and world_enu while recording and loading
        # try:
        #     # Get transform from source frame to world_ned
        #     transform = self.tf_buffer.lookup_transform('world_ned', msg.header.frame_id, 
        #                                                 rclpy.time.Time(), timeout=Duration(seconds=1.0))
        #     # Apply transform to pose (do_transform_pose expects Pose, not PoseStamped)
        #     pose_ned = tf2_geometry_msgs.do_transform_pose(msg.pose, transform)
        #     goal = (pose_ned.position.x, pose_ned.position.y)
            
        #     self.get_logger().info(f"Goal received in {msg.header.frame_id}: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        #     self.get_logger().info(f"Goal transformed to world_ned: ({goal[0]:.2f}, {goal[1]:.2f})")
        # except tf2_ros.TransformException as ex:
        #     self.get_logger().error(f"Transform failed: {ex}")
        #     return
        
        # Using goal directly without transform because the frames are already aligned while recording and loading
        # This assumes msg.header.frame_id is already in the planner's coordinate frame
        goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"Goal received in {msg.header.frame_id}: ({goal[0]:.2f}, {goal[1]:.2f})")
        self.get_logger().info(f"Using goal directly: ({goal[0]:.2f}, {goal[1]:.2f})")
        
        # The following debug logs were to help identify why the planner was failing to find a path. 
        # It checks the occupancy of the start and goal cells, and whether they are free according to
        #  the planner's is_point_free function. This helps confirm that the issue is not with the planner's collision checking, 
        # but rather with the input positions or map data.
        if self.planner is None:
            self.get_logger().error("Planner not initialized yet")
            return
        
        self.get_logger().info(f"[DEBUG] Map origin: {self.map_origin}, Resolution: {self.map_resolution}")
        
        # Use last known pose or origin as start
        if self.current_pose is None:
            start = (0.0, 0.0)
            self.get_logger().warn("No current pose available, using origin (0, 0). Make sure /turtlebot/odom_ground_truth is publishing!")
        else:
            start = self.current_pose
        
        self.get_logger().info(f"[PLANNING] START: ({start[0]:.3f}, {start[1]:.3f}) GOAL: ({goal[0]:.3f}, {goal[1]:.3f})")
        
        # Debug: Check goal occupancy in grid
        goal_cell = self.planner.world_to_cell(goal)
        goal_cell_x, goal_cell_y = goal_cell
        self.get_logger().info(f"[DEBUG] Goal cell: [{goal_cell_x}, {goal_cell_y}]")
        
        if 0 <= goal_cell_x < self.planner.width and 0 <= goal_cell_y < self.planner.height:
            goal_occupancy = self.planner.grid[goal_cell_y, goal_cell_x]
            self.get_logger().info(f"[DEBUG] Goal occupancy: {goal_occupancy}")
            if goal_occupancy > 50:
                self.get_logger().error(f"[DEBUG] Goal cell is OCCUPIED (value={goal_occupancy} > 50)")
            elif goal_occupancy == -1:
                self.get_logger().error(f"[DEBUG] Goal cell is UNKNOWN (value=-1)")
            else:
                self.get_logger().info(f"[DEBUG] Goal cell is FREE (value={goal_occupancy})")
        else:
            self.get_logger().error(f"[DEBUG] Goal cell OUT OF BOUNDS (width={self.planner.width}, height={self.planner.height})")
        
        # Debug: Check start occupancy in grid
        start_cell = self.planner.world_to_cell(start)
        start_cell_x, start_cell_y = start_cell
        self.get_logger().info(f"[DEBUG] Start cell: [{start_cell_x}, {start_cell_y}]")
        
        if 0 <= start_cell_x < self.planner.width and 0 <= start_cell_y < self.planner.height:
            start_occupancy = self.planner.grid[start_cell_y, start_cell_x]
            self.get_logger().info(f"[DEBUG] Start occupancy: {start_occupancy}")
            if start_occupancy > 50:
                self.get_logger().error(f"[DEBUG] Start cell is OCCUPIED (value={start_occupancy} > 50)")
            elif start_occupancy == -1:
                self.get_logger().error(f"[DEBUG] Start cell is UNKNOWN (value=-1)")
            else:
                self.get_logger().info(f"[DEBUG] Start cell is FREE (value={start_occupancy})")
        else:
            self.get_logger().error(f"[DEBUG] Start cell OUT OF BOUNDS")
        
        # Check if start is collision-free using planner's is_point_free
        if not self.planner.is_point_free(start[0], start[1]):
            self.get_logger().error(f"[DEBUG] Start position ({start[0]:.3f}, {start[1]:.3f}) fails is_point_free() check!")
        else:
            self.get_logger().info(f"[DEBUG] Start position ({start[0]:.3f}, {start[1]:.3f}) passes is_point_free() check")
        
        # Plan path (start should be current robot position from last odometry callback)
        path = self.planner.plan(start, goal)
        
        if path is None:
            self.get_logger().error("[PLANNING] Path planning FAILED - start or goal in collision!")
            return
        
        self.get_logger().info(f"[PLANNING] SUCCESS - Found path with {len(path)} waypoints")
        
        # Log path details
        self.get_logger().info(f"[PLANNING] Path found with {len(path)} waypoints")
        if len(path) > 0:
            first_wp = path[0]
            last_wp = path[-1]
            self.get_logger().info(f"[PATH] First waypoint: ({first_wp[0]:.3f}, {first_wp[1]:.3f}), Last waypoint: ({last_wp[0]:.3f}, {last_wp[1]:.3f})")
            dist_start_to_first_wp = np.linalg.norm(np.array(start) - np.array(first_wp))
            self.get_logger().info(f"[PATH] Distance from robot to first waypoint: {dist_start_to_first_wp:.3f}m")
        
        # Publish waypoints
        self.publish_waypoints(path)
        
        # Visualize path
        # self.visualize_path(path)
        
        # Visualize RRT* tree
        self.visualize_rrt_tree()
    
    def publish_waypoints(self, path):
        """Publish waypoints as nav_msgs::Path."""
        path_msg = Path()
        path_msg.header.frame_id = 'world_enu'
        # path_msg.header.frame_id = 'odom' # for rosbag
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in path:
            pose = PoseStamped()
            pose.header.frame_id = 'world_enu'
            # pose.header.frame_id = 'odom' # for rosbag
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            path_msg.poses.append(pose)
        
        self.waypoint_pub.publish(path_msg)
    
    # def visualize_path(self, path):
    #     """Publish path visualization markers."""
    #     from geometry_msgs.msg import Point
    #     markers = MarkerArray()
        
    #     # Path line
    #     line_marker = Marker()
    #     line_marker.header.frame_id = 'world_enu'
    #     # line_marker.header.frame_id = 'odom' # for rosbag
    #     line_marker.header.stamp = self.get_clock().now().to_msg()
    #     line_marker.type = Marker.LINE_STRIP
    #     line_marker.action = Marker.ADD
    #     line_marker.id = 0
    #     line_marker.color.g = 1.0
    #     line_marker.color.a = 0.8
    #     line_marker.scale.x = 0.05
        
    #     for x, y in path:
    #         p = Point()
    #         p.x = float(x)
    #         p.y = float(y)
    #         p.z = 0.0
    #         line_marker.points.append(p)
        
    #     markers.markers.append(line_marker)
        
    #     # Waypoint markers
    #     for i, (x, y) in enumerate(path):
    #         sphere = Marker()
    #         sphere.header.frame_id = 'world_enu'
    #         # sphere.header.frame_id = 'odom' # for rosbag
    #         sphere.header.stamp = self.get_clock().now().to_msg()
    #         sphere.type = Marker.SPHERE
    #         sphere.action = Marker.ADD
    #         sphere.id = i + 1
    #         sphere.pose.position.x = float(x)
    #         sphere.pose.position.y = float(y)
    #         sphere.pose.position.z = 0.0
    #         sphere.scale.x = 0.1
    #         sphere.scale.y = 0.1
    #         sphere.scale.z = 0.1
    #         sphere.color.r = 1.0
    #         sphere.color.a = 0.8
            
    #         markers.markers.append(sphere)
        
    #     self.marker_pub.publish(markers)
    
    def visualize_rrt_tree(self):
        """Publish RRT* tree visualization markers."""
        tree_markers = self.planner.get_tree_visualization_markers()
        # Add current timestamp to markers
        for marker in tree_markers.markers:
            marker.header.stamp = self.get_clock().now().to_msg()
        self.tree_viz_pub.publish(tree_markers)
        self.get_logger().info(f"[VIZ] Published RRT* tree with {len(tree_markers.markers)} marker groups")


def main():
    rclpy.init()
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
