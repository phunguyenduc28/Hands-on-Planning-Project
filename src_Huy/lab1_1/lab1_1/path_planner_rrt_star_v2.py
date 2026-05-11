#!/usr/bin/env python3
"""
RRT* Path Planning Node with Online Replanning
Loads occupancy map, subscribes to /goal_pose topic, performs RRT* path planning,
and periodically validates and replans paths while navigating.
Features:
  - Real-time path validation every 1 second
  - Stops robot if path becomes invalid
  - Replan if obstacles block current trajectory
  - Continues planning while map is being built
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist, Quaternion, Pose
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
        """
        for i in range(num_checks + 1):
            t = i / num_checks
            check_x = pos1[0] + t * (pos2[0] - pos1[0])
            check_y = pos1[1] + t * (pos2[1] - pos1[1])
            
            if not self.is_point_free(check_x, check_y):
                return False
        return True
    
    def is_point_free(self, x, y):
        """Check if point with robot radius is free from obstacles."""
        cell_x, cell_y = self.world_to_cell((x, y))
        
        # Check cells within robot radius
        radius_cells = int(np.ceil(self.robot_radius / self.resolution))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cx = cell_x + dx
                cy = cell_y + dy
                
                if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                    return False
                
                occupancy = self.grid[cy, cx]
                if occupancy > 50:  # Only occupied cells block
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
                    x = self.origin[0] + random.uniform(0, self.width * self.resolution)
                    y = self.origin[1] + random.uniform(0, self.height * self.resolution)
                    random_point = (x, y)
                
                # Find nearest node in tree
                nearest_id = min(tree.keys(), 
                               key=lambda n: np.linalg.norm(np.array(tree[n][0]) - np.array(random_point)))
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
                    path = self.extract_path(tree, new_node_id + 1)
                    self.last_tree = tree
                    if self.logger:
                        self.logger.info(f"[RRT*] Path found in {iteration} iterations!")
                    return path
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"[RRT*] Exception in planning loop: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
            return None
        
        if self.logger:
            self.logger.error(f"[RRT*] Failed after {self.max_iterations} iterations. Tree size: {len(tree)}")
        
        self.last_tree = tree
        return None
    
    def extract_path(self, tree, node_id):
        """Extract path from tree by backtracking from node_id to root."""
        path = []
        current_id = node_id
        
        while current_id != -1:
            path.append(tree[current_id][0])
            current_id = tree[current_id][1]
        
        return list(reversed(path))
    
    def get_tree_visualization_markers(self):
        """Generate visualization markers for the RRT* tree."""
        from geometry_msgs.msg import Point
        markers = MarkerArray()
        
        if self.last_tree is None or len(self.last_tree) == 0:
            return markers
        
        tree = self.last_tree
        
        # Tree edges
        edge_marker = Marker()
        edge_marker.header.frame_id = 'world_enu'
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.id = 0
        edge_marker.color.r = 0.5
        edge_marker.color.g = 0.5
        edge_marker.color.b = 0.5
        edge_marker.color.a = 0.6
        edge_marker.scale.x = 0.02
        
        # Tree nodes
        node_marker = Marker()
        node_marker.header.frame_id = 'world_enu'
        node_marker.type = Marker.LINE_STRIP
        node_marker.action = Marker.ADD
        node_marker.id = 1
        node_marker.color.r = 0.0
        node_marker.color.g = 0.8
        node_marker.color.b = 0.0
        node_marker.color.a = 0.8
        node_marker.scale.x = 0.05
        
        # Add tree edges and nodes
        for node_id, (pos, parent_id, cost) in tree.items():
            p = Point()
            p.x = float(pos[0])
            p.y = float(pos[1])
            p.z = 0.0
            node_marker.points.append(p)
            
            if parent_id != -1 and parent_id in tree:
                parent_pos = tree[parent_id][0]
                p_start = Point()
                p_start.x = float(parent_pos[0])
                p_start.y = float(parent_pos[1])
                p_start.z = 0.0
                edge_marker.points.append(p_start)
                
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
        self.declare_parameter('replan_interval', 1.0)  # Replan every 1 second
        
        self.robot_radius = self.get_parameter('robot_radius').value
        self.map_file = self.get_parameter('map_file').value
        self.max_iterations = self.get_parameter('rrt_max_iterations').value
        self.step_size = self.get_parameter('rrt_step_size').value
        self.replan_interval = self.get_parameter('replan_interval').value
        
        # State
        self.planner = None
        self.current_pose = None
        self.current_goal = None
        self.current_path = None
        self.last_logged_pose = None
        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_origin = None
        self.map_grid_data = None
        
        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.waypoint_pub = self.create_publisher(Path, '/plan', 10)
        # self.marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)
        self.tree_viz_pub = self.create_publisher(MarkerArray, '/rrt_tree_visualization', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10)
        
        # Load map
        # self.load_map_from_file()
        
        # Timers
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)
        self.replan_timer = self.create_timer(self.replan_interval, self.replan_callback)
        
        # Subscriptions
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(Odometry, '/turtlebot/odom', self.current_pose_callback, 10)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Path Planner Node (v2 with Online Replanning) initialized")
        self.get_logger().info(f"Replan interval: {self.replan_interval}s")
        if self.planner is not None:
            self.get_logger().info(" Planner created from MAP FILE")
        else:
            self.get_logger().info(" Waiting for /map topic...")
        self.get_logger().info("=" * 60)
    
    def load_map_from_file(self):
        """Load occupancy map from YAML + PGM files."""
        map_file = os.path.expanduser(self.map_file)
        
        if not os.path.exists(map_file):
            self.get_logger().error(f"Map file not found: {map_file}")
            return
        
        try:
            with open(map_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            resolution = yaml_data['resolution']
            origin = yaml_data['origin'][:2]
            
            pgm_file = os.path.join(os.path.dirname(map_file), yaml_data['image'])
            
            from PIL import Image
            img = Image.open(pgm_file)
            grid_data = np.array(img, dtype=np.uint8)
            
            # Convert PGM to ROS format
            grid_ros = np.zeros_like(grid_data, dtype=np.int8)
            for i in range(grid_data.shape[0]):
                for j in range(grid_data.shape[1]):
                    pgm_val = grid_data[i, j]
                    if pgm_val >= 250:
                        grid_ros[i, j] = 0
                    elif pgm_val <= 10:
                        grid_ros[i, j] = 100
                    else:
                        grid_ros[i, j] = -1
            
            self.map_height, self.map_width = grid_ros.shape
            self.map_resolution = resolution
            self.map_origin = origin
            self.map_grid_data = grid_ros
            
            self.publish_map()
            
            self.planner = RRTStarPlanner(
                grid_ros, resolution, origin,
                robot_radius=self.robot_radius,
                max_iterations=self.max_iterations,
                step_size=self.step_size,
                logger=self.get_logger()
            )
            
            self.get_logger().info(f"Loaded map from {map_file}")
            
        except Exception as e:
            self.get_logger().error(f"Error loading map: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def map_callback(self, msg):
        """Update planner when map is updated (from occupancy grid node)."""
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        grid_data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        
        if self.planner is None:
            # Initialize planner from first map message
            self.planner = RRTStarPlanner(
                grid_data, resolution, origin,
                robot_radius=self.robot_radius,
                max_iterations=self.max_iterations,
                step_size=self.step_size,
                logger=self.get_logger()
            )
            self.map_height = height
            self.map_width = width
            self.map_resolution = resolution
            self.map_origin = origin
            self.map_grid_data = grid_data
            self.get_logger().info("[MAP] Planner initialized from /map topic")
        else:
            # Update the grid in the planner with latest map
            self.planner.grid = grid_data
            self.planner.height, self.planner.width = grid_data.shape
            self.map_grid_data = grid_data
            
            self.get_logger().debug(f"[MAP UPDATE] Grid updated from /map topic")
    
    def current_pose_callback(self, msg):
        """Update robot pose from odometry."""
        new_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        if self.current_pose != new_pose:
            self.current_pose = new_pose
            
            # Log periodically (every 0.1m)
            should_log = False
            if self.last_logged_pose is None:
                should_log = True
            else:
                dist_moved = ((self.current_pose[0] - self.last_logged_pose[0])**2 + 
                            (self.current_pose[1] - self.last_logged_pose[1])**2) ** 0.5
                if dist_moved > 0.1:
                    should_log = True
            
            if should_log and self.planner is not None:
                cell = self.planner.world_to_cell(self.current_pose)
                occupancy = self.planner.grid[cell[1], cell[0]] if (0 <= cell[0] < self.planner.width and 0 <= cell[1] < self.planner.height) else -1
                self.get_logger().info(f"[POSE] Robot at ({self.current_pose[0]:.2f}, {self.current_pose[1]:.2f}), Occupancy: {occupancy}")
                self.last_logged_pose = self.current_pose
    
    def goal_pose_callback(self, msg):
        """Handle new goal pose."""
        # try:
        #     transform = self.tf_buffer.lookup_transform('world_enu', msg.header.frame_id, 
        #                                                 rclpy.time.Time(), timeout=Duration(seconds=1.0))
        #     pose_ned = tf2_geometry_msgs.do_transform_pose(msg.pose, transform)
        #     goal = (pose_ned.position.x, pose_ned.position.y)
            
        #     self.get_logger().info(f"[GOAL] Received goal: ({goal[0]:.2f}, {goal[1]:.2f})")
        # except tf2_ros.TransformException as ex:
        #     self.get_logger().error(f"Transform failed: {ex}")
        #     return
        
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
        
        
        if self.planner is None:
            self.get_logger().error("Planner not initialized yet")
            return
        
        # Store goal and plan
        self.current_goal = goal
        self.plan_path()
    
    def plan_path(self):
        """Plan path to current goal."""
        if self.current_goal is None or self.current_pose is None:
            return
        
        self.get_logger().info(f"[PLANNING] Planning path from ({self.current_pose[0]:.2f}, {self.current_pose[1]:.2f}) to ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})")
        
        path = self.planner.plan(self.current_pose, self.current_goal)
        
        if path is None:
            self.get_logger().error("[PLANNING] Path planning FAILED")
            self.stop_robot()
            return
        
        self.current_path = path
        self.get_logger().info(f"[PLANNING] SUCCESS - Found path with {len(path)} waypoints")
        
        # Publish
        self.publish_waypoints(path)
        # self.visualize_path(path)
        self.visualize_rrt_tree()
    
    def replan_callback(self):
        """Check if current path is still valid and replan if needed."""
        if self.current_path is None or self.current_goal is None or self.current_pose is None or self.planner is None:
            return
        
        # Find closest point on path to robot
        path_valid = True
        for i in range(len(self.current_path) - 1):
            if not self.planner.is_collision_free(self.current_path[i], self.current_path[i+1]):
                self.get_logger().warn(f"[REPLAN] Path segment {i} is blocked! Replanning...")
                path_valid = False
                break
        
        if not path_valid:
            self.stop_robot()
            self.plan_path()
    
    def stop_robot(self):
        """Stop the robot immediately."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        self.get_logger().warn("[ROBOT] STOPPING!")
    
    def publish_map(self):
        """Publish occupancy map."""
        if self.map_grid_data is None:
            return
        
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world_enu'
        msg.info.resolution = float(self.map_resolution)
        msg.info.width = int(self.map_width)
        msg.info.height = int(self.map_height)
        msg.info.origin = Pose()
        msg.info.origin.position.x = float(self.map_origin[0])
        msg.info.origin.position.y = float(self.map_origin[1])
        msg.info.origin.orientation.w = 1.0
        
        msg.data = self.map_grid_data.flatten().tolist()
        self.map_pub.publish(msg)
    
    def publish_waypoints(self, path):
        """Publish waypoints as Path message."""
        path_msg = Path()
        path_msg.header.frame_id = 'world_enu'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in path:
            pose = PoseStamped()
            pose.header.frame_id = 'world_enu'
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            path_msg.poses.append(pose)
        
        self.waypoint_pub.publish(path_msg)
        self.get_logger().info(f"[PUBLISH] Path published to /plan with {len(path_msg.poses)} waypoints")
    
    # def visualize_path(self, path):
    #     """Visualize path with markers."""
    #     from geometry_msgs.msg import Point
    #     markers = MarkerArray()
        
    #     line_marker = Marker()
    #     line_marker.header.frame_id = 'world_enu'
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
    #         line_marker.points.append(p)
        
    #     markers.markers.append(line_marker)
        
    #     for i, (x, y) in enumerate(path):
    #         sphere = Marker()
    #         sphere.header.frame_id = 'world_enu'
    #         sphere.header.stamp = self.get_clock().now().to_msg()
    #         sphere.type = Marker.SPHERE
    #         sphere.action = Marker.ADD
    #         sphere.id = i + 1
    #         sphere.pose.position.x = float(x)
    #         sphere.pose.position.y = float(y)
    #         sphere.scale.x = 0.1
    #         sphere.scale.y = 0.1
    #         sphere.scale.z = 0.1
    #         sphere.color.r = 1.0
    #         sphere.color.a = 0.8
    #         markers.markers.append(sphere)
        
    #     self.marker_pub.publish(markers)
    
    def visualize_rrt_tree(self):
        """Visualize RRT* tree."""
        tree_markers = self.planner.get_tree_visualization_markers()
        for marker in tree_markers.markers:
            marker.header.stamp = self.get_clock().now().to_msg()
        self.tree_viz_pub.publish(tree_markers)


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
