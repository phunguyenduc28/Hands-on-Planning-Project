import rclpy
from rclpy.node import Node
import numpy as np
import math

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

# Assuming these are available in your workspace
from online_motion_planning.rrt_star import RRT_STAR
from online_motion_planning.Point import Point as PointRRT

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        
        # Robot State
        self.robot_pose = None
        self.goal_pose = None
        
        # Map variables
        self.map = None
        self.origin = None
        self.resolution = None

        # RRT star parameters
        self.max_iterations = 2000
        self.delta_q = 8
        self.p = 0.3
        self.max_depth = round(math.log(self.delta_q, 2)) + 1 
        self.min_dist = 5
        self.radius = 5
        self.threshold_path_rewire_dist = 5

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/turtlebot/odom', self.odom_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/inflated_map', self.map_callback, 10)
        self.goal_pose_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 10)

        # Publisher for RViz
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)

        # Timer for planning (runs once per second if a goal exists)
        self.path_timer = self.create_timer(1.0, self.path_planning_loop)
        
        self.get_logger().info("Path Planner Node Started. Waiting for Map and Goal...")

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose.position
        
    def goal_pose_callback(self, msg):
        self.goal_pose = msg.pose.position
        self.get_logger().info(f"Received new goal: ({self.goal_pose.x:.2f}, {self.goal_pose.y:.2f})")

    def map_callback(self, msg):
        info = msg.info
        self.resolution = info.resolution
        self.origin = np.array([info.origin.position.x, info.origin.position.y])
        
        # Reshape map data
        self.map = np.array(msg.data, dtype=float).reshape(info.height, info.width)
        self.map = np.where(self.map > 50, 1, 0)

    def path_planning_loop(self):
        # Only plan if we have all necessary data
        if self.map is None or self.robot_pose is None or self.goal_pose is None:  
            return
        
        # 1. Coordinate Conversion (Robot/Goal to Cell)
        # Maintaining the Y, X swap from original code for RRT logic consistency
        q_goal = (np.array([self.goal_pose.y, self.goal_pose.x]) - self.origin) / self.resolution
        q_start = (np.array([self.robot_pose.y, self.robot_pose.x]) - self.origin) / self.resolution
        
        q_goal_point = PointRRT(q_goal[0], q_goal[1])
        q_start_point = PointRRT(q_start[0], q_start[1])

        # 2. Initialize RRT*
        rrt_star = RRT_STAR(self.delta_q, self.p, self.max_depth, self.min_dist, self.radius, self.threshold_path_rewire_dist)
        
        # Check if goal is valid
        if rrt_star.is_point_occupied(q_goal_point, self.map): 
            self.get_logger().warn("Goal point is inside an obstacle!")
            return

        # 3. Sample Path
        self.get_logger().info("Planning path...")
        G, edges, iterations = rrt_star.sample(self.map, self.max_iterations, q_start[0], q_start[1], q_goal[0], q_goal[1])
        
        if iterations == self.max_iterations and len(edges) == 0:
            self.get_logger().warn("Path not found within iteration limit.")
            return

        # 4. Process and Smooth Path
        G, edges, path_indices = rrt_star.fill_path(G, edges)
        path_indices = rrt_star.smoothing(self.map, G, path_indices)
        
        # 5. Convert indices back to world coordinates
        waypoints_world = []
        for idx in path_indices:
            node = G[idx]
            # Reverse the Y, X swap back to X, Y
            world_x = node.y * self.resolution + self.origin[0]
            world_y = node.x * self.resolution + self.origin[1]
            waypoints_world.append((world_x, world_y))

        # 6. Publish to RViz
        if waypoints_world:
            self.get_logger().info(f"Path found with {len(waypoints_world)} nodes. Publishing markers.")
            self.publish_markers(waypoints_world)

    def publish_markers(self, positions):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Line Strip (The Path)
        if len(positions) > 1:
            line_marker = Marker()
            line_marker.header.frame_id = "world_enu"
            line_marker.header.stamp = now
            line_marker.ns = "path_line"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.05  # Line thickness
            line_marker.color.r, line_marker.color.g, line_marker.color.b, line_marker.color.a = (0.0, 0.5, 1.0, 1.0)
            
            for (x, y) in positions:
                p = Point()
                p.x, p.y, p.z = float(x), float(y), 0.05
                line_marker.points.append(p)
            marker_array.markers.append(line_marker)

        # Spheres (The Waypoints)
        for i, (x, y) in enumerate(positions):
            sphere = Marker()
            sphere.header.frame_id = "world_enu"
            sphere.header.stamp = now
            sphere.ns = "waypoints"
            sphere.id = i + 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = float(x), float(y), 0.06
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.15
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = (1.0, 0.0, 0.0, 1.0)
            marker_array.markers.append(sphere)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()