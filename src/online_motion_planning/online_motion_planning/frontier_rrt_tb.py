import rclpy
from rclpy.node import Node
import numpy as np
import math

from matplotlib import pyplot as plt

from geometry_msgs.msg import Pose, PoseStamped, Twist, Point
from nav_msgs.msg import Odometry, OccupancyGrid

from visualization_msgs.msg import Marker, MarkerArray

from online_motion_planning.rrt_star import RRT_STAR
from online_motion_planning.Point import Point as PointRRT

import cv2
import copy

class SamplingTurtlebot(Node):
    def __init__(self):
        super().__init__('sampling_turtlebot')
        self.robot_pose = None
        self.goal_pose = None
        self.acceptance_radius = 0.1
        self.wheel_base_distance = 0.230
        self.max_linear_velocity = 0.3
        self.max_angular_velocity = 0.3
        self.kv = 0.5
        self.kw = 1.0

        # Map variables
        self.occupancy_map = None
        self.binary_map = None
        self.origin  = None
        self.resolution = None
        self.height = None
        self.width = None

        # Frontier variables
        self.kdist = 1
        self.karea = 2
        self.find_frontier = True

        # RRT star variable
        self.max_iterations = 2000
        self.delta_q = 8
        self.p = 0.3
        self.max_depth = round(math.log(self.delta_q, 2)) + 1 # max iteration for considering a segment free or not
        self.min_dist = 5
        self.radius = 5
        self.threshold_path_rewire_dist = 5

        self.waypoints = None

        self.collide_robot_next_waypoint = False
        # Motion controller node
        self.complete_a_path = True
        
        self.declare_parameter('map_frame', 'world_enu')
        self.binary_map_frame = self.get_parameter('map_frame').value

        # Subsrciber
        self.odom_sub = self.create_subscription(Odometry, '/turtlebot/odom', self.odom_callback, 10)
        self.binary_map_sub = self.create_subscription(OccupancyGrid, '/inflated_map', self.map_callback, 10)

        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtlebot/cmd_vel', 10) # publishe cmd_vel

        # # Timer
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.path_timer = self.create_timer(1, self.path_planning_loop)
        self.viewpoint_timer = self.create_timer(2, self.frontier_viewpoint)

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose.position
        # Convert Quaternion to Yaw (Z-axis rotation)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def map_callback(self, msg):
        info = msg.info

        self.resolution = info.resolution

        self.width = info.width
        self.height = info.height 
        
        origin_x = info.origin.position.x 
        origin_y = info.origin.position.y
        self.origin = np.array([origin_x, origin_y])

        self.occupancy_map = np.array(msg.data, dtype = float).reshape(self.height, self.width)

        self.binary_map = np.where(copy.deepcopy(self.occupancy_map) > 50, 1, 0)

    def frontier_cost(self, area, cX, cY):
        q_goal = np.array([cY, cX])
        q_start = (np.array([self.robot_pose.y, self.robot_pose.x]) - self.origin) / self.resolution
        q_goal_point = PointRRT(q_goal[0], q_goal[1])
        q_start_point = PointRRT(q_start[0], q_start[1])
        dist = q_start_point.dist(q_goal_point)

        # self.get_logger().info(f"Dist {dist} area {area}")

        cost = self.kdist*dist - self.karea*area
        return cost

    def frontier_viewpoint(self):
        self.get_logger().info(f"In frontier viewpoint")
        if self.occupancy_map is None or self.robot_pose is None or self.find_frontier == False:
            return
        
        frontier_cells = np.zeros((self.height, self.width), dtype=np.uint8)

        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.occupancy_map[y, x] == 0:  # Cell is Free
                    neighbors = [self.occupancy_map[y-1, x], self.occupancy_map[y+1, x], 
                                self.occupancy_map[y, x-1], self.occupancy_map[y, x+1]]
                    if 50.0 in neighbors:
                        if 100.0 not in neighbors:
                            frontier_cells[y, x] = 255
        
        output = cv2.connectedComponentsWithStats(frontier_cells, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        cost_list = np.full(numLabels, np.inf) # Initialize with infinity

        for i in range(1, numLabels):
            # Stats are: [left, top, width, height, area]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            # Ignore very small noise clusters
            if area < 5:
                continue

            cost = self.frontier_cost(area, cX, cY)
            cost_list[i] = cost
        
        # Find the index of the minimum cost
        if numLabels > 1:
            best_index = np.argmin(cost_list)
            if cost_list[best_index] != np.inf:
                best_cX, best_cY = centroids[best_index]
                
                # Convert pixel (best_cX, best_cY) to world coordinates
                goal_x = best_cX * self.resolution + self.origin[0]
                goal_y = best_cY * self.resolution + self.origin[1]
                
                self.get_logger().info(f"Best Goal: {goal_x}, {goal_y} with cost {cost_list[best_index]}")
                self.goal_pose = [goal_x, goal_y]
                frontier_goal = [(goal_x, goal_y)]
                self.publish_positions_as_markers(frontier_goal)
                self.find_frontier = False
        elif numLabels == 1:
            self.get_logger().info(f"No more frontier regions. Stop exploring!")
            return

    def path_planning_loop(self):
        if self.binary_map is None or self.robot_pose is None or self.goal_pose is None:  
            return
        
        # self.get_logger().info(f"In path loop")
        # Conversion of robot pose and goal pose to cell coordinate
        q_goal = (np.array([self.goal_pose[1], self.goal_pose[0]]) - self.origin) / self.resolution # why do we need to swap between the x and y
        q_start = (np.array([self.robot_pose.y, self.robot_pose.x]) - self.origin) / self.resolution
        q_goal_point = PointRRT(q_goal[0], q_goal[1])
        q_start_point = PointRRT(q_start[0], q_start[1])

        rrt_star = RRT_STAR(self.delta_q, self.p, self.max_depth, self.min_dist, self.radius, self.threshold_path_rewire_dist)
        is_goal_occupied = rrt_star.is_point_occupied(q_goal_point, self.binary_map)
        if is_goal_occupied: 
            self.get_logger().warn(f"Goal point is not valid (on obstacle). Select another goal")
            self.find_frontier = True
            return

        if self.waypoints is not None:
            next_waypoint = self.waypoints[0]
            q_next = (np.array([next_waypoint[1], next_waypoint[0]]) - self.origin) / self.resolution
            q_next_point = PointRRT(q_next[0], q_next[1])
            self.collide_robot_next_waypoint = not rrt_star.is_segment_free_bisection(q_start_point, q_next_point, self.binary_map, 0) # might test transposing the map input when running with real robot
            if self.collide_robot_next_waypoint:
                self.get_logger().warn(f"Path to next waypoint is not clear. Stop and replan")
                stop_msg = Twist()
                self.cmd_vel_pub.publish(stop_msg)

        if self.complete_a_path is False and not self.collide_robot_next_waypoint: 
            return
        else:
            path = []
            self.get_logger().info(f"Planning a path from {q_start} to {q_goal}")
            G, edges, iter = rrt_star.sample(self.binary_map, self.max_iterations, q_start[0], q_start[1], q_goal[0], q_goal[1]) # might test transposing the map input when running with real robot
            if iter == self.max_iterations and len(edges) == 0:
                self.get_logger().warn("Cannot find a path within a defined iteration numbers")
                self.find_frontier = True
            else:
                G, edges, path = rrt_star.fill_path(G, edges)
                path = rrt_star.smoothing(self.binary_map, G, path)
                self.get_logger().info("Find a path")
                rrt_star.plot(self.binary_map, G, edges, path)

            waypoints = []
            for i in range(1, len(path)):
                q = G[path[i]]
                q = np.array([q.y, q.x])
                coordinate = q * self.resolution + self.origin
                waypoints.append(np.array([coordinate[0], coordinate[1]]))

            if len(waypoints) > 0: 
                self.waypoints = waypoints
                waypoints_viz = [np.array([self.robot_pose.x, self.robot_pose.y])] + waypoints # add robot start position to path
                self.publish_positions_as_markers(waypoints_viz)
            
            self.complete_a_path = False
            self.collide_robot_next_waypoint = False

    def control_loop(self):
        self.get_logger().info(f"In control loop received waypoint {self.waypoints}")
        if self.waypoints is None or len(self.waypoints) == 0:
            # If no waypoints, ensure robot is stopped
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            self.complete_a_path = True
            return
        
        self.complete_a_path = False
        self.get_logger().info(f"Controlling the robot")
        
        next_waypoint = self.waypoints[0]
        
        inc_x = next_waypoint[0] - self.robot_pose.x
        inc_y = next_waypoint[1] - self.robot_pose.y
        dist = np.sqrt(inc_x**2 + inc_y**2)

        if dist < self.acceptance_radius:
            self.waypoints.pop(0) # Remove the reached waypoint
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            
            if len(self.waypoints) == 0:
                self.get_logger().info("All waypoints reached. Stopping.")
                self.waypoints = None
                self.goal_pose = None # Clear goal so planner stops
                stop_msg = Twist()
                self.cmd_vel_pub.publish(stop_msg)
                self.complete_a_path = True
                self.find_frontier = True
                return
        else: 
            desired_yaw = math.atan2(inc_y, inc_x)
            angle_diff = desired_yaw - self.current_yaw
            
            # Standardize angle to [-pi, pi]
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

            cmd_vel = Twist()
            w = min(self.kw * angle_diff, self.max_angular_velocity)
            
            # Only move forward if we are roughly facing the right way
            if abs(angle_diff) <= 0.3:
                v = min(self.kv * dist, self.max_linear_velocity)
            else:
                v = 0.0 # Pivot in place
            
            cmd_vel.linear.x = v
            cmd_vel.angular.z = w
            
            self.cmd_vel_pub.publish(cmd_vel)
            # self.get_logger().info(f"Target: {next_waypoint}, Dist: {dist:.2f}, Linear: {v:.2f}, Angular: {w:.2f}")
        
    def publish_positions_as_markers(self, positions):
        """
        Takes an array of (x, y) tuples and publishes them as a MarkerArray,
        including spheres for points and a line strip to link them.
        """
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # --- 1. Create the Line Strip Marker (The Links) ---
        if len(positions) > 1:
            line_marker = Marker()
            line_marker.header.frame_id = self.binary_map_frame
            line_marker.header.stamp = now
            line_marker.ns = "links"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            
            # Line width
            line_marker.scale.x = 0.05  
            
            # Color (Blue-ish)
            line_marker.color.r = 0.0
            line_marker.color.g = 0.5
            line_marker.color.b = 1.0
            line_marker.color.a = 0.8
            
            line_marker.pose.orientation.w = 1.0

            # Add all positions to the line's points list
            for (x, y) in positions:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                line_marker.points.append(p)
                
            marker_array.markers.append(line_marker)

        # --- 2. Create the Sphere Markers (The Points) ---
        for i, (x, y) in enumerate(positions):
            marker = Marker()
            marker.header.frame_id = self.binary_map_frame
            marker.header.stamp = now
            marker.ns = "positions"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.01  # Slightly higher than line to avoid flickering
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

def main(args=None):
    rclpy.init(args=args)

    sampling_turtlebot = SamplingTurtlebot()

    rclpy.spin(sampling_turtlebot)

    sampling_turtlebot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
