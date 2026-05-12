import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import numpy as np
import cv2

class FrontierBasedExploration(Node):
    def __init__(self):
        super().__init__('frontier_based_exploration')

        # 1. Create Subscriber to /inflated_map
        # Using 10 as QoS depth. 
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/inflated_map',
            self.map_callback,
            10)

        # 2. Create Publisher to /goal_pose
        self.publisher = self.create_publisher(
            PoseStamped, 
            '/goal_pose', 
            10)

        self.map = None
        self.frame_id = None
        self.width = None
        self.num_cells_height = None

        #   Timer
        self.viewpoint = self.create_timer(1, self.frontier_viewpoint)

    def map_callback(self, msg: OccupancyGrid):
        self.frame_id = msg.header.frame_id
        info = msg.info

        self.resolution = info.resolution

        self.width = info.width
        self.height = info.height 

        origin_x = info.origin.position.x 
        origin_y = info.origin.position.y
        self.origin = np.array([origin_x, origin_y])

        self.map = np.array(msg.data, dtype = float).reshape(self.height, self.width)
        # self.map = np.where(self.map > 50, 1, 0)

    def frontier_viewpoint(self):
        # 100 occupied, 0 free, 50 unknown
        frontier_cells = np.zeros((self.height, self.width))

        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.map[y, x] == 0:  # Cell is Free
                    # Check 4-neighbors for "Unknown" (50)
                    neighbors = [self.map[y-1, x], self.map[y+1, x], self.map[y, x-1], self.map[y, x+1]]
                    if 50 in neighbors:
                        if 100 not in neighbors:
                            frontier_cells[y,x] = 255
        
        thresh = cv2.threshold(frontier_cells, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        goal = PoseStamped()
        
        # Always set the frame_id to match your map (usually 'map')
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.frame_id

        # Example coordinates (Center of the map)
        goal.pose.position.x = 1.0
        goal.pose.position.y = 1.0
        goal.pose.orientation.w = 1.0

        # Publish the goal
        self.publisher.publish(goal)
        self.get_logger().info('Published a goal pose!')

def main(args=None):
    rclpy.init(args=args)
    node = FrontierBasedExploration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()