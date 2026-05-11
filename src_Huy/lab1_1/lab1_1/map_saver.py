#!/usr/bin/env python3
"""
Map Saver Node
Subscribes to /map topic and saves the occupancy grid as PGM image + YAML metadata.
https://wiki.ros.org/map_server#map_saver <- inspiration for code structure and file format
https://docs.ros.org/en/humble/p/nav2_map_server/ <- ROS 2 version of map_saver
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import os
from datetime import datetime


class MapSaver(Node):
    def __init__(self):
        super().__init__('map_saver')
        
        # Parameter for save directory
        self.declare_parameter('save_dir', os.path.expanduser('~/maps'))
        self.save_dir = self.get_parameter('save_dir').value
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Subscription to map topic
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        
        self.map_received = False
        self.get_logger().info(f"Map Saver started. Maps will be saved to {self.save_dir}")
        self.get_logger().info("Send a trigger by publishing empty message to /save_map or call service")
        
        # Service to trigger map saving
        from std_srvs.srv import Empty
        self.save_service = self.create_service(
            Empty, '/save_map', self.save_map_service_callback
        )
    
    def map_callback(self, msg):
        """Store latest occupancy grid."""
        self.latest_map = msg
        self.map_received = True
    
    def save_map_service_callback(self, request, response):
        """Service callback to save the current map."""
        if not self.map_received:
            self.get_logger().error("No map received yet!")
            return response
        
        self.save_occupancy_grid(self.latest_map)
        return response
    
    def save_occupancy_grid(self, map_msg):
        """
        Save occupancy grid as PGM image + YAML metadata.
        Follows standard ROS map format.
        """
        # Extract grid data
        width = map_msg.info.width
        height = map_msg.info.height
        resolution = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y
        
        # Convert to numpy array
        grid_data = np.array(map_msg.data, dtype=np.int8).reshape((height, width))
        
        # Convert to PGM format (0-254, where 0=occupied, 254=free, -1=unknown becomes 127)
        pgm_data = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                value = grid_data[i, j]
                if value == -1:  # Unknown
                    pgm_data[i, j] = 127
                else:
                    # ROS uses 0-100, PGM uses 0-254
                    # ROS: 0=free, 100=occupied
                    # PGM: 0=occupied, 254=free (inverted!)
                    pgm_data[i, j] = max(0, min(254, int((100 - value) * 2.54)))
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_name = f"map_{timestamp}"
        
        pgm_file = os.path.join(self.save_dir, f"{map_name}.pgm")
        yaml_file = os.path.join(self.save_dir, f"{map_name}.yaml")
        
        # Save PGM image
        try:
            from PIL import Image
            img = Image.fromarray(pgm_data)
            img.save(pgm_file)
            self.get_logger().info(f"Saved PGM image to {pgm_file}")
        except ImportError:
            # Fallback: write PGM manually without PIL
            self.write_pgm_manually(pgm_file, pgm_data)
        
        # Save YAML metadata
        self.write_yaml_metadata(yaml_file, map_name, resolution, origin_x, origin_y)
        
        self.get_logger().info(f"Map saved successfully: {map_name}")
    
    def write_pgm_manually(self, filename, data):
        """Write PGM file without PIL (fallback method)."""
        height, width = data.shape
        with open(filename, 'wb') as f:
            # PGM header
            f.write(b'P5\n')
            f.write(f'{width} {height}\n'.encode())
            f.write(b'255\n')
            # Data
            f.write(data.tobytes())
        self.get_logger().info(f"Saved PGM image (no PIL) to {filename}")
    
    def write_yaml_metadata(self, filename, map_name, resolution, origin_x, origin_y):
        """Write YAML metadata file for the map (ROS format)."""
        yaml_content = f"""image: {map_name}.pgm
        resolution: {resolution}
        origin: [{origin_x}, {origin_y}, 0.0]
        occupied_thresh: 0.65
        free_thresh: 0.196
        negate: 0
        """
        with open(filename, 'w') as f:
            f.write(yaml_content)
        self.get_logger().info(f"Saved YAML metadata to {filename}")


def main():
    rclpy.init()
    node = MapSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
