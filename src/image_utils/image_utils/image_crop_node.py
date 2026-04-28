#!/usr/bin/env python3
"""
image_crop_node.py

Crops the bottom N% of RGB and depth images to remove floor from
RTAB-Map's field of view, preventing false loop closures from
repetitive floor patterns.

Topics subscribed:
  /turtlebot/camera/color/image_color       (sensor_msgs/Image)
  /turtlebot/camera/color/camera_info       (sensor_msgs/CameraInfo)
  /turtlebot/camera/depth/image_depth       (sensor_msgs/Image)
  /turtlebot/camera/depth/camera_info       (sensor_msgs/CameraInfo)

Topics published:
  /turtlebot/camera/color/image_cropped     (sensor_msgs/Image)
  /turtlebot/camera/color/camera_info_cropped (sensor_msgs/CameraInfo)
  /turtlebot/camera/depth/image_cropped     (sensor_msgs/Image)
  /turtlebot/camera/depth/camera_info_cropped (sensor_msgs/CameraInfo)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import numpy as np


class ImageCropNode(Node):
    def __init__(self):
        super().__init__('image_crop_node')

        # --- Parameters ---
        # Fraction of image height to crop from the bottom (0.0 - 1.0)
        # 0.3 = crop bottom 30% of the image
        self.declare_parameter('crop_bottom_fraction', 0.49)
        self.crop_frac = self.get_parameter('crop_bottom_fraction').value

        self.get_logger().info(
            f"ImageCropNode started. Cropping bottom {self.crop_frac*100:.0f}% of images.")

        # --- RGB subscribers/publishers ---
        self.rgb_sub = self.create_subscription(
            Image,
            '/turtlebot/camera/color/image_color',
            self.rgb_callback, 10)

        self.rgb_info_sub = self.create_subscription(
            CameraInfo,
            '/turtlebot/camera/color/camera_info',
            self.rgb_info_callback, 10)
        
        # self.rgb_sub = self.create_subscription(
        #     Image,
        #     '/turtlebot/camera/color/image_compressed',
        #     self.rgb_callback, 10)

        # self.rgb_info_sub = self.create_subscription(
        #     CameraInfo,
        #     '/turtlebot/camera/color/camera_info',
        #     self.rgb_info_callback, 10)

        self.rgb_pub = self.create_publisher(
            Image,
            '/turtlebot/camera/color/image_cropped', 10)

        self.rgb_info_pub = self.create_publisher(
            CameraInfo,
            '/turtlebot/camera/color/camera_info_cropped', 10)

        # --- Depth subscribers/publishers ---
        self.depth_sub = self.create_subscription(
            Image,
            '/turtlebot/camera/depth/image_depth',
            self.depth_callback, 10)
        
        # self.depth_sub = self.create_subscription(
        #     Image,
        #     '/turtlebot/camera/depth/image_rect_raw',
        #     self.depth_callback, 10)

        self.depth_info_sub = self.create_subscription(
            CameraInfo,
            '/turtlebot/camera/depth/camera_info',
            self.depth_info_callback, 10)

        self.depth_pub = self.create_publisher(
            Image,
            '/turtlebot/camera/depth/image_cropped', 10)

        self.depth_info_pub = self.create_publisher(
            CameraInfo,
            '/turtlebot/camera/depth/camera_info_cropped', 10)

    # Core crop logic
    def crop_image_msg(self, msg: Image) -> Image:
        """Crop bottom fraction from a sensor_msgs/Image."""
        height = msg.height
        width  = msg.width

        # How many rows to keep
        keep_rows = int(height * (1.0 - self.crop_frac))

        # Bytes per row
        row_step = msg.step  # bytes per full row

        # Slice the raw data (bytearray/bytes)
        raw = bytes(msg.data)
        cropped_data = raw[:keep_rows * row_step]

        out = Image()
        out.header   = msg.header
        out.encoding = msg.encoding
        out.width    = width
        out.height   = keep_rows
        out.step     = row_step
        out.is_bigendian = msg.is_bigendian
        out.data     = cropped_data
        return out

    def crop_camera_info(self, msg: CameraInfo, new_height: int) -> CameraInfo:
        """
        Adjust CameraInfo for cropped image.
        Only the principal point cy changes — cropping the bottom
        does NOT shift cy because we crop from the bottom, not the top.
        However height must be updated so downstream nodes allocate
        correctly sized buffers.
        """
        out = CameraInfo()
        out.header = msg.header
        out.width  = msg.width
        out.height = new_height

        # K matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        # cx, cy, fx, fy unchanged — we crop bottom, origin stays top-left
        out.k = list(msg.k)
        out.d = list(msg.d)
        out.r = list(msg.r)
        out.p = list(msg.p)
        out.distortion_model = msg.distortion_model
        out.binning_x = msg.binning_x
        out.binning_y = msg.binning_y
        out.roi = msg.roi
        return out

    # Callbacks
    def rgb_callback(self, msg: Image):
        cropped = self.crop_image_msg(msg)
        self.rgb_pub.publish(cropped)

    def rgb_info_callback(self, msg: CameraInfo):
        new_height = int(msg.height * (1.0 - self.crop_frac))
        self.rgb_info_pub.publish(self.crop_camera_info(msg, new_height))

    def depth_callback(self, msg: Image):
        cropped = self.crop_image_msg(msg)
        self.depth_pub.publish(cropped)

    def depth_info_callback(self, msg: CameraInfo):
        new_height = int(msg.height * (1.0 - self.crop_frac))
        self.depth_info_pub.publish(self.crop_camera_info(msg, new_height))


def main():
    rclpy.init()
    node = ImageCropNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
