#!/usr/bin/env python3
"""
image_crop_node.py

Crops depth and RGB images for two downstream consumers:

  1. RTAB-Map  — bottom N% removed to exclude floor from loop-closure features.
     Publishes to /turtlebot/camera/{color,depth}/image_cropped.

  2. depthimage_to_laserscan (fake laser)
                — top AND bottom N% removed to exclude robot arm and floor.
     Publishes to /turtlebot/camera/depth/image_scan_cropped
                  /turtlebot/camera/depth/camera_info_scan_cropped

Topics subscribed:
  /turtlebot/camera/color/image_color       (sensor_msgs/Image)
  /turtlebot/camera/color/camera_info       (sensor_msgs/CameraInfo)
  /turtlebot/camera/depth/image_depth       (sensor_msgs/Image)
  /turtlebot/camera/depth/camera_info       (sensor_msgs/CameraInfo)

Topics published:
  /turtlebot/camera/color/image_cropped          (sensor_msgs/Image)
  /turtlebot/camera/color/camera_info_cropped    (sensor_msgs/CameraInfo)
  /turtlebot/camera/depth/image_cropped          (sensor_msgs/Image)
  /turtlebot/camera/depth/camera_info_cropped    (sensor_msgs/CameraInfo)
  /turtlebot/camera/depth/image_scan_cropped     (sensor_msgs/Image)
  /turtlebot/camera/depth/camera_info_scan_cropped (sensor_msgs/CameraInfo)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import numpy as np


class ImageCropNode(Node):
    def __init__(self):
        super().__init__('image_crop_node')

        # Fraction of image height to crop from the bottom (for RTAB-Map)
        self.declare_parameter('crop_bottom_fraction', 0.49)
        self.crop_frac = self.get_parameter('crop_bottom_fraction').value

        # Top and bottom crop fractions for the fake laser scan depth output.
        # Top crop removes the robot arm; bottom crop removes the floor.
        self.declare_parameter('scan_crop_top_fraction', 0.25)
        self.scan_crop_top_frac = self.get_parameter('scan_crop_top_fraction').value

        self.declare_parameter('scan_crop_bottom_fraction', 0.49)
        self.scan_crop_bottom_frac = self.get_parameter('scan_crop_bottom_fraction').value

        # Spatial outlier filter for the scan depth image.
        # Pixels whose depth exceeds their 3×3 neighbourhood median by more than
        # depth_outlier_threshold (metres) are replaced with NaN.  This removes
        # pixels that 'look through' or 'over' nearby walls due to the arm or
        # camera geometry, which would otherwise produce spurious far readings.
        self.declare_parameter('depth_filter_enabled', True)
        self.depth_filter_enabled = self.get_parameter('depth_filter_enabled').value
        self.declare_parameter('depth_outlier_threshold', 0.3)
        self.depth_outlier_threshold = self.get_parameter('depth_outlier_threshold').value

        self.get_logger().info(
            f'ImageCropNode: rtab_bottom={self.crop_frac*100:.0f}%  '
            f'scan_top={self.scan_crop_top_frac*100:.0f}%  '
            f'scan_bottom={self.scan_crop_bottom_frac*100:.0f}%  '
            f'filter={self.depth_filter_enabled} '
            f'(thr={self.depth_outlier_threshold}m)')

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

        # Top-cropped depth for depthimage_to_laserscan (arm excluded)
        self.scan_depth_pub = self.create_publisher(
            Image,
            '/turtlebot/camera/depth/image_scan_cropped', 10)

        self.scan_depth_info_pub = self.create_publisher(
            CameraInfo,
            '/turtlebot/camera/depth/camera_info_scan_cropped', 10)

    # ── Core crop helpers ─────────────────────────────────────────────────────

    def _crop_bottom(self, msg: Image, frame_id: str) -> Image:
        """Remove the bottom crop_frac rows (removes floor for RTAB-Map)."""
        keep_rows = int(msg.height * (1.0 - self.crop_frac))
        raw = bytes(msg.data)
        out = Image()
        out.header = msg.header
        out.header.frame_id = frame_id
        out.encoding = msg.encoding
        out.width = msg.width
        out.height = keep_rows
        out.step = msg.step
        out.is_bigendian = msg.is_bigendian
        out.data = raw[:keep_rows * msg.step]
        return out

    def _crop_bottom_info(self, msg: CameraInfo, new_height: int,
                          frame_id: str) -> CameraInfo:
        """Adjust CameraInfo after bottom crop — cy unchanged (origin top-left)."""
        out = CameraInfo()
        out.header = msg.header
        out.header.frame_id = frame_id
        out.width = msg.width
        out.height = new_height
        out.k = list(msg.k)
        out.d = list(msg.d)
        out.r = list(msg.r)
        out.p = list(msg.p)
        out.distortion_model = msg.distortion_model
        out.binning_x = msg.binning_x
        out.binning_y = msg.binning_y
        out.roi = msg.roi
        return out

    def _crop_top_bottom(self, msg: Image, frame_id: str) -> Image:
        """Remove top (arm) and bottom (floor) rows for the fake laser scan."""
        top_rows = int(msg.height * self.scan_crop_top_frac)
        bottom_rows = int(msg.height * self.scan_crop_bottom_frac)
        keep_rows = max(1, msg.height - top_rows - bottom_rows)
        raw = bytes(msg.data)
        start = top_rows * msg.step
        out = Image()
        out.header = msg.header
        out.header.frame_id = frame_id
        out.encoding = msg.encoding
        out.width = msg.width
        out.height = keep_rows
        out.step = msg.step
        out.is_bigendian = msg.is_bigendian
        out.data = raw[start: start + keep_rows * msg.step]
        return out

    def _crop_top_bottom_info(self, msg: CameraInfo, top_rows: int,
                              new_height: int, frame_id: str) -> CameraInfo:
        """Adjust CameraInfo after top+bottom crop — cy shifts up by top_rows."""
        out = CameraInfo()
        out.header = msg.header
        out.header.frame_id = frame_id
        out.width = msg.width
        out.height = new_height
        out.k = list(msg.k)
        out.k[5] = msg.k[5] - float(top_rows)   # cy shifts up
        out.d = list(msg.d)
        out.r = list(msg.r)
        out.p = list(msg.p)
        out.p[6] = msg.p[6] - float(top_rows)   # cy in projection matrix
        out.distortion_model = msg.distortion_model
        out.binning_x = msg.binning_x
        out.binning_y = msg.binning_y
        out.roi = msg.roi
        return out

    # ── Depth outlier filter ───────────────────────────────────────────────────

    def _spatial_outlier_remove(self, arr: np.ndarray,
                                threshold: float) -> np.ndarray:
        """Replace pixels more than `threshold` units farther than their
        3×3 neighbourhood median with NaN.

        Removes depth pixels that 'look through' or 'over' nearby walls due to
        arm geometry or sensor noise near depth discontinuities.
        Works on float arrays; caller converts to/from native encoding.
        """
        h, w = arr.shape
        # Build 8-neighbour stack without scipy — pad edges then slice
        p = np.pad(arr, 1, mode='edge')
        nbrs = np.stack([
            p[0:h,   0:w],   p[0:h,   1:w+1], p[0:h,   2:w+2],
            p[1:h+1, 0:w],                     p[1:h+1, 2:w+2],
            p[2:h+2, 0:w],   p[2:h+2, 1:w+1], p[2:h+2, 2:w+2],
        ], axis=0)
        med = np.nanmedian(nbrs, axis=0)
        result = arr.copy()
        valid = (np.isfinite(arr) & (arr > 0)
                 & np.isfinite(med) & (med > 0))
        result[valid & ((arr - med) > threshold)] = np.nan
        return result

    def _filter_scan_depth(self, msg: Image) -> Image:
        """Apply spatial outlier filter to a depth Image message."""
        enc = msg.encoding
        if enc == '32FC1':
            arr = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(
                msg.height, msg.width).copy()
            arr[arr <= 0] = np.nan
            arr = self._spatial_outlier_remove(arr, self.depth_outlier_threshold)
            data = np.nan_to_num(arr, nan=0.0).astype(np.float32).tobytes()
        elif enc == '16UC1':
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(
                msg.height, msg.width).astype(np.float32)
            arr[arr == 0] = np.nan
            # threshold in mm (depth_outlier_threshold is in metres)
            arr = self._spatial_outlier_remove(
                arr, self.depth_outlier_threshold * 1000.0)
            data = np.nan_to_num(arr, nan=0.0).astype(np.uint16).tobytes()
        else:
            return msg  # unsupported encoding — return unchanged
        out = Image()
        out.header = msg.header
        out.encoding = enc
        out.width = msg.width
        out.height = msg.height
        out.step = msg.step
        out.is_bigendian = msg.is_bigendian
        out.data = data
        return out

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def rgb_callback(self, msg: Image):
        self.rgb_pub.publish(
            self._crop_bottom(msg, frame_id='camera_color_optical_frame'))

    def rgb_info_callback(self, msg: CameraInfo):
        new_h = int(msg.height * (1.0 - self.crop_frac))
        self.rgb_info_pub.publish(
            self._crop_bottom_info(msg, new_h, frame_id='camera_color_optical_frame'))

    def depth_callback(self, msg: Image):
        # Bottom-cropped → RTAB-Map (floor removed)
        self.depth_pub.publish(
            self._crop_bottom(msg, frame_id='camera_depth_optical_frame'))
        # Top+bottom-cropped + outlier-filtered → depthimage_to_laserscan
        scan_img = self._crop_top_bottom(msg, frame_id='camera_depth_optical_frame')
        if self.depth_filter_enabled:
            scan_img = self._filter_scan_depth(scan_img)
        self.scan_depth_pub.publish(scan_img)

    def depth_info_callback(self, msg: CameraInfo):
        # Bottom-cropped info → RTAB-Map
        new_h = int(msg.height * (1.0 - self.crop_frac))
        self.depth_info_pub.publish(
            self._crop_bottom_info(msg, new_h, frame_id='camera_depth_optical_frame'))
        # Top+bottom-cropped info → depthimage_to_laserscan
        top_rows = int(msg.height * self.scan_crop_top_frac)
        bottom_rows = int(msg.height * self.scan_crop_bottom_frac)
        new_h_scan = max(1, msg.height - top_rows - bottom_rows)
        self.scan_depth_info_pub.publish(
            self._crop_top_bottom_info(msg, top_rows, new_h_scan,
                                       frame_id='camera_depth_optical_frame'))


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
