import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan
import message_filters


class RtabmapSyncNode(Node):
    """Synchronise the five streams RTAB-Map needs before they arrive.

    RTAB-Map drops frames when its input topics are not time-aligned.  This
    node collects:
        • cropped colour image
        • cropped depth image
        • colour camera_info
        • depth camera_info
        • fake laser scan (from depthimage_to_laserscan)

    and publishes them together only when all five arrive within `slop`
    seconds of each other.  Each republished message keeps its original
    content but has its stamp overwritten to the colour image stamp so
    RTAB-Map sees a single coherent time base.

    Parameters
    ----------
    slop                   (float, default 0.1 s) — max allowed timestamp spread
    queue_size             (int,   default 10)    — per-topic message buffer
    color_in               input colour topic
    depth_in               input depth topic
    camera_info_in         input colour camera_info topic
    depth_camera_info_in   input depth camera_info topic
    scan_in                input fake scan topic
    color_out              republished colour topic
    depth_out              republished depth topic
    camera_info_out        republished colour camera_info topic
    depth_camera_info_out  republished depth camera_info topic
    scan_out               republished scan topic
    """

    def __init__(self):
        super().__init__('rtabmap_sync_node')

        self.declare_parameter('slop',       0.2)
        self.declare_parameter('queue_size', 10000)

        self.declare_parameter('color_in',
            '/turtlebot/camera/color/image_cropped')
        self.declare_parameter('depth_in',
            '/turtlebot/camera/depth/image_cropped')
        self.declare_parameter('camera_info_in',
            '/turtlebot/camera/color/camera_info_cropped')
        self.declare_parameter('depth_camera_info_in',
            '/turtlebot/camera/depth/camera_info_cropped')
        self.declare_parameter('scan_in',
            '/turtlebot/fake_scan')

        self.declare_parameter('color_out',
            '/sync/camera/color/image_cropped')
        self.declare_parameter('depth_out',
            '/sync/camera/depth/image_cropped')
        self.declare_parameter('camera_info_out',
            '/sync/camera/color/camera_info_cropped')
        self.declare_parameter('depth_camera_info_out',
            '/sync/camera/depth/camera_info_cropped')
        self.declare_parameter('scan_out',
            '/sync/fake_scan')

        slop       = self.get_parameter('slop').value
        queue_size = self.get_parameter('queue_size').value

        color_in             = self.get_parameter('color_in').value
        depth_in             = self.get_parameter('depth_in').value
        camera_info_in       = self.get_parameter('camera_info_in').value
        depth_camera_info_in = self.get_parameter('depth_camera_info_in').value
        scan_in              = self.get_parameter('scan_in').value

        color_out             = self.get_parameter('color_out').value
        depth_out             = self.get_parameter('depth_out').value
        camera_info_out       = self.get_parameter('camera_info_out').value
        depth_camera_info_out = self.get_parameter('depth_camera_info_out').value
        scan_out              = self.get_parameter('scan_out').value

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_color      = self.create_publisher(Image,      color_out,             10)
        self._pub_depth      = self.create_publisher(Image,      depth_out,             10)
        self._pub_info       = self.create_publisher(CameraInfo, camera_info_out,       10)
        self._pub_depth_info = self.create_publisher(CameraInfo, depth_camera_info_out, 10)
        self._pub_scan       = self.create_publisher(LaserScan,  scan_out,              10)

        # ── Subscribers via message_filters ───────────────────────────────────
        self._sub_color      = message_filters.Subscriber(self, Image,      color_in)
        self._sub_depth      = message_filters.Subscriber(self, Image,      depth_in)
        self._sub_info       = message_filters.Subscriber(self, CameraInfo, camera_info_in)
        self._sub_depth_info = message_filters.Subscriber(self, CameraInfo, depth_camera_info_in)
        self._sub_scan       = message_filters.Subscriber(self, LaserScan,  scan_in)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_color, self._sub_depth, self._sub_info,
             self._sub_depth_info, self._sub_scan],
            queue_size=queue_size,
            slop=slop,
            allow_headerless=False,
        )
        self._sync.registerCallback(self._sync_cb)

        self.get_logger().info(
            f'RtabmapSyncNode ready  slop={slop}s\n'
            f'  IN:  {color_in}\n'
            f'       {depth_in}\n'
            f'       {camera_info_in}\n'
            f'       {depth_camera_info_in}\n'
            f'       {scan_in}\n'
            f'  OUT: {color_out}\n'
            f'       {depth_out}\n'
            f'       {camera_info_out}\n'
            f'       {depth_camera_info_out}\n'
            f'       {scan_out}')

    def _sync_cb(self, color: Image, depth: Image,
                 info: CameraInfo, depth_info: CameraInfo, scan: LaserScan):
        """All five messages arrived within slop — republish with unified stamp."""
        stamp = color.header.stamp

        color.header.stamp      = stamp
        depth.header.stamp      = stamp
        info.header.stamp       = stamp
        depth_info.header.stamp = stamp
        scan.header.stamp       = stamp

        self._pub_color.publish(color)
        self._pub_depth.publish(depth)
        self._pub_info.publish(info)
        self._pub_depth_info.publish(depth_info)
        self._pub_scan.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = RtabmapSyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
