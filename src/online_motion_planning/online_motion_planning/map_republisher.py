import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


class MapRepublisher(Node):
    """Republishes the latest RTAB-Map /map at a fixed rate.

    RTAB-Map publishes /map only when the map changes, which can be very
    infrequent on the real robot.  Nodes that depend on a steady map stream
    (frontier detection, global costmap) stall waiting for the next message.
    This node caches the last received map and re-publishes it at publish_rate
    Hz so downstream nodes keep running with up-to-date data.

    Parameters
    ----------
    input_topic   (str,   default '/map')          — RTAB-Map map topic
    output_topic  (str,   default '/map_fast')     — high-rate republished topic
    publish_rate  (float, default 2.0)             — republish frequency (Hz)
    """

    def __init__(self):
        super().__init__('map_republisher')

        self.declare_parameter('input_topic',  '/map')
        self.declare_parameter('output_topic', '/map_fast')
        self.declare_parameter('publish_rate', 2.0)

        in_topic  = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value
        rate      = self.get_parameter('publish_rate').value

        self._latest_map = None

        self._pub = self.create_publisher(OccupancyGrid, out_topic, 10)
        self.create_subscription(OccupancyGrid, in_topic, self._map_cb, 10)
        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f'MapRepublisher: {in_topic} → {out_topic} @ {rate:.1f} Hz')

    def _map_cb(self, msg: OccupancyGrid):
        if self._latest_map is None:
            self.get_logger().info('First map received from RTAB-Map — republishing started')
        self._latest_map = msg

    def _publish(self):
        if self._latest_map is not None:
            self._pub.publish(self._latest_map)


def main(args=None):
    rclpy.init(args=args)
    node = MapRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
