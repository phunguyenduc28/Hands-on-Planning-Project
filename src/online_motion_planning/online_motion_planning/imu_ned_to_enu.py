import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuNedToEnu(Node):
    """Convert IMU data from world_NED to world_ENU.

    The stonefish simulation publishes the IMU orientation in the NED
    (North-East-Down) convention.  All other nodes work in ENU
    (East-North-Up), so yaw must be negated before fusion.

    Conversion applied:
        yaw_enu   = -yaw_ned
        omega_z_enu = -omega_z_ned   (yaw rate sign flips with frame)
        ax_enu  = ax_ned,  ay_enu = -ay_ned   (NED Y→ENU -Y)

    The output quaternion is rebuilt from yaw_enu only (roll=pitch=0)
    because this is a planar robot and only yaw is fused by the EKF.
    """

    def __init__(self):
        super().__init__('imu_ned_to_enu')

        self.declare_parameter('input_topic',  '/turtlebot/sensors/imu_data')
        self.declare_parameter('output_topic', '/turtlebot/sensors/imu_enu')

        in_topic  = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value

        self._pub = self.create_publisher(Imu, out_topic, 10)
        self.create_subscription(Imu, in_topic, self._cb, 10)
        self.get_logger().info(f'IMU NED→ENU: {in_topic} → {out_topic}')

    def _yaw_from_quat(self, q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _quat_from_yaw(self, yaw):
        return math.sin(yaw / 2.0), math.cos(yaw / 2.0)  # (z, w)

    def _cb(self, msg: Imu):
        out = Imu()
        out.header = msg.header
        out.header.frame_id = 'turtlebot/imu_enu'

        # Negate yaw to convert NED → ENU, rebuild quaternion (planar robot)
        yaw_ned = self._yaw_from_quat(msg.orientation)
        yaw_enu = -yaw_ned
        qz, qw = self._quat_from_yaw(yaw_enu)
        out.orientation.x = 0.0
        out.orientation.y = 0.0
        out.orientation.z = qz
        out.orientation.w = qw
        out.orientation_covariance = msg.orientation_covariance

        # Yaw rate: negate z component (frame flip)
        out.angular_velocity.x =  msg.angular_velocity.x
        out.angular_velocity.y =  msg.angular_velocity.y
        out.angular_velocity.z = -msg.angular_velocity.z
        out.angular_velocity_covariance = msg.angular_velocity_covariance

        # Linear acceleration: NED→ENU flips Y and Z signs
        out.linear_acceleration.x =  msg.linear_acceleration.x
        out.linear_acceleration.y = -msg.linear_acceleration.y
        out.linear_acceleration.z = -msg.linear_acceleration.z
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImuNedToEnu()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
