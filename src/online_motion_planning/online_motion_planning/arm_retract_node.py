import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState


class ArmRetractNode(Node):
    """Arm retraction service node.

    Drives the SwiftPro arm to its safe retracted configuration using a
    proportional joint-velocity controller.  Active only in simulation
    (is_sim=true); on a real robot the service responds immediately with
    success so callers need not special-case the robot type.

    Automatically retracts at startup when is_sim=true.

    Service
      /arm/retract  (std_srvs/Trigger) — start retraction; returns immediately

    Publishers
      /arm/is_retracted (std_msgs/Bool) — True once arm is within tolerance
      /turtlebot/swiftpro/joint_velocity_controller/command (Float64MultiArray)

    Subscribers
      /turtlebot/joint_states (sensor_msgs/JointState)
    """

    ARM_Q2_RETRACT = 0.040
    ARM_Q3_RETRACT = -1.45
    ARM_RETRACT_TOL = 0.06
    ARM_KP = 2.0
    ARM_MAX_VEL = 0.3

    def __init__(self):
        super().__init__('arm_retract_node')

        self.declare_parameter('is_sim', False)
        self.is_sim = self.get_parameter('is_sim').value

        self.arm_q2 = None
        self.arm_q3 = None
        # On the real robot (no arm) treat arm as always retracted so the
        # path planner is never blocked.
        self._retracted = not self.is_sim
        self._retracting = False

        self.arm_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/turtlebot/swiftpro/joint_velocity_controller/command', 10)
        self.retracted_pub = self.create_publisher(
            Bool, '/arm/is_retracted', 10)

        self.create_subscription(
            JointState, '/turtlebot/joint_states', self._joint_state_cb, 10)

        self.create_service(Trigger, '/arm/retract', self._handle_retract)

        self.create_timer(0.1, self._control_tick)

        if self.is_sim:
            self._retracting = True
            self.get_logger().info(
                'is_sim=True — auto-retracting arm at startup')
        else:
            self.get_logger().info(
                'is_sim=False — arm retraction disabled (real robot)')

    def _joint_state_cb(self, msg):
        pos_map = dict(zip(msg.name, msg.position))
        if 'turtlebot/swiftpro/joint2' in pos_map:
            self.arm_q2 = pos_map['turtlebot/swiftpro/joint2']
        if 'turtlebot/swiftpro/joint3' in pos_map:
            self.arm_q3 = pos_map['turtlebot/swiftpro/joint3']

    def _handle_retract(self, request, response):
        if not self.is_sim:
            response.success = True
            response.message = 'Arm retraction skipped (real robot mode)'
            return response
        if self._retracted:
            response.success = True
            response.message = 'Arm already retracted'
            return response
        self._retracting = True
        self._retracted = False
        response.success = True
        response.message = 'Arm retraction started'
        self.get_logger().info('Arm retraction requested via service')
        return response

    def _control_tick(self):
        msg = Bool()
        msg.data = self._retracted
        self.retracted_pub.publish(msg)

        if not self._retracting or not self.is_sim:
            return
        if self.arm_q2 is None or self.arm_q3 is None:
            return

        err2 = self.ARM_Q2_RETRACT - self.arm_q2
        err3 = self.ARM_Q3_RETRACT - self.arm_q3

        if abs(err2) < self.ARM_RETRACT_TOL and abs(err3) < self.ARM_RETRACT_TOL:
            stop = Float64MultiArray()
            stop.data = [0.0, 0.0, 0.0, 0.0]
            self.arm_cmd_pub.publish(stop)
            self._retracting = False
            self._retracted = True
            self.get_logger().info('Arm retracted successfully')
            return

        dq2 = float(np.clip(self.ARM_KP * err2, -self.ARM_MAX_VEL, self.ARM_MAX_VEL))
        dq3 = float(np.clip(self.ARM_KP * err3, -self.ARM_MAX_VEL, self.ARM_MAX_VEL))

        # Soft joint-limit guards
        if self.arm_q2 >= 0.045 and dq2 > 0:
            dq2 = 0.0
        if self.arm_q2 <= -1.50 and dq2 < 0:
            dq2 = 0.0
        if self.arm_q3 >= 0.045 and dq3 > 0:
            dq3 = 0.0
        if self.arm_q3 <= -1.50 and dq3 < 0:
            dq3 = 0.0

        cmd = Float64MultiArray()
        cmd.data = [0.0, dq2, dq3, 0.0]
        self.arm_cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ArmRetractNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
