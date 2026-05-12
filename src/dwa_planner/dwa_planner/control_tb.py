import rclpy
from rclpy.node import Node
import numpy as np
import math

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

from dwa_interfaces.srv import ComputeVelocity


class DWAServiceNode(Node):
    """DWA local planner exposed as a ROS2 service.

    Maintains its own subscriptions to /turtlebot/odom and /inflated_map so
    callers only need to supply the current waypoint goal (goal_x, goal_y).
    Trajectory candidates are published to /dwa_trajectories as a side-effect
    of each service call so RViz visualisation works without any extra node.

    Service: /dwa/compute_velocity  (dwa_interfaces/srv/ComputeVelocity)
      Request : float64 goal_x, float64 goal_y
      Response: float64 linear_x, float64 angular_z, bool success
    """

    def __init__(self):
        super().__init__('dwa_service_node')

        # ── Robot state (updated by odom subscription) ───────────────────────
        self.robot_pose = None
        self.current_yaw = 0.0
        self.current_vel = [0.0, 0.0]
        self.grid_map = None

        # ── DWA kinematic limits ─────────────────────────────────────────────
        self.max_speed = 0.35
        self.max_yaw_rate = 1.8
        self.max_accel = 0.8
        self.max_delta_yaw = 1.2
        self.dt = 0.1
        self.predict_time = 2.5
        self.viz_time = 4.0

        # ── Cost weights ─────────────────────────────────────────────────────
        self.heading_w = 8.0
        self.dist_w = 6.0
        self.obstacle_w = 8.0
        self.velocity_w = 0.5

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('inflated_map_topic', '/inflated_map')
        map_topic = self.get_parameter('inflated_map_topic').value

        # Minimum velocities the real robot responds to (dead zone).
        # Set to 0.0 (default) in simulation — no clamping applied.
        self.declare_parameter('vel_deadzone_linear', 0.0)
        self.vel_deadzone_linear = self.get_parameter('vel_deadzone_linear').value
        self.declare_parameter('vel_deadzone_angular', 0.0)
        self.vel_deadzone_angular = self.get_parameter('vel_deadzone_angular').value

        # ── Publishers / subscribers ─────────────────────────────────────────
        self.marker_pub = self.create_publisher(MarkerArray, '/dwa_trajectories', 10)
        self.create_subscription(Odometry, '/turtlebot/odom', self._odom_cb, 10)
        self.create_subscription(OccupancyGrid, map_topic, self._map_cb, 10)
        self.get_logger().info(f'DWA costmap topic: {map_topic}')

        # ── Service server ───────────────────────────────────────────────────
        self.create_service(
            ComputeVelocity, '/dwa/compute_velocity', self._handle_compute)

        self.get_logger().info('DWA service node ready at /dwa/compute_velocity')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        self.robot_pose = msg.pose.pose.position
        self.current_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2))

    def _map_cb(self, msg):
        self.grid_map = msg

    # ── Service handler ──────────────────────────────────────────────────────

    def _handle_compute(self, request, response):
        if self.robot_pose is None or self.grid_map is None:
            response.success = False
            return response

        v, w, paths = self._compute_dwa(request.goal_x, request.goal_y)

        # Dead zone compensation: bump non-zero commands up to the minimum
        # velocity the real robot responds to.  No-ops when deadzone == 0.0.
        if self.vel_deadzone_linear > 0.0 and 0.0 < v < self.vel_deadzone_linear:
            v = self.vel_deadzone_linear
        if self.vel_deadzone_angular > 0.0 and 0.0 < abs(w) < self.vel_deadzone_angular:
            w = math.copysign(self.vel_deadzone_angular, w)

        self._publish_paths(paths, v, w)

        response.linear_x = float(v)
        response.angular_z = float(w)
        response.success = True
        return response

    # ── Obstacle query ───────────────────────────────────────────────────────

    def _get_cell_value(self, x, y):
        if self.grid_map is None:
            return None
        info = self.grid_map.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        if 0 <= gx < info.width and 0 <= gy < info.height:
            return self.grid_map.data[gy * info.width + gx]
        return None

    def _obstacle_cost(self, traj):
        """Soft penalty along sampled trajectory points.

        Lethal threshold >= 99 (inscribed + wall) matches the binary_map
        threshold used by BiRRT* so DWA and the global planner agree on what
        is passable.  Values 10-98 (inflation gradient) get soft penalties so
        DWA still prefers paths away from walls.
        """
        if self.grid_map is None:
            return 0.0
        penalty = 0.0
        for x, y, _ in traj[::3]:
            val = self._get_cell_value(x, y)
            if val is None or val >= 99:
                return float('inf')
            if val > 50:
                penalty += 1.0
            elif val > 30:
                penalty += 0.5
            elif val > 10:
                penalty += 0.2
        return penalty

    # ── DWA core ─────────────────────────────────────────────────────────────

    def _dynamic_window(self):
        v, w = self.current_vel
        v_min = max(0.0, v - self.max_accel * self.dt)
        v_max = min(self.max_speed, v + self.max_accel * self.dt)
        w_min = max(-self.max_yaw_rate, w - self.max_delta_yaw * self.dt)
        w_max = min(self.max_yaw_rate, w + self.max_delta_yaw * self.dt)
        return v_min, v_max, w_min, w_max

    def _simulate(self, v, w, horizon=None):
        if horizon is None:
            horizon = self.predict_time
        x, y, yaw = self.robot_pose.x, self.robot_pose.y, self.current_yaw
        traj = []
        for _ in range(max(1, int(horizon / self.dt))):
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += w * self.dt
            traj.append((x, y, yaw))
        return traj

    def _compute_dwa(self, goal_x, goal_y):
        dist_to_goal = math.hypot(
            goal_x - self.robot_pose.x, goal_y - self.robot_pose.y)
        # Clip horizon so trajectories never extend past the waypoint into
        # unknown space, which would make every trajectory return obs=inf.
        horizon = max(0.5, min(self.predict_time,
                               dist_to_goal / max(self.max_speed, 0.01)))

        v_min, v_max, w_min, w_max = self._dynamic_window()
        best_v, best_w = 0.0, 0.0
        best_cost = float('inf')
        all_paths = []

        for v in np.arange(v_min, v_max + 0.01, 0.03):
            for w in np.arange(w_min, w_max + 0.01, 0.06):
                traj = self._simulate(v, w, horizon)
                obs = self._obstacle_cost(traj)
                if obs == float('inf'):
                    continue
                lx, ly, lyaw = traj[-1]
                ga = math.atan2(goal_y - ly, goal_x - lx)
                heading_err = abs(math.atan2(
                    math.sin(ga - lyaw), math.cos(ga - lyaw)))
                dist = math.hypot(goal_x - lx, goal_y - ly)
                cost = (self.heading_w * heading_err +
                        self.dist_w * dist +
                        self.obstacle_w * obs +
                        self.velocity_w * (self.max_speed - v))
                all_paths.append({'v': v, 'w': w, 'traj': traj, 'cost': cost})
                if cost < best_cost:
                    best_cost = cost
                    best_v, best_w = v, w

        return best_v, best_w, all_paths

    # ── Trajectory visualisation ─────────────────────────────────────────────

    def _publish_paths(self, paths, bv, bw):
        if self.grid_map is None:
            return
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        frame = self.grid_map.header.frame_id
        lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()

        cid = 0
        for p in paths[::3]:
            if abs(p['v'] - bv) < 1e-3 and abs(p['w'] - bw) < 1e-3:
                continue
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = now
            m.ns = 'dwa_candidates'
            m.id = cid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.02
            m.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.35)
            m.pose.orientation.w = 1.0
            m.lifetime = lifetime
            for x, y, _ in self._simulate(p['v'], p['w'], self.viz_time):
                pt = Point()
                pt.x = float(x)
                pt.y = float(y)
                pt.z = 0.05
                m.points.append(pt)
            marker_array.markers.append(m)
            cid += 1

        # Best trajectory drawn last so it renders on top
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = now
        m.ns = 'dwa_best'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.10
        m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        m.pose.orientation.w = 1.0
        m.lifetime = lifetime
        for x, y, _ in self._simulate(bv, bw, self.viz_time):
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = 0.30
            m.points.append(pt)
        marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = DWAServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
