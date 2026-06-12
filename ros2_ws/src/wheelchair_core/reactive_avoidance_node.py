#!/usr/bin/env python3
"""Reactive LiDAR-based obstacle avoidance for prototype operation.

Pipeline:
  /cmd_vel_raw -> reactive_avoidance_node -> /cmd_vel_safe

Behavior:
  - Critical zone (-15..+15 deg):
      If obstacle < critical_distance, stop forward motion and rotate away.
  - Warning zone (-40..+40 deg):
      If obstacle < warning_distance, reduce forward speed and steer away.
  - Clear path:
      Pass command through for differential-drive axes only.

Fail-safe:
  - If /scan data becomes stale, publish a full stop.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class ReactiveAvoidanceNode(Node):
    def __init__(self) -> None:
        super().__init__('reactive_avoidance_node')

        # Topics
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_in_topic', '/cmd_vel_raw')
        self.declare_parameter('cmd_out_topic', '/cmd_vel_safe')

        # Detection zones and distances
        self.declare_parameter('critical_angle_deg', 15.0)
        self.declare_parameter('warning_angle_deg', 40.0)
        self.declare_parameter('critical_distance_m', 0.5)
        self.declare_parameter('warning_distance_m', 0.8)

        # Control shaping
        self.declare_parameter('critical_turn_speed_rad_s', 0.6)
        self.declare_parameter('warning_steer_gain_rad_s', 0.8)

        # Safety and timing
        self.declare_parameter('scan_timeout_s', 0.6)
        self.declare_parameter('publish_rate_hz', 20.0)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.cmd_in_topic = self.get_parameter('cmd_in_topic').value
        self.cmd_out_topic = self.get_parameter('cmd_out_topic').value

        self.critical_angle_deg = float(self.get_parameter('critical_angle_deg').value)
        self.warning_angle_deg = float(self.get_parameter('warning_angle_deg').value)
        self.critical_distance = float(self.get_parameter('critical_distance_m').value)
        self.warning_distance = float(self.get_parameter('warning_distance_m').value)

        self.critical_turn_speed = float(self.get_parameter('critical_turn_speed_rad_s').value)
        self.warning_steer_gain = float(self.get_parameter('warning_steer_gain_rad_s').value)

        self.scan_timeout_s = float(self.get_parameter('scan_timeout_s').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        if self.warning_distance <= self.critical_distance:
            self.get_logger().warn(
                'warning_distance_m should be > critical_distance_m. '
                'Adjusting warning_distance_m automatically.'
            )
            self.warning_distance = self.critical_distance + 0.1

        # Input command to protect
        self.last_cmd_raw = Twist()

        # Last scan-derived metrics
        self.last_scan_stamp = None
        self.front_critical_min = float('inf')
        self.front_warning_min = float('inf')
        self.left_warning_min = float('inf')
        self.right_warning_min = float('inf')

        # QoS: LaserScan is typically best-effort sensor data.
        scan_qos = QoSProfile(depth=10)
        scan_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        cmd_qos = QoSProfile(depth=10)
        cmd_qos.reliability = ReliabilityPolicy.RELIABLE

        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, scan_qos)
        self.create_subscription(Twist, self.cmd_in_topic, self.cmd_raw_cb, cmd_qos)
        self.cmd_safe_pub = self.create_publisher(Twist, self.cmd_out_topic, cmd_qos)

        period = 1.0 / max(1.0, self.publish_rate_hz)
        self.create_timer(period, self.control_loop)

        self.get_logger().info(
            f'Reactive avoidance active: {self.cmd_in_topic} -> {self.cmd_out_topic}, '
            f'scan={self.scan_topic}'
        )

    def cmd_raw_cb(self, msg: Twist) -> None:
        if msg.linear.x < 0.0:
            self.get_logger().warn(
                'Reverse command requested; clamping linear.x to 0.0 (forward-only policy)',
                throttle_duration_sec=1.0,
            )
        self.last_cmd_raw = self._sanitize_diff_drive_cmd(msg)

    def scan_cb(self, msg: LaserScan) -> None:
        self.last_scan_stamp = self.get_clock().now()

        self.front_critical_min = float('inf')
        self.front_warning_min = float('inf')
        self.left_warning_min = float('inf')
        self.right_warning_min = float('inf')

        angle = msg.angle_min
        for r in msg.ranges:
            if self._is_valid_range(r, msg.range_min, msg.range_max):
                deg = math.degrees(angle)

                if -self.critical_angle_deg <= deg <= self.critical_angle_deg:
                    self.front_critical_min = min(self.front_critical_min, r)

                if -self.warning_angle_deg <= deg <= self.warning_angle_deg:
                    self.front_warning_min = min(self.front_warning_min, r)
                    if deg >= 0.0:
                        self.left_warning_min = min(self.left_warning_min, r)
                    else:
                        self.right_warning_min = min(self.right_warning_min, r)

            angle += msg.angle_increment

    def control_loop(self) -> None:
        # Differential-drive contract: only linear.x and angular.z are controlled.
        safe = self._sanitize_diff_drive_cmd(self.last_cmd_raw)

        # Fail-safe: stop if scan stream is stale.
        if self._scan_is_stale():
            self.cmd_safe_pub.publish(Twist())
            self.get_logger().warn('LiDAR scan timeout; publishing stop command', throttle_duration_sec=1.0)
            return

        # Only clamp forward commands; do not increase speed in any case.
        is_forward = safe.linear.x > 0.0

        if is_forward and self.front_critical_min < self.critical_distance:
            safe.linear.x = 0.0
            safe.angular.z = self._away_turn_direction() * self.critical_turn_speed
            self.get_logger().warn(
                f'Critical obstacle {self.front_critical_min:.2f}m < {self.critical_distance:.2f}m; rotating away',
                throttle_duration_sec=0.5,
            )

        elif is_forward and self.front_warning_min < self.warning_distance:
            # Scale forward speed between [critical_distance, warning_distance].
            span = self.warning_distance - self.critical_distance
            ratio = (self.front_warning_min - self.critical_distance) / span
            ratio = max(0.0, min(1.0, ratio))

            safe.linear.x = min(safe.linear.x, safe.linear.x * ratio)

            steer_mag = self.warning_steer_gain * (1.0 - ratio)
            safe.angular.z = safe.angular.z + self._away_turn_direction() * steer_mag

            self.get_logger().info(
                f'Warning obstacle {self.front_warning_min:.2f}m; speed scaled to {ratio:.2f}',
                throttle_duration_sec=1.0,
            )

        self.cmd_safe_pub.publish(safe)

    def _away_turn_direction(self) -> float:
        """Return angular sign to rotate away from closest side.

        +1 => turn left (CCW), -1 => turn right (CW)
        """
        if self.left_warning_min < self.right_warning_min:
            return -1.0  # obstacle on left -> turn right
        if self.right_warning_min < self.left_warning_min:
            return 1.0   # obstacle on right -> turn left

        # Tie-breaker: preserve current steering sign if available, else default left.
        if self.last_cmd_raw.angular.z < 0.0:
            return -1.0
        return 1.0

    def _scan_is_stale(self) -> bool:
        if self.last_scan_stamp is None:
            return True
        age = (self.get_clock().now() - self.last_scan_stamp).nanoseconds * 1e-9
        return age > self.scan_timeout_s

    @classmethod
    def _sanitize_diff_drive_cmd(cls, msg: Twist) -> Twist:
        cmd = Twist()
        # Forward-only policy: reverse motion is disabled at ROS layer.
        cmd.linear.x = max(0.0, cls._finite_or_zero(msg.linear.x))
        cmd.angular.z = cls._finite_or_zero(msg.angular.z)
        return cmd

    @staticmethod
    def _is_valid_range(value: float, min_r: float, max_r: float) -> bool:
        if value is None:
            return False
        if math.isnan(value) or math.isinf(value):
            return False
        return min_r <= value <= max_r

    @staticmethod
    def _finite_or_zero(value: float) -> float:
        if value is None:
            return 0.0
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ReactiveAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
