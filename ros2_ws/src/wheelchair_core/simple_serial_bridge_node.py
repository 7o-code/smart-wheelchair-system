#!/usr/bin/env python3
"""Simple ROS2 -> STM32 serial bridge using text commands.

Protocol:
  M,<linear_velocity>,<angular_velocity>\n
Example:
  M,0.40,0.00

Design goals:
  - deterministic, easy to debug over serial monitor
  - minimal dependencies
  - fail-safe timeout: send stop command if command stream is stale
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
import serial


class SimpleSerialBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('simple_serial_bridge_node')

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('cmd_topic', '/cmd_vel_safe')
        self.declare_parameter('send_rate_hz', 20.0)
        self.declare_parameter('cmd_timeout_s', 0.5)
        self.declare_parameter('max_linear_speed', 1.0)
        self.declare_parameter('max_angular_speed', 1.5)

        self.port = self.get_parameter('port').value
        self.baud = int(self.get_parameter('baud').value)
        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.send_rate_hz = float(self.get_parameter('send_rate_hz').value)
        self.cmd_timeout_s = float(self.get_parameter('cmd_timeout_s').value)
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(Twist, self.cmd_topic, self.cmd_cb, qos)

        self.last_cmd = Twist()
        self.last_cmd_time = None
        self.last_sent_line = ''

        self.ser = None
        self._open_serial()

        period = 1.0 / max(1.0, self.send_rate_hz)
        self.create_timer(period, self.send_loop)

        self.get_logger().info(
            f'Simple serial bridge active on {self.port} @ {self.baud}, topic={self.cmd_topic}'
        )

    def _open_serial(self) -> None:
        import glob
        ports_to_try = [self.port]
        # Auto-discover other common serial ports if the default fails
        ports_to_try.extend(glob.glob('/dev/ttyACM*'))
        ports_to_try.extend(glob.glob('/dev/ttyUSB*'))
        
        # Remove duplicates while preserving order
        seen = set()
        ports_to_try = [x for x in ports_to_try if not (x in seen or seen.add(x))]

        for p in ports_to_try:
            try:
                self.ser = serial.Serial(p, self.baud, timeout=0.05)
                self.ser.dtr = True
                self.ser.rts = True
                self.get_logger().info(f'Serial port opened successfully on {p} with DTR/RTS active')
                self.port = p # Update active port
                return
            except Exception as exc:
                self.ser = None
                
        self.get_logger().error(f'Could not open any serial port. Tried: {ports_to_try}')

    def cmd_cb(self, msg: Twist) -> None:
        self.last_cmd = self._sanitize_diff_drive_cmd(msg)
        self.last_cmd_time = self.get_clock().now()

    def send_loop(self) -> None:
        if self.ser is None or not self.ser.is_open:
            self._open_serial()
            return

        cmd = Twist()
        if self._is_cmd_fresh():
            cmd = self.last_cmd

        # Forward-only policy: do not transmit reverse linear velocity.
        linear = self._clamp(cmd.linear.x, 0.0, self.max_linear_speed)
        angular = self._clamp(cmd.angular.z, -self.max_angular_speed, self.max_angular_speed)
        linear_mm = int(linear * 1000)
        angular_mrad = int(angular * 1000)
        line = f'M,{linear_mm},{angular_mrad}\n'

        try:
            self.ser.write(line.encode('ascii'))
            self.last_sent_line = line.strip()
            # self.get_logger().info('Sent: ' + line.strip())
        except Exception as exc:
            self.get_logger().error(f'Serial write failed: {exc}')
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def _is_cmd_fresh(self) -> bool:
        if self.last_cmd_time is None:
            return False
        age_s = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        return age_s <= self.cmd_timeout_s

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @classmethod
    def _sanitize_diff_drive_cmd(cls, msg: Twist) -> Twist:
        cmd = Twist()
        cmd.linear.x = max(0.0, cls._finite_or_zero(msg.linear.x))
        cmd.angular.z = cls._finite_or_zero(msg.angular.z)
        return cmd

    @staticmethod
    def _finite_or_zero(value: float) -> float:
        if value is None:
            return 0.0
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)

    def destroy_node(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimpleSerialBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
