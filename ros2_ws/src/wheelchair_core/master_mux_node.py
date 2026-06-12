#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32

class MasterMuxNode(Node):
    def __init__(self):
        super().__init__('master_mux_node')
        
        # State: 0 = Patient (Eye), 1 = Companion (ESP)
        self.mode = 0
        
        self.create_subscription(Int32, '/wheelchair_mode', self.mode_cb, 10)
        self.create_subscription(Twist, '/cmd_vel_raw', self.eye_cb, 10)
        self.create_subscription(Twist, '/cmd_vel_esp', self.esp_cb, 10)
        
        self.pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        
        self.get_logger().info("Master Mux Node active. Mode: Patient (Eye)")

    def mode_cb(self, msg):
        if msg.data != self.mode:
            self.mode = msg.data
            mode_str = "Companion (ESP)" if self.mode == 1 else "Patient (Eye)"
            self.get_logger().info(f"Switched to mode: {mode_str}")

    def eye_cb(self, msg):
        if self.mode == 0:
            self.pub.publish(msg)

    def esp_cb(self, msg):
        if self.mode == 1:
            self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MasterMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
