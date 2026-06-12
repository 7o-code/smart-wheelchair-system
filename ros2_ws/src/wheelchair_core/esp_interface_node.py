#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Float32
import serial
import json
import threading
import time

class ESPInterfaceNode(Node):
    def __init__(self):
        super().__init__('esp_interface_node')
        
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('linear_scale', 0.5)
        self.declare_parameter('angular_scale', 0.8)
        self.declare_parameter('smoothing_factor', 0.2)
        
        self.port = self.get_parameter('port').value
        self.baud = self.get_parameter('baud').value
        self.linear_scale = self.get_parameter('linear_scale').value
        self.angular_scale = self.get_parameter('angular_scale').value
        self.alpha = self.get_parameter('smoothing_factor').value
        
        self.mode_pub = self.create_publisher(Int32, '/wheelchair_mode', 10)
        self.volt_pub = self.create_publisher(Float32, '/battery/voltage', 10)
        self.bat_pub = self.create_publisher(Float32, '/battery/percentage', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_esp', 10)
        
        self.mode_sub = self.create_subscription(Int32, '/set_wheelchair_mode', self.set_mode_cb, 10)
        
        # Latest data from ESP
        self.last_joy_x = 0.0
        self.last_joy_y = 0.0
        self.last_mode = 0
        self.last_data_time = 0
        
        # Smoothed state
        self.current_linear = 0.0
        self.current_angular = 0.0
        
        self.ser = None
        self.running = True
        self.serial_thread = threading.Thread(target=self.serial_rx_loop)
        self.serial_thread.start()
        
        # Control Loop Timer (10Hz) to ensure continuous stream and smoothing
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info(f"ESP Interface Node started on {self.port} with 10Hz stabilization")

    def serial_rx_loop(self):
        while self.running and rclpy.ok():
            if self.ser is None or not self.ser.is_open:
                try:
                    self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
                    self.get_logger().info(f"Connected to ESP8266 on {self.port}")
                except Exception as e:
                    time.sleep(2.0)
                    continue
            
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('{') and line.endswith('}'):
                    data = json.loads(line)
                    self.last_data_time = time.time()
                    
                    if 'mode' in data:
                        self.last_mode = int(data['mode'])
                        mode_msg = Int32()
                        mode_msg.data = self.last_mode
                        self.mode_pub.publish(mode_msg)
                        
                    if 'x' in data and 'y' in data:
                        self.last_joy_x = float(data['x'])
                        self.last_joy_y = float(data['y'])
                    
                    if 'volt' in data:
                        self.volt_pub.publish(Float32(data=float(data['volt'])))
                    if 'bat' in data:
                        self.bat_pub.publish(Float32(data=float(data['bat'])))
                        
            except Exception as e:
                pass

    def control_loop(self):
        """10Hz Loop to publish smoothed twist commands."""
        # Only publish if we are in Companion Mode (1) and data is relatively fresh (2s)
        if self.last_mode == 1 and (time.time() - self.last_data_time < 2.0):
            twist = Twist()
            
            # --- Kinematics Math ---
            # Standard mapping (supports both analog joystick and discrete button inputs)
            # If buttons are used, the app sends ±255 for full deflection
            target_linear = self.linear_scale * (self.last_joy_y / 255.0)
            target_angular = -self.angular_scale * (self.last_joy_x / 255.0)
            
            # "One-Wheel-At-A-Time" logic for pure turns (Common for button interfaces)
            # If the user clicks a "Turn" button with no forward input
            if abs(self.last_joy_y) < 15 and abs(self.last_joy_x) > 100:
                # Use a calibrated turn speed (0.4 matches eye tracker)
                target_angular = -0.4 * (self.last_joy_x / 255.0)
                # To make v_inner = 0, target_linear must be target_angular * (wheel_base / 2)
                # b=0.6, so b/2 = 0.3.
                target_linear = abs(target_angular * 0.3)
            
            # --- Smoothing (Low Pass Filter) ---
            # Matches the 'eye_twist_publisher_node' behavior
            self.current_linear = (1.0 - self.alpha) * self.current_linear + self.alpha * target_linear
            self.current_angular = (1.0 - self.alpha) * self.current_angular + self.alpha * target_angular
            
            twist.linear.x = self.current_linear
            twist.angular.z = self.current_angular
            self.cmd_pub.publish(twist)
        else:
            # Gradually ramp to stop if mode changes or data lost
            self.current_linear *= 0.8
            self.current_angular *= 0.8
            if abs(self.current_linear) < 0.01: self.current_linear = 0.0
            if abs(self.current_angular) < 0.01: self.current_angular = 0.0

    def set_mode_cb(self, msg):
        if self.ser and self.ser.is_open:
            cmd = {"cmd": "set_mode", "val": msg.data}
            try:
                self.ser.write(f">{json.dumps(cmd)}\n".encode('utf-8'))
            except Exception:
                pass

    def destroy_node(self):
        self.running = False
        if self.serial_thread.is_alive():
            self.serial_thread.join()
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ESPInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
