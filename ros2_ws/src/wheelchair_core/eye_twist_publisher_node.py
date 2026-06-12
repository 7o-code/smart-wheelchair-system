#!/usr/bin/env python3

import sys
import os

# MediaPipe must be imported before OpenCV to prevent C++ binding symbol conflicts!
try:
    import mediapipe
except Exception:
    pass

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import threading
import cv2

# Add the AI directory to Python path so it can import its local modules
ai_path = os.path.join(os.path.expanduser('~'), 'ros_ws', 'eye_tracking_pi')
sys.path.append(ai_path)

# Ensure ML relative paths load correctly
os.chdir(ai_path)

from model.inference import InferenceEngine
from interface.command_interface import CommandInterface, EyeCommand
from model.safety import SafetyGate, BlinkStateManager

class EyeTwistPublisher(Node):
    def __init__(self):
        super().__init__('eye_twist_publisher_node')
        
        # ROS Parameters
        self.declare_parameter('cmd_topic', '/cmd_vel_raw')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('forward_speed', 0.2) # Conservative speed
        self.declare_parameter('turn_speed', 0.4)    # Predictable turning
        self.declare_parameter('turn_forward_speed', 0.1) # Move slightly forward while turning
        self.declare_parameter('headless', True)
        
        cmd_topic = self.get_parameter('cmd_topic').value
        self.camera_index = self.get_parameter('camera_index').value
        self.forward_speed = self.get_parameter('forward_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value
        self.turn_forward_speed = self.get_parameter('turn_forward_speed').value
        self.headless = self.get_parameter('headless').value
        
        # Automatic SSH/No-Display detection
        if not self.headless and 'DISPLAY' not in os.environ:
            self.get_logger().warning("No DISPLAY detected (running over SSH?). Falling back to HEADLESS mode automatically.")
            self.headless = True
        
        # Smoothing state (Low-Pass Filter)
        self.declare_parameter('smoothing_factor', 0.2) # 0.0 to 1.0 (Lower = Smoother/Slower)
        self.smoothing_factor = self.get_parameter('smoothing_factor').value
        self.current_linear_x = 0.0
        self.current_angular_z = 0.0
        
        self.config_path = os.path.join(ai_path, 'config', 'model_config.yaml')

        self.publisher_ = self.create_publisher(Twist, cmd_topic, 10)
        
        # Load AI config
        import yaml
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.get_logger().warning(f"Config file not found at {self.config_path}, using defaults")
            self.config = {'system': {'frame_width': 640, 'frame_height': 480}, 'model': {'force_rule': False}, 'calibration': {}}

        # Init AI components
        try:
            from feature_extraction.mediapipe_features import EyeGazeExtractor
            tmp = EyeGazeExtractor(self.config)
            if hasattr(tmp, 'landmarker') and tmp.landmarker is not None:
                self.extractor = tmp
                self.get_logger().info("Using MediaPipe FaceLandmarker extractor.")
            elif hasattr(tmp, 'face_mesh') and tmp.face_mesh is not None: 
                self.extractor = tmp
                self.get_logger().info("Using MediaPipe FaceMesh extractor (Legacy).")
            else:
                raise RuntimeError("MediaPipe FaceMesh/Landmarker not usable")
        except Exception as e:
            self.get_logger().warning(f"MediaPipe extractor not available: {e}. Falling back to OpenCV extractor.")
            from feature_extraction.opencv_features import OpenCVEyeGazeExtractor
            self.extractor = OpenCVEyeGazeExtractor(self.config)
            
        self.inference = InferenceEngine(self.config, force_rule=self.config.get('model', {}).get('force_rule', False))
        self.safety = SafetyGate(self.inference, n=3, confidence_threshold=0.65)
        self.blink_manager = BlinkStateManager(ear_threshold=0.20, double_blink_window=0.6)
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.get_logger().warning(f"Could not open webcam {self.camera_index}. Auto-scanning other ports...")
            for i in range(40):
                if i == self.camera_index: continue
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    self.get_logger().info(f"Successfully auto-detected webcam at index {i}!")
                    self.camera_index = i
                    break
                    
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not find any working webcam on ports 0-40.")
            raise RuntimeError("Webcam not found")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['system'].get('frame_width', 640))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['system'].get('frame_height', 480))
        
        # Load calibration offsets if available to avoid staring at a wall initially
        if 'calibration' in self.config and 'gaze_offset_x' in self.config['calibration']:
            gx = self.config['calibration']['gaze_offset_x']
            gy = self.config['calibration']['gaze_offset_y']
            self.inference.set_offsets(gx, gy)
            self.get_logger().info(f"Loaded calibration offsets: gx={gx}, gy={gy}")

        # Auto-Calibration on startup
        calib_cfg = self.config.get('calibration', {})
        auto_calibrate = calib_cfg.get('auto_calibrate', False)
        auto_calibrate_always = calib_cfg.get('auto_calibrate_always', False)
        auto_calibrate_seconds = float(calib_cfg.get('auto_calibrate_seconds', 3.0))

        if auto_calibrate and (auto_calibrate_always or (self.inference.offset_x == 0.0 and self.inference.offset_y == 0.0)):
            fps = float(self.config['system'].get('fps', 30))
            self.inference.calib_target_count = max(10, int(auto_calibrate_seconds * fps))
            self.inference.start_calibration()
            self.get_logger().info(f"Auto-calibration started for ~{auto_calibrate_seconds:.1f}s")
            self.prev_is_calibrating = self.inference.is_calibrating
        else:
            self.prev_is_calibrating = False

        self.running = True
        self.thread = threading.Thread(target=self.process_loop)
        self.thread.start()
        
        self.get_logger().info(f"Eye Twist Publisher initialized. Publishing to {cmd_topic}")

    def process_loop(self):
        while self.running and rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warning("Failed to grab frame.")
                continue
                
            data = self.extractor.process_frame(frame)
            command_str = "NO_FACE"
            is_running = True
            
            # Map enum string to target velocities
            target_linear_x = 0.0
            target_angular_z = 0.0
            confidence = 0.0
            is_running = True
            
            if data:
                features = data['features']
                debug_info = data['debug_info']
                eye_crop = data.get('eye_crop')
                
                # Use stable prediction from SafetyGate to eliminate jitter
                command_str, confidence = self.safety.predict(features, eye_crop=eye_crop)
                
                # Triple Blink pause/resume (only safety mechanism)
                current_ear = features[2]
                is_running = self.blink_manager.process(current_ear)
                
                if not is_running:
                    command_str = "STOP"
            else:
                command_str = "STOP"
            
            if command_str == "FORWARD":
                target_linear_x = float(self.forward_speed)
                target_angular_z = 0.0
            elif command_str == "LEFT":
                target_linear_x = float(self.turn_forward_speed)
                target_angular_z = float(self.turn_speed)  # positive Z is left turn in ROS
            elif command_str == "RIGHT":
                target_linear_x = float(self.turn_forward_speed)
                target_angular_z = -float(self.turn_speed) # negative Z is right turn in ROS
            # STOP or NO_FACE leaves targets at 0.0
            
            # Map enum string to Twist safely clamped values
            msg = Twist()
            
            # Smooth the values using a simple Low-Pass Filter (LPF)
            # current = (1 - alpha) * current + alpha * target
            alpha = self.smoothing_factor
            self.current_linear_x = (1.0 - alpha) * self.current_linear_x + alpha * target_linear_x
            self.current_angular_z = (1.0 - alpha) * self.current_angular_z + alpha * target_angular_z
            
            # Apply to message
            msg.linear.x = self.current_linear_x
            msg.angular.z = self.current_angular_z
                
            self.publisher_.publish(msg)

            if not self.headless:
                # Use targets for display to show what the AI "wants"
                cv2.putText(frame, f"CMD: {command_str} (Conf: {confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Show actual smoothed velocity for debug
                cv2.putText(frame, f"v: {msg.linear.x:.2f} w: {msg.angular.z:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if not is_running:
                    cv2.putText(frame, "PAUSED (Double Blink to Resume)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("ROS2 Eye Tracker", frame)
                cv2.waitKey(1)
                
    def destroy_node(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = EyeTwistPublisher()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error starting Eye Tracker ROS 2 bridge: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
