import cv2
import yaml
import logging
import time
import numpy as np
from feature_extraction.mediapipe_features import EyeGazeExtractor
from model.inference import InferenceEngine
from interface.command_interface import CommandInterface, EyeCommand

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MainNode")

def get_config_path():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "config", "model_config.yaml")

def load_config():
    config_path = get_config_path()
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

def save_config(config):
    config_path = get_config_path()
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

def demo_loop(safety, interface, w=640, h=480):
    # Deterministic demo sequence of synthetic features and labels for presentation
    demo_features = {
        'FORWARD': [0.5, 0.0, 0.3, 0.0, 0.0],
        'LEFT': [0.2, 0.0, 0.3, -0.2, 0.0],
        'RIGHT': [0.8, 0.0, 0.3, 0.2, 0.0],
        'STOP': [0.5, 0.0, 0.1, 0.0, 0.2], # Pitch for Nod
    }
    order = ['FORWARD', 'LEFT', 'RIGHT', 'STOP']
    idx = 0
    frame = 255 * np.ones((h, w, 3), dtype=np.uint8)
    logger.info("Running demo sequence. Press 'q' to quit.")
    while True:
        label = order[idx % len(order)]
        features = demo_features[label]
        cmd, conf = safety.predict(features)
        try:
            command_enum = EyeCommand(cmd)
            interface.publish_command(command_enum, conf)
        except ValueError:
            pass

        disp = frame.copy()
        cv2.putText(disp, f"DEMO: {label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 255), 3)
        cv2.putText(disp, f"CMD: {cmd} ({conf:.2f})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow("Eye Control AI Testbed - DEMO", disp)
        key = cv2.waitKey(1500) & 0xFF
        if key == ord('q'):
            break
        idx += 1


def main(demo=False, force_rule=False, debug=False, log_csv=None, headless=False):
    logger.info("Starting Eye Control AI Node...")
    config = load_config()
    force_rule = force_rule or config.get('model', {}).get('force_rule', False)
    headless = headless or config.get('system', {}).get('headless', False)
    
    # Initialize components
    # Prefer MediaPipe extractor; fallback to OpenCV extractor automatically
    extractor = None
    try:
        tmp = EyeGazeExtractor(config)
        if hasattr(tmp, 'landmarker') and tmp.landmarker is not None:
            extractor = tmp
            logger.info("Using MediaPipe FaceLandmarker extractor.")
        elif hasattr(tmp, 'face_mesh') and tmp.face_mesh is not None: 
            extractor = tmp
            logger.info("Using MediaPipe FaceMesh extractor (Legacy).")
        else:
            raise RuntimeError("MediaPipe FaceMesh/Landmarker not usable")
    except Exception as e:
        logger.warning(f"MediaPipe extractor not available: {e}. Falling back to OpenCV extractor.")
        from feature_extraction.opencv_features import OpenCVEyeGazeExtractor
        extractor = OpenCVEyeGazeExtractor(config)
        logger.info("Using OpenCV-based extractor.")

    inference = InferenceEngine(config, force_rule=force_rule)
    
    # Interface with Cooldown
    cooldown = config['system'].get('command_cooldown', 2.0)
    interface = CommandInterface(cooldown=cooldown)

    # Safety gate wrapper (debounce + confidence threshold)
    from model.safety import SafetyGate, BlinkStateManager
    safety = SafetyGate(inference, n=3, confidence_threshold=0.65)
    blink_manager = BlinkStateManager(ear_threshold=0.20, double_blink_window=0.6) # Short window for double blink

    if demo:
        demo_loop(safety, interface, w=config['system']['frame_width'], h=config['system']['frame_height'])
        logger.info("Demo finished.")
        return

    # Optional auto-calibration on start (useful for headless runs)
    calib_cfg = config.get('calibration', {})
    auto_calibrate = calib_cfg.get('auto_calibrate', False)
    auto_calibrate_always = calib_cfg.get('auto_calibrate_always', False)
    auto_calibrate_seconds = float(calib_cfg.get('auto_calibrate_seconds', 3.0))
    auto_calibrate_save = calib_cfg.get('auto_calibrate_save', True)

    if auto_calibrate and (auto_calibrate_always or (inference.offset_x == 0.0 and inference.offset_y == 0.0)):
        fps = float(config['system'].get('fps', 30))
        inference.calib_target_count = max(10, int(auto_calibrate_seconds * fps))
        inference.start_calibration()
        logger.info(f"Auto-calibration started for ~{auto_calibrate_seconds:.1f}s")

    # Camera setup
    cap = cv2.VideoCapture(config['system']['camera_index'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['system']['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['system']['frame_height'])

    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return

    logger.info("System initialized. Press 'q' to quit.")
    
    import os
    frame_idx = 0
    debug_every = config['system'].get('debug_every', 30)
    csv_fh = None
    if debug:
        if log_csv is None:
            log_dir = os.path.join("data", "debug_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_csv = os.path.join(log_dir, "gaze_log.csv")
        try:
            csv_fh = open(log_csv, "w", encoding="utf-8")
            csv_fh.write("t,cmd,conf,gx_raw,gx,gy_raw,gy,ear,yaw,pitch,l_gx,r_gx,presence\n")
        except Exception as e:
            logger.warning(f"Could not open log file {log_csv}: {e}")
            csv_fh = None
    prev_is_calibrating = inference.is_calibrating
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame.")
                break
                
            # 1. Feature Extraction
            data = extractor.process_frame(frame)
            
            command_str = "NO_FACE"
            confidence = 0.0
            
            if data:
                features = data['features']
                debug_info = data['debug_info']
                eye_crop = data.get('eye_crop')
                
                # 2. Inference with safety gating
                command_str, confidence = safety.predict(features, eye_crop=eye_crop)
                
                # --- SAFETY FAILSAFE (AI Engineer Requirement) ---
                # Presence check: If landmarks are unreliable, force STOP
                presence_lvl = debug_info.get('presence_conf', 1.0)
                if presence_lvl < 0.5:
                    command_str = "STOP"
                    confidence = 1.0
                    cv2.putText(frame, "SAFETY STOP: LOSS OF TRACKING", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # BLINK CONTROL (Double Blink to Toggle STOP/RESUME)
                # Extract EAR from features (index 2)
                current_ear = features[2]
                is_running = blink_manager.process(current_ear)
                
                if not is_running:
                    # System Paused via Double Blink
                    command_str = "STOP"
                    confidence = 1.0
                    cv2.putText(frame, "PAUSED (Triple Blink to Resume)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # System Running
                    # User requested to avoid "mistakenly stopped". 
                    # If the model predicts "STOP" (Nod), we should probably ignore it 
                    # and rely EXCLUSIVELY on Triple Blink for stopping.
                    if command_str == "STOP":
                         command_str = "FORWARD" # Default to Forward/Center if model thinks it's a nod
                
                # 3. Command Interface
                try:
                    command_enum = EyeCommand(command_str)
                    interface.publish_command(command_enum, confidence)
                except ValueError:
                    pass # Unknown command

                # Visualization: Iris Highlighting & Eye Box
                for iris_key in ['l_iris', 'r_iris']:
                    if iris_key in debug_info:
                        center = debug_info[iris_key].astype(int)
                        cv2.circle(frame, tuple(center), 5, (255, 0, 0), -1) # Blue
                
                x1, y1, x2, y2 = debug_info['eye_box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Right Eye Tracked", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if debug:
                    frame_idx += 1
                    if frame_idx % debug_every == 0:
                        logger.info(
                            f"DBG: gaze_x={features[0]:+.3f} gaze_y={features[1]:+.3f} "
                            f"ear={features[2]:.3f} yaw={features[3]:+.3f} pitch={features[4]:+.3f} "
                            f"cmd={command_str} conf={confidence:.2f}"
                        )
                    if csv_fh:
                        csv_fh.write(
                            f"{time.time():.3f},{command_str},{confidence:.3f},"
                            f"{debug_info.get('gaze_x_raw', 0.0):.5f},{features[0]:.5f},"
                            f"{debug_info.get('gaze_y_raw', 0.0):.5f},{features[1]:.5f},"
                            f"{features[2]:.5f},{features[3]:.5f},{features[4]:.5f},"
                            f"{debug_info.get('l_gaze_x', 0.0):.5f},{debug_info.get('r_gaze_x', 0.0):.5f},"
                            f"{debug_info.get('presence_conf', 0.0):.3f}\n"
                        )
                        csv_fh.flush()

                # Visual debug: draw 1D gaze axis
                if debug:
                    gx = max(-1.0, min(1.0, float(features[0])))
                    bar_w = 300
                    bar_h = 8
                    bar_x = 10
                    bar_y = config['system']['frame_height'] - 25
                    center_x = bar_x + bar_w // 2
                    pos_x = int(center_x + gx * (bar_w / 2))
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
                    cv2.line(frame, (center_x, bar_y - 4), (center_x, bar_y + bar_h + 4), (200, 200, 200), 1)
                    cv2.circle(frame, (pos_x, bar_y + bar_h // 2), 5, (0, 255, 255), -1)
                
            # Save auto-calibration once it finishes
            if prev_is_calibrating and not inference.is_calibrating and auto_calibrate_save:
                if 'calibration' not in config or config['calibration'] is None:
                    config['calibration'] = {}
                config['calibration']['gaze_offset_x'] = float(inference.offset_x)
                config['calibration']['gaze_offset_y'] = float(inference.offset_y)
                config['calibration']['target_count'] = int(inference.calib_target_count)
                try:
                    save_config(config)
                    logger.info("Auto-calibration saved to config.")
                except Exception as e:
                    logger.warning(f"Failed to save auto-calibration: {e}")
            prev_is_calibrating = inference.is_calibrating

            # Overlay Command
            cv2.putText(frame, f"CMD: {command_str} ({confidence:.2f})", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if not headless:
                cv2.imshow("Eye Control AI Testbed", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('i') and hasattr(extractor, "invert_gaze_x"):
                    extractor.invert_gaze_x = not extractor.invert_gaze_x
                    logger.info(f"invert_gaze_x set to {extractor.invert_gaze_x}")
                if key == ord('c') and data:
                    inference.set_offsets(features[0], features[1])
                    if 'calibration' not in config or config['calibration'] is None:
                        config['calibration'] = {}
                    config['calibration']['gaze_offset_x'] = float(features[0])
                    config['calibration']['gaze_offset_y'] = float(features[1])
                    try:
                        save_config(config)
                        logger.info("Calibration saved to config.")
                    except Exception as e:
                        logger.warning(f"Failed to save calibration: {e}")
            else:
                # Keep loop timing sane when headless
                time.sleep(1.0 / max(1.0, float(config['system'].get('fps', 30))))
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        if csv_fh:
            csv_fh.close()
        logger.info("Shutting down.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run deterministic demo sequence')
    parser.add_argument('--rule', action='store_true', help='Force rule-based inference (no ML model)')
    parser.add_argument('--debug', action='store_true', help='Log debug features and predictions')
    parser.add_argument('--log-csv', default=None, help='Path to CSV log file (debug mode)')
    parser.add_argument('--headless', action='store_true', help='Run without any GUI windows')
    args = parser.parse_args()

    main(demo=args.demo, force_rule=args.rule, debug=args.debug, log_csv=args.log_csv, headless=args.headless)
