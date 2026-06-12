import numpy as np
import logging
from .model_utils import load_model, load_scaler

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    TFLiteInterpreter = None

logger = logging.getLogger("InferenceEngine")

def _build_rule_model(config):
    thresholds = config.get('thresholds', {})
    gaze_sensitivity = thresholds.get('gaze_sensitivity', 0.25)
    blink_ratio = thresholds.get('blink_ratio', 0.18)
    deadzone = thresholds.get('deadzone', 0.1)
    return RuleBasedModel(
        gaze_thresh_left=-abs(gaze_sensitivity),
        gaze_thresh_right=abs(gaze_sensitivity),
        ear_blink_thresh=blink_ratio,
        deadzone=deadzone,
    )

class RuleBasedModel:
    """Simple interpretable rules for direction detection based on features.
    Expects features: [gaze_x, gaze_y, ear, head_yaw, head_pitch]
    Returns (command, confidence)
    """
    def __init__(self, gaze_thresh_left=-0.25, gaze_thresh_right=0.25, ear_blink_thresh=0.18, deadzone=0.1):
        self.gaze_left = gaze_thresh_left
        self.gaze_right = gaze_thresh_right
        self.ear_blink = ear_blink_thresh
        self.deadzone = deadzone

    def predict_command(self, features, eye_crop=None):
        gx, gy, ear, yaw, pitch = features
        
        # Blink -> STOP
        if ear < self.ear_blink:
            return "STOP", 0.95
            
        # Deadzone: small movements near center
        if abs(gx) < self.deadzone:
            return "FORWARD", 0.8

        # Looking left/right
        if gx < self.gaze_left:
            return "LEFT", 0.9
        if gx > self.gaze_right:
            return "RIGHT", 0.9
            
        # Default forward
        return "FORWARD", 0.8


class InferenceEngine:
    def __init__(self, config, force_rule=False):
        self.config = config
        self.model_path = config['model']['model_path']
        self.scaler_path = config['model']['scaler_path']
        self.classes = config['model']['classes']
        self.debug_counter = 0  # For throttled logging
        
        # Calibration State
        self.is_calibrating = False
        self.calib_samples = []
        self.calib_target_count = 20
        self.offset_x = 0.0
        self.offset_y = 0.0

        # --- TFLite Model Setup ---
        self.tflite_interpreter = None
        self.tflite_input_details = None
        self.tflite_output_details = None
        self.tflite_ready = False
        
        # Matches the alphabetical order of training: 
        # ['close_look', 'forward_look', 'left_look', 'right_look']
        self.tflite_class_names = ["STOP", "FORWARD", "LEFT", "RIGHT"]

        import os
        tflite_path = config.get('tflite_path', 'model/weights/best_model.tflite') 
        use_tflite = config.get('model', {}).get('use_tflite', True)

        if not force_rule and use_tflite:
            interpreter_cls = None
            if tf is not None:
                interpreter_cls = tf.lite.Interpreter
            elif TFLiteInterpreter is not None:
                interpreter_cls = TFLiteInterpreter
            else:
                logger.warning("No TFLite runtime available. Install TensorFlow to enable TFLite inference.")

            if interpreter_cls is not None:
                try:
                    self.tflite_interpreter = interpreter_cls(model_path=tflite_path)
                    self.tflite_interpreter.allocate_tensors()
                    self.tflite_input_details = self.tflite_interpreter.get_input_details()
                    self.tflite_output_details = self.tflite_interpreter.get_output_details()
                    self.tflite_ready = True
                    logger.info("TFLite single-eye model loaded successfully.")
                except Exception as e:
                    logger.warning(f"TFLite model not loaded ({e}).")
                    self.tflite_interpreter = None
        else:
            if not use_tflite:
                logger.info("TFLite inference disabled via config.")
            else:
                logger.info("Rule-based inference forced. TFLite disabled.")
        
        # Fallback
        self.model = load_model(self.model_path)
        self.scaler = load_scaler(self.scaler_path)
        
        # If explicitly forced into rule mode or no model is available, use RuleBasedModel
        if force_rule or (self.model is None and not self.tflite_ready):
            self.model = _build_rule_model(config)
            logger.info("Using RuleBasedModel for inference.")
        elif self.model is None and self.tflite_ready:
            # Provide a safe fallback in case eye crops are unavailable.
            self.model = _build_rule_model(config)
            logger.info("Using RuleBasedModel as fallback when eye crop is missing.")
        
        if self.model is None and self.tflite_interpreter is None:
            logger.warning("No trained model found. Inference will fail until trained.")

        # Load Calibration
        calib = config.get('calibration', {})
        self.offset_x = calib.get('gaze_offset_x', 0.0)
        self.offset_y = calib.get('gaze_offset_y', 0.0)
        self.calib_target_count = int(calib.get('target_count', self.calib_target_count))
        if self.offset_x != 0.0 or self.offset_y != 0.0:
            logger.info(f"Calibration applied: X_off={self.offset_x:.3f}, Y_off={self.offset_y:.3f}")

    def start_calibration(self):
        """Start gathering samples for a new gaze center."""
        self.is_calibrating = True
        self.calib_samples = []
        logger.info("Gaze calibration started. Please look forward.")
    
    def set_offsets(self, gaze_x, gaze_y):
        """Immediately set calibration offsets."""
        self.offset_x = float(gaze_x)
        self.offset_y = float(gaze_y)
        self.is_calibrating = False
        self.calib_samples = []
        logger.info(f"Calibration set: X_off={self.offset_x:.3f}, Y_off={self.offset_y:.3f}")

    def predict_command(self, feature_vector, eye_crop=None):
        """
        Predicts the command based on the feature vector and/or the eye_crop.
        """
        # Handle Calibration Mode
        if self.is_calibrating:
            self.calib_samples.append(feature_vector[0])
            if len(self.calib_samples) >= self.calib_target_count:
                self.offset_x = np.mean(self.calib_samples)
                self.is_calibrating = False
                logger.info(f"Calibration finished. Gaze Offset X: {self.offset_x:.3f}")
            return "CALIBRATING", 0.0

        # Apply Calibration Offsets
        feature_vector = list(feature_vector)
        feature_vector[0] -= self.offset_x
        feature_vector[1] -= self.offset_y

        # 1. TFLite Model
        self.debug_counter += 1
        if self.tflite_interpreter is not None and eye_crop is not None:
            try:
                input_data = np.expand_dims(eye_crop, axis=0).astype(np.float32) # 1x224x224x3
                self.tflite_interpreter.set_tensor(self.tflite_input_details[0]['index'], input_data)
                self.tflite_interpreter.invoke()
                output_data = self.tflite_interpreter.get_tensor(self.tflite_output_details[0]['index'])
                
                pred_idx = np.argmax(output_data[0])
                confidence = output_data[0][pred_idx]
                command = self.tflite_class_names[pred_idx]

                # throttled debug log
                if self.debug_counter % 15 == 0:
                    probs_str = ", ".join([f"{name}:{prob:.2f}" for name, prob in zip(self.tflite_class_names, output_data[0])])
                    logger.info(f"GAZE DEBUG: backend=tflite x={feature_vector[0]:+.3f} y={feature_vector[1]:+.3f} ear={feature_vector[2]:.3f} | cmd={command} | probs: {probs_str}")

                return command, float(confidence)
            except Exception as e:
                logger.error(f"TFLite Prediction error: {e}")
                # Fallback to standard
        
        # 2. Standard Model Fallback
        if self.model is None:
            return "NO_MODEL", 0.0

        if isinstance(self.model, RuleBasedModel):
            command, confidence = self.model.predict_command(feature_vector, eye_crop)
            if self.debug_counter % 30 == 0:
                logger.info(f"GAZE DEBUG: backend=rule x={feature_vector[0]:+.3f} y={feature_vector[1]:+.3f} ear={feature_vector[2]:.3f} | cmd={command} conf={confidence:.2f}")
            return command, confidence

        # Preprocess for SL Models
        features = np.array(feature_vector).reshape(1, -1)
        if self.scaler:
            features = self.scaler.transform(features)

        # Predict
        try:
            probs = self.model.predict_proba(features)[0]
            max_idx = np.argmax(probs)
            confidence = probs[max_idx]
            command = self.classes[max_idx]
            
            if self.debug_counter % 30 == 0:
                logger.info(f"GAZE DEBUG: backend=sklearn x={feature_vector[0]:+.3f} y={feature_vector[1]:+.3f} ear={feature_vector[2]:.3f} | cmd={command} conf={confidence:.2f}")
            return command, float(confidence)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "ERROR", 0.0
