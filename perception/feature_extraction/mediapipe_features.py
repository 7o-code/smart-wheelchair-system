import cv2
import mediapipe as mp
import numpy as np
import logging
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, t, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev 

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class EMAFilter:
    """Exponential Moving Average filter as requested by the Senior Engineer."""
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.value = None

    def filter(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

logger = logging.getLogger("FeatureExtractor")

class EyeGazeExtractor:
    # Landmark indices (Same as before, consistent with 468/478 Face Mesh)
    L_H_LEFT = 33
    L_H_RIGHT = 133
    R_H_LEFT = 362
    R_H_RIGHT = 263
    LEFT_IRIS = [468, 469, 470, 471]
    RIGHT_IRIS = [472, 473, 474, 475]

    @staticmethod
    def _order_by_x(p1, p2):
        # Ensure left->right ordering in image coordinates
        return (p1, p2) if p1[0] <= p2[0] else (p2, p1)

    def __init__(self, config):
        self.config = config
        self.landmarker = None

        try:
            # Use absolute path relative to this file to find the model
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed/face_landmarker.task"))
            
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model file not found at: {model_path}")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False, 
                output_facial_transformation_matrixes=False,
                num_faces=config['mediapipe']['max_num_faces'],
                min_face_detection_confidence=config['mediapipe']['min_detection_confidence'],
                min_face_presence_confidence=config['mediapipe']['min_tracking_confidence']
            )
            
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker (Tasks API) initialized successfully.")

        except Exception as e:
            logger.warning("MediaPipe FaceLandmarker initialization failed. Feature extraction disabled.")
            logger.error(f"Error details: {e}")
            self.landmarker = None

        self.mock_state = {
            'gaze_x': 0.5,
            'gaze_y': 0.0,
            'ear': 0.3,
            'yaw': 0.0,
            'pitch': 0.0
        }
        
        # Initialize Filters
        # min_cutoff: 0.1 (very smooth/slow) to 1.0 (responsive)
        self.filters = {
            'gaze_x': OneEuroFilter(min_cutoff=0.8, beta=0.1),
            'gaze_y': OneEuroFilter(min_cutoff=0.8, beta=0.1),
            'yaw': OneEuroFilter(min_cutoff=0.5, beta=0.1),
            'pitch': OneEuroFilter(min_cutoff=0.5, beta=0.1),
            'eye_x1': OneEuroFilter(min_cutoff=1.0, beta=0.01),
            'eye_y1': OneEuroFilter(min_cutoff=1.0, beta=0.01),
            'eye_x2': OneEuroFilter(min_cutoff=1.0, beta=0.01),
            'eye_y2': OneEuroFilter(min_cutoff=1.0, beta=0.01),
        }
        self.use_head_pose = self.config.get('features', {}).get('use_head_pose', True)
        self.use_blink = self.config.get('features', {}).get('use_blink', True)
        self.invert_gaze_x = self.config.get('features', {}).get('invert_gaze_x', False)
        self.gaze_gain = float(self.config.get('features', {}).get('gaze_gain', 1.0))

    def set_mock_state(self, gaze_x, gaze_y, blink):
        self.mock_state['gaze_x'] = gaze_x
        self.mock_state['gaze_y'] = gaze_y
        self.mock_state['ear'] = 0.0 if blink else 0.3
        self.mock_state['yaw'] = (gaze_x - 0.5) * 0.5
        self.mock_state['pitch'] = gaze_y * 0.5

    def process_frame(self, frame):
        try:
            h, w, _ = frame.shape
            
            if self.landmarker is None:
                return None

            # MediaPipe Tasks Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Detect
            result = self.landmarker.detect(mp_image)

            if not result or not result.face_landmarks:
                return None

            # Get the first face
            landmarks = result.face_landmarks[0]
            
            # Convert landmarks to pixel coordinates 
            mesh_points = np.array([np.array([p.x * w, p.y * h]).astype(int) for p in landmarks])
            
            # Ensure required landmarks exist
            if len(landmarks) <= 477:
                return None
            
            # --- 1D Gaze Direction (Predictable Ratios for Gating) ---
            # Right Eye (Camera Perspective Left)
            l_left, l_right = self._order_by_x(mesh_points[33], mesh_points[133])
            l_iris = np.mean(mesh_points[self.LEFT_IRIS], axis=0)
            l_gaze_x = (l_iris[0] - l_left[0]) / (l_right[0] - l_left[0] + 1e-6)
            
            # Left Eye (Camera Perspective Right)
            r_left, r_right = self._order_by_x(mesh_points[362], mesh_points[263])
            r_iris = np.mean(mesh_points[self.RIGHT_IRIS], axis=0)
            r_gaze_x = (r_iris[0] - r_left[0]) / (r_right[0] - r_left[0] + 1e-6)
            
            # Center-zero (0.0 = forward, negative = left, positive = right)
            # Center-zero (0.0 = forward, negative = left, positive = right)
            avg_gaze_x_raw = ((l_gaze_x + r_gaze_x) / 2.0) - 0.5
            avg_gaze_x = -avg_gaze_x_raw if self.invert_gaze_x else avg_gaze_x_raw
            avg_gaze_x *= self.gaze_gain
            if avg_gaze_x > 1.0:
                avg_gaze_x = 1.0
            elif avg_gaze_x < -1.0:
                avg_gaze_x = -1.0

            # Vertical (Relative to eye center line)
            l_top, l_bottom = mesh_points[159], mesh_points[145]
            r_top, r_bottom = mesh_points[386], mesh_points[374]
            l_gaze_y = (l_iris[1] - (l_top[1] + l_bottom[1])/2) / (np.linalg.norm(l_top - l_bottom) + 1e-6)
            r_gaze_y = (r_iris[1] - (r_top[1] + r_bottom[1])/2) / (np.linalg.norm(r_top - r_bottom) + 1e-6)
            avg_gaze_y_raw = (l_gaze_y + r_gaze_y) / 2.0
            avg_gaze_y = avg_gaze_y_raw

            # EAR for blink
            avg_ear = (np.linalg.norm(l_top - l_bottom) / (np.linalg.norm(l_right - l_left) + 1e-6) + 
                       np.linalg.norm(r_top - r_bottom) / (np.linalg.norm(r_right - r_left) + 1e-6)) / 2.0

            # 3. Head Pose
            nose_tip = mesh_points[1]
            face_center_x = (mesh_points[234][0] + mesh_points[454][0]) / 2
            face_center_y = (mesh_points[10][1] + mesh_points[152][1]) / 2
            head_yaw = (nose_tip[0] - face_center_x) / w 
            head_pitch = (nose_tip[1] - face_center_y) / h

            # Apply Smoothing
            t = time.time()
            avg_gaze_x = self.filters['gaze_x'].filter(t, avg_gaze_x)
            avg_gaze_y = self.filters['gaze_y'].filter(t, avg_gaze_y)
            head_yaw = self.filters['yaw'].filter(t, head_yaw)
            head_pitch = self.filters['pitch'].filter(t, head_pitch)

            if not self.use_head_pose:
                head_yaw = 0.0
                head_pitch = 0.0
            if not self.use_blink:
                avg_ear = 1.0

            features = [avg_gaze_x, avg_gaze_y, avg_ear, head_yaw, head_pitch]
            
            # --- Eye Crop with Rotation Normalization ---
            p1, p2 = mesh_points[33], mesh_points[133]
            d_x, d_y = p2[0] - p1[0], p2[1] - p1[1]
            angle = math.degrees(math.atan2(d_y, d_x))
            
            eye_center_raw = np.mean([p1, p2], axis=0)
            eye_center = (float(eye_center_raw[0]), float(eye_center_raw[1]))
            rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            rotated_frame = cv2.warpAffine(frame, rot_mat, (w, h))
            
            margin = 35
            rx1, ry1 = max(0, min(p1[0], p2[0]) - margin), max(0, min(p1[1], p2[1]) - margin)
            rx2, ry2 = min(w, max(p1[0], p2[0]) + margin), min(h, max(p1[1], p2[1]) + margin)
            
            # 3. Use raw box (don't filter box boundaries to match training data)
            x1, y1 = int(rx1), int(ry1)
            x2, y2 = int(rx2), int(ry2)

            eye_crop = rotated_frame[y1:y2, x1:x2]
            normalized_eye = None
            
            if eye_crop.shape[0] > 0 and eye_crop.shape[1] > 0:
                # IMPORTANT: Rotate landmarks to match the 'rotated_frame' coordinate system
                # This ensures the mask is perfectly aligned with the eye in the crop
                ones = np.ones(shape=(len(mesh_points), 1))
                points_ones = np.concatenate((mesh_points, ones), axis=1)
                rotated_mesh = points_ones.dot(rot_mat.T)
                
                gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                mask = np.zeros_like(enhanced)
                eye_outline_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                
                # Use rotated_mesh coordinates
                eye_pts = np.array([[rotated_mesh[idx][0] - x1, rotated_mesh[idx][1] - y1] for idx in eye_outline_idx], dtype=np.int32)
                
                cv2.fillPoly(mask, [eye_pts], 255)
                masked_eye = cv2.bitwise_and(enhanced, enhanced, mask=mask)
                resized_eye = cv2.resize(cv2.cvtColor(masked_eye, cv2.COLOR_GRAY2RGB), (224, 224))
                normalized_eye = resized_eye.astype("float32") / 255.0
            
            # Safety checks
            presence_conf = getattr(landmarks[33], 'visibility', 1.0)
            if presence_conf is None: presence_conf = 1.0
            
            debug_info = {
                'mesh_points': mesh_points,
                'l_iris': l_iris,
                'r_iris': r_iris,
                'gaze_x_raw': avg_gaze_x_raw,
                'gaze_x': avg_gaze_x,
                'gaze_y_raw': avg_gaze_y_raw,
                'gaze_y': avg_gaze_y,
                'l_gaze_x': l_gaze_x,
                'r_gaze_x': r_gaze_x,
                'ear': avg_ear,
                'eye_box': (x1, y1, x2, y2), 'presence_conf': presence_conf
            }

            return {'features': features, 'eye_crop': normalized_eye, 'debug_info': debug_info}
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            import traceback
            traceback.print_exc()
            return None

    def close(self):
        if self.landmarker:
            self.landmarker.close()
