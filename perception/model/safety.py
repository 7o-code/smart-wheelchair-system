import collections
import logging
import time

logger = logging.getLogger("InferenceSafety")


class Debouncer:
    """Require N consecutive identical commands above confidence threshold."""
    def __init__(self, n=3, confidence_threshold=0.65):
        self.n = n
        self.confidence_threshold = confidence_threshold
        self.window = collections.deque(maxlen=n)

    def input(self, command, confidence):
        """Feed a new prediction. Returns stable command or None if not stable."""
        if confidence < self.confidence_threshold:
            # treat as no detection
            self.window.clear()
            return None

        self.window.append(command)
        if len(self.window) == self.n and all(c == self.window[0] for c in self.window):
            logger.debug(f"Debounced to command: {command}")
            return command
        return None


class SafetyGate:
    """Wraps an inference engine to apply thresholds and debouncing."""
    def __init__(self, inference_engine, n=3, confidence_threshold=0.65):
        self.engine = inference_engine
        self.debouncer = Debouncer(n, confidence_threshold)

    def predict(self, features, eye_crop=None):
        cmd, conf = self.engine.predict_command(features, eye_crop=eye_crop)
        stable = self.debouncer.input(cmd, conf)
        
        # User requested to remove "NO_COMMAND" condition.
        # If not stable yet, we will fallback to the raw command (cmd)
        # potentially confusing but requested.
        # Alternatively, we could hold the last stable command.
        
        if stable is None:
             return cmd, conf # Return raw command instead of NO_COMMAND
             
        return stable, conf


class BlinkStateManager:
    """
    Manages system state based on blink patterns.
    - Single Blink + Wait -> Toggle PAUSE/RUNNING (or specific requested behavior: Blink once = STOP)
    - Double Blink -> RESUME / START
    
    Requested Logic:
    - Blink Once -> STOP (Pause)
    - Blink Twice -> RESUME (Read directions)
    """
    def __init__(self, ear_threshold=0.20, double_blink_window=1.0):
        self.ear_threshold = ear_threshold
        self.double_blink_window = double_blink_window
        
        self.is_paused = False # Default state (or set to True if safe start needed)
        
        self.blink_count = 0
        self.last_blink_time = 0
        self.in_blink = False
        
    def process(self, ear):
        """
        Process the current EAR and update state.
        Returns: current state (RUNNING=True, PAUSED=False) or maybe status string
        """
        now = time.time()
        
        # 1. Detect Blink Edge (Open -> Closed)
        if ear < self.ear_threshold:
            if not self.in_blink:
                self.in_blink = True
                # Blink started
        else:
            if self.in_blink:
                self.in_blink = False
                # Blink ended (Rising edge) -> Count it
                
                # Check if this is part of a sequence or a new one
                if now - self.last_blink_time < self.double_blink_window:
                    self.blink_count += 1
                else:
                    self.blink_count = 1
                
                self.last_blink_time = now
                logger.debug(f"Blink detected! Count: {self.blink_count}")

        # 2. Logic Evaluation
        # Check for double blink
        if self.blink_count >= 3:
            # Toggle State
            self.is_paused = not self.is_paused
            state_str = "PAUSED" if self.is_paused else "RUNNING"
            logger.info(f"Triple Blink Detected! System State: {state_str}")
            
            self.blink_count = 0 # Reset immediately after toggle
            self.last_blink_time = 0 # Prevent quadruple blink from toggling again immediately

        # Timeout for single blink reset (if user blinked once and stopped)
        elif self.blink_count > 0 and (now - self.last_blink_time > self.double_blink_window):
             self.blink_count = 0 # Reset, it was just a random blink

        return not self.is_paused
