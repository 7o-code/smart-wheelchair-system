import logging

logger = logging.getLogger("OpenCVFeatureExtractor")


class OpenCVEyeGazeExtractor:
    """
    Minimal OpenCV fallback extractor.
    This does not provide real gaze estimation; it prevents hard crashes
    if MediaPipe is unavailable.
    """
    def __init__(self, config):
        self.config = config
        logger.warning("OpenCV fallback extractor is a stub. MediaPipe is required for real inference.")

    def process_frame(self, frame):
        return None

    def close(self):
        return
