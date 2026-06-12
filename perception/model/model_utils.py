import pickle
import os
import logging

logger = logging.getLogger("ModelUtils")

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")

def load_model(path):
    if not os.path.exists(path):
        logger.warning(f"Model file not found at {path}")
        return None
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load model from {path}: {e}")
        return None

def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load scaler from {path}: {e}")
        return None
