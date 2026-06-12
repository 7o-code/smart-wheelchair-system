import time
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CommandInterface")

class EyeCommand(Enum):
    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STOP = "STOP"
    NO_COMMAND = "NO_COMMAND"

class CommandInterface:
    """
    Interface for sending commands. 
    Currently prints to console/logs. 
    In the future, this will publish to a ROS2 topic.
    """
    def __init__(self, cooldown=2.0):
        self.last_command = EyeCommand.NO_COMMAND
        self.last_command_time = 0
        self.cooldown = cooldown # Seconds between commands
        logger.info(f"Command Interface Initialized (Cooldown: {cooldown}s)")

    def publish_command(self, command: EyeCommand, confidence: float = 1.0):
        """
        Publishes the command.
        Args:
            command (EyeCommand): The command to publish.
            confidence (float): The confidence of the prediction.
        """
        if not isinstance(command, EyeCommand):
            logger.error(f"Invalid command type: {type(command)}")
            return

        current_time = time.time()
        
        # Debounce / Cooldown Logic
        if current_time - self.last_command_time < self.cooldown:
            return # Too soon

        # In a real ROS2 node, this would be: self.publisher.publish(msg)
        if command != self.last_command:
            logger.info(f"Command Published: {command.value} (Confidence: {confidence:.2f})")
            self.last_command = command
            self.last_command_time = current_time
        
        # For simulation/debug purposes, we might want to return the command string
        return command.value

    def get_available_commands(self):
        return [c.value for c in EyeCommand]
