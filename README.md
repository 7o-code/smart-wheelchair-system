# Smart Wheelchair System 🦼
# WISE🧠 - Wheelchair with Intelligent Sight and
Eye-control

An accessible, AI-driven assistive mobility platform utilizing a **Master/Slave (Brain/Reflex)** architecture. This repository contains the full software stack for the Smart Wheelchair Graduation Project.

## 🏗️ System Architecture

*   **Master (Raspberry Pi 5)**: Runs the ROS 2 Humble stack. Handles AI Perception (MediaPipe Iris) and High-level Navigation.
*   **Slave (STM32F411)**: Low-level real-time control (20kHz PWM) and hardware-level safety interrupts.
*   **Bridge (ESP8266)**: Handles auxiliary telemetry and cloud interface.

## 📂 Repository Structure

```text
├── firmware/
│   ├── stm32-reflex/         # STM32F411 "Reflex" Firmware (Arduino)
│   └── esp-interface/        # ESP8266 Telemetry Bridge (PlatformIO)
├── ros2_ws/
│   └── src/
│       └── wheelchair_core/  # ROS 2 Humble Package (Logic & Nodes)
├── perception/               # MediaPipe Eye-Tracking & Gaze Model
├── README.md                 # Project Documentation
└── LICENSE                   # MIT License
```

## 🚀 Getting Started

### Prerequisites
*   Ubuntu 22.04 LTS
*   ROS 2 Humble
*   Python 3.10+
*   Arduino IDE / PlatformIO (for firmware)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/smart-wheelchair-system.git
    ```
2.  Install ROS 2 dependencies:
    ```bash
    cd ros2_ws
    rosdep install --from-paths src --ignore-src -r -y
    colcon build
    ```
3.  Install Perception requirements:
    ```bash
    cd perception
    pip install -r requirements_rpi.txt
    ```

## 🚨 Safety First
The system implements a multi-layered safety strategy:
1.  **Level 1 (Hardware)**: STM32 Ultrasonic interrupts (300mm Emergency Stop).
2.  **Level 2 (Software)**: ROS 2 `reactive_avoidance_node` (600mm Speed Scaling).
3.  **Level 3 (Supervisor)**: 200ms Command Watchdog.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
