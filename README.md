# PiVision Control: Gesture-Based Computer Interaction

![PiVision Control GUI](https://place-hold.it/800x450/1e1e1e/dddddd&text=PiVision+GUI+Screenshot)

**PiVision Control** is a sophisticated Python application that transforms your webcam into a powerful, gesture-based input device. It allows you to control your computer's mouse, keyboard, and applications using a customizable set of hand gestures, offering a futuristic and hands-free way to interact with your digital environment.

The project has evolved significantly from its original concept, transitioning from a dedicated hardware solution to a self-contained desktop application.

---

## 📜 Project Evolution

The name "PiVision" is a nod to the project's origins. It was initially designed as a client-server system centered around a Raspberry Pi.

### Phase 1: The Raspberry Pi External Device (`/Pi`)

The first iteration was built as a plug-and-play external gesture recognition module. The architecture was split into two main parts:

1.  **The Raspberry Pi (Server)**:
    *   A Raspberry Pi, equipped with a camera module, served as the dedicated processing unit.
    *   The core logic resided in `Pi/DeviceControl.py`. This script captured the video feed, performed hand tracking with MediaPipe, and classified gestures using an ONNX model.
    *   It hosted a **Flask web server** (defined in `Pi/WebServer.py` and the `Pi/webserver/` package) that provided a web-based interface for configuration. Users could connect to the Pi's local IP address to manage user accounts, customize gesture mappings, and view a live stream of the camera feed.
    *   User data and mappings were stored in a **SQLite database** (`Pi/Database.py`).
    *   Upon recognizing a gesture, the Pi would send the corresponding action command over a **TCP socket** to a client application running on the user's main computer.

2.  **The User's Computer (Client)**:
    *   A lightweight client application (`Device/PiVision Connection Software.exe`) would run on the user's Windows PC.
    *   Its sole purpose was to listen for commands from the Raspberry Pi's TCP server and translate them into keyboard and mouse inputs using libraries like `pydirectinput`.

This model was powerful but required dedicated hardware, network configuration, and managing two separate applications.

### Phase 2: The Local Desktop Application (`/Gui.py`)

To make the project more accessible and easier to use, it was refactored into a single, self-contained desktop application that runs entirely on the user's local machine.

*   **All-in-One Architecture**: The `Gui.py` script is the new entry point. It uses **PySide6 (Qt for Python)** to create a modern, feature-rich graphical user interface.
*   **No External Hardware**: The application now uses the computer's built-in or connected webcam directly, eliminating the need for a Raspberry Pi.
*   **Integrated Logic**: The gesture recognition pipeline (MediaPipe + ONNX) from `DeviceControl.py` was integrated into a `CameraThread` within the GUI application. This allows for non-blocking video processing and a responsive user interface.
*   **Local Configuration**: The complex Flask web server and SQLite database were replaced with a simple and portable JSON configuration file (`config.json`). This file is stored in the user's local application data directory and holds all settings and gesture mapping presets.
*   **Direct Input**: Instead of sending commands over a network, the application now uses `pydirectinput` to execute actions directly on the host machine.

This evolution transformed PiVision Control from a hardware project into a distributable software utility that anyone can run on their computer.

---

## ✨ Features

- **Real-Time Gesture Recognition**: Utilizes a custom-trained ONNX model for fast and accurate classification of 18 different hand gestures.
- **Advanced Hand Tracking**: Leverages Google's MediaPipe framework to detect and track up to two hands simultaneously in the video feed.
- **Comprehensive Control Modes**:
  - **Standard Actions**: Map gestures to single key presses (`'a'`), hotkeys (`'ctrl+c'`), and mouse clicks.
  - **Continuous Actions**: Configure gestures to be "held" for continuous actions like scrolling, holding a key down, or dragging with the mouse.
  - **Mouse Control Mode**: Dedicate a gesture to take full control of the mouse cursor, moving it based on your hand's position.
  - **Game Control Mode**: Assign a gesture to a virtual joystick, translating hand movements into WASD-style inputs for gaming.
- **Preset Management**: Create, save, rename, and switch between different sets of gesture mappings. Tailor your controls for different applications, games, or users.
- **Fine-Tuned Settings**: A dedicated settings page in the GUI allows you to adjust mouse sensitivity, scroll speed, gesture hold duration, and more.
- **Modern & Intuitive UI**: A clean, dark-themed interface built with PySide6 makes it easy to manage settings, customize mappings, and view the live camera feed.
- **Hardware Acceleration**: Automatically attempts to use your GPU for faster model inference (via DirectML on Windows or CUDA on Linux/NVIDIA), with a seamless fallback to CPU.

---

## 🛠️ Technical Deep Dive

The core of PiVision Control is a multi-stage pipeline that runs on every frame from the camera.

1.  **Frame Capture**: A multi-threaded camera wrapper (`WebcamVideoStream`) captures frames from the webcam at a high frame rate without blocking the main GUI thread.

2.  **Hand Detection & Tracking**: The captured frame is passed to a `GestureController` instance. It uses `mediapipe.solutions.hands` to locate and track the 3D landmarks of each hand present in the frame. The system is configured to track up to two hands, identifying them as 'left' or 'right'.

3.  **Gesture Classification**:
    *   For each detected hand, the image is cropped around the hand's bounding box.
    *   This cropped image is pre-processed (resized to 224x224, normalized) and fed into the pre-trained ONNX model (`gesture_model_v4_handcrop.onnx`).
    *   The **ONNX Runtime** engine performs inference to get a prediction. It intelligently selects the best execution provider available (DirectML, CUDA, or CPU) to maximize performance.
    *   The model's output (a logit array) is converted into a probability distribution, and the gesture with the highest probability is chosen as the recognized gesture.

4.  **State Management & Action Dispatching**:
    *   The system maintains a `HandState` for each hand to prevent jitter and accidental inputs. A gesture must be held for a configurable number of frames (`MIN_HOLD_FRAMES`) before it is considered active.
    *   When a gesture becomes active, the system looks up the corresponding action in the currently active preset from the `config.json` file.
    *   The action is dispatched via the `perform_action` function, which uses `pydirectinput` to simulate keyboard and mouse events. This library is used for its compatibility with games and applications that might ignore inputs from other libraries.
    *   The system correctly handles "press" (one-time) and "hold" (continuous) actions, sending `keyDown`/`mouseDown` events when a hold gesture begins and `keyUp`/`mouseUp` events when it ends.

---

## 📂 Project Structure

```
PiVision/
├── Device/
│   └── PiVision Connection Software.exe  # (Legacy) Client for the Pi version
├── Gui.py                                # Entry point for the modern Desktop GUI application
├── Models/
│   └── gesture_model_v4_handcrop.onnx    # The core gesture classification model
├── Pi/                                   # (Legacy) Code for the Raspberry Pi server
│   ├── DeviceControl.py                  # Main gesture detection script for the Pi
│   ├── WebServer.py                      # Flask web server entry point
│   ├── webserver/                        # Flask blueprints, routes, and logic
│   ├── Database.py                       # SQLite database management
│   ├── static/                           # CSS and images for the web interface
│   └── ...
└── README.md                             # This file

```

---

## 🚀 Getting Started (Local GUI)

1.  **Prerequisites**:
    *   Python 3.8+
    *   A webcam

2.  **Installation**:
    *   Clone the repository:
        ```bash
        git clone https://github.com/your-username/PiVisionCont.git
        cd PiVisionCont
        ```
    *   Install the required Python packages. It is highly recommended to use a virtual environment.
        ```bash
        # You may need to create a requirements.txt file first
        pip install opencv-python pydirectinput onnxruntime-directml mediapipe PySide6
        ```

3.  **Running the Application**:
    *   Execute the `Gui.py` script:
        ```bash
        python Gui.py
        ```

4.  **Usage**:
    *   From the home screen, click **▶ Start Gesture Recognition**.
    *   To customize controls, stop recognition, go back home, and navigate to the **🎮 Gesture Mappings** page.
    *   To adjust performance and feel, visit the **⚙ Settings** page.

Enjoy a new way of interacting with your computer!