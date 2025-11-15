from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QApplication, QLabel, QPushButton, QVBoxLayout,
    QWidget, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QHBoxLayout, QDialog,
    QGridLayout, QMessageBox, QScrollArea, QFrame, QInputDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QEvent
from PySide6.QtGui import QImage, QPixmap, QKeySequence
import cv2
import numpy as np
import json
import os
from pathlib import Path
from copy import deepcopy
import onnxruntime as ort
import mediapipe as mp
import threading
import time

import pydirectinput

from Pi.webserver.config.paths import PROJECT_ROOT

# ---------- User Settings Management (local to GUI) ----------

DEFAULT_USER_SETTINGS = {
    "MOVE_INTERVAL": 0.03,
    "SCROLL_AMOUNT": 100,
    "MOUSE_SENSITIVITY": 5,
    "MIN_HOLD_FRAMES": 3,
    "MOUSE_HAND": "right",
    "GAME_HAND": "left",
    "MOVE_MARGIN": 30,
    "presets": {"default": {
        "call": [
            "esc",
            "press"
        ],
        "dislike": [
            "scroll_down",
            "hold"
        ],
        "fist": [
            "delete",
            "press"
        ],
        "four": [
            "tab",
            "press"
        ],
        "like": [
            "scroll_up",
            "hold"
        ],
        "mute": [
            "volume_toggle",
            "press"
        ],
        "ok": [
            "enter",
            "press"
        ],
        "one": [
            "left_click",
            "hold"
        ],
        "palm": [
            "space",
            "press"
        ],
        "peace": [
            "winleft",
            "press"
        ],
        "peace_inverted": [
            "alt",
            "hold"
        ],
        "rock": [
            "w",
            "press"
        ],
        "stop": [
            "mouse",
            "move"
        ],
        "stop_inverted": [
            "game",
            "hold"
        ],
        "three": [
            "shift",
            "hold"
        ],
        "three2": [
            "left_click",
            "press"
        ],
        "two_up": [
            "right_click",
            "press"
        ],
        "two_up_inverted": [
            "ctrl",
            "hold"
        ]
    }},
    "active_preset": "default"
}

def get_config_path():
    """Return the platform-specific path to the PiVision config file."""
    # Windows → AppData\Roaming\PiVision
    # Linux/Mac → ~/.config/PiVision
    if os.name == "nt":
        base_dir = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base_dir = Path.home() / ".config"
    pivision_dir = base_dir / "PiVision"
    pivision_dir.mkdir(parents=True, exist_ok=True)
    return pivision_dir / "config.json"

SETTINGS_FILE = get_config_path()

def load_user_settings():
    """Load settings from local JSON, or create default."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
                # fill in missing keys if new defaults are added later
                for k, v in DEFAULT_USER_SETTINGS.items():
                    if k not in data:
                        data[k] = v
                return data
        except Exception as e:
            print(f"[WARN] Could not read user_settings.json: {e}")
    save_user_settings(DEFAULT_USER_SETTINGS)
    return DEFAULT_USER_SETTINGS.copy()

def save_user_settings(settings):
    """Save settings to local JSON."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"[INFO] User settings saved → {SETTINGS_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save settings: {e}")

# ===============================================================
# -------------------- MAIN WINDOW -------------------------------
# ===============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.user_settings = load_user_settings()
        print("[INFO] Loaded user settings:", self.user_settings)

        self.setWindowTitle("PiVision Control Panel")
        self.resize(1000, 700)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # PAGES
        self.home = HomePage(self)
        self.recognition = RecognitionPage(self)
        self.settings = SettingsPage(self)
        self.mappings = MappingsPage(self)

        # ADD TO STACK
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.recognition)
        self.stack.addWidget(self.settings)
        self.stack.addWidget(self.mappings)

        # CONNECT BUTTONS
        self.home.start_button.clicked.connect(self.start_recognition)
        self.home.settings_button.clicked.connect(self.go_settings)
        self.home.mapping_button.clicked.connect(self.go_mappings)

        self.recognition.stop_button.clicked.connect(self.go_home)
        self.settings.back_button.clicked.connect(self.go_home)
        self.mappings.back_button.clicked.connect(self.go_home)

        # Apply modern dark style
        self.apply_stylesheet()

    # ------------- Navigation --------------
    def start_recognition(self):
        self.recognition.update_active_preset_display()
        self.stack.setCurrentWidget(self.recognition)
        self.recognition.start_camera()

    def go_home(self):
        self.recognition.stop_camera()
        self.home.update_active_preset_display()  # Update preset label
        self.stack.setCurrentWidget(self.home)

    def go_settings(self):
        self.settings.load_settings_from_json()
        self.stack.setCurrentWidget(self.settings)

    def go_mappings(self):
        self.stack.setCurrentWidget(self.mappings)

    # ------------- Stylesheet --------------
    def apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #dddddd;
                font-family: Segoe UI, Roboto, Arial;
                font-size: 15px;
            }

            QLabel {
                color: #f0f0f0;
            }

            QPushButton {
                background-color: #2d2d30;
                color: #ffffff;
                padding: 10px 20px;
                border-radius: 10px;
                border: 1px solid #3e3e42;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a3a3d;
            }
            QPushButton:pressed {
                background-color: #0078d7;
            }
            QPushButton#DeleteButton {
                color: #ffdddd; border-color: #9e2a2b;
            }
            QPushButton#DeleteButton:hover {
                background-color: #540b0e;
            }

            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #2b2b2b;
                border: 1px solid #3e3e42;
                padding: 5px;
                border-radius: 6px;
                color: #ffffff;
            }

            QScrollArea {
                background: transparent;
                border: none;
            }

            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 10px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
            }

            QHeaderLabel {
                font-size: 22px;
                font-weight: bold;
                color: #ffffff;
            }
        """)
    #------------ Functionality of updating JSON ---------------
    def update_gesture_mapping(self, gesture_name, action_key, action_type):
        """Update one gesture mapping and save to local JSON."""
        active_preset = self.user_settings.get("active_preset", "default")

        # Make sure the preset exists
        if "presets" not in self.user_settings:
            self.user_settings["presets"] = {}
        if active_preset not in self.user_settings["presets"]:
            self.user_settings["presets"][active_preset] = {}

        # Update mapping
        self.user_settings["presets"][active_preset][gesture_name] = [action_key, action_type]

        # Save immediately
        save_user_settings(self.user_settings)
        print(f"[INFO] Updated '{gesture_name}' → {action_key}, {action_type}")

# ===============================================================
# -------------------- HOME PAGE --------------------------------
# ===============================================================

class HomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        main_layout = QVBoxLayout()

        # --- Central content ---
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.setSpacing(20)
        
        title = QLabel("PiVision Control Panel")
        title.setStyleSheet("font-size: 28px; font-weight: bold; margin-bottom: 40px;")

        self.start_button = QPushButton("▶ Start Gesture Recognition")
        self.settings_button = QPushButton("⚙ Settings")
        self.mapping_button = QPushButton("🎮 Gesture Mappings")
        
        for btn in [self.start_button, self.settings_button, self.mapping_button]:
            btn.setFixedWidth(300)
            btn.setFixedHeight(50)

        center_layout.addWidget(title)
        center_layout.addWidget(self.start_button)
        center_layout.addWidget(self.settings_button)
        center_layout.addWidget(self.mapping_button)

        # --- Main layout structure ---
        main_layout.addStretch(1)  # Pushes content down
        main_layout.addLayout(center_layout)
        main_layout.addStretch(1)  # Pushes preset label down

        # Preset display label
        self.preset_label = QLabel()
        self.preset_label.setStyleSheet("font-size: 12px; color: #888888; padding-right: 10px;")
        self.preset_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(self.preset_label)

        self.setLayout(main_layout)
        self.update_active_preset_display()

    def update_active_preset_display(self):
        """Reads the active preset from settings and updates the label."""
        active_preset = self.parent_window.user_settings.get("active_preset", "default")
        self.preset_label.setText(f"Active Preset: {active_preset}")

# ===============================================================
# -------------------- CAMERA THREAD -----------------------------
# ===============================================================

# --- Action Performers ---
SCROLL_AMOUNT = 0
MOVE_INTERVAL = 0

# Keep track of held states globally for pyautogui
active_mouse_holds = {}
active_key_holds = {}

def continuous_scroll(direction):
    """Scroll continuously while held."""
    """SCROLL AMOUNT IS NOT IMPLEMENT YET, SHOULD BE A PARAMETER IN THE SCROLL FUNCTION"""
    while active_mouse_holds.get(direction, False):
        if direction == "scroll_up":
            pydirectinput.scroll(clicks=1, wheel_delta=SCROLL_AMOUNT)
            # pyautogui.scroll(SCROLL_AMOUNT)
        elif direction == "scroll_down":
            pydirectinput.scroll(clicks=-1, wheel_delta=SCROLL_AMOUNT)
            # pyautogui.scroll(-SCROLL_AMOUNT)
        time.sleep(MOVE_INTERVAL)

def move_mouse(distance_x: int, distance_y: int):
    """One-time mouse move for per frame movements."""
    # pyautogui.moveRel(distance_x, distance_y)
    pydirectinput.moveRel(distance_x, distance_y)

def perform_action(msg):
    parts = msg.strip().split(" ", 3)
    if len(parts) < 2:
        print(f"Ignoring invalid command: {msg}")
        return

    command = parts[0].lower()
    key = parts[1].lower()

    # === Handle mouse inputs ===
    if key in [
        "left_click", "right_click",
        "scroll_up", "scroll_down",
        "mouse"
    ]:
        if key == "mouse":
            move_mouse(int(parts[2]), int(parts[3]))
            return

        if key in ["scroll_up", "scroll_down"]:
            # Continuous scrolling
            if command == "hold":
                if not active_mouse_holds.get(key, False):
                    active_mouse_holds[key] = True
                    threading.Thread(target=continuous_scroll, args=(key,), daemon=True).start()
                    print(f"Started continuous {key}")
            elif command == "release":
                active_mouse_holds[key] = False
                print(f"Stopped continuous {key}")
            elif command == "press":
                pydirectinput.scroll(1 if key == "scroll_up" else -1)
                # pyautogui.scroll(SCROLL_AMOUNT if key == "scroll_up" else -SCROLL_AMOUNT)
            return

        if key in ["left_click", "right_click"]:
            button = key.replace("_click", "")
            if command == "press":
                pydirectinput.click(button=button)
                # pyautogui.click(button=button)
            elif command == "hold":
                pydirectinput.mouseDown(button=button)
                # pyautogui.mouseDown(button=button)
                print(f"Holding {button} click")
            elif command == "release":
                pydirectinput.mouseUp(button=button)
                # pyautogui.mouseUp(button=button)
                print(f"Released {button} click")
            return

    # === Handle keyboard inputs ===
    keys = [k.strip() for k in key.split('+') if k.strip()]

    try:
        if command == "press":
            if len(keys) > 1:
                pydirectinput.hotkey(*keys)
                # pyautogui.hotkey(*keys)
            else:
                pydirectinput.press(keys[0])
                # pyautogui.press(keys[0])

        elif command == "hold":
            for k in keys:
                if not active_key_holds.get(k, False):
                    pydirectinput.keyDown(k)
                    # pyautogui.keyDown(k)
                    active_key_holds[k] = True
            print(f"Holding {'+'.join(keys)}")

        elif command == "release":
            for k in reversed(keys):
                if active_key_holds.get(k, False):
                    pydirectinput.keyUp(k)
                    # pyautogui.keyUp(k)
                    active_key_holds[k] = False
            print(f"Released {'+'.join(keys)}")

        else:
            print(f"Unknown command: {command}")

    except Exception as e:
        print(f"Error performing action '{msg}': {e}")

def reset_active_holds():
    """Release any held keys or mouse actions when the connection ends."""
    for key in list(active_mouse_holds.keys()):
        active_mouse_holds[key] = False

    for key, held in list(active_key_holds.items()):
        if held:
            try:
                pydirectinput.keyUp(key)
                # pyautogui.keyUp(key)
            except Exception as exc:
                print(f"Error releasing key '{key}': {exc}")
        active_key_holds[key] = False

# --- Gesture and State Management Classes ---

class HandState:
    """Encapsulates the state of a single hand."""

    def __init__(self):
        self.previous_gesture = ""
        self.frame_count = 0
        self.hold_gesture = False
        self.input_sent = False
        self.prev_coords = None
        self.game_coords = None
        self.holds = []

    def reset(self):
        self.previous_gesture = ""
        self.frame_count = 0
        self.hold_gesture = False
        self.input_sent = False
        self.prev_coords = None
        self.game_coords = None
        self.holds = []

class GestureController:
    """Manages gesture detection, state, and action dispatching."""
    global SCROLL_AMOUNT, MOVE_INTERVAL

    def __init__(self, settings):
        global SCROLL_AMOUNT, MOVE_INTERVAL

        self.settings = settings
        self.preset = settings["active_preset"]
        self.mappings = settings.get("presets", {}).get(self.preset)
        self.min_hold_frames = settings["MIN_HOLD_FRAMES"]
        self.mouse_sensitivity = settings["MOUSE_SENSITIVITY"]
        self.mouse_hand = settings["MOUSE_HAND"]
        self.game_hand = settings["GAME_HAND"]
        self.move_margin = settings["MOVE_MARGIN"]
        SCROLL_AMOUNT = settings["SCROLL_AMOUNT"]
        MOVE_INTERVAL = settings["MOVE_INTERVAL"]

        # Model setup
        available_providers = ort.get_available_providers()
        print(f"Available ONNX Providers: {available_providers}")
        provider = "CPUExecutionProvider"  # Default
        if "DmlExecutionProvider" in available_providers:
            provider = "DmlExecutionProvider"
            print("Using DirectML Execution Provider (GPU).")
        elif "CUDAExecutionProvider" in available_providers:
            provider = "CUDAExecutionProvider"
            print("Using CUDA Execution Provider (NVIDIA GPU).")
        else:
            print("Using CPU Execution Provider.")

        model_path = os.path.join(PROJECT_ROOT, "Models", "gesture_model_v4_handcrop.onnx")
        self.session = ort.InferenceSession(model_path, providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.classes = [
            'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
            'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
            'three', 'three2', 'two_up', 'two_up_inverted'
        ]

        # Mediapipe setup
        self.mp_hands = mp.solutions.hands.Hands(
            model_complexity=0,  # Use the lighter-weight model (0) instead of the default (1)
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils

        # State tracking for each hand
        self.hand_states = {'left': HandState(), 'right': HandState()}

    def process_frame(self, frame: np.ndarray, hand_landmarks) -> str:
        """Process the frame and then pass it through ML model to detect gesture."""
        h, w, _ = frame.shape
        # Find the coordinates of all plotted hand points
        xs = [int(p.x * w) for p in hand_landmarks.landmark]
        ys = [int(p.y * h) for p in hand_landmarks.landmark]
        margin = 30

        # Crop the image around the hand to feed into the ML model
        x1, y1 = max(min(xs) - margin, 0), max(min(ys) - margin, 0)
        x2, y2 = min(max(xs) + margin, w), min(max(ys) + margin, h)
        hand_img = frame[y1:y2, x1:x2]

        # Double check that the cropped image is valid
        if hand_img.size == 0:
            return "none"

        # Preprocess the image before passing into ML model
        img = cv2.resize(hand_img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :].astype(np.float32)

        # Pass the image into the model and get the output
        outputs = self.session.run(None, {self.input_name: img})
        pred_idx = np.argmax(outputs[0])
        label = self.classes[pred_idx]

        # Return the detected gesture
        return label

    def _handle_gesture_change(self, state: HandState):
        """Handle logic when a gesture changes."""
        if self.mappings.get(state.previous_gesture, ["", ""])[0] == "game":
            for button in state.holds:
                msg = f"release {button}"
                perform_action(msg)
        if state.input_sent and self.mappings.get(state.previous_gesture, ["", ""])[
            1] == "hold":
            msg = f"release {self.mappings[state.previous_gesture][0]}"
            perform_action(msg)
        state.reset()

    def _calculate_and_perform_mouse_move(self, state: HandState, hand_landmarks, frame_shape):
        """Calculates mouse movement based on index fingertip and performs the move."""
        h, w, _ = frame_shape
        # Get the coordinates of the index fingertip
        index_finger_tip = hand_landmarks.landmark[
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

        # If the hand moved
        if state.prev_coords:
            px, py = state.prev_coords
            dx = (cx - px) * self.mouse_sensitivity
            dy = (cy - py) * self.mouse_sensitivity
            if dx != 0 or dy != 0:
                msg = f"press mouse {dx} {dy}"
                perform_action(msg)

        # Update the stored coords to ensure correct calculation in the next frame
        state.prev_coords = (cx, cy)

        return (cx, cy)

    def _calculate_and_perform_game_input(self, state: HandState, hand_landmarks, frame_shape):
        """Calculates game movement based on index fingertip and performs the input."""
        h, w, _ = frame_shape
        # Get the coordinates of the index fingertip
        index_finger_tip = hand_landmarks.landmark[
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

        # Is this the first frame where the user is doing this gesture
        if state.game_coords is None:
            state.game_coords = cx, cy  # Save the starting position for movement
        # Starting point already exists
        else:
            # Calculate how far the user moved their hand
            px, py = state.game_coords
            dx = (cx - px)
            dy = (cy - py)

            # Has the index point moved vertically outside the margin area
            if dy >= self.move_margin and "s" not in state.holds:
                msg = "hold s"
                perform_action(msg)
                state.holds.append("s")
            elif "s" in state.holds and not dy >= self.move_margin:
                state.holds.remove("s")
                msg = "release s"
                perform_action(msg)

            elif dy <= -self.move_margin and "w" not in state.holds:
                msg = "hold w"
                perform_action(msg)
                state.holds.append("w")
            elif "w" in state.holds and not dy <= -self.move_margin:
                state.holds.remove("w")
                msg = "release w"
                perform_action(msg)

            # Has the index point moved horizontally outside the margin area
            if dx >= self.move_margin and "d" not in state.holds:
                msg = "hold d"
                perform_action(msg)
                state.holds.append("d")
            elif "d" in state.holds and not dx >= self.move_margin:
                state.holds.remove("d")
                msg = "release d"
                perform_action(msg)

            elif dx <= -self.move_margin and "a" not in state.holds:
                msg = "hold a"
                perform_action(msg)
                state.holds.append("a")
            elif "a" in state.holds and not dx <= -self.move_margin:
                state.holds.remove("a")
                msg = "release a"
                perform_action(msg)

    def run_detection(self, frame):
        """Processes a single camera frame for gesture detection."""
        frame = cv2.flip(frame, 1)  # Flip the camera input so that right and left is correct
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)  # Find the hands in the frame

        detected_hands = set()
        # If there is a hand in frame
        if results.multi_hand_landmarks:
            # Loop through detected hands to recognise gesture
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get the hand that is being processed
                hand_label = results.multi_handedness[idx].classification[0].label.lower()
                detected_hands.add(hand_label)
                state = self.hand_states[hand_label]

                # Pass the frame through the ML model to get the performed gesture and probability
                label = self.process_frame(frame, hand_landmarks)

                # Is the currently detected gesture different from the one in the previous frame
                # for this hand?
                if label != state.previous_gesture:
                    self._handle_gesture_change(state)
                    state.previous_gesture = label

                # Has the gesture been held for a sufficient amount or is it still being held?
                if state.frame_count >= self.min_hold_frames or state.hold_gesture:
                    # Has the gesture just started to be held?
                    if not state.hold_gesture:
                        state.hold_gesture = True  # Mark the gesture as held
                        print(f"{hand_label.capitalize()} Hand Detected Gesture: {label}")

                    # Get the input mapped to this gesture
                    action_key, action_type = self.mappings.get(label, [None, None])

                    # Is the user moving the mouse?
                    if action_key == "mouse" and hand_label == self.mouse_hand:
                        # Move the mouse based on hand movements
                        coords = self._calculate_and_perform_mouse_move(state, hand_landmarks,
                                                                        frame.shape)
                        cv2.circle(frame, coords, 10, (255, 0, 255), cv2.FILLED)
                        state.input_sent = True  # Mark input as sent to prevent double movement
                    # Is the user holding a mouse button?
                    elif "click" in action_key and action_type == "hold" and hand_label == self.mouse_hand:
                        # Move the mouse based on hand movements
                        coords = self._calculate_and_perform_mouse_move(state, hand_landmarks,
                                                                        frame.shape)
                        cv2.circle(frame, coords, 10, (255, 0, 255), cv2.FILLED)
                    # The user is using the wrong hand for mouse movements
                    elif action_key == "mouse" or ("click" in action_key and action_type ==
                                                   "hold") and hand_label != self.mouse_hand:
                        state.input_sent = True  # Mark input as sent so that it is not interpreted

                    # Is the user performing game input
                    if action_key == "game" and hand_label == self.game_hand:
                        # Send inputs based on hand position
                        self._calculate_and_perform_game_input(state, hand_landmarks,
                                                               frame.shape)

                        # Draw a box outline with half sidelength being move_margin around game_coords
                        if state.game_coords:
                            gx, gy = state.game_coords
                            cv2.rectangle(frame, (gx - self.move_margin, gy - self.move_margin),
                                          (gx + self.move_margin, gy + self.move_margin),
                                          (0, 255, 0), 2)

                        state.input_sent = True  # Mark input as sent to prevent double inputs
                    # The user is using the wrong hand
                    elif action_key == "game" and hand_label != self.game_hand:
                        state.input_sent = True  # Mark input as sent so that it is not interpreted

                    # Has the input been sent yet?
                    if not state.input_sent:
                        msg = f"{action_type} {action_key}"  # Create the input message
                        perform_action(msg)
                        state.input_sent = True

                else:  # Gesture is the same, but not held long enough yet
                    state.frame_count += 1

                # Drawing
                self.mp_draw.draw_landmarks(frame, hand_landmarks,
                                            mp.solutions.hands.HAND_CONNECTIONS)
                data = {"gesture": label,
                        "confidence": 0.0}  # Confidence removed for performance
                add_text(frame, data, hand_label)

        # Handle hands that are no longer detected
        for hand_label in ['left', 'right']:
            if hand_label not in detected_hands:
                state = self.hand_states[hand_label]
                if state.previous_gesture:
                    self._handle_gesture_change(state)

        return frame

# Default mappings if not in config
classes = [
    'call', 'dislike', 'fist', 'four',
    'like', 'mute', 'ok', 'one',
    'palm', 'peace', 'peace_inverted',
    'rock', 'stop', 'stop_inverted',
    'three', 'three2', 'two_up', 'two_up_inverted'
]

# --- UI and Helper Functions ---

def add_text(frame: np.ndarray, gesture: dict, hand_label: str):
    """Add text to the frame to show the most likely gesture, positioning right hand text on the right."""
    yd, yh = 70, 30
    if hand_label == 'right':
        # Position right hand text on the right side of the frame
        x_offset = frame.shape[1] - 160
    else:
        x_offset = 10
    cv2.putText(frame, f"Hand: {hand_label.capitalize()}", (x_offset, yh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Confidence calculation was removed for performance, so we just show the gesture.
    # If you re-enable it, you can use the old text format.
    text = f"Gesture: {gesture.get('gesture')}"
    # text = f"{gesture.get('gesture')}: {gesture.get('confidence', 0.0) * 100:.1f}%"
    cv2.putText(frame, text, (x_offset, yd),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

class WebcamVideoStream:
    """A threaded wrapper for cv2.VideoCapture to improve performance."""

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # Set camera properties here
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

def main():
    """Main function to run the gesture detection loop."""
    settings = load_user_settings()

    # Initialize camera
    print("Starting threaded video stream...")
    vs = WebcamVideoStream(src=0).start()

    if not vs.stream.isOpened():
        print("Error: Could not open video stream.")
        return

    controller = GestureController(settings)
    print("Gesture detection started. Press 'q' to quit.")

    while True:
        try:
            frame = vs.read()
            if frame is None:
                print("Info: End of video stream.")
                break

            processed_frame = controller.run_detection(frame)

            cv2.imshow("Gesture Detector + Classifier", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
    vs.stopped = True
    cv2.destroyAllWindows()
    reset_active_holds()

class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)
    gesture_ready = Signal(str)
    vs = None

    def run(self):
        global vs
        settings = load_user_settings()

        vs = WebcamVideoStream(src=0).start()

        if not vs.stream.isOpened():
            print("Error: Could not open video stream.")
            return

        controller = GestureController(settings)
        print("Gesture detection started.")

        while True:
            frame = vs.read()
            if frame is None:
                print("Info: End of video stream.")
                break

            processed_frame = controller.run_detection(frame)
            self.frame_ready.emit(processed_frame)
        vs.stopped = True
        reset_active_holds()

    def stop(self):
        if vs is not None:
            vs.stopped = True
            reset_active_holds()

# ===============================================================
# -------------------- RECOGNITION PAGE --------------------------
# ===============================================================

class RecognitionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        layout = QVBoxLayout()

        header = QLabel("Gesture Recognition")
        header.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 20px;")

        self.video_label = QLabel("Camera Feed")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 10px;")

        self.gesture_label = QLabel("Gesture: ...")
        self.gesture_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stop_button = QPushButton("⏹ Stop Recognition")

        layout.addWidget(header, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.gesture_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.stop_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()  # Pushes the preset label to the bottom

        # Preset display label
        self.preset_label = QLabel()
        self.preset_label.setStyleSheet("font-size: 12px; color: #888888; padding-right: 10px;")
        self.preset_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.preset_label)

        self.update_active_preset_display()

        self.setLayout(layout)

        self.thread = CameraThread()
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.gesture_ready.connect(self.update_gesture)

    def start_camera(self):
        self.thread.start()

    def stop_camera(self):
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.terminate()

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def update_gesture(self, gesture):
        self.gesture_label.setText(f"Gesture: {gesture}")

    def update_active_preset_display(self):
        """Reads the active preset from settings and updates the label."""
        active_preset = self.parent_window.user_settings.get("active_preset", "default")
        self.preset_label.setText(f"Active Preset: {active_preset}")

# ===============================================================
# -------------------- SETTINGS PAGE -----------------------------
# ===============================================================

class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.user_settings = self.parent_window.user_settings

        layout = QVBoxLayout()
        header = QLabel("⚙ Settings")
        header.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 20px;")

        # --- Form Layout ---
        form = QFormLayout()

        # Mouse sensitivity
        self.mouse_sensitivity = QDoubleSpinBox()
        self.mouse_sensitivity.setRange(0.1, 10.0)
        self.mouse_sensitivity.setSingleStep(0.1)
        form.addRow("Mouse Sensitivity (0.1 - 10.0):", self.mouse_sensitivity)

        # Scroll speed
        self.scroll_amount = QSpinBox()
        self.scroll_amount.setRange(10, 1000)
        form.addRow("Scroll Speed (px, 10 - 1000):", self.scroll_amount)

        # Min hold frames
        self.min_hold_frames = QSpinBox()
        self.min_hold_frames.setRange(1, 120)
        form.addRow("Minimum Hold Frames (1 - 120):", self.min_hold_frames)

        # Mouse hand
        self.mouse_hand = QComboBox()
        self.mouse_hand.addItems(["right", "left"])
        form.addRow("Mouse Hand:", self.mouse_hand)

        # Game hand
        self.game_hand = QComboBox()
        self.game_hand.addItems(["right", "left"])
        form.addRow("Game Hand:", self.game_hand)

        # Move Interval
        self.move_interval = QDoubleSpinBox()
        self.move_interval.setRange(0.001, 1)
        self.move_interval.setSingleStep(0.01)
        form.addRow("Move Interval (0.001 - 1.0):", self.move_interval)

        # Move Margin
        self.move_margin = QDoubleSpinBox()
        self.move_margin.setRange(1, 100)
        self.move_margin.setSingleStep(5)
        form.addRow("Move Margin (1 - 100.0):", self.move_margin)

        # --- Buttons ---
        self.save_button = QPushButton("💾 Save Settings")
        self.revert_button = QPushButton("↩ Revert to Default")
        self.back_button = QPushButton("← Back")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.revert_button)
        button_layout.addWidget(self.back_button)

        layout.addWidget(header, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(form)
        layout.addSpacing(20)
        layout.addLayout(button_layout)
        layout.addStretch()
        self.setLayout(layout)

        # Connect signals
        self.save_button.clicked.connect(self.save_settings)
        self.revert_button.clicked.connect(self.revert_to_default)

        # Load initial data
        self.load_settings_from_json()

    # ------------------------------
    # Load current JSON values
    # ------------------------------
    def load_settings_from_json(self):
        """Read user settings and populate input widgets."""
        settings = self.parent_window.user_settings
        self.mouse_sensitivity.setValue(settings.get("MOUSE_SENSITIVITY", 5))
        self.scroll_amount.setValue(settings.get("SCROLL_AMOUNT", 100))
        self.min_hold_frames.setValue(settings.get("MIN_HOLD_FRAMES", 3))
        self.mouse_hand.setCurrentText(settings.get("MOUSE_HAND", "right"))
        self.game_hand.setCurrentText(settings.get("GAME_HAND", "left"))
        self.move_interval.setValue(settings.get("MOVE_INTERVAL", 0.03))
        self.move_margin.setValue(settings.get("MOVE_MARGIN", 30))

    # ------------------------------
    # Save updated values to JSON
    # ------------------------------
    def save_settings(self):
        """Update user_settings.json with the new values."""
        settings = self.parent_window.user_settings

        settings["MOUSE_SENSITIVITY"] = self.mouse_sensitivity.value()
        settings["SCROLL_AMOUNT"] = self.scroll_amount.value()
        settings["MIN_HOLD_FRAMES"] = self.min_hold_frames.value()
        settings["MOUSE_HAND"] = self.mouse_hand.currentText().lower()
        settings["GAME_HAND"] = self.game_hand.currentText().lower()
        settings["MOVE_INTERVAL"] = self.move_interval.value()
        settings["MOVE_MARGIN"] = self.move_margin.value()

        save_user_settings(settings)
        self.parent_window.user_settings = load_user_settings()

        QMessageBox.information(self, "Settings Saved", "Your settings were successfully saved!")

    # ------------------------------
    # Revert to default values
    # ------------------------------
    def revert_to_default(self):
        """Reset settings to their default values."""
        default_settings = {
            "MOVE_INTERVAL": 0.03,
            "SCROLL_AMOUNT": 100,
            "MOUSE_SENSITIVITY": 5,
            "MIN_HOLD_FRAMES": 3,
            "MOUSE_HAND": "right",
            "GAME_HAND": "left",
            "MOVE_MARGIN": 30,
        }

        reply = QMessageBox.question(
            self, "Confirm Reset",
            "Are you sure you want to reset all settings to default?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        save_user_settings(default_settings)
        self.parent_window.user_settings = load_user_settings()
        self.load_settings_from_json()

        QMessageBox.information(self, "Reverted", "Settings have been reset to default values.")

# ===============================================================
# -------------------- MAPPINGS PAGE -----------------------------
# ===============================================================

class MappingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # keep parent ref for saving
        self.parent_window = parent

        layout = QVBoxLayout()
        header = QLabel("🎮 Gesture Mappings")
        header.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(header, alignment=Qt.AlignmentFlag.AlignCenter)

        # Preset selector (kept for UI; active_preset is used)
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_selector = QComboBox()
        # fill presets from user_settings if available
        presets = list(self.parent_window.user_settings.get("presets", {}).keys())
        if not presets:
            presets = ["default"]
        self.preset_selector.addItems(presets)
        self.preset_selector.setCurrentText(self.parent_window.user_settings.get("active_preset", presets[0]))
        self.preset_selector.currentTextChanged.connect(self.on_preset_changed)

        self.new_preset_button = QPushButton("New")
        self.new_preset_button.setStyleSheet("padding: 5px 10px; font-size: 13px;")
        self.new_preset_button.clicked.connect(self.create_new_preset)

        self.rename_preset_button = QPushButton("Rename")
        self.rename_preset_button.setStyleSheet("padding: 5px 10px; font-size: 13px;")
        self.rename_preset_button.clicked.connect(self.rename_current_preset)

        self.delete_preset_button = QPushButton("Delete")
        self.delete_preset_button.setObjectName("DeleteButton") # For specific styling
        self.delete_preset_button.setStyleSheet("padding: 5px 10px; font-size: 13px;")
        self.delete_preset_button.clicked.connect(self.delete_current_preset)

        preset_layout.addWidget(self.preset_selector)
        preset_layout.addWidget(self.new_preset_button)
        preset_layout.addWidget(self.rename_preset_button)
        preset_layout.addWidget(self.delete_preset_button)
        layout.addLayout(preset_layout)

        # Scroll area for mappings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.grid = QGridLayout()
        container.setLayout(self.grid)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # bottom buttons
        btn_layout = QHBoxLayout()
        self.revert_button = QPushButton("↩ Revert to Default")
        self.back_button = QPushButton("← Back")
        btn_layout.addWidget(self.revert_button)
        btn_layout.addWidget(self.back_button)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # storage for widget refs
        self.bind_buttons = {}     # gesture -> QPushButton (shows current binding)
        self.duration_boxes = {}   # gesture -> QComboBox (press/hold)
        self.mouse_buttons = {}    # gesture -> QPushButton
        self.game_buttons = {}     # gesture -> QPushButton
        self.listening_gesture = None  # currently listening gesture name or None

        # populate UI
        self.populate_from_settings()

        # wire buttons
        self.revert_button.clicked.connect(self.revert_mappings)
        # back_button already connected by MainWindow in your setup

        # install event filter on application to capture keys & mouse while listening
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

    # ---------------- UI population ----------------
    def populate_from_settings(self):
        # clear grid
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # header row
        self.grid.addWidget(QLabel("Gesture"), 0, 0)
        self.grid.addWidget(QLabel("Binding"), 0, 1)
        self.grid.addWidget(QLabel("Mode"), 0, 2)
        self.grid.addWidget(QLabel("Quick Assign"), 0, 3, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        self.grid.addWidget(QLabel("Info"), 0, 5)


        # get current preset data
        preset_name = self.preset_selector.currentText()
        presets = self.parent_window.user_settings.get("presets", {})
        preset_data = presets.get(preset_name, {})

        gestures = sorted(preset_data.keys())
        for i, g in enumerate(gestures, start=1):
            lbl = QLabel(g)

            action_key, current_mode = preset_data.get(g, ["custom_key", "press"])
            # binding button (click to listen)
            btn = QPushButton(self.display_text_for_action(action_key))
            btn.setToolTip("Click to change binding; then press a key or mouse button.")
            btn.clicked.connect(lambda _, gesture=g: self.start_listening_for(gesture))
            btn.setFixedWidth(220)

            # duration combo
            mode_combo = QComboBox()
            mode_combo.addItems(["press", "hold"])
            mode_combo.setCurrentText(current_mode if current_mode in ("press", "hold") else "press")
            mode_combo.currentTextChanged.connect(lambda _, gesture=g: self.on_mode_changed(gesture))

            # Stylesheet for the quick assign buttons
            quick_assign_style = """
                QPushButton { padding: 4px 8px; font-size: 13px; }
                QPushButton:checked {
                    background-color: #0078d7;
                    border: 1px solid #005a9e;
                }
            """

            # Quick assign buttons
            mouse_btn = QPushButton("Mouse")
            mouse_btn.setCheckable(True)
            mouse_btn.setChecked(action_key == "mouse")
            mouse_btn.setStyleSheet(quick_assign_style)
            mouse_btn.clicked.connect(lambda _, gesture=g: self.on_quick_assign(gesture, "mouse"))

            game_btn = QPushButton("Game")
            game_btn.setCheckable(True)
            game_btn.setChecked(action_key == "game")
            game_btn.setStyleSheet(quick_assign_style)
            game_btn.clicked.connect(lambda _, gesture=g: self.on_quick_assign(gesture, "game"))

            quick_assign_layout = QHBoxLayout()
            quick_assign_layout.addWidget(mouse_btn)
            quick_assign_layout.addWidget(game_btn)
            quick_assign_layout.setContentsMargins(0, 0, 0, 0)

            # info button
            info = QPushButton("🛈")
            info.setFixedWidth(60)
            info.clicked.connect(lambda _, gesture=g: self.show_info(gesture))

            # Add widgets to grid
            self.grid.addWidget(lbl, i, 0)
            self.grid.addWidget(btn, i, 1)
            self.grid.addWidget(mode_combo, i, 2)
            self.grid.addLayout(quick_assign_layout, i, 3, 1, 2)
            self.grid.addWidget(info, i, 5)

            # store refs
            self.bind_buttons[g] = btn
            self.duration_boxes[g] = mode_combo
            self.mouse_buttons[g] = mouse_btn
            self.game_buttons[g] = game_btn

    # ---------------- helpers ----------------
    def on_preset_changed(self, preset_name):
        # update active preset in memory and reload UI
        self.parent_window.user_settings["active_preset"] = preset_name
        save_user_settings(self.parent_window.user_settings)
        self.populate_from_settings()

    def rename_current_preset(self):
        """Rename the currently selected preset."""
        current_name = self.preset_selector.currentText()
        if not current_name:
            QMessageBox.warning(self, "Rename Preset", "No preset selected to rename.")
            return

        if current_name == "default":
            QMessageBox.warning(self, "Rename Preset", "The 'default' preset cannot be renamed.")
            return

        new_name, ok = QInputDialog.getText(
            self, "Rename Preset", f"Enter a new name for '{current_name}':", text=current_name
        )

        if ok and new_name:
            new_name = new_name.strip()
            if not new_name:
                QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty.")
                return

            all_presets = self.parent_window.user_settings.get("presets", {})
            if new_name in all_presets and new_name != current_name:
                QMessageBox.warning(self, "Name Exists", f"A preset named '{new_name}' already exists.")
                return

            # Update settings
            all_presets[new_name] = all_presets.pop(current_name)
            self.parent_window.user_settings["active_preset"] = new_name
            save_user_settings(self.parent_window.user_settings)

            # Update UI
            self.preset_selector.blockSignals(True)
            current_index = self.preset_selector.findText(current_name)
            self.preset_selector.setItemText(current_index, new_name)
            self.preset_selector.setCurrentText(new_name)
            self.preset_selector.blockSignals(False)

            QMessageBox.information(self, "Success", f"Preset '{current_name}' was renamed to '{new_name}'.")

    def delete_current_preset(self):
        """Delete the currently selected preset."""
        current_name = self.preset_selector.currentText()
        if not current_name:
            QMessageBox.warning(self, "Delete Preset", "No preset selected to delete.")
            return

        if current_name == "default":
            QMessageBox.warning(self, "Delete Preset", "The 'default' preset cannot be deleted.")
            return

        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to permanently delete the preset '{current_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            all_presets = self.parent_window.user_settings.get("presets", {})
            if current_name in all_presets:
                del all_presets[current_name]

            # Switch back to default and save
            self.parent_window.user_settings["active_preset"] = "default"
            save_user_settings(self.parent_window.user_settings)

            # Repopulate the dropdown and set the default preset as active
            self.preset_selector.blockSignals(True)
            self.preset_selector.clear()
            self.preset_selector.addItems(list(all_presets.keys()))
            self.preset_selector.setCurrentText("default")
            self.preset_selector.blockSignals(False)

            self.populate_from_settings()
            QMessageBox.information(self, "Success", f"Preset '{current_name}' has been deleted.")

    def create_new_preset(self):
        """Create a new preset by copying the default."""
        new_name, ok = QInputDialog.getText(self, "New Preset", "Enter a name for the new preset:")
        if ok and new_name:
            new_name = new_name.strip()
            if not new_name:
                QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty.")
                return

            all_presets = self.parent_window.user_settings.get("presets", {})
            if new_name in all_presets:
                QMessageBox.warning(self, "Name Exists", f"A preset named '{new_name}' already exists.")
                return

            # Create new preset by copying default and save
            all_presets[new_name] = deepcopy(DEFAULT_USER_SETTINGS["presets"]["default"])
            self.parent_window.user_settings["active_preset"] = new_name
            save_user_settings(self.parent_window.user_settings)

            # Repopulate the dropdown and set the new preset as active
            self.preset_selector.blockSignals(True)
            self.preset_selector.clear()
            self.preset_selector.addItems(list(all_presets.keys()))
            self.preset_selector.setCurrentText(new_name)
            self.preset_selector.blockSignals(False)

            self.populate_from_settings()

    def display_text_for_action(self, action_key: str) -> str:
        """Make a human-friendly label for a raw action string."""
        if not action_key:
            return "Unassigned"
        ak = action_key.lower()
        mapping = {
            "left_click": "Left Click",
            "right_click": "Right Click",
            "middle_click": "Middle Click",
            "scroll_up": "Scroll Up",
            "scroll_down": "Scroll Down",
            "mouse": "Mouse Control",
            "game": "Game Control"
        }
        if ak in mapping:
            return mapping[ak]
        # typical keyboard string (e.g. "ctrl+a" or "space")
        return action_key

    def start_listening_for(self, gesture_name):
        """Begin listening for next key or mouse button for given gesture."""
        # if already listening something, ignore or replace
        if self.listening_gesture is not None:
            QMessageBox.information(self, "Listening", f"Already listening for '{self.listening_gesture}'. Press a key or cancel.")
            return
        self.listening_gesture = gesture_name
        # update button visual
        btn = self.bind_buttons.get(gesture_name)
        if btn:
            btn.setText("Listening... (press key or click mouse)")
            btn.setStyleSheet("background-color: #444; font-weight: bold; font-size: 10px;")
        # give user focus hint
        self.grabKeyboard()
        self.grabMouse()

    def stop_listening(self):
        """Stop listening mode."""
        if self.listening_gesture:
            # restore button text from current settings
            preset = self.parent_window.user_settings.get("active_preset", "default")
            action = self.parent_window.user_settings.get("presets", {}).get(preset, {}).get(self.listening_gesture, [""])[0]
            btn = self.bind_buttons.get(self.listening_gesture)
            if btn:
                btn.setText(self.display_text_for_action(action))
                btn.setStyleSheet("background-color: #2d2d30; font-size: 15px;")
        self.listening_gesture = None
        try:
            self.releaseKeyboard()
            self.releaseMouse()
        except Exception:
            pass

    def on_mode_changed(self, gesture_name):
        """When press/hold changed — save mapping immediately (keep action the same)."""
        preset = self.parent_window.user_settings.get("active_preset", "default")
        action = self.parent_window.user_settings.get("presets", {}).get(preset, {}).get(gesture_name, ["unassigned", "press"])[0]
        mode = self.duration_boxes[gesture_name].currentText().lower()
        # persist
        self.parent_window.update_gesture_mapping(gesture_name, action, mode)

    def on_quick_assign(self, gesture_name, assign_type):
        """Handle 'Mouse' or 'Game' button clicks."""
        other_btn = self.game_buttons[gesture_name] if assign_type == "mouse" else self.mouse_buttons[gesture_name]
        other_btn.setChecked(False)

        action_key = assign_type
        mode = self.duration_boxes[gesture_name].currentText().lower()
        self.parent_window.update_gesture_mapping(gesture_name, action_key, mode)

        btn = self.bind_buttons.get(gesture_name)
        if btn:
            btn.setText(self.display_text_for_action(action_key))

    def on_mode_changed(self, gesture_name):
        """When press/hold changed — save mapping immediately (keep action the same)."""
        preset = self.parent_window.user_settings.get("active_preset", "default")
        action = self.parent_window.user_settings.get("presets", {}).get(preset, {}).get(gesture_name, ["unassigned", "press"])[0]
        mode = self.duration_boxes[gesture_name].currentText().lower()
        # persist
        self.parent_window.update_gesture_mapping(gesture_name, action, mode)

    def save_mappings(self):
        # ensure saved (MainWindow.update_gesture_mapping already persisted per change)
        save_user_settings(self.parent_window.user_settings)
        QMessageBox.information(self, "Saved", "Mappings saved to user_settings.json.")

    def revert_mappings(self):
        """Restore the default preset mappings from DEFAULT_USER_SETTINGS."""
        confirm = QMessageBox.question(
            self, "Revert Mappings",
            "Are you sure you want to revert all mappings to default?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        # deep copy default preset to avoid shared references
        default_presets = deepcopy(DEFAULT_USER_SETTINGS.get("presets", {}))

        # overwrite user_settings
        self.parent_window.user_settings["presets"] = default_presets
        self.parent_window.user_settings["active_preset"] = "default"

        # update preset selector in UI
        self.preset_selector.clear()
        self.preset_selector.addItems(list(default_presets.keys()))
        self.preset_selector.setCurrentText("default")

        save_user_settings(self.parent_window.user_settings)
        QMessageBox.information(self, "Reverted", "Mappings reverted to default.")
        self.populate_from_settings()

    def show_info(self, gesture):
        QMessageBox.information(self, "Gesture Preview", f"Preview for '{gesture}' gesture (image/video to be added).")

    # ---------------- event filter to capture input ----------------
    def eventFilter(self, obj, event):
        # only act when listening
        if self.listening_gesture is None:
            return super().eventFilter(obj, event)

        # capture keyboard events
        if event.type() == QEvent.KeyPress:
            try:
                action_key = self.key_event_to_action(event)
                mode = self.duration_boxes[self.listening_gesture].currentText().lower()
                self.parent_window.update_gesture_mapping(self.listening_gesture, action_key, mode)
                btn = self.bind_buttons.get(self.listening_gesture)
                if btn:
                    btn.setText(self.display_text_for_action(action_key))
                # Uncheck quick assign buttons
                self.mouse_buttons[self.listening_gesture].setChecked(False)
                self.game_buttons[self.listening_gesture].setChecked(False)

                self.stop_listening()
                return True  # consume event
            except Exception as e:
                print("Error capturing key event:", e)
                self.stop_listening()
                return True

        # capture mouse button events
        if event.type() == QEvent.MouseButtonPress:
            try:
                action_key = self.mouse_event_to_action(event)
                mode = self.duration_boxes[self.listening_gesture].currentText().lower()
                self.parent_window.update_gesture_mapping(self.listening_gesture, action_key, mode)
                btn = self.bind_buttons.get(self.listening_gesture)
                if btn:
                    btn.setText(self.display_text_for_action(action_key))
                # Uncheck quick assign buttons
                self.mouse_buttons[self.listening_gesture].setChecked(False)
                self.game_buttons[self.listening_gesture].setChecked(False)

                self.stop_listening()
                return True
            except Exception as e:
                print("Error capturing mouse event:", e)
                self.stop_listening()
                return True

        # capture mouse wheel events
        if event.type() == QEvent.Wheel:
            try:
                action_key = self.mouse_scroll_event_to_action(event)
                mode = self.duration_boxes[self.listening_gesture].currentText().lower()
                self.parent_window.update_gesture_mapping(self.listening_gesture, action_key, mode)
                btn = self.bind_buttons.get(self.listening_gesture)
                if btn:
                    btn.setText(self.display_text_for_action(action_key))
                # Uncheck quick assign buttons
                self.mouse_buttons[self.listening_gesture].setChecked(False)
                self.game_buttons[self.listening_gesture].setChecked(False)

                self.stop_listening()
                return True
            except Exception as e:
                print("Error capturing wheel event:", e)
                self.stop_listening()
                return True

        return super().eventFilter(obj, event)

    def mouse_scroll_event_to_action(self, event: QEvent) -> str:
        delta = event.angleDelta().y()
        if delta > 0:
            action_key = "scroll_up"
        else:
            action_key = "scroll_down"
        return action_key

    def key_event_to_action(self, event):
        """Convert the key input into a mapping."""
        key = event.key()
        return QKeySequence(key).toString().lower() or f"key_{int(key)}"

    def mouse_event_to_action(self, event):
        """Convert QMouseEvent into canonical action string."""
        btn = event.button()
        if btn == Qt.LeftButton:
            return "left_click"
        if btn == Qt.RightButton:
            return "right_click"
        if btn == Qt.MiddleButton:
            return "middle_click"
        return "mouse_button"

    # ensure cleanup: uninstall filter if widget destroyed
    def closeEvent(self, ev):
        app = QApplication.instance()
        if app:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass
        super().closeEvent(ev)

# ===============================================================
# -------------------- MAIN ENTRY --------------------------------
# ===============================================================

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
