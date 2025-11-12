import cv2
import onnxruntime as ort
import numpy as np
import mediapipe as mp
import threading
import time
import os
import json
from pathlib import Path
import pyautogui

from Pi.webserver.config.paths import PROJECT_ROOT

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

CONFIG_PATH = get_config_path()

# --- Constants ---
DEFAULT_SETTINGS = {
    "MOVE_INTERVAL": 0.03,
    "SCROLL_AMOUNT": 100,
    "MOUSE_SENSITIVITY": 5,
    "MIN_HOLD_FRAMES": 3,
    "MOUSE_HAND": "right",
    "GAME_HAND": "left",
    "MOVE_MARGIN": 30,
    "MAPPINGS": {
        "call": ["esc", "press"],
        "dislike": ["scroll_down", "hold"],
        "fist": ["delete", "press"],
        "four": ["tab", "press"],
        "like": ["scroll_up", "hold"],
        "mute": ["volume_toggle", "press"],
        "ok": ["enter", "press"],
        "one": ["left_click", "hold"],
        "palm": ["space", "press"],
        "peace": ["winleft", "press"],
        "peace_inverted": ["alt", "hold"],
        "rock": ["w", "press"],
        "stop": ["mouse", "move"],
        "stop_inverted": ["game", "hold"],
        "three": ["shift", "hold"],
        "three2": ["left_click", "press"],
        "two_up": ["right_click", "press"],
        "two_up_inverted": ["ctrl", "hold"]
    }
}

def load_json():
    """Load settings from JSON file or use defaults if missing."""
    defaults = DEFAULT_SETTINGS.copy()

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
                defaults.update(data)
        except Exception as e:
            print(f"Warning: Failed to load config, using defaults: {e}")

    return defaults

def save_json(settings):
    """Save settings to JSON file."""
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"Settings saved to {CONFIG_PATH}")
    except Exception as e:
        print(f"Error saving config: {e}")

# --- Action Performers ---

# Keep track of held states globally for pyautogui
active_mouse_holds = {}
active_key_holds = {}

def continuous_scroll(direction):
    """Scroll continuously while held."""
    while active_mouse_holds.get(direction, False):
        if direction == "scroll_up":
            pyautogui.scroll(SCROLL_AMOUNT)
        elif direction == "scroll_down":
            pyautogui.scroll(-SCROLL_AMOUNT)
        time.sleep(MOVE_INTERVAL)

def move_mouse(distance_x: int, distance_y: int):
    """One-time mouse move for per frame movements."""
    pyautogui.moveRel(distance_x, distance_y)

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
                pyautogui.scroll(SCROLL_AMOUNT if key == "scroll_up" else -SCROLL_AMOUNT)
            return

        if key in ["left_click", "right_click"]:
            button = key.replace("_click", "")
            if command == "press":
                pyautogui.click(button=button)
            elif command == "hold":
                pyautogui.mouseDown(button=button)
                print(f"Holding {button} click")
            elif command == "release":
                pyautogui.mouseUp(button=button)
                print(f"Released {button} click")
            return

    # === Handle keyboard inputs ===
    keys = [k.strip() for k in key.split('+') if k.strip()]

    try:
        if command == "press":
            if len(keys) > 1:
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(keys[0])

        elif command == "hold":
            for k in keys:
                if not active_key_holds.get(k, False):
                    pyautogui.keyDown(k)
                    active_key_holds[k] = True
            print(f"Holding {'+'.join(keys)}")

        elif command == "release":
            for k in reversed(keys):
                if active_key_holds.get(k, False):
                    pyautogui.keyUp(k)
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
                pyautogui.keyUp(key)
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
    def __init__(self, settings):
        self.settings = settings
        self.mappings = settings["MAPPINGS"]
        self.min_hold_frames = settings["MIN_HOLD_FRAMES"]
        self.mouse_sensitivity = settings["MOUSE_SENSITIVITY"]
        self.mouse_hand = settings["MOUSE_HAND"]
        self.game_hand = settings["GAME_HAND"]
        self.move_margin = settings["MOVE_MARGIN"]

        # Model setup
        available_providers = ort.get_available_providers()
        print(f"Available ONNX Providers: {available_providers}")
        provider = "CPUExecutionProvider" # Default
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
        if state.input_sent and self.mappings.get(state.previous_gesture, ["", ""])[1] == "hold":
            msg = f"release {self.mappings[state.previous_gesture][0]}"
            perform_action(msg)
        state.reset()

    def _calculate_and_perform_mouse_move(self, state: HandState, hand_landmarks, frame_shape):
        """Calculates mouse movement based on index fingertip and performs the move."""
        h, w, _ = frame_shape
        # Get the coordinates of the index fingertip
        index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
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
        index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

        # Is this the first frame where the user is doing this gesture
        if state.game_coords is None:
            state.game_coords = cx, cy # Save the starting position for movement
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
        frame = cv2.flip(frame, 1) # Flip the camera input so that right and left is correct
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame) # Find the hands in the frame

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
                        state.hold_gesture = True # Mark the gesture as held
                        print(f"{hand_label.capitalize()} Hand Detected Gesture: {label}")

                    # Get the input mapped to this gesture
                    action_key, action_type = self.mappings.get(label, [None, None])

                    # Is the user moving the mouse?
                    if action_key == "mouse" and hand_label == self.mouse_hand:
                        # Move the mouse based on hand movements
                        coords = self._calculate_and_perform_mouse_move(state, hand_landmarks,
                                                                        frame.shape)
                        cv2.circle(frame, coords, 10, (255, 0, 255), cv2.FILLED)
                        state.input_sent = True # Mark input as sent to prevent double movement
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
                        self._calculate_and_perform_game_input(state, hand_landmarks, frame.shape)

                        # Draw a box outline with half sidelength being move_margin around game_coords
                        if state.game_coords:
                            gx, gy = state.game_coords
                            cv2.rectangle(frame, (gx - self.move_margin, gy - self.move_margin), (gx + self.move_margin, gy + self.move_margin), (0, 255, 0), 2)

                        state.input_sent = True # Mark input as sent to prevent double inputs
                    # The user is using the wrong hand
                    elif action_key == "game" and hand_label != self.game_hand:
                        state.input_sent = True # Mark input as sent so that it is not interpreted

                    # Has the input been sent yet?
                    if not state.input_sent:
                        msg = f"{action_type} {action_key}" # Create the input message
                        perform_action(msg)
                        state.input_sent = True

                else: # Gesture is the same, but not held long enough yet
                    state.frame_count += 1

                # Drawing
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                data = {"gesture": label, "confidence": 0.0} # Confidence removed for performance
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
mappings = {
    "call": ["esc", "press"],
    "dislike": ["scroll_down", "hold"],
    "fist": ["delete", "press"],
    "four": ["tab", "press"],
    "like": ["scroll_up", "hold"],
    "mute": ["volume_toggle", "press"],
    "ok": ["enter", "press"],
    "one": ["left_click", "hold"],
    "palm": ["space", "press"],
    "peace": ["winleft", "press"],
    "peace_inverted": ["alt", "hold"],
    "rock": ["w", "press"],
    "stop": ["mouse", "move"],
    "stop_inverted": ["game", "hold"],
    "three": ["shift", "hold"],
    "three2": ["left_click", "press"],
    "two_up": ["right_click", "press"],
    "two_up_inverted": ["ctrl", "hold"]
}

if "MAPPINGS" not in DEFAULT_SETTINGS or not DEFAULT_SETTINGS["MAPPINGS"]:
    DEFAULT_SETTINGS["MAPPINGS"] = mappings

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
    user_settings = load_json()
    # Update global settings used by pyautogui functions
    global MOVE_DISTANCE, MOVE_INTERVAL, SCROLL_AMOUNT, MOUSE_HAND
    MOVE_DISTANCE = user_settings["MOVE_DISTANCE"]
    MOVE_INTERVAL = user_settings["MOVE_INTERVAL"]
    SCROLL_AMOUNT = user_settings["SCROLL_AMOUNT"]
    MOUSE_HAND = user_settings["MOUSE_HAND"]

    # Initialize camera
    print("Starting threaded video stream...")
    vs = WebcamVideoStream(src=0).start()

    if not vs.stream.isOpened():
        print("Error: Could not open video stream.")
        return

    controller = GestureController(user_settings)
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

if __name__ == "__main__":
    main()