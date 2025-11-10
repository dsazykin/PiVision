import cv2
import onnxruntime as ort
import numpy as np
import mediapipe as mp
import threading
import time
import queue
import os
import json
import tkinter as tk
from pathlib import Path
from tkinter.scrolledtext import ScrolledText

import pyautogui

from Pi.webserver.config.paths import (FRAME_PATH, MAPPINGS_PATH, PROJECT_ROOT)
from Pi.SaveJson import update_current_gesture, update_password_gesture
from Pi.ReadJson import check_loggedin, check_entering_password, get_new_mappings

log_queue = queue.Queue()

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

def load_json():
    """Load settings from JSON file or use defaults if missing."""
    defaults = {
        "MOVE_DISTANCE": 20,
        "MOVE_INTERVAL": 0.03,
        "SCROLL_AMOUNT": 100
    }

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

def log_event(message, status=None):
    """Log a message to the console and queue it for the GUI."""
    print(message)
    log_queue.put((message, status))

# Keep track of held states
active_mouse_holds = {}
active_key_holds = {}

# Adjustable sensitivity
MOVE_DISTANCE = 20     # pixels per step
MOVE_INTERVAL = 0.03   # seconds between steps
SCROLL_AMOUNT = 100    # scroll step size

def continuous_mouse_move(direction):
    """Move the mouse continuously in a given direction while held."""
    while active_mouse_holds.get(direction, False):
        if direction == "mouse_left":
            pyautogui.moveRel(-MOVE_DISTANCE, 0)
        elif direction == "mouse_right":
            pyautogui.moveRel(MOVE_DISTANCE, 0)
        elif direction == "mouse_up":
            pyautogui.moveRel(0, -MOVE_DISTANCE)
        elif direction == "mouse_down":
            pyautogui.moveRel(0, MOVE_DISTANCE)
        time.sleep(MOVE_INTERVAL)

def continuous_scroll(direction):
    """Scroll continuously while held."""
    while active_mouse_holds.get(direction, False):
        if direction == "scroll_up":
            pyautogui.scroll(SCROLL_AMOUNT)
        elif direction == "scroll_down":
            pyautogui.scroll(-SCROLL_AMOUNT)
        time.sleep(MOVE_INTERVAL)

def continuous_mouse_move_once(direction):
    """One-time mouse move for press actions."""
    if direction == "mouse_left":
        pyautogui.moveRel(-MOVE_DISTANCE, 0)
    elif direction == "mouse_right":
        pyautogui.moveRel(MOVE_DISTANCE, 0)
    elif direction == "mouse_up":
        pyautogui.moveRel(0, -MOVE_DISTANCE)
    elif direction == "mouse_down":
        pyautogui.moveRel(0, MOVE_DISTANCE)

def perform_action(msg):
    parts = msg.strip().split(" ", 1)
    if len(parts) != 2:
        log_event(f"Ignoring invalid command: {msg}")
        return

    command = parts[0].lower()
    key = parts[1].lower()

    # === Handle mouse inputs ===
    if key in [
        "left_click", "right_click",
        "mouse_left", "mouse_right",
        "mouse_up", "mouse_down",
        "scroll_up", "scroll_down"
    ]:
        if key in ["mouse_left", "mouse_right", "mouse_up", "mouse_down"]:
            # Continuous movement
            if command == "hold":
                if not active_mouse_holds.get(key, False):
                    active_mouse_holds[key] = True
                    threading.Thread(target=continuous_mouse_move, args=(key,), daemon=True).start()
                    log_event(f"Started continuous {key}")
            elif command == "release":
                active_mouse_holds[key] = False
                log_event(f"Stopped continuous {key}")
            elif command == "press":
                continuous_mouse_move_once(key)
            return

        if key in ["scroll_up", "scroll_down"]:
            # Continuous scrolling
            if command == "hold":
                if not active_mouse_holds.get(key, False):
                    active_mouse_holds[key] = True
                    threading.Thread(target=continuous_scroll, args=(key,), daemon=True).start()
                    log_event(f"Started continuous {key}")
            elif command == "release":
                active_mouse_holds[key] = False
                log_event(f"Stopped continuous {key}")
            elif command == "press":
                pyautogui.scroll(SCROLL_AMOUNT if key == "scroll_up" else -SCROLL_AMOUNT)
            return

        if key in ["left_click", "right_click"]:
            button = key.replace("_click", "")
            if command == "press":
                pyautogui.click(button=button)
            elif command == "hold":
                pyautogui.mouseDown(button=button)
                log_event(f"Holding {button} click")
            elif command == "release":
                pyautogui.mouseUp(button=button)
                log_event(f"Released {button} click")
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
            log_event(f"Holding {'+'.join(keys)}")

        elif command == "release":
            for k in reversed(keys):
                if active_key_holds.get(k, False):
                    pyautogui.keyUp(k)
                    active_key_holds[k] = False
            log_event(f"Released {'+'.join(keys)}")

        else:
            log_event(f"Unknown command: {command}")

    except Exception as e:
        log_event(f"Error performing action '{msg}': {e}")

def reset_active_holds():
    """Release any held keys or mouse actions when the connection ends."""
    for key in list(active_mouse_holds.keys()):
        active_mouse_holds[key] = False

    for key, held in list(active_key_holds.items()):
        if held:
            try:
                pyautogui.keyUp(key)
            except Exception as exc:
                log_event(f"Error releasing key '{key}': {exc}")
        active_key_holds[key] = False

def start_gui():
    root = tk.Tk()
    root.title("Laptop Server Monitor")

    status_var = tk.StringVar(value="Starting server...")

    # === STATUS BAR ===
    status_label = tk.Label(root, textvariable=status_var, font=("Segoe UI", 12, "bold"))
    status_label.pack(padx=10, pady=(10, 5), anchor="w")

    # === SETTINGS BUTTON ===
    def open_settings():
        settings_window = tk.Toplevel(root)
        settings_window.title("Settings")

        tk.Label(settings_window, text="Mouse move distance (pixels):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        move_dist_var = tk.IntVar(value=MOVE_DISTANCE)
        tk.Entry(settings_window, textvariable=move_dist_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(settings_window, text="Move interval (seconds):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        move_interval_var = tk.DoubleVar(value=MOVE_INTERVAL)
        tk.Entry(settings_window, textvariable=move_interval_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(settings_window, text="Scroll amount (pixels per step):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        scroll_amount_var = tk.IntVar(value=SCROLL_AMOUNT)
        tk.Entry(settings_window, textvariable=scroll_amount_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        def save_settings():
            global MOVE_DISTANCE, MOVE_INTERVAL, SCROLL_AMOUNT
            MOVE_DISTANCE = move_dist_var.get()
            MOVE_INTERVAL = move_interval_var.get()
            SCROLL_AMOUNT = scroll_amount_var.get()

            # Save persistently
            save_json({
                "MOVE_DISTANCE": MOVE_DISTANCE,
                "MOVE_INTERVAL": MOVE_INTERVAL,
                "SCROLL_AMOUNT": SCROLL_AMOUNT
            })

            log_event(
                f"Settings updated and saved: MOVE_DISTANCE={MOVE_DISTANCE}, MOVE_INTERVAL={MOVE_INTERVAL}, SCROLL_AMOUNT={SCROLL_AMOUNT}")
            settings_window.destroy()

        tk.Button(settings_window, text="Save", command=save_settings).grid(row=3, column=0, columnspan=2, pady=10)

    tk.Button(root, text="⚙ Settings", command=open_settings).pack(padx=10, pady=(0, 10), anchor="e")

    # === LOG WINDOW ===
    log_frame = tk.Frame(root)
    log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    log_text = ScrolledText(log_frame, wrap=tk.WORD, height=20, state=tk.DISABLED)
    log_text.pack(fill=tk.BOTH, expand=True)

    # === QUEUE PROCESSING ===
    def process_queue():
        while True:
            try:
                message, status = log_queue.get_nowait()
            except queue.Empty:
                break

            log_text.configure(state=tk.NORMAL)
            log_text.insert(tk.END, message + "\n")
            log_text.configure(state=tk.DISABLED)
            log_text.see(tk.END)

            if status:
                status_var.set(status)

        root.after(100, process_queue)

    def on_close():
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, process_queue)
    root.mainloop()

# --------------- MODEL SETUP ----------------
model_path = os.path.join(PROJECT_ROOT, "Models", "gesture_model_v4_handcrop.onnx")

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

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
    "one": ["left_click", "press"],
    "palm": ["space", "press"],
    "peace": ["winleft", "press"],
    "peace_inverted": ["alt", "hold"],
    "rock": ["w", "press"],
    "stop": ["mouse_up", "hold"],
    "stop_inverted": ["mouse_down", "hold"],
    "three": ["mouse_right", "hold"],
    "three2": ["mouse_left", "hold"],
    "two_up": ["right_click", "press"],
    "two_up_inverted": ["ctrl", "hold"]
}

# --------------- MAPPINGS UPDATE SETUP ----------------

# --------------- MEDIAPIPE SETUP ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
mp_hands = mp.solutions.hands.Hands(max_num_hands=2,
    min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

previous_gesture = ""   # The detected gesture in the previous frame
minimum_hold = 3        # How many frames the gesture has to be held in order to be valid
frame_count = 0         # How many frames the current gesture has been held
hold_gesture = False    # If the detected gesture is being held

input_sent = False      # Has this gesture input already been sent to the connected device

# --------------- GESTURE DETECTION METHODS ----------------
def process_frame(frame: np.ndarray, hand_landmarks) -> tuple[str, list[tuple[str, float]]]:
    """Process the frame and then pass it through ML model to detect gesture."""
    h, w, _ = frame.shape # Get the dimensions of the frame
    xs = [int(p.x * w) for p in hand_landmarks.landmark] # Get the x and y coordinates of the hand
    ys = [int(p.y * h) for p in hand_landmarks.landmark]
    margin = 30
    x1 = max(min(xs) - margin, 0) # Cut out an image of the hand
    y1 = max(min(ys) - margin, 0)
    x2 = min(max(xs) + margin, w)
    y2 = min(max(ys) + margin, h)
    hand_img = frame[y1:y2, x1:x2]

    # Prevent crash if crop is empty
    if hand_img.size == 0:
        return "none", []

    # Preprocess for model
    img = cv2.resize(hand_img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :].astype(np.float32)

    # Feed hand image into the ML model to detect gesture
    outputs = session.run(None, {input_name: img})

    logits = outputs[0][0]  # raw model outputs
    probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax
    top3_idx = np.argsort(probs)[-3:][::-1]  # indices of top 3 classes
    top3 = [(classes[i], probs[i]) for i in top3_idx]

    pred = np.argmax(outputs[0])
    label = classes[pred]
    return label, top3

def add_text(frame: np.ndarray, top3: list[tuple[str, float]]):
    """Add text to the frame to show the 3 most likely gestures."""
    y0, dy = 40, 30
    for rank, (cls, prob) in enumerate(top3):
        text = f"{rank + 1}. {cls}: {prob * 100:.1f}%"
        y = y0 + rank * dy
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

# --------------- MAIN LOOP ----------------
if __name__ == "__main__":
    user_settings = load_json()

    MOVE_DISTANCE = user_settings["MOVE_DISTANCE"]
    MOVE_INTERVAL = user_settings["MOVE_INTERVAL"]
    SCROLL_AMOUNT = user_settings["SCROLL_AMOUNT"]

    while True:
        try:
            data = {"gesture": "none", "confidence": 0.0} # Initialise the default json

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            if results.multi_hand_landmarks: # Is there a hand detected?
                for hand_landmarks in results.multi_hand_landmarks:

                    # Pass into the ML model to recognise gestures
                    label, top3 = process_frame(frame, hand_landmarks)

                    # If the detected gesture changed from the previous frame
                    if label != previous_gesture:
                        # Was the previous input sent, and was it a hold input?
                        if input_sent and mappings.get(previous_gesture)[1] == "hold":
                            # Send release command to device
                            msg = "release" + " " + mappings.get(previous_gesture)[0]
                            perform_action(msg)

                        # Reset variables
                        hold_gesture = False
                        input_sent = False
                        previous_gesture = label
                        frame_count = 0

                    # Has the current gesture been held long enough?
                    if frame_count == minimum_hold or hold_gesture:
                        hold_gesture = True
                        frame_count = 0
                        print("Detected Gesture: " + label)
                        data = {"gesture": label, "confidence": float(top3[0][1])}

                        # Has this input already been sent to the device?
                        if not input_sent:
                            msg = mappings.get(label)[1] + " " + mappings.get(label)[0]
                            perform_action(msg)

                        input_sent = True

                    # Gesture hasn't changed but is still held
                    elif label == previous_gesture:
                        frame_count += 1

                    mp_draw.draw_landmarks(frame, hand_landmarks,
                                           mp.solutions.hands.HAND_CONNECTIONS)

                    add_text(frame, top3) # Add text to the frame to show the 3 most likely gestures
            # No hand detected in frame
            else:
                # Was there a hand in the previous frame?
                if previous_gesture != "":
                    # Was the previous input sent, and was it a hold input?
                    if input_sent and mappings.get(previous_gesture)[1] == "hold":
                        # Send release command to device
                        msg = "release" + " " + mappings.get(previous_gesture)[0]
                        perform_action(msg)

                    # Reset variables
                    hold_gesture = False
                    input_sent = False
                    frame_count = 0
                    previous_gesture = ""

            cv2.imwrite(FRAME_PATH, frame) # Store the frame so that the webserver can fetch it
            update_current_gesture(data) # Store the detected gesture

            time.sleep(0.05)

            cv2.imshow("Gesture Detector + Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            print("\nStopped by user.")
            cap.release()
            cv2.destroyAllWindows()
        finally:
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(FRAME_PATH, empty_frame)  # Clear the frame
            update_current_gesture({"gesture": "none", "confidence": 0.0}) # Clear last detected gesture