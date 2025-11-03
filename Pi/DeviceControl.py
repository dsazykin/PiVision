import cv2
import onnxruntime as ort
import numpy as np
import mediapipe as mp
import os, time, socket
import threading
import inotify.adapters

from Pi.webserver.config.paths import (FRAME_PATH, MAPPINGS_PATH, PROJECT_ROOT)
from Pi.SaveJson import update_current_gesture, update_password_gesture
from Pi.ReadJson import check_loggedin, check_entering_password, get_new_mappings
os.environ["QT_QPA_PLATFORM"] = "offscreen"

sendPassword = False

# --------------- TCP CONNECTION SETUP ----------------
possible_ips = ["192.168.137.1", "192.168.5.2", "192.168.178.16"]
PORT = 9000

def connect_to_server(possible_ips, port):
    while True:
        for ip in possible_ips:
            try:
                print(f"Trying {ip}...")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                s.connect((ip, port))
                print(f"✅ Connected to {ip}:{port}")
                return s
            except Exception as e:
                print(f"Failed to connect to {ip}: {e}")
                time.sleep(1)

# Connect before starting gesture recognition
sock = connect_to_server(possible_ips, PORT)

def send_gesture(label):    
    try:
        sock.sendall((label + "\n").encode())
        print("Sent: " + label)
    except Exception as e:
        print("Send failed, reconnecting:", e)
        sock.close()
        time.sleep(1)
        # Try to reconnect and resend
        new_sock = connect_to_server(possible_ips, PORT)
        new_sock.sendall((label + "\n").encode())
        return new_sock
    return sock

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
# Watch for mapping updates (using inotify)
def watch_for_mapping_updates_inotify():
    global mappings

    i = inotify.adapters.Inotify()

    i.add_watch(MAPPINGS_PATH)

    print("Watching for mapping file updates via inotify...")

    for event in i.event_gen(yield_nones=False): # If the watched file is modified read the new
        (_, type_names, path, filename) = event  # contents and update the mappings
        if "IN_CLOSE_WRITE" in type_names:
            mappings = get_new_mappings()

# Start inotify watcher in a background thread
threading.Thread(target=watch_for_mapping_updates_inotify, daemon=True).start()

# --------------- MEDIAPIPE SETUP ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
mp_hands = mp.solutions.hands.Hands(max_num_hands=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

previous_gesture = ""   # The detected gesture in the previous frame
minimum_hold = 3        # How many frames the gesture has to be held in order to be valid
frame_count = 0         # How many frames the current gesture has been held
hold_gesture = False    # If the detected gesture is being held

input_sent = False      # Has this gesture input already been sent to the connected device

# --------------- GESTURE DETECTION METHODS ----------------
def send_password_gestures():
    """Gesture recognition code used only to recognise gestures and send them for password input """
    global sendPassword, previous_gesture, frame_count, minimum_hold, hold_gesture

    try:
        while sendPassword:
            data = {"gesture": "none"} # Initialise the default json

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            if results.multi_hand_landmarks: # Is there a hand detected?
                for hand_landmarks in results.multi_hand_landmarks:

                    label, top3 = process_frame(frame, hand_landmarks) # Pass into the ML model
                                                                       # to recognise gestures

                    if label != previous_gesture: # If the detected gesture changed from the
                                                  # previous frame
                        previous_gesture = label
                        frame_count = 0 # Reset how long gesture has been held
                        hold_gesture = False

                    if frame_count == minimum_hold or hold_gesture: # Has the current gesture been held long enough?
                        frame_count = 0
                        print("detected gesture: ", label)
                        data = {"gesture": label} # Update json so that is stores detected gesture
                        hold_gesture = True

                        if label == "stop": # Is the user submitting the form?
                            sendPassword = False # Stop password input
                            hold_gesture = False

                    elif label == previous_gesture: # Gesture hasn't changed but is still held
                        frame_count += 1

                    mp_draw.draw_landmarks(frame, hand_landmarks,
                                           mp.solutions.hands.HAND_CONNECTIONS)

                    add_text(frame, top3) # Add text to the frame to show the 3 most likely gestures
            else: # No hand is detected in frame
                if previous_gesture != "": # Was there a hand in the previous frame
                    frame_count = 0
                    previous_gesture = ""
                    hold_gesture = False

            cv2.imwrite(FRAME_PATH, frame) # Store the frame so that the webserver can fetch it
            update_password_gesture(data) # Store the detected gesture

            sendPassword = check_entering_password() # Check if the user is still inputting a
                                                     # password
            time.sleep(0.05)

        update_password_gesture({"gesture": "none"}) # Once password input stops reset stored
                                                     # gesture to prevent unwanted behaviour

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(FRAME_PATH, empty_frame)  # Clear the frame
        update_password_gesture({"gesture": "none"}) # Clear last detected gesture

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
isLoggedIn = False
try:
    while True:
        isLoggedIn = check_loggedin() # Is the user logged in?

        sendPassword = check_entering_password() # Is the user inputting a password

        if sendPassword:
            send_password_gestures()

        if isLoggedIn:
            try:
                while isLoggedIn:
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
                                    send_gesture(msg)

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
                                    send_gesture(msg)

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
                            if input_sent and mappings.get(previous_gesture)[1] == "hold":\
                                # Send release command to device
                                msg = "release" + " " + mappings.get(previous_gesture)[0]
                                send_gesture(msg)

                            # Reset variables
                            hold_gesture = False
                            input_sent = False
                            frame_count = 0
                            previous_gesture = ""

                    cv2.imwrite(FRAME_PATH, frame) # Store the frame so that the webserver can fetch it
                    update_current_gesture(data) # Store the detected gesture

                    isLoggedIn = check_loggedin() # Check if the user is still logged in

                    time.sleep(0.05)
            finally:
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.imwrite(FRAME_PATH, empty_frame)  # Clear the frame
                update_current_gesture({"gesture": "none", "confidence": 0.0}) # Clear last detected gesture

except KeyboardInterrupt:
    print("\nStopped by user.")
    cap.release()
    sock.close()
finally:
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(FRAME_PATH, empty_frame)  # Clear the frame
    update_current_gesture({"gesture": "none", "confidence": 0.0})  # Clear last detected gesture