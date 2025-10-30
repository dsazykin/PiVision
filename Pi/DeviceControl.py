import cv2
import onnxruntime as ort
import numpy as np
import mediapipe as mp
import os, time, json, socket
import threading
import inotify.adapters
from webserver.paths import (LOGGEDIN_PATH, FRAME_PATH, JSON_PATH, MAPPINGS_PATH, PROJECT_ROOT,
                             PASSWORD_PATH, PASSWORD_GESTURE_PATH)
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

def recognize_gestures():
    global sendPassword

    previous_gesture = ""
    gesture_count = 0
    minimum_hold = 3

    try:
        while sendPassword:
            data = {"gesture": "none"}

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    h, w, _ = frame.shape
                    xs = [int(p.x * w) for p in hand_landmarks.landmark]
                    ys = [int(p.y * h) for p in hand_landmarks.landmark]
                    margin = 30
                    x1 = max(min(xs) - margin, 0)
                    y1 = max(min(ys) - margin, 0)
                    x2 = min(max(xs) + margin, w)
                    y2 = min(max(ys) + margin, h)
                    hand_img = frame[y1:y2, x1:x2]

                    # Preprocess for model
                    img = cv2.resize(hand_img, (224, 224))
                    img = img.astype(np.float32) / 255.0
                    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                    img = np.transpose(img, (2, 0, 1))[np.newaxis, :].astype(np.float32)

                    outputs = session.run(None, {input_name: img})

                    logits = outputs[0][0]  # raw model outputs
                    probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax
                    top3_idx = np.argsort(probs)[-3:][::-1]  # indices of top 3 classes
                    top3 = [(classes[i], probs[i]) for i in top3_idx]

                    pred = np.argmax(outputs[0])
                    label = classes[pred]

                    if (label != previous_gesture):
                        previous_gesture = label
                        gesture_count = 0

                    if (gesture_count == minimum_hold):
                        gesture_count = 0
                        print("detected gesture: ", label)
                        data = {"gesture": label}

                        if label == "stop":
                            sendPassword = False

                    elif (label == previous_gesture):
                        gesture_count += 1

                    mp_draw.draw_landmarks(frame, hand_landmarks,
                                           mp.solutions.hands.HAND_CONNECTIONS)

                    y0, dy = 40, 30
                    for rank, (cls, prob) in enumerate(top3):
                        text = f"{rank + 1}. {cls}: {prob * 100:.1f}%"
                        y = y0 + rank * dy
                        cv2.putText(frame, text, (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                if (previous_gesture != ""):
                    input_sent = False
                    gesture_count = 0
                    previous_gesture = ""

            cv2.imwrite(FRAME_PATH, frame)

            try:
                with open(PASSWORD_GESTURE_PATH, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                print("Error writing JSON: ", e)

            time.sleep(0.05)

        with open(PASSWORD_GESTURE_PATH, 'w') as f:
            json.dump({"gesture": "none"}, f)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(FRAME_PATH, empty_frame)  # Clear the frame
        with open(JSON_PATH, 'w') as f:
            json.dump({"gesture": "none"}, f)

# Watch for mapping updates (using inotify)
def watch_for_mapping_updates_inotify():
    global mappings

    i = inotify.adapters.Inotify()

    i.add_watch(MAPPINGS_PATH)

    print("Watching for mapping file updates via inotify...")

    for event in i.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event
        if "IN_CLOSE_WRITE" in type_names:
            try:
                with open(MAPPINGS_PATH, "r") as f:
                    new_map = json.load(f)
                mappings = new_map
                print("🔄 Mappings reloaded (file changed).")
            except Exception as e:
                print("⚠️ Failed to reload mappings:", e)

# Start inotify watcher in a background thread
threading.Thread(target=watch_for_mapping_updates_inotify, daemon=True).start()

# --------------- MEDIAPIPE SETUP ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
mp_hands = mp.solutions.hands.Hands(max_num_hands=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

previous_gesture = ""
gesture_count = 0
minimum_hold = 3
hold_gesture = False

hold_input = True
input_sent = False

# --------------- MAIN LOOP ----------------
isLoggedIn = False

def gesture_control():
    global isLoggedIn, previous_gesture, gesture_count, hold_gesture, hold_input, input_sent, minimum_hold

    try:
        while isLoggedIn:
            data = {"gesture": "none", "confidence": 0.0}

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    h, w, _ = frame.shape
                    xs = [int(p.x * w) for p in hand_landmarks.landmark]
                    ys = [int(p.y * h) for p in hand_landmarks.landmark]
                    margin = 30
                    x1 = max(min(xs) - margin, 0)
                    y1 = max(min(ys) - margin, 0)
                    x2 = min(max(xs) + margin, w)
                    y2 = min(max(ys) + margin, h)
                    hand_img = frame[y1:y2, x1:x2]

                    # Preprocess for model
                    img = cv2.resize(hand_img, (224, 224))
                    img = img.astype(np.float32) / 255.0
                    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                    img = np.transpose(img, (2, 0, 1))[np.newaxis, :].astype(np.float32)

                    outputs = session.run(None, {input_name: img})

                    logits = outputs[0][0]  # raw model outputs
                    probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax
                    top3_idx = np.argsort(probs)[-3:][::-1]  # indices of top 3 classes
                    top3 = [(classes[i], probs[i]) for i in top3_idx]

                    pred = np.argmax(outputs[0])
                    label = classes[pred]

                    if label != previous_gesture:
                        if input_sent and mappings.get(previous_gesture)[1] == "hold":
                            msg = "release" + " " + mappings.get(previous_gesture)[0]
                            send_gesture(msg)

                        hold_gesture = False
                        input_sent = False
                        previous_gesture = label
                        gesture_count = 0

                    if gesture_count == minimum_hold or hold_gesture:
                        hold_gesture = True
                        gesture_count = 0
                        print("Detected Gesture: " + label)
                        data = {"gesture": label, "confidence": float(top3[0][1])}

                        if not input_sent:
                            msg = mappings.get(label)[1] + " " + mappings.get(label)[0]
                            send_gesture(msg)

                        input_sent = True

                    elif label == previous_gesture:
                        gesture_count += 1

                    mp_draw.draw_landmarks(frame, hand_landmarks,
                                           mp.solutions.hands.HAND_CONNECTIONS)

                    y0, dy = 40, 30
                    for rank, (cls, prob) in enumerate(top3):
                        text = f"{rank + 1}. {cls}: {prob * 100:.1f}%"
                        y = y0 + rank * dy
                        cv2.putText(frame, text, (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                if previous_gesture != "":
                    if input_sent and mappings.get(previous_gesture)[1] == "hold":
                        msg = "release" + " " + mappings.get(previous_gesture)[0]
                        send_gesture(msg)

                    hold_gesture = False
                    input_sent = False
                    gesture_count = 0
                    previous_gesture = ""

            cv2.imwrite(FRAME_PATH, frame)

            try:
                with open(JSON_PATH, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                print("Error writing JSON: ", e)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        sock.close()
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(FRAME_PATH, empty_frame)  # Clear the frame
        with open(JSON_PATH, 'w') as f:
            json.dump({"gesture": "none", "confidence": 0.0}, f)

while True:
    try:
        with open(LOGGEDIN_PATH) as handle:
            jsonValue = json.load(handle)
    except Exception:
        jsonValue = {"loggedIn": False}
    isLoggedIn = jsonValue.get("loggedIn")

    if not sendPassword:
        try:
            with open(PASSWORD_PATH) as handle:
                jsonValue = json.load(handle)
        except Exception:
            jsonValue = {"value": False}
        sendPassword = jsonValue.get("value")

        if sendPassword:
            recognize_gestures()

    if isLoggedIn:
        gesture_control()