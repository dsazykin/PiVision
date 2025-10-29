import cv2
import onnxruntime as ort
import numpy as np
import torchvision.transforms as T
import mediapipe as mp
import os, time, json, socket
import threading
os.environ["QT_QPA_PLATFORM"] = "offscreen"

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

listener_thread = threading.Thread(target=listen_for_updates, args=(sock,), daemon=True)
listener_thread.start()

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

def update_mappings(new_map):
    global mappings
    mappings = new_map
    print("Mappings updated from server")

def listen_for_updates(sock):
    while True:
        try:
            raw = sock.recv(4096).decode().strip()
            if raw.startswith("UPDATE_MAPPINGS"):
                json_str = raw.replace("UPDATE_MAPPINGS", "").strip()
                new_map = json.loads(json_str)
                update_mappings(new_map)
        except Exception as e:
            print("Mapping update listener error:", e)
            time.sleep(1)

# --------------- MODEL SETUP ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
temp_dir = os.path.join(project_root, "WebServerStream")
os.makedirs(temp_dir, exist_ok=True)

FRAME_PATH = os.path.join(temp_dir, "latest.jpg")
JSON_PATH = os.path.join(temp_dir, "latest.json")

model_path = os.path.join(project_root, "Models", "gesture_model_v4_handcrop.onnx")

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

#Goal: mapping from the database to Pi (in DeviceControl.py)
# Get it from the database at the start
# Update the database when change and also update the Pi

# --------------- MEDIAPIPE SETUP ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
mp_hands = mp.solutions.hands.Hands(max_num_hands=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# frame_count = 0
# PROCESS_EVERY = 10

previous_gesture = ""
gesture_count = 0
minimum_hold = 3
hold_gesture = False

hold_input = True
input_sent = False

# --------------- MAIN LOOP ----------------

isLoggedIn = False
BOOLEAN_PATH = os.path.join(temp_dir, "boolean.json")

while not isLoggedIn:
    try:
        with open(BOOLEAN_PATH) as handle:
            jsonValue = json.load(handle)
    except Exception:
        jsonValue = {"loggedIn": False}
    isLoggedIn = jsonValue.get("loggedIn")

try:
    while True:
        data = {"gesture": "None", "confidence": 0.0}

        ret, frame = cap.read()
        if not ret:
            break

        # frame_count += 1
        # if frame_count % PROCESS_EVERY != 0:
        #     continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                h, w, _ = frame.shape
                xs = [int(p.x * w) for p in hand_landmarks.landmark]
                ys = [int(p.y * h) for p in hand_landmarks.landmark]
                margin = 30
                x1 = max(min(xs)-margin, 0)
                y1 = max(min(ys)-margin, 0)
                x2 = min(max(xs)+margin, w)
                y2 = min(max(ys)+margin, h)
                hand_img = frame[y1:y2, x1:x2]

                # Preprocess for model
                img = cv2.resize(hand_img, (224,224))
                img = img.astype(np.float32) / 255.0
                img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
                img = np.transpose(img, (2,0,1))[np.newaxis, :].astype(np.float32)

                outputs = session.run(None, {input_name: img})

                logits = outputs[0][0]                      # raw model outputs
                probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax
                top3_idx = np.argsort(probs)[-3:][::-1]     # indices of top 3 classes
                top3 = [(classes[i], probs[i]) for i in top3_idx]

                pred = np.argmax(outputs[0])
                label = classes[pred]

                if(label != previous_gesture):
                    # print("Detected gesture changed")
                    # print("Input sent: ", input_sent)
                    # print("Previous Gesture: ", previous_gesture)
                    # if (previous_gesture != ""):
                    #     print("Hold/Press: ", mappings.get(previous_gesture)[1])
                    if(input_sent and mappings.get(previous_gesture)[1] == "hold"):
                        msg = "release" + " " + mappings.get(previous_gesture)[0]
                        send_gesture(msg)

                    hold_gesture = False
                    input_sent = False
                    previous_gesture = label
                    gesture_count = 0

                if(gesture_count == minimum_hold or hold_gesture):
                        hold_gesture = True
                        gesture_count = 0
                        print("Detected Gesture: " + label)
                        data = {"gesture": label, "confidence": float(top3[0][1])}

                        if(not input_sent):
                            msg = mappings.get(label)[1] + " " + mappings.get(label)[0]
                            send_gesture(msg)

                        input_sent = True

                elif(label == previous_gesture):
                    gesture_count += 1

                mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                y0, dy = 40, 30
                for rank, (cls, prob) in enumerate(top3):
                    text = f"{rank+1}. {cls}: {prob*100:.1f}%"
                    y = y0 + rank * dy
                    cv2.putText(frame, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            if (previous_gesture != ""):
                if(input_sent and mappings.get(previous_gesture)[1] == "hold"):
                    msg = "release" + " " + mappings.get(previous_gesture)[0]
                    send_gesture(msg)

                hold_gesture = False
                input_sent = False
                gesture_count = 0
                previous_gesture = ""
                    
        cv2.imwrite(FRAME_PATH, frame)

        try:
            with open(JSON_PATH, 'w')as f:
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
        json.dump({"gesture": "None", "confidence": 0.0}, f)