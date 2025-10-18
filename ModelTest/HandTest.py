import cv2
import onnxruntime as ort
import numpy as np
import torchvision.transforms as T
import mediapipe as mp
import os
import time
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

temp_dir = os.path.join(project_root, "WebServerStream")
os.makedirs(temp_dir, exist_ok=True)

FRAME_PATH = os.path.join(temp_dir, "latest.jpg")
JSON_PATH = os.path.join(temp_dir, "latest.json")

# Create inference session
session = ort.InferenceSession("Models/gesture_model_v4_handcrop.onnx", providers=["CPUExecutionProvider"])

# Print input/output details
print("Model inputs:", session.get_inputs())
print("Model outputs:", session.get_outputs())

# List of gesture class names (in same order as during training)
classes = [
    'train_val_call',
    'train_val_dislike',
    'train_val_fist',
    'train_val_four',
    'train_val_like',
    'train_val_mute',
    'train_val_ok',
    'train_val_one',
    'train_val_palm',
    'train_val_peace',
    'train_val_peace_inverted',
    'train_val_rock',
    'train_val_stop',
    'train_val_stop_inverted',
    'train_val_three',
    'train_val_three2',
    'train_val_two_up',
    'train_val_two_up_inverted'
]

# Run inference
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# frame_count = 0
# PROCESS_EVERY = 10

previous_gesture = ""
gesture_count = 0
minimum_hold = 10
hold_gesture = False

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
                hold_gesture = False
                previous_gesture = label
                gesture_count = 0

            if(gesture_count == minimum_hold or hold_gesture):
                    hold_gesture = True
                    gesture_count = 0
                    print("Detected Gesture: " + label)
                    data = {"gesture": label, "confidence": float(top3[0][1])}
            elif(label == previous_gesture):
                gesture_count += 1

            mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            y0, dy = 40, 30
            for rank, (cls, prob) in enumerate(top3):
                text = f"{rank+1}. {cls}: {prob*100:.1f}%"
                y = y0 + rank * dy
                cv2.putText(frame, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(FRAME_PATH, frame)

    try:
        with open(JSON_PATH, 'w')as f:
            json.dump(data, f)
    except Exception as e:
        print("Error writing JSON: ", e)

    time.sleep(0.05)

    cv2.imshow("Gesture Detector + Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()