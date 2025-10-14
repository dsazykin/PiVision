import cv2
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as T
import mediapipe as mp
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "gesture_model_v3.onnx")

# Create inference session
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

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
cap.set(cv2.CAP_PROP_FPS, 25)
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

frame_count = 0
PROCESS_EVERY = 10

previous_gesture = ""
gesture_count = 0
minimum_hold = 10
hold_gesture = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % PROCESS_EVERY != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Preprocess for model
            img = cv2.resize(frame, (224,224))
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
            elif(label == previous_gesture):
                gesture_count += 1

            mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Detector + Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()