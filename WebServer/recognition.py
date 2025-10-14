import cv2
import onnxruntime as ort
import numpy as np
import mediapipe as mp
import pyautogui
import database
import threading
import time


def run_recognition(username):
    """Run gesture recognition for the given user (non-blocking)."""
    # Load gesture mappings from database
    gesture_mappings = database.get_user_mappings(username)

    print(f"[INFO] Starting gesture detection for user: {username}")
    print(f"[INFO] Loaded mappings: {gesture_mappings}")

    # Load ONNX model
    session = ort.InferenceSession("models/gesture_model_v3.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # List of gesture labels
    classes = [
        'train_val_call', 'train_val_dislike', 'train_val_fist', 'train_val_four',
        'train_val_like', 'train_val_mute', 'train_val_ok', 'train_val_one',
        'train_val_palm', 'train_val_peace', 'train_val_peace_inverted',
        'train_val_rock', 'train_val_stop', 'train_val_stop_inverted',
        'train_val_three', 'train_val_three2', 'train_val_two_up',
        'train_val_two_up_inverted'
    ]

    mp_hands = mp.solutions.hands.Hands(max_num_hands=1,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    print("[INFO] Press 'q' to quit gesture detection.")
    while True:
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
                x1, y1 = max(min(xs) - margin, 0), max(min(ys) - margin, 0)
                x2, y2 = min(max(xs) + margin, w), min(max(ys) + margin, h)
                hand_img = frame[y1:y2, x1:x2]

                if hand_img.size == 0:
                    continue

                # Preprocess for model
                img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
                img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                img = np.transpose(img, (2, 0, 1))[np.newaxis, :].astype(np.float32)

                outputs = session.run(None, {input_name: img})
                pred = np.argmax(outputs[0])
                label = classes[pred]

                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # === Gesture to Key Mapping ===
                for gesture, key in gesture_mappings.items():
                    # Normalize naming differences
                    if gesture.lower().replace(" ", "_") in label.lower():
                        print(f"[ACTION] {gesture} → press '{key}'")
                        pyautogui.press(key)
                        time.sleep(0.3)  # avoid key spam

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Gesture detection stopped.")
