import cv2
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as T
import mediapipe as mp

# Create inference session
session = ort.InferenceSession("gesture_model.onnx", providers=["CPUExecutionProvider"])

# Print input/output details
print("Model inputs:", session.get_inputs())
print("Model outputs:", session.get_outputs())

# List of gesture class names (in same order as during training)
classes = ["call", "dislike", "fist", "like", "mute", "ok", "peace", "palm", "rock", "stop", "three", "two_up", "two_up_inverted", "thumbs_down", "thumbs_up", "one", "smile", "grip_close"]

# Load and preprocess image
img_path = "test_call.jpg"  # your local image
img = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

x = transform(img).unsqueeze(0).numpy()

# Run inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: x})
pred_idx = np.argmax(outputs[0])
print("Predicted gesture:", classes[pred_idx])

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

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
            x1 = max(min(xs)-margin, 0)
            y1 = max(min(ys)-margin, 0)
            x2 = min(max(xs)+margin, w)
            y2 = min(max(ys)+margin, h)
            hand_img = frame[y1:y2, x1:x2]

            # Preprocess for model
            hand = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(hand_img, (224,224))
            img = img.astype(np.float32) / 255.0
            img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
            img = np.transpose(img, (2,0,1))[np.newaxis, :].astype(np.float32)

            outputs = session.run(None, {input_name: img})
            pred = np.argmax(outputs[0])
            label = classes[pred]

            mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Detector + Classifier", frame)
    cv2.imshow("Cropped Hand", hand)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()