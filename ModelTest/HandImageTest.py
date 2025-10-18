import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import os
import mediapipe as mp
import os
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

temp_dir = os.path.join(project_root, "WebServerStream")
os.makedirs(temp_dir, exist_ok=True)

FRAME_PATH = os.path.join(temp_dir, "latest.jpg")
JSON_PATH = os.path.join(temp_dir, "latest.json")
model_path = os.path.join(script_dir, "..", "Models", "gesture_model_v4_handcrop.onnx")

# Create inference session
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Print input/output details
print("Model inputs:", session.get_inputs())
print("Model outputs:", session.get_outputs())

# List of gesture class names (in same order as during training)
classes = [
    'call',
    'dislike',
    'fist',
    'four',
    'like',
    'mute',
    'ok',
    'one',
    'palm',
    'peace',
    'peaceinverted',
    'rock',
    'stop',
    'stopinverted',
    'three',
    'three2',
    'twoup',
    'twoupinverted'
]

base_dir = Path(script_dir).parent / "ModelTest" / "TestImages"
print("Looking in:", base_dir.resolve())

mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

total = 0
correct = 0

for ext in ["*.jpg", "*.jpeg", "*.png"]:
    for img_path in base_dir.rglob(ext):
        total += 1

        # Load with OpenCV (which returns a NumPy array)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = mp_hands.process(img)

        if not results.multi_hand_landmarks:
            print(f"{img_path.name:<25}  No hand detected ❌")
            continue

        # Get bounding box around the hand
        h, w, _ = img.shape
        hand_landmarks = results.multi_hand_landmarks[0]
        xs = [int(p.x * w) for p in hand_landmarks.landmark]
        ys = [int(p.y * h) for p in hand_landmarks.landmark]
        x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
        y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)

        # Crop to the detected hand
        img = img[y_min:y_max, x_min:x_max]

        img = Image.fromarray(img)

        true_label = img_path.stem.lower()  # filename without extension
        if("_" in true_label):
            true_label = true_label.split("_")[1]

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
        pred_label = classes[pred_idx]

        is_correct = (pred_label.lower() in true_label)
        if(is_correct): 
            correct+= 1

        print(f"{img_path.name:<25}  True: {true_label:<12}  Pred: {pred_label:<12}  {'✅' if is_correct else '❌'}")

accuracy = correct / total if total > 0 else 0
print(f"\nTotal images: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy*100:.2f}%")