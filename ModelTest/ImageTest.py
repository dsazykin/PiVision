import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

# Create inference session
session = ort.InferenceSession("models/gesture_model_v3.onnx", providers=["CPUExecutionProvider"])

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

base_dir = Path("ModelTest/TestImages")
print(base_dir)

total = 0
correct = 0

for ext in ["*.jpg", "*.jpeg", "*.png"]:
    for img_path in base_dir.rglob(ext):
        total += 1

        img = Image.open(img_path).convert("RGB")

        true_label = img_path.stem.lower()  # filename without extension
        # if filename is like call_1.jpg, keep only the part before "_"
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