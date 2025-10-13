import tkinter as tk
from tkinter import messagebox
import cv2
import onnxruntime as ort
import numpy as np
import mediapipe as mp
import pyautogui

# ====== Default gesture mappings ======
gesture_mappings = {
    "Swipe Left": "a",
    "Swipe Right": "d",
    "Fist": "Enter",
    "Open Palm": "Space",
    "Ok": "q",
}

loggedIn = False

# ====== Login Screen ======
def show_login_screen():
    login_root = tk.Tk()
    login_root.title("Login")
    login_root.geometry("400x400")

    tk.Label(login_root, text="Username:").pack(pady=(10, 0))
    entry_username = tk.Entry(login_root)
    entry_username.pack()

    tk.Label(login_root, text="Password:").pack(pady=(10, 0))
    entry_password = tk.Entry(login_root, show="*")
    entry_password.pack()

    def login():
        username = entry_username.get()
        password = entry_password.get()
        if username == "admin" and password == "1234":
            loggedIn = True
            login_root.destroy()
            show_welcome_screen(username)
        else:
            messagebox.showerror("Login Failed", "Invalid credentials!")

    tk.Button(login_root, text="Login", command=login).pack(pady=10)
    login_root.mainloop()

# ====== Welcome Screen ======
def show_welcome_screen(username):
    welcome_root = tk.Tk()
    welcome_root.title("Welcome")
    welcome_root.geometry("400x400")

    tk.Label(welcome_root, text=f"Welcome, {username}!", font=("Arial", 16)).pack(pady=15)

    tk.Button(welcome_root, text="Start Gesture Control", command=lambda: start_gesture_detection(welcome_root)).pack(pady=10)

    keystroke_label = tk.Label(welcome_root, text="Press any key...", font=("Arial", 12))
    keystroke_label.pack()

    def on_key_press(event):
        keystroke_label.config(text=f"You pressed: {repr(event.char)} (KeyCode: {event.keycode})")

    welcome_root.bind("<Key>", on_key_press)

    def logout():
        welcome_root.destroy()
        show_login_screen()

    def open_mapping_screen():
        welcome_root.destroy()
        show_mapping_screen(username)

    tk.Button(welcome_root, text="Change Mappings", command=open_mapping_screen).pack(pady=10)
    tk.Button(welcome_root, text="Log Out", command=logout).pack(pady=5)

    welcome_root.mainloop()

# ====== Mapping Screen ======
def show_mapping_screen(username):
    map_root = tk.Tk()
    map_root.title("Gesture Mapping")
    map_root.geometry("400x400")

    tk.Label(map_root, text="Gesture to Key Mappings", font=("Arial", 14)).pack(pady=10)

    mapping_frames = {}

    # Temporary state to track which gesture is waiting for input
    waiting_for_input = {"gesture": None}

    # Callback to bind key to gesture
    def on_key_press(event):
        gesture = waiting_for_input["gesture"]
        if gesture:
            new_key = event.keysym
            gesture_mappings[gesture] = new_key
            label = mapping_frames[gesture]["label"]
            label.config(text=f"Key: {new_key}")
            waiting_for_input["gesture"] = None
            map_root.unbind("<Key>")
            messagebox.showinfo("Updated", f"'{gesture}' is now mapped to '{new_key}'.")

    # Generate UI rows
    for gesture, key in gesture_mappings.items():
        frame = tk.Frame(map_root)
        frame.pack(pady=5)

        tk.Label(frame, text=gesture + ":", width=15, anchor="w").pack(side="left")

        key_label = tk.Label(frame, text=f"Key: {key}", width=10)
        key_label.pack(side="left", padx=5)

        def make_change_func(gesture_name):
            def change():
                waiting_for_input["gesture"] = gesture_name
                messagebox.showinfo("Key Bind", f"Press a key to bind to '{gesture_name}'")
                map_root.bind("<Key>", on_key_press)
            return change

        change_button = tk.Button(frame, text="Change", command=make_change_func(gesture))
        change_button.pack(side="left")

        mapping_frames[gesture] = {"label": key_label, "button": change_button}

    def save_and_return():
        map_root.destroy()
        show_welcome_screen(username)


    tk.Button(map_root, text="Save & Return", command=save_and_return).pack(pady=20)

    map_root.mainloop()


def start_gesture_detection(root):
    root.destroy()  # close the GUI before opening camera

    # Load ONNX model
    session = ort.InferenceSession("models/gesture_model_v3.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Class list (same as in test.py)
    classes = [
        'train_val_call', 'train_val_dislike', 'train_val_fist', 'train_val_four',
        'train_val_like', 'train_val_mute', 'train_val_ok', 'train_val_one',
        'train_val_palm', 'train_val_peace', 'train_val_peace_inverted',
        'train_val_rock', 'train_val_stop', 'train_val_stop_inverted',
        'train_val_three', 'train_val_three2', 'train_val_two_up',
        'train_val_two_up_inverted'
    ]

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands.Hands(max_num_hands=1,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    print("[INFO] Starting gesture detection. Press 'q' to quit.")
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
                x1, y1 = max(min(xs)-margin, 0), max(min(ys)-margin, 0)
                x2, y2 = min(max(xs)+margin, w), min(max(ys)+margin, h)
                hand_img = frame[y1:y2, x1:x2]

                if hand_img.size == 0:
                    continue

                # Preprocess
                img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
                img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                img = np.transpose(img, (2, 0, 1))[np.newaxis, :].astype(np.float32)

                outputs = session.run(None, {input_name: img})
                pred = np.argmax(outputs[0])
                label = classes[pred]

                # Draw results
                mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # === Key mapping logic ===
                for gesture, key in gesture_mappings.items():
                    if gesture.lower().replace(" ", "_") in label.lower():
                        print(f"[ACTION] {gesture} → press '{key}'")
                        pyautogui.press(key)

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ====== Start the app ======
show_login_screen()