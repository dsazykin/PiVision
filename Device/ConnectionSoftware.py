import socket
import threading
import time
import queue
import os
import json
import tkinter as tk
from pathlib import Path
from tkinter.scrolledtext import ScrolledText

import pyautogui

HOST = "0.0.0.0"
PORT = 9000

log_queue = queue.Queue()

def get_config_path():
    """Return the platform-specific path to the PiVision config file."""
    # Windows → AppData\Roaming\PiVision
    # Linux/Mac → ~/.config/PiVision
    if os.name == "nt":
        base_dir = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base_dir = Path.home() / ".config"
    pivision_dir = base_dir / "PiVision"
    pivision_dir.mkdir(parents=True, exist_ok=True)
    return pivision_dir / "config.json"

CONFIG_PATH = get_config_path()

def load_json():
    """Load settings from JSON file or use defaults if missing."""
    defaults = {
        "MOVE_DISTANCE": 20,
        "MOVE_INTERVAL": 0.03,
        "SCROLL_AMOUNT": 100
    }

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
                defaults.update(data)
        except Exception as e:
            print(f"Warning: Failed to load config, using defaults: {e}")

    return defaults

def save_json(settings):
    """Save settings to JSON file."""
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"Settings saved to {CONFIG_PATH}")
    except Exception as e:
        print(f"Error saving config: {e}")

def log_event(message, status=None):
    """Log a message to the console and queue it for the GUI."""
    print(message)
    log_queue.put((message, status))

# Keep track of held states
active_mouse_holds = {}
active_key_holds = {}

# Adjustable sensitivity
MOVE_DISTANCE = 20     # pixels per step
MOVE_INTERVAL = 0.03   # seconds between steps
SCROLL_AMOUNT = 100    # scroll step size

def continuous_mouse_move(direction):
    """Move the mouse continuously in a given direction while held."""
    while active_mouse_holds.get(direction, False):
        if direction == "mouse_left":
            pyautogui.moveRel(-MOVE_DISTANCE, 0)
        elif direction == "mouse_right":
            pyautogui.moveRel(MOVE_DISTANCE, 0)
        elif direction == "mouse_up":
            pyautogui.moveRel(0, -MOVE_DISTANCE)
        elif direction == "mouse_down":
            pyautogui.moveRel(0, MOVE_DISTANCE)
        time.sleep(MOVE_INTERVAL)

def continuous_scroll(direction):
    """Scroll continuously while held."""
    while active_mouse_holds.get(direction, False):
        if direction == "scroll_up":
            pyautogui.scroll(SCROLL_AMOUNT)
        elif direction == "scroll_down":
            pyautogui.scroll(-SCROLL_AMOUNT)
        time.sleep(MOVE_INTERVAL)

def continuous_mouse_move_once(direction):
    """One-time mouse move for press actions."""
    if direction == "mouse_left":
        pyautogui.moveRel(-MOVE_DISTANCE, 0)
    elif direction == "mouse_right":
        pyautogui.moveRel(MOVE_DISTANCE, 0)
    elif direction == "mouse_up":
        pyautogui.moveRel(0, -MOVE_DISTANCE)
    elif direction == "mouse_down":
        pyautogui.moveRel(0, MOVE_DISTANCE)

def perform_action(msg):
    parts = msg.strip().split(" ", 1)
    if len(parts) != 2:
        log_event(f"Ignoring invalid command: {msg}")
        return

    command = parts[0].lower()
    key = parts[1].lower()

    # === Handle mouse inputs ===
    if key in [
        "left_click", "right_click",
        "mouse_left", "mouse_right",
        "mouse_up", "mouse_down",
        "scroll_up", "scroll_down"
    ]:
        if key in ["mouse_left", "mouse_right", "mouse_up", "mouse_down"]:
            # Continuous movement
            if command == "hold":
                if not active_mouse_holds.get(key, False):
                    active_mouse_holds[key] = True
                    threading.Thread(target=continuous_mouse_move, args=(key,), daemon=True).start()
                    log_event(f"Started continuous {key}")
            elif command == "release":
                active_mouse_holds[key] = False
                log_event(f"Stopped continuous {key}")
            elif command == "press":
                continuous_mouse_move_once(key)
            return

        if key in ["scroll_up", "scroll_down"]:
            # Continuous scrolling
            if command == "hold":
                if not active_mouse_holds.get(key, False):
                    active_mouse_holds[key] = True
                    threading.Thread(target=continuous_scroll, args=(key,), daemon=True).start()
                    log_event(f"Started continuous {key}")
            elif command == "release":
                active_mouse_holds[key] = False
                log_event(f"Stopped continuous {key}")
            elif command == "press":
                pyautogui.scroll(SCROLL_AMOUNT if key == "scroll_up" else -SCROLL_AMOUNT)
            return

        if key in ["left_click", "right_click"]:
            button = key.replace("_click", "")
            if command == "press":
                pyautogui.click(button=button)
            elif command == "hold":
                pyautogui.mouseDown(button=button)
                log_event(f"Holding {button} click")
            elif command == "release":
                pyautogui.mouseUp(button=button)
                log_event(f"Released {button} click")
            return

    # === Handle keyboard inputs ===
    keys = [k.strip() for k in key.split('+') if k.strip()]

    try:
        if command == "press":
            if len(keys) > 1:
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(keys[0])

        elif command == "hold":
            for k in keys:
                if not active_key_holds.get(k, False):
                    pyautogui.keyDown(k)
                    active_key_holds[k] = True
            log_event(f"Holding {'+'.join(keys)}")

        elif command == "release":
            for k in reversed(keys):
                if active_key_holds.get(k, False):
                    pyautogui.keyUp(k)
                    active_key_holds[k] = False
            log_event(f"Released {'+'.join(keys)}")

        else:
            log_event(f"Unknown command: {command}")

    except Exception as e:
        log_event(f"Error performing action '{msg}': {e}")

def reset_active_holds():
    """Release any held keys or mouse actions when the connection ends."""
    for key in list(active_mouse_holds.keys()):
        active_mouse_holds[key] = False

    for key, held in list(active_key_holds.items()):
        if held:
            try:
                pyautogui.keyUp(key)
            except Exception as exc:
                log_event(f"Error releasing key '{key}': {exc}")
        active_key_holds[key] = False

def handle_client(conn, addr):
    log_event(f"Connected by {addr}", status=f"Connected to {addr}")
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break

            msg = data.decode(errors="replace").strip()
            if not msg:
                continue

            log_event(f"Received: {msg}")
            try:
                perform_action(msg)
            except Exception as e:
                log_event(f"Error performing key action '{msg}': {e}")

    reset_active_holds()
    log_event("Connection closed", status="Waiting for connection...")

def run_server():
    log_event("Starting server thread")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            log_event(f"Listening on {HOST}:{PORT}", status=f"Listening on {HOST}:{PORT}")

            while True:
                try:
                    conn, addr = s.accept()
                except OSError as exc:
                    log_event(f"Socket error while accepting connection: {exc}")
                    continue

                handle_client(conn, addr)
    except Exception as exc:
        log_event(f"Server error: {exc}", status="Server stopped")

def start_gui():
    root = tk.Tk()
    root.title("Laptop Server Monitor")

    status_var = tk.StringVar(value="Starting server...")

    # === STATUS BAR ===
    status_label = tk.Label(root, textvariable=status_var, font=("Segoe UI", 12, "bold"))
    status_label.pack(padx=10, pady=(10, 5), anchor="w")

    # === SETTINGS BUTTON ===
    def open_settings():
        settings_window = tk.Toplevel(root)
        settings_window.title("Settings")

        tk.Label(settings_window, text="Mouse move distance (pixels):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        move_dist_var = tk.IntVar(value=MOVE_DISTANCE)
        tk.Entry(settings_window, textvariable=move_dist_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(settings_window, text="Move interval (seconds):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        move_interval_var = tk.DoubleVar(value=MOVE_INTERVAL)
        tk.Entry(settings_window, textvariable=move_interval_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(settings_window, text="Scroll amount (pixels per step):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        scroll_amount_var = tk.IntVar(value=SCROLL_AMOUNT)
        tk.Entry(settings_window, textvariable=scroll_amount_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        def save_settings():
            global MOVE_DISTANCE, MOVE_INTERVAL, SCROLL_AMOUNT
            MOVE_DISTANCE = move_dist_var.get()
            MOVE_INTERVAL = move_interval_var.get()
            SCROLL_AMOUNT = scroll_amount_var.get()

            # Save persistently
            save_json({
                "MOVE_DISTANCE": MOVE_DISTANCE,
                "MOVE_INTERVAL": MOVE_INTERVAL,
                "SCROLL_AMOUNT": SCROLL_AMOUNT
            })

            log_event(
                f"Settings updated and saved: MOVE_DISTANCE={MOVE_DISTANCE}, MOVE_INTERVAL={MOVE_INTERVAL}, SCROLL_AMOUNT={SCROLL_AMOUNT}")
            settings_window.destroy()

        tk.Button(settings_window, text="Save", command=save_settings).grid(row=3, column=0, columnspan=2, pady=10)

    tk.Button(root, text="⚙ Settings", command=open_settings).pack(padx=10, pady=(0, 10), anchor="e")

    # === LOG WINDOW ===
    log_frame = tk.Frame(root)
    log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    log_text = ScrolledText(log_frame, wrap=tk.WORD, height=20, state=tk.DISABLED)
    log_text.pack(fill=tk.BOTH, expand=True)

    # === QUEUE PROCESSING ===
    def process_queue():
        while True:
            try:
                message, status = log_queue.get_nowait()
            except queue.Empty:
                break

            log_text.configure(state=tk.NORMAL)
            log_text.insert(tk.END, message + "\n")
            log_text.configure(state=tk.DISABLED)
            log_text.see(tk.END)

            if status:
                status_var.set(status)

        root.after(100, process_queue)

    def on_close():
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, process_queue)
    root.mainloop()


if __name__ == "__main__":
    user_settings = load_json()

    MOVE_DISTANCE = user_settings["MOVE_DISTANCE"]
    MOVE_INTERVAL = user_settings["MOVE_INTERVAL"]
    SCROLL_AMOUNT = user_settings["SCROLL_AMOUNT"]

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    start_gui()