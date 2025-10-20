import socket
import pyautogui
import threading
import time

HOST = "0.0.0.0"
PORT = 9000

# Keep track of which keys or directions are currently held
active_mouse_holds = {}
active_key_holds = {}

# Adjustable sensitivity
MOVE_DISTANCE = 20     # pixels per step
MOVE_INTERVAL = 0.03   # seconds between steps
SCROLL_AMOUNT = 300    # scroll step size

def continuous_mouse_move(direction):
    """Move the mouse continuously in the given direction while held."""
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

def perform_action(msg):
    parts = msg.strip().split(" ", 1)
    if len(parts) != 2:
        print(f"Ignoring invalid command: {msg}")
        return

    command = parts[0].lower()
    key = parts[1].lower()

    # === Handle mouse actions ===
    if key in [
        "left_click", "right_click",
        "mouse_left", "mouse_right",
        "mouse_up", "mouse_down",
        "scroll_up", "scroll_down"
    ]:
        # ---- Continuous movement ----
        if key.startswith("mouse_") and key not in ["mouse_left", "mouse_right", "mouse_up", "mouse_down"]:
            pass  # no movement hold here

        if key in ["mouse_left", "mouse_right", "mouse_up", "mouse_down"]:
            if command == "hold":
                if not active_mouse_holds.get(key, False):
                    active_mouse_holds[key] = True
                    threading.Thread(target=continuous_mouse_move, args=(key,), daemon=True).start()
                    print(f"Started continuous {key}")
            elif command == "release":
                active_mouse_holds[key] = False
                print(f"Stopped continuous {key}")
            elif command == "press":
                # one-time movement
                continuous_mouse_move_once(key)
            return

        # ---- Scroll actions ----
        if key == "scroll_up" and command in ["press", "hold"]:
            pyautogui.scroll(SCROLL_AMOUNT)
        elif key == "scroll_down" and command in ["press", "hold"]:
            pyautogui.scroll(-SCROLL_AMOUNT)
        elif key == "left_click":
            if command == "press":
                pyautogui.click(button="left")
            elif command == "hold":
                pyautogui.mouseDown(button="left")
            elif command == "release":
                pyautogui.mouseUp(button="left")
        elif key == "right_click":
            if command == "press":
                pyautogui.click(button="right")
            elif command == "hold":
                pyautogui.mouseDown(button="right")
            elif command == "release":
                pyautogui.mouseUp(button="right")
        return

    # === Handle keyboard actions ===
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
            print(f"Holding {'+'.join(keys)}")

        elif command == "release":
            for k in reversed(keys):
                if active_key_holds.get(k, False):
                    pyautogui.keyUp(k)
                    active_key_holds[k] = False
            print(f"Released {'+'.join(keys)}")

        else:
            print(f"Unknown command: {command}")

    except Exception as e:
        print(f"Error performing action '{msg}': {e}")

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

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"Listening on {HOST}:{PORT}")
    conn, addr = s.accept()
    with conn:
        print("Connected by", addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            msg = data.decode().strip()
            print("Received:", msg)
            try:
                perform_action(msg)
            except Exception as e:
                print(f"Error performing key action '{msg}': {e}")