import socket
import pyautogui

HOST = "0.0.0.0"
PORT = 9000

def perform_key_action(key_input: str):
    # Normalize input (e.g., remove spaces and lowercase)
    key_input = key_input.strip().lower()

    # Handle combinations like "ctrl+w" or "alt+tab"
    if '+' in key_input:
        keys = [k.strip() for k in key_input.split('+')]
        pyautogui.hotkey(*keys)
    else:
        # Handle single keys like "space", "enter", "a", etc.
        pyautogui.press(key_input)

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
                perform_key_action(msg)
            except Exception as e:
                print(f"Error performing key action '{msg}': {e}")