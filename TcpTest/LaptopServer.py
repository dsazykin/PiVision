import socket
import pyautogui

HOST = "0.0.0.0"
PORT = 9000

def perform_action(mapped_key):
    if mapped_key == "scroll_up":
        pyautogui.scroll(300)
    elif mapped_key == "scroll_down":
        pyautogui.scroll(-300)
    elif mapped_key == "mouse_up":
        pyautogui.moveRel(0, -50)
    elif mapped_key == "mouse_down":
        pyautogui.moveRel(0, 50)
    elif mapped_key == "mouse_left":
        pyautogui.moveRel(-50, 0)
    elif mapped_key == "mouse_right":
        pyautogui.moveRel(50, 0)
    elif mapped_key == "left_click":
        pyautogui.click(button='left')
    elif mapped_key == "right_click":
        pyautogui.click(button='right')
    elif mapped_key == "mouse_left":
        pyautogui.click(button='left')
    elif mapped_key == "volume_toggle":
        pyautogui.press("volumemute")
    else:
        # normal key press
        pyautogui.press(mapped_key)

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