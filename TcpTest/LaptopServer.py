import socket, pyautogui

HOST = "0.0.0.0"
PORT = 9000

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
            # interpret gesture command
            if msg == "SWIPE_LEFT":
                pyautogui.hotkey("alt", "tab")
            elif msg == "FIST":
                pyautogui.hotkey("ctrl", "w")
            elif msg == "OPEN_PALM":
                pyautogui.press("space")