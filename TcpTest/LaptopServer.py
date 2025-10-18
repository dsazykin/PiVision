import socket, threading, time, pyautogui

TCP_PORT = 9000
UDP_PORT = 50000
BROADCAST_INTERVAL = 2  # seconds

def tcp_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", TCP_PORT))
        s.listen(1)
        print(f"[TCP] Listening on port {TCP_PORT}")
        conn, addr = s.accept()
        print(f"[TCP] Connected by {addr}")
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                msg = data.decode().strip()
                print(f"[TCP] Received: {msg}")
                # example actions
                if msg == "SWIPE_LEFT":
                    pyautogui.hotkey("alt", "tab")
                elif msg == "FIST":
                    pyautogui.hotkey("ctrl", "w")
                elif msg == "OPEN_PALM":
                    pyautogui.press("space")

def udp_broadcast():
    """Periodically broadcast our IP and TCP port."""
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    while True:
        msg = f"{ip}:{TCP_PORT}".encode()
        udp.sendto(msg, ("<broadcast>", UDP_PORT))
        time.sleep(BROADCAST_INTERVAL)

if __name__ == "__main__":
    threading.Thread(target=udp_broadcast, daemon=True).start()
    tcp_server()