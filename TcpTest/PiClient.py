import socket, time

possible_ips = [
    "192.168.5.2",     # Ethernet static IP
    "192.168.178.16"   # Daniel Home Local IP
]
PORT = 9000

connected = False

while not connected:
    for ip in possible_ips:
        try:
            print(f"Trying {ip}...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect((ip, PORT))
            print(f"Connected to {ip}")
            connected = True
            break
        except Exception as e:
            print(f"Failed to connect to {ip}: {e}")
            time.sleep(1)
            continue

def send_gesture(gesture):
    s.sendall(gesture.encode() + b"\n")

# Example loop
while True:
    # Imagine this is your gesture recognizer output
    send_gesture("SWIPE_LEFT")
    time.sleep(3)