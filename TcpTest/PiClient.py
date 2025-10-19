import socket, time

possible_ips = [
    "192.168.5.2",     # Ethernet static IP
    "192.168.178.16"   # Daniel Home Local IP
]
PORT = 9000

connected = False

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

while(not connected):
    for ip in possible_ips:
        try:
            print(f"Trying {ip}...")
            s.connect((ip, PORT))
            print(f"Connected to {ip}")
            connected = True
            break
        except:
            continue

def send_gesture(gesture):
    s.sendall(gesture.encode() + b"\n")

# Example loop
while True:
    # Imagine this is your gesture recognizer output
    send_gesture("SWIPE_LEFT")
    time.sleep(100)