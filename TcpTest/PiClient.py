import socket, time

LAPTOP_IP = "192.168.5.2"
PORT = 9000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((LAPTOP_IP, PORT))

def send_gesture(gesture):
    s.sendall(gesture.encode() + b"\n")

# Example loop
while True:
    # Imagine this is your gesture recognizer output
    send_gesture("SWIPE_LEFT")
    time.sleep(100)