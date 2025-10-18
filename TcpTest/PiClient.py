import socket, threading, time

UDP_PORT = 50000
TCP_PORT_FALLBACKS = [( "192.168.5.2", 9000)]  # static Ethernet guesses
connected = False

def listen_for_broadcast():
    global connected, server_addr
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp.bind(("", UDP_PORT))
    print("[UDP] Listening for broadcasts...")
    while not connected:
        data, addr = udp.recvfrom(1024)
        msg = data.decode().strip()
        ip, port = msg.split(":")
        print(f"[UDP] Discovered server at {ip}:{port}")
        try_connect(ip, int(port))

def try_connect(ip, port):
    global connected
    if connected:
        return
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect((ip, port))
        connected = True
        print(f"[TCP] Connected to {ip}:{port}")
        # example send loop
        while True:
            s.sendall(b"SWIPE_LEFT\n")
            time.sleep(5)
    except Exception as e:
        s.close()

def try_fallbacks():
    for ip, port in TCP_PORT_FALLBACKS:
        if connected:
            break
        print(f"[Fallback] Trying {ip}:{port}")
        try_connect(ip, port)
        time.sleep(2)

if __name__ == "__main__":
    threading.Thread(target=listen_for_broadcast, daemon=True).start()
    try_fallbacks()
    # keep alive
    while not connected:
        time.sleep(1)