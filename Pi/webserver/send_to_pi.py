import socket, json

PI_HOST = "192.168.137.1"   # same IP Pi connects to
PI_PORT = 9000

def send_mappings_to_pi(mappings: dict):
    """Sends new gesture mappings to the Pi for live update."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((PI_HOST, PI_PORT))

        message = "UPDATE_MAPPINGS " + json.dumps(mappings)
        s.sendall(message.encode())
        s.close()
        print("Sent mapping update to Pi")

    except Exception as e:
        print("Could not send mappings to Pi:", e)