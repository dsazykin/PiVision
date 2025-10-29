import json, os

# Shared location where both the Flask app and DeviceControl can access
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
temp_dir = os.path.join(project_root, "WebServerStream")
os.makedirs(temp_dir, exist_ok=True)

MAPPINGS_PATH = os.path.join(temp_dir, "mappings.json")

def update_gestures(mappings: dict):
    """Write updated mappings to a JSON file for DeviceControl to detect."""
    try:
        os.makedirs(os.path.dirname(MAPPINGS_PATH), exist_ok=True)
        with open(MAPPINGS_PATH, "w") as f:
            json.dump(mappings, f)
        print("Wrote updated mappings to shared file.")
    except Exception as e:
        print("Failed to write mappings:", e)