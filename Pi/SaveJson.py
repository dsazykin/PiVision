import json, os

from .webserver.paths import MAPPINGS_PATH, PASSWORD_PATH, TEMP_DIR, JSON_PATH, LOGGEDIN_PATH

os.makedirs(TEMP_DIR, exist_ok=True)

def update_gestures(mappings: dict):
    """Write updated mappings to a JSON file for DeviceControl to detect."""
    try:
        os.makedirs(os.path.dirname(MAPPINGS_PATH), exist_ok=True)
        with open(MAPPINGS_PATH, "w") as f:
            json.dump(mappings, f)
        print("Wrote updated mappings to shared file.")
    except Exception as e:
        print("Failed to write mappings:", e)

def entering_password(value: bool):
    """Mark that the user is entering their password, so gesture recognition should be enabled."""
    try:
        os.makedirs(os.path.dirname(PASSWORD_PATH), exist_ok=True)
        with open(PASSWORD_PATH, "w") as f:
            json.dump({"value": value}, f)
        print("Password can now be entered.")
    except Exception as e:
        print("Failed to mark entering password:", e)

def update_current_gesture(data: dict):
    """Save the current detected gesture for use in the webserver."""
    try:
        with open(JSON_PATH, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print("Error saving current gesture: ", e)

def update_loggedin(value: bool):
    """Set whether the user is logged in."""
    try:
        with open(LOGGEDIN_PATH, 'w') as f:
            json.dump({"loggedIn": value}, f)
    except Exception as e:
        print("Error updating logged in value:", e)