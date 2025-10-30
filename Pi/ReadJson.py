import json

from .webserver.paths import MAPPINGS_PATH, PASSWORD_PATH, TEMP_DIR, JSON_PATH, LOGGEDIN_PATH, PASSWORD_GESTURE_PATH

def check_loggedin():
    """Check to see if the user is currently logged in."""
    try:
        with open(LOGGEDIN_PATH) as handle:
            jsonValue = json.load(handle)
    except Exception:
        jsonValue = {"loggedIn": False}
    return jsonValue.get("loggedIn")

def check_entering_password():
    """Check if the user is attempting to enter their gesture password."""
    try:
        with open(PASSWORD_PATH) as handle:
            jsonValue = json.load(handle)
    except Exception:
        jsonValue = {"value": False}
    return jsonValue.get("value")

def get_password_gesture():
    """Get the detected gesture for the password input."""
    try:
        with open(PASSWORD_GESTURE_PATH) as handle:
            jsonValue = json.load(handle)
    except Exception:
        jsonValue = {"gesture": "none"}

    return jsonValue.get("gesture")

def get_new_mappings():
    try:
        with open(MAPPINGS_PATH, "r") as f:
            new_map = json.load(f)
            print("🔄 Mappings reloaded (file changed).")
        return new_map
    except Exception as e:
        print("⚠️ Failed to reload mappings:", e)