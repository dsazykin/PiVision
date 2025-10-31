"""Shared filesystem paths for the Pi webserver package."""
from __future__ import annotations

import os

WEBSERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PI_DIR = os.path.abspath(os.path.join(WEBSERVER_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PI_DIR, ".."))

STATIC_DIR = os.path.join(PI_DIR, "static")
TEMP_DIR = os.path.join(PROJECT_ROOT, "WebServerStream")

FRAME_PATH = os.path.join(TEMP_DIR, "latest.jpg")
JSON_PATH = os.path.join(TEMP_DIR, "latest.json")
LOGGEDIN_PATH = os.path.join(TEMP_DIR, "loggedIn.json")
PASSWORD_PATH = os.path.join(TEMP_DIR, "password.json")
PASSWORD_GESTURE_PATH = os.path.join(TEMP_DIR, "password_gesture.json")
MAPPINGS_PATH = os.path.join(TEMP_DIR, "mappings.json")

CONNECTION_SOFTWARE_PATH = os.path.join(PROJECT_ROOT, "Device", "PiVision Connection Software.exe")

os.makedirs(TEMP_DIR, exist_ok=True)
