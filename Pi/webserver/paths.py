"""Shared filesystem paths for the Pi webserver package."""
from __future__ import annotations

import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PI_DIR = os.path.abspath(os.path.join(_PACKAGE_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PI_DIR, ".."))

STATIC_DIR = os.path.join(PI_DIR, "static")
STREAM_DIR = os.path.join(PROJECT_ROOT, "WebServerStream")
FRAME_PATH = os.path.join(STREAM_DIR, "latest.jpg")
JSON_PATH = os.path.join(STREAM_DIR, "latest.json")
CONNECTION_SOFTWARE_PATH = os.path.join(PROJECT_ROOT, "Device", "PiVision Connection Software.exe")

os.makedirs(STREAM_DIR, exist_ok=True)
