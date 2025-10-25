"""Streaming helpers for the Pi webserver."""
from __future__ import annotations

import json
import time
from typing import Iterator

import cv2
from flask import Response, jsonify

from .paths import FRAME_PATH, JSON_PATH


def read_gesture_payload() -> dict[str, object]:
    try:
        with open(JSON_PATH) as handle:
            return json.load(handle)
    except Exception:
        return {"gesture": "None", "confidence": 0.0}


def stream_frames() -> Iterator[bytes]:
    while True:
        try:
            frame = cv2.imread(FRAME_PATH)
            if frame is None:
                time.sleep(0.05)
                continue
            success, jpeg = cv2.imencode(".jpg", frame)
            if not success:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        except Exception:
            time.sleep(0.05)


def gesture_response() -> Response:
    return jsonify(read_gesture_payload())
