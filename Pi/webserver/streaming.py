"""Streaming helpers for the Pi webserver."""
from __future__ import annotations

import json
import time
from typing import Iterator

import cv2
from flask import Response

from .paths import FRAME_PATH, JSON_PATH


def read_gesture_payload() -> dict[str, object]:
    try:
        with open(JSON_PATH) as handle:
            return json.load(handle)
    except Exception:
        return {"gesture": "None", "confidence": 0.0}


def stream_frames() -> Iterator[bytes]:
    previous_frame = None
    while True:
        try:
            current_frame = cv2.imread(FRAME_PATH)
            if current_frame is None or (
                    previous_frame is not None and (current_frame == previous_frame).all()):
                time.sleep(0.1)
                continue
            success, jpeg = cv2.imencode(".jpg", current_frame)
            previous_frame = current_frame
            if not success:
                time.sleep(3)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        except Exception:
            time.sleep(3)


def stream_gestures() -> Iterator[str]:
    last_payload = None
    while True:
        payload = read_gesture_payload()
        if payload != last_payload:
            yield f"data: {json.dumps(payload)}\n\n"
            last_payload = payload
        time.sleep(0.1)
