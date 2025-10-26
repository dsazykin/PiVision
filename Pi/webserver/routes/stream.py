"""Streaming and status routes."""
from __future__ import annotations

from flask import Blueprint, Response

from ..middleware import SessionManager
from ..streaming import stream_gestures, stream_frames


def create_blueprint(_: SessionManager) -> Blueprint:
    bp = Blueprint("stream", __name__)

    @bp.route("/gesture")
    def gesture() -> Response:
        return Response(stream_gestures(), mimetype="text/event-stream")

    @bp.route("/stream")
    def stream() -> Response:
        return Response(
            stream_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return bp
