"""Routes for laptop server downloads."""
from __future__ import annotations

import os

from flask import Blueprint, Response, abort, request, send_file, url_for

from ..middleware import SessionManager, get_request_session
from ..paths import LAPTOP_SERVER_PATH


def create_blueprint(session_manager: SessionManager) -> Blueprint:
    bp = Blueprint("downloads", __name__)
    require_login = session_manager.require_login

    @bp.route("/download-laptopserver")
    @require_login
    def download_page() -> str:
        session = get_request_session(request)
        username = session.get("user_name") if session else ""
        exists = os.path.exists(LAPTOP_SERVER_PATH)
        status_message = (
            "LaptopServer.py is available for download."
            if exists
            else "LaptopServer.py could not be found on the server."
        )
        download_button = (
            f"<a href='{url_for('downloads.download_file')}'><button>Download LaptopServer.py</button></a>"
            if exists
            else ""
        )
        return f"""
        <h1>Download Laptop Server Script</h1>
        <p>{status_message}</p>
        {download_button}
        <br><br>
        <a href='{url_for('main.main_page', username=username)}'><button>Back to Home</button></a>
        """

    @bp.route("/download-laptopserver/file")
    @require_login
    def download_file() -> Response:
        if not os.path.exists(LAPTOP_SERVER_PATH):
            abort(404)
        return send_file(
            LAPTOP_SERVER_PATH,
            as_attachment=True,
            download_name="LaptopServer.py",
        )

    return bp
