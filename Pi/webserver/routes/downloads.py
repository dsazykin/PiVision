"""Routes for laptop server downloads."""
from __future__ import annotations

import os

from flask import Blueprint, Response, abort, request, send_file, url_for

from ..middleware import SessionManager, get_request_session
from ..paths import CONNECTION_SOFTWARE_PATH


def create_blueprint(session_manager: SessionManager) -> Blueprint:
    bp = Blueprint("downloads", __name__)
    require_login = session_manager.require_login

    @bp.route("/download-software")
    @require_login
    def download_page():
        exists = os.path.exists(CONNECTION_SOFTWARE_PATH)
        status_message = (
            "PiVision Connection Software is available for download."
            if exists
            else "PiVision Connection Software could not be found on the server."
        )

        download_button = (
            f"<a href='{url_for('downloads.download_file')}'><button>Download "
            f"PiVision Connection Software</button></a>"
            if exists
            else ""
        )

        return f"""
            <h1>Download PiVision Connection Software</h1>
            <p>{status_message}</p>
            {download_button}
            <br><br>
            <a href='
{url_for('main.main_page', username=request.session['user_name'])}'><button>Back to 
Home</button></a>
        """

    @bp.route("/download-software/file")
    @require_login
    def download_file():
        if not os.path.exists(CONNECTION_SOFTWARE_PATH):
            abort(404)
        return send_file(
            CONNECTION_SOFTWARE_PATH,
            as_attachment=True,
            download_name="PiVision Connection Software.exe",
        )

    return bp
