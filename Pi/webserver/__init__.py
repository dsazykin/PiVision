"""Application factory for the Pi webserver."""
from __future__ import annotations

from flask import Flask, Response, url_for

from .middleware import SessionManager
from Pi.webserver.config.paths import STATIC_DIR
from .routes import register_blueprints
from .. import Database


def create_app() -> Flask:
    Database.initialize_database()

    app = Flask(__name__, static_folder=STATIC_DIR)

    session_manager = SessionManager()
    session_manager.init_app(app)

    register_blueprints(app, session_manager)

    @app.after_request
    def inject_css(response: Response) -> Response:
        if response.content_type == "text/html; charset=utf-8":
            css_link = (
                f'<link rel="stylesheet" type="text/css" '
                f'href="{url_for("static", filename="style.css")}">'
            )
            html = response.get_data(as_text=True)
            if "<head>" in html:
                html = html.replace("<head>", f"<head>{css_link}")
            else:
                html = css_link + html
            response.set_data(html)
        return response

    return app
