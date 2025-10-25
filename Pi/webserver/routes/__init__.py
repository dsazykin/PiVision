"""Blueprint registration for the Pi webserver."""
from __future__ import annotations

from flask import Flask

from ..middleware import SessionManager
from . import auth, downloads, main, mappings, stream

BLUEPRINTS = [
    auth.create_blueprint,
    main.create_blueprint,
    mappings.create_blueprint,
    downloads.create_blueprint,
    stream.create_blueprint,
]


def register_blueprints(app: Flask, session_manager: SessionManager) -> None:
    for factory in BLUEPRINTS:
        blueprint = factory(session_manager)
        app.register_blueprint(blueprint)
