"""Middleware and authentication helpers for the Pi webserver."""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional

from flask import Flask, Request, g, redirect, request, url_for

import Database

ViewFunc = Callable[..., Any]


class SessionManager:
    """Handle session lookup and enforce login requirements."""

    def __init__(self) -> None:
        self._app: Optional[Flask] = None

    def init_app(self, app: Flask) -> None:
        self._app = app

        @app.before_request
        def _load_session() -> None:
            token = request.cookies.get("session_token")
            session = Database.get_session(token) if token else None
            g.active_session = session

    @staticmethod
    def get_active_session() -> Optional[dict[str, Any]]:
        return getattr(g, "active_session", None)

    def require_login(self, view: ViewFunc) -> ViewFunc:
        @wraps(view)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.get_active_session() is None:
                return redirect(url_for("auth.login"))
            return view(*args, **kwargs)

        return wrapped


def get_request_session(req: Request) -> Optional[dict[str, Any]]:
    """Convenience helper mirroring the old request.session attribute."""
    session = SessionManager.get_active_session()
    if session is not None:
        return session
    return getattr(req, "session", None)
