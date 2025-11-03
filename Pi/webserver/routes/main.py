"""Main navigation and informational routes."""
from __future__ import annotations

import html as h

from flask import Blueprint, Response, redirect, request, url_for

import Database

from ..middleware import SessionManager, get_request_session


def create_blueprint(session_manager: SessionManager) -> Blueprint:
    bp = Blueprint("main", __name__)
    require_login = session_manager.require_login

    @bp.route("/")
    def index() -> str:
        return """
            <!doctype html>
            <html>
            <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width,initial-scale=1">
            <title>Welcome - Gesture Control</title>
            <link rel="stylesheet" href="/static/css/style.css">
            </head>
            <body>
            <div class="container">
                <div class="header">
                <div class="brand">
                    <div class="brand-mark">GC</div>
                    <h1>Gesture Control</h1>
                </div>
                </div>

                <div class="homepage_content_div center" style="max-width:600px;margin:0 auto;">
                <h2>Welcome to the Gesture Control Web App</h2>
                <p class="lead">Manage users and gesture recognition from this interface.</p>
                <div style="margin-top:18px;">
                    <a href="/login" class="btn">Login</a>
                    <a href="/signup" class="btn ghost" style="margin-left:8px;">Sign Up</a>
                </div>
                </div>
            </div>
            </body>
            </html>
            """


    @bp.route("/video")
    @require_login
    def video():
        session = getattr(request, "session", None)
        username = None
        if session is not None:
            try:
                username = session["user_name"]
            except (TypeError, KeyError):
                username = session.get("user_name") if hasattr(session, "get") else None

        if username:
            target_url = f"/main/{username}"
            button_label = "Back to home page"
        else:
            target_url = "/login"
            button_label = "Back to login page"

        return f"""<html><body style=\"text-align:center\">
                  <h2>Processed Gesture Feed</h2>
                  <img src=\"/stream\" width=\"640\" height=\"480\">
                  <br><a href=\"{target_url}\"><button>{button_label}</button></a>
                  </body></html>"""

    @bp.route("/main/<username>")
    @require_login
    def main_page(username: str) -> Response | str:
        session = get_request_session(request)
        if not session:
            return redirect(url_for("auth.login"))
        session_user = session["user_name"]
        if session_user != username:
            return redirect(url_for("main.main_page", username=session_user))

        download_url = url_for("downloads.download_page")
        safe_name = h.escape(session["user_name"])

        return f"""<!doctype html>
            <html>
            <head>
            <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
            <title>Pi Vision Gestures</title>
            <link rel="stylesheet" href="/static/css/style.css">
            <script>
            const eventSource = new EventSource("/gesture");
            eventSource.onmessage = function(event) {{
                const data = JSON.parse(event.data);
                document.getElementById('g').innerText = data.gesture;
                document.getElementById('c').innerText = (data.confidence*100).toFixed(1)+'%';
            }};
            </script>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <div class="brand">
                            <div class="brand-mark">
                                <img src="{{ url_for('static', filename='images/PiVision_full_logo.png') }}" alt="Logo">
                            </div>
                            <h1>PiVision</h1>
                        </div>
                        <div style="align-self:center;">
                            <a href="/logout" class="btn ghost">Log Out</a>
                        </div>
                    </div>
                    <div class="homepage_container_div">
                        <div class="homepage_content_div center">
                            <div style="max-width: 420px;">
                                <h2>Welcome, {safe_name}</h2>
                                <p class="lead">Choose an action</p>

                                <div style="margin-top:12px; display:flex; flex-direction:column; gap:10px; align-items:center;">
                                    <a href="/mappings/{safe_name}" class="btn">Edit Gesture Mappings</a>
                                    <a href="{download_url}" class="btn green">Download Connection Software</a>
                                    <a href="/delete/{safe_name}" class="btn danger">Delete My Account</a>
                                    <a href="/logout" class="btn ghost">Log Out</a>
                                </div>
                            </div>
                        </div>
                        <div class="homepage_content_div gesture-display">
                            <h1 id="g">Loading...</h1>
                            <p>Confidence: <strong id="c">--%</strong></p>
                            <a href="/video" class="btn ghost">View Live Stream ▶</a>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """


        #return f"""
        """
        <html><head><title>Pi Vision Gestures</title>
        <script>
        const eventSource = new EventSource("/gesture");
        eventSource.onmessage = function(event) {{
            const data = JSON.parse(event.data);
            document.getElementById('g').innerText = data.gesture;
            document.getElementById('c').innerText = (data.confidence*100).toFixed(1)+'%';
        }};
        </script></head>
        <div class='homepage_container_div'>
            <div class='homepage_content_div'>
                <body style="text-align:center;font-family:sans-serif;margin-top:40px">
                <h1>Welcome, {safe_name}</h1>
                <p>Choose an action:</p>
                <a href="/mappings/{safe_name}"><button>Edit Gesture Mappings</button></a><br><br>
                <a href="{download_url}"><button style='background:green;'>Download Connection Software</button></a><br><br>
                <a href="/delete/{safe_name}"><button style='background:red;'>Delete My Account</button></a><br><br>
                <a href="/logout"><button>Log Out</button></a>
            </div>
            <div class='homepage_content_div'>
                <h1 id="g">Loading...</h1>
                <p>Confidence: <span id="c">--%</span></p>
                <a href="/video">View Live Stream ▶</a>
            </div>
        </div>
        </body></html>
        """

    @bp.route("/database")
    def show_database() -> str:
        databaseinfo = []
        for user_name  in Database.get_all_users():
            mappings = Database.get_user_mappings(user_name)
            password = Database.get_user_password(user_name)
            if isinstance(password, bytes):
                password = password.decode("utf-8")
            databaseinfo.append(
                {
                    "user_name": user_name,
                    "mappings": mappings,
                    "password": password,
                }
            )

        # after you build databaseinfo[]
        html = """<!doctype html>
        <html><head>
        <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Database - Users</title>
        <link rel="stylesheet" href="/static/css/style.css">
        </head><body>
        <div class="container">
            <div class="header">
            <div class="brand">
                <div class="brand-mark">DB</div>
                <h1>Database Overview</h1>
            </div>
            </div>

            <div class="db_container_div">
        """
        for user in databaseinfo:
            safename = h.escape(user['user_name'])
            html += f"""
            <div class="db_entry_div">
                <h2>User: {safename}</h2>
                <ul>
            """
            for gesture, action in user["mappings"].items():
                html += f"<li>{h.escape(str(gesture))}: {h.escape(str(action))}</li>"
            html += f"""</ul>
                <p><strong>Hashed Password:</strong> {h.escape(user['password'])}</p>
            </div>
            """
        html += """
            </div>
            <div style="margin-top:18px;">
            <a href="/" class="btn ghost">Back Home</a>
            </div>
        </div>
        </body></html>
        """
        return html


    @bp.route("/sessions", methods=["GET"])
    def show_sessions() -> str:
        sessions = Database.get_all_sessions()
        if not sessions:
            return "<h2>No active sessions found.</h2>"

        rows = "".join(
            f"<tr><td>{s['session_id']}</td><td>{h.escape(s['user_name'])}</td>"
            f"<td class='token-cell'>{h.escape(s['session_token'])}</td>"
            f"<td>{h.escape(str(s['created_at']))}</td><td>{h.escape(str(s['expires_at']))}</td></tr>"
            for s in sessions
        )

        return f"""<!doctype html>
        <html><head>
        <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Active Sessions</title>
        <link rel="stylesheet" href="/static/css/style.css">
        </head><body>
        <div class="container">
            <div class="header">
            <div class="brand">
                <div class="brand-mark">S</div>
                <h1>Active Sessions</h1>
            </div>
            </div>

            <div class="table-wrap">
            <table class="sessions-table">
                <thead>
                <tr><th>ID</th><th>User</th><th>Session Token</th><th>Created At</th><th>Expires At</th></tr>
                </thead>
                <tbody>
                {rows}
                </tbody>
            </table>
            </div>

            <div style="margin-top:18px;">
            <a href="/" class="btn">Back Home</a>
            </div>
        </div>
        </body></html>
        """


    @bp.route("/delete/<username>")
    @require_login
    def delete_user(username: str) -> str:
        session = get_request_session(request)
        if not session:
            return redirect(url_for("auth.login"))
        session_user = session["user_name"]
        if session_user != username:
            return redirect(url_for("main.main_page", username=session_user))

        deleted = Database.delete_user(username)
        safe = h.escape(username)
        if deleted == 0:
            return f"""<!doctype html><html><head>
                <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
                <link rel="stylesheet" href="/static/css/style.css">
                <title>Deletion Failed</title></head><body>
                <div class="container">
                <div class="homepage_content_div center" style="max-width:640px;margin:20px auto;">
                    <h2>Deletion Failed</h2>
                    <p style="color:var(--danger)">User '{h.escape(username)}' not found.</p>
                    <a href="/" class="btn ghost">Return Home</a>
                </div>
                </div>
                </body></html>"""

        return f"""<!doctype html><html><head>
            <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
            <link rel="stylesheet" href="/static/css/style.css">
            <title>Account Deleted</title></head><body>
            <div class="container">
            <div class="homepage_content_div center" style="max-width:640px;margin:20px auto;">
                <h2>Account Deleted</h2>
                <p>User '{h.escape(username)}' has been removed.</p>
                <a href="/" class="btn">Return Home</a>
            </div>
            </div>
            </body></html>"""


    return bp
