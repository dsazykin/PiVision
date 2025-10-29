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
        return (
            "<h1>Welcome to the Gesture Control Web App</h1>"
            "<p>This web interface lets you manage users and gesture recognition.</p>"
            "<a href=\"/login\"><button>Login</button></a>"
            "<a href=\"/signup\"><button>Sign Up</button></a>"
        )

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

        return f"""
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

        html = "<h1>Database Page</h1><div class='db_container_div'>"
        for user in databaseinfo:
            safename = h.escape(user['user_name'][0])
            html += "<div class='db_entry_div'>"
            html += (
                f"<h2>User: {safename} </h2><ul>"
            )
            for gesture, action in user["mappings"].items():
                html += f"<li>{gesture}: {action}</li>"
            html += (
                f"</ul><p><strong>Hashed Password:</strong> {user['password']}</p></div>"
            )
        html += "</div>"
        return html

    @bp.route("/sessions", methods=["GET"])
    def show_sessions() -> str:
        sessions = Database.get_all_sessions()
        if not sessions:
            return "<h2>No active sessions found.</h2>"

        rows = "".join(
            f"<tr><td>{s['session_id']}</td><td>{h.escape(s['user_name'])}</td>"
            f"<td class='token-cell'>{s['session_token']}</td>"
            f"<td>{s['created_at']}</td><td>{s['expires_at']}</td></tr>"
            for s in sessions
        )

        return """
        <h1>Active Sessions</h1>
        <style>
            table {{
                border-collapse: collapse;
                margin-top: 20px;
                width: 90%;
                font-family: Arial, sans-serif;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 10px;
                text-align: center;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .token-cell {{
                max-width: 320px;
                overflow-wrap: anywhere;
                font-family: monospace;
                color: #333;
            }}
            h1 {{
                font-family: Arial, sans-serif;
                text-align: center;
            }}
            body {{
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
        </style>
        <table>
            <tr>
                <th>ID</th>
                <th>User</th>
                <th>Session Token</th>
                <th>Created At</th>
                <th>Expires At</th>
            </tr>
        """ + rows + """
        </table>
        <br><a href='/'><button style='padding:10px 20px; border:none; background-color:#4CAF50; color:white; border-radius:5px; cursor:pointer;'>Back Home</button></a>
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
            return (
                f"<h1>Deletion Failed</h1><p style='color:red;'>User '{safe}' not found.</p>"
            )
        return (
            f"<h1>Account Deleted</h1><p>User '{safe}' has been removed.</p>"
            "<a href='/'>Return Home</a>"
        )

    return bp
