"""Routes for gesture mapping management."""
from __future__ import annotations

from flask import Blueprint, Response, redirect, request, url_for

import Database

from ..send_to_pi import send_mappings_to_pi

from ..middleware import SessionManager, get_request_session

def create_blueprint(session_manager: SessionManager) -> Blueprint:
    bp = Blueprint("mappings", __name__)
    require_login = session_manager.require_login

    @bp.route("/mappings/<username>", methods=["GET", "POST"])
    @require_login
    def mappings(username: str) -> Response | str:
        session = get_request_session(request)
        if not session:
            return redirect(url_for("auth.login"))
        session_user = session["user_name"]
        if session_user != username:
            return redirect(url_for("mappings.mappings", username=session_user))

        if request.method == "POST":
            gesture = request.form.get("gesture")
            new_action = request.form.get("action")
            new_duration = request.form.get("duration")
            if gesture and new_action and new_duration:
                Database.update_gesture_mapping(
                    username, gesture, new_action, new_duration
                )

                updated_map = Database.get_user_mappings(username)
                send_mappings_to_pi(updated_map)

            return redirect(url_for("mappings.mappings", username=username))

        mappings_data = Database.get_user_mappings(username)
        rows = "".join(
            f"<tr><form method='POST'>"
            f"<td><strong>{gesture}</strong></td>"
            f"<td><input type='text' id='input_{gesture}' name='action' value='{action}' required readonly>"
            f"<button type='button' onclick=\"startListening('{gesture}')\">Edit</button></td>"
            f"<td><select name='duration'>"
            f"<option value='press' {'selected' if duration == 'press' else ''}>press</option>"
            f"<option value='hold' {'selected' if duration == 'hold' else ''}>hold</option>"
            f"</select></td>"
            f"<td><input type='hidden' name='gesture' value='{gesture}'>"
            f"<input type='submit' value='Update'></td></form></tr>"
            for gesture, (action, duration) in mappings_data.items()
        )

        return f"""
        <h1>Gesture Mappings for {username}</h1>
        <style>
            table {{
                border-collapse: collapse;
                margin-top: 10px;
                width: 85%;
            }}
            th, td {{
                padding: 8px 12px;
                border: 1px solid #ccc;
                text-align: center;
            }}
            input[type='text'], select {{
                padding: 5px;
                border-radius: 4px;
                border: 1px solid #888;
            }}
            input[type='submit'], button {{
                padding: 5px 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            input[type='submit']:hover, button:hover {{
                background-color: #45a049;
            }}
        </style>
        <script>
            let listening = false;
            let currentInput = null;
            let startX = 0, startY = 0;
            let moveTimeout = null;

            function startListening(gestureId) {{
                if (listening) return;
                listening = true;
                currentInput = document.getElementById('input_' + gestureId);
                currentInput.value = 'Listening... (key, click, scroll, or move mouse)';
                currentInput.style.backgroundColor = '#ffeeaa';

                function stopListening(name) {{
                    listening = false;
                    currentInput.value = name;
                    currentInput.style.backgroundColor = '';
                    window.removeEventListener('keydown', onKey, true);
                    window.removeEventListener('mousedown', onClick, true);
                    window.removeEventListener('mousemove', onMove, true);
                    window.removeEventListener('wheel', onScroll, true);
                    if (moveTimeout) clearTimeout(moveTimeout);
                }}

                function onKey(e) {{
                    e.preventDefault();
                    let key = e.key.toLowerCase();
                    if (key === ' ') key = 'space';
                    if (key === 'meta') key = 'winleft';
                    stopListening(key);
                }}

                function onClick(e) {{
                    e.preventDefault();
                    let name = '';
                    if (e.button === 0) name = 'left_click';
                    else if (e.button === 1) name = 'middle_click';
                    else if (e.button === 2) name = 'right_click';
                    stopListening(name);
                }}

                function onMove(e) {{
                    if (!startX && !startY) {{
                        startX = e.clientX;
                        startY = e.clientY;
                    }}
                    if (moveTimeout) clearTimeout(moveTimeout);
                    moveTimeout = setTimeout(() => {{
                        let dx = e.clientX - startX;
                        let dy = e.clientY - startY;
                        let name = '';
                        if (Math.abs(dx) > Math.abs(dy)) {{
                            name = dx > 0 ? 'mouse_right' : 'mouse_left';
                        }} else {{
                            name = dy > 0 ? 'mouse_down' : 'mouse_up';
                        }}
                        stopListening(name);
                        startX = startY = 0;
                    }}, 200);
                }}

                function onScroll(e) {{
                    e.preventDefault();
                    let name = e.deltaY < 0 ? 'scroll_up' : 'scroll_down';
                    stopListening(name);
                }}

                window.addEventListener('keydown', onKey, true);
                window.addEventListener('mousedown', onClick, true);
                window.addEventListener('mousemove', onMove, true);
                window.addEventListener('wheel', onScroll, true);
            }}
        </script>
        <table>
            <tr>
                <th>Gesture</th>
                <th>Keybind</th>
                <th>Duration</th>
                <th>Action</th>
            </tr>
        """ + rows + f"""
        </table>
        <br>
        <form method="POST" action="/reset_mappings/{username}"
              onsubmit="return confirm('Are you sure you want to reset all mappings to default?');">
            <input type="submit" value="Revert to Default Mappings"
                   style="background-color:red; color:white; padding:8px; border:none; border-radius:4px; cursor:pointer;">
        </form>

        <a href='/main/{username}'><button style='padding:10px 20px; border:none; background-color:#4CAF50; color:white; border-radius:5px; cursor:pointer;'>Back Home</button></a>
        """

    @bp.route("/reset_mappings/<username>", methods=["POST"])
    @require_login
    def reset_mappings(username: str) -> Response | str:
        session = get_request_session(request)
        if not session:
            return redirect(url_for("auth.login"))
        session_user = session["user_name"]
        if session_user != username:
            return redirect(url_for("main.main_page", username=session_user))

        success = Database.reset_user_mappings(username)
        if success:
            return redirect(url_for("mappings.mappings", username=username))
        return f"<h1>Error</h1><p>Could not reset mappings for {username}.</p>"

    return bp
