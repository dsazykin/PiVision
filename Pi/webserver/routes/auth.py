"""Authentication routes."""
from __future__ import annotations

import html as h
from flask import Blueprint, Response, make_response, redirect, request, url_for

import Database
import json
import os
import threading
import random
import uuid

from ..UpdateMappings import update_gestures
from ..middleware import SessionManager

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
temp_dir = os.path.join(project_root, "WebServerStream")
BOOLEAN_PATH = os.path.join(temp_dir, "boolean.json")


def create_blueprint(session_manager: SessionManager) -> Blueprint:
    bp = Blueprint("auth", __name__)

    def _generate_default_username() -> str:
        while True:
            name_list = [
                "Traveler", "Explorer", "Guest", "Builder",
                "Creator", "Nomad", "ZeroKool", "Seeker", "SyntaxError404",
                "User",
            ]

            base_name = random.choice(name_list)
            unique_signature = str(random.randint(0, 100000))

            candidate_username = f"{base_name}{unique_signature}"
            if not Database.get_user(candidate_username):
                return candidate_username

    @bp.route("/login", methods=["GET", "POST"])
    def login_step1() -> Response | str:
        active_session = session_manager.get_active_session()
        if active_session:
            return redirect(
                url_for("main.main_page", username=active_session["user_name"])
            )

        if request.method == "POST":
            username = request.form.get("username_select")
            if username:
                return redirect(url_for("auth.login_step2", username=username))
            # Keep behaviour: simple message if nothing selected
            return "Please select a username."

        # GET request - display all usernames in a dropdown
        all_users = Database.get_all_users()
        # escape usernames for HTML safety
        user_options = "".join(
            f'<option value="{h.escape(u)}">{h.escape(u)}</option>' for u in all_users
        )

        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Login — Select User</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <div class="container">
    <div class="homepage_content_div" style="max-width:620px;margin:24px auto;">
      <h2>Select Username</h2>
      <p class="lead">Choose your username from the list below.</p>
      <form method="POST" style="margin-top:12px; display:flex; flex-direction:column; gap:10px; align-items:flex-start;">
        <label for="username_select">Username:</label>
        <select id="username_select" name="username_select" required style="width:260px;padding:10px;border-radius:8px;">
            {user_options}
        </select>

        <div style="margin-top:10px;">
          <button type="submit" class="btn">Next</button>
          <a href="/signup" class="btn ghost" style="margin-left:10px;">Sign up</a>
        </div>
      </form>
      <div style="margin-top:14px;">
        <a href="/signup" class="btn ghost">Don't have an account? Sign up here</a>
      </div>
    </div>
  </div>
</body>
</html>
"""

    @bp.route("/login/password", methods=["GET", "POST"])
    def login_step2() -> Response | str:
        active_session = session_manager.get_active_session()
        if active_session:
            return redirect(
                url_for("main.main_page", username=active_session["user_name"])
            )

        # Handle GET request to show password form
        if request.method == "GET":
            username = request.args.get("username")
            if not username:
                return redirect(url_for("auth.login_step1"))  # Go back if no username

            safe_username = h.escape(username)
            return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Login — Password</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <div class="container">
    <div class="homepage_content_div" style="max-width:520px;margin:24px auto;">
      <h2>Login: Enter Password</h2>
      <p class="lead">Signing in as <strong>{safe_username}</strong></p>
      <form method="POST" style="margin-top:12px; display:flex; flex-direction:column; gap:10px;">
        <input type="hidden" name="username" value="{safe_username}">
        <label for="password">Password</label>
        <input id="password" type="password" name="password" required>
        <div style="margin-top:8px;">
          <button type="submit" class="btn">Login</button>
          <a href="/login" class="btn ghost" style="margin-left:8px;">Back</a>
        </div>
      </form>
      <div style="margin-top:14px;"><a href="/login">Go back and change username</a></div>
    </div>
  </div>
</body>
</html>
"""

        # Handle POST request to verify password
        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")

            if username and password and Database.verify_user(username, password):
                user_mappings = Database.get_user_mappings(username)
                thread = threading.Thread(target=update_gestures, args=(user_mappings,))
                thread.start()

                user = Database.get_user(username)
                token = Database.create_session(user["user_id"])
                response = make_response(
                    redirect(url_for("main.main_page", username=username))
                )
                response.set_cookie(
                    "session_token",
                    token,
                    httponly=True,
                    samesite="Lax",
                    max_age=7200,
                )
                try:
                    with open(BOOLEAN_PATH, "w") as f:
                        json.dump({"loggedIn": True}, f)
                except Exception as e:
                    print("Error writing JSON: ", e)

                return response

            # Failed login — keep same behaviour but return a styled page
            safe_username = h.escape(username) if username else ""
            return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Login Failed</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <div class="container">
    <div class="homepage_content_div center" style="max-width:640px;margin:24px auto;">
      <h2>Login Failed</h2>
      <p style="color:var(--danger)">Invalid password.</p>
      <div style="margin-top:12px;">
        <a href="/login/password?username={safe_username}" class="btn">Try again</a>
        <a href="/login" class="btn ghost" style="margin-left:8px;">Choose different user</a>
      </div>
    </div>
  </div>
</body>
</html>
"""

    @bp.route("/signup", methods=["GET", "POST"])
    def signup_step1() -> Response | str:
        active_session = session_manager.get_active_session()
        if active_session:
            return redirect(
                url_for("main.main_page", username=active_session["user_name"])
            )

        if request.method == "POST":
            username = request.form.get("username")
            if not username:
                return "<h1>Error</h1><p>Username is required.</p>"

            if Database.get_user(username):
                return (
                    "<h1>Error</h1><p style='color:red;'>Username already taken.</p>"
                    f"<a href='{url_for('auth.signup_step1')}'>Try Again</a>"
                )

            return redirect(url_for("auth.signup_step2", username=username))

        default_username = _generate_default_username()

        safe_default = h.escape(default_username)
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sign Up — Step 1</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <div class="container">
    <div class="homepage_content_div" style="max-width:640px;margin:24px auto;">
      <h2>Sign Up (Step 1 of 2)</h2>
      <p class="lead">We've generated a unique name for you. Feel free to change it!</p>

      <form method="POST" style="margin-top:12px; display:flex; flex-direction:column; gap:10px;">
        <label for="username">Choose a Username</label>
        <input id="username" type="text" name="username" value="{safe_default}" required>
        <div style="margin-top:8px;">
          <button type="submit" class="btn">Next: Set Password</button>
          <a href="/login" class="btn ghost" style="margin-left:8px;">Already have an account?</a>
        </div>
      </form>
    </div>
  </div>
</body>
</html>
"""

    @bp.route("/signup/password", methods=["GET", "POST"])
    def signup_step2() -> Response | str:
        active_session = session_manager.get_active_session()
        if active_session:
            return redirect(
                url_for("main.main_page", username=active_session["user_name"])
            )

        if request.method == "GET":
            username = request.args.get("username")
            if not username:
                return redirect(url_for("auth.signup_step1"))

            safe_username = h.escape(username)
            return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sign Up — Step 2</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <div class="container">
    <div class="homepage_content_div" style="max-width:640px;margin:24px auto;">
      <h2>Sign Up (Step 2 of 2)</h2>
      <p class="lead">Creating account for <strong>{safe_username}</strong></p>
      <form method="POST" style="margin-top:12px; display:flex; flex-direction:column; gap:10px;">
        <input type="hidden" name="username" value="{safe_username}">
        <label for="password">Choose a Password</label>
        <input id="password" type="password" name="password" required>
        <div style="margin-top:8px;">
          <button type="submit" class="btn">Create Account</button>
          <a href="/signup" class="btn ghost" style="margin-left:8px;">Back</a>
        </div>
      </form>
      <div style="margin-top:14px;"><a href="/signup">Go back and change username</a></div>
    </div>
  </div>
</body>
</html>
"""

        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")

            if not username or not password:
                return "<h1>Error</h1><p>Username and password are required.</p>"

            try:
                Database.add_user(user_name=username, user_password=password)
                user = Database.get_user(username)
                token = Database.create_session(user["user_id"])
                updated_map = Database.get_user_mappings(username)
                thread = threading.Thread(target=update_gestures, args=(updated_map,))
                thread.start()
                response = make_response(
                    redirect(url_for("main.main_page", username=username))
                )
                response.set_cookie(
                    "session_token",
                    token,
                    httponly=True,
                    samesite="Lax",
                    max_age=7200,
                )

                try:
                    with open(BOOLEAN_PATH, "w") as f:
                        json.dump({"loggedIn": True}, f)
                except Exception as e:
                    print("Error writing JSON: ", e)

                return response

            except ValueError as exc:
                return f"""<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign Up Error</title>
<link rel="stylesheet" href="/static/css/style.css">
</head><body>
  <div class="container">
    <div class="homepage_content_div center" style="max-width:640px;margin:24px auto;">
      <h2>Error</h2>
      <p style="color:var(--danger)">{h.escape(str(exc))}</p>
      <div style="margin-top:10px;">
        <a href="/signup" class="btn">Try Again</a>
      </div>
    </div>
  </div>
</body>
</html>
"""

            except Exception as exc:
                return f"""<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unexpected Error</title>
<link rel="stylesheet" href="/static/css/style.css">
</head><body>
  <div class="container">
    <div class="homepage_content_div center" style="max-width:640px;margin:24px auto;">
      <h2>Unexpected Error</h2>
      <p>{h.escape(str(exc))}</p>
      <div style="margin-top:10px;">
        <a href="/signup" class="btn">Back</a>
      </div>
    </div>
  </div>
</body>
</html>
"""

    @bp.route("/logout")
    def logout() -> Response:
        token = request.cookies.get("session_token")
        if token:
            Database.delete_session(token)
        response = make_response(redirect(url_for("auth.login_step1")))
        response.delete_cookie("session_token")
        try:
            with open(BOOLEAN_PATH, "w") as f:
                json.dump({"loggedIn": False}, f)
        except Exception as e:
            print("Error writing JSON: ", e)
        return response

    return bp
