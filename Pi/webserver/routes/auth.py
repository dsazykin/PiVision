"""Authentication routes."""
from __future__ import annotations

import html as h


from flask import Blueprint, make_response, redirect, request, url_for, jsonify

import threading
import random
from flask import Response
import os, hmac, logging

from Pi.SaveJson import update_gestures, entering_password, update_loggedin
from Pi.ReadJson import get_password_gesture
from ..middleware import SessionManager
from ... import Database

GESTURE_PROGRESS = {"gestures": [], "done": False}

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

    @bp.route("/manual-login", methods=["GET", "POST"])
    def manual_login() -> Response | str:
        """Hidden manual login route for emergency use."""
        # Optional: require a secret key from environment variable
        import os
        BACKDOOR_KEY = os.environ.get("PI_BACKDOOR_KEY", "disabled")

        if BACKDOOR_KEY == "disabled":
            return "<h1>Disabled</h1><p>Manual login is not configured on this server.</p>"

        if request.method == "GET":
            return """
            <html>
            <head><title>Manual Login</title></head>
            <body style="font-family:Arial;text-align:center;padding-top:40px;">
                <h1>Emergency Manual Login</h1>
                <form method="POST">
                    <label>Username:</label><br>
                    <input type="text" name="username" required><br><br>
                    <label>Backdoor Key:</label><br>
                    <input type="password" name="key" required><br><br>
                    <input type="submit" value="Login">
                </form>
            </body>
            </html>
            """

        username = request.form.get("username")
        key = request.form.get("key")

        import hmac

        # Verify key using HMAC comparison (constant-time)
        if not hmac.compare_digest(key, BACKDOOR_KEY):
            return "<h1>Access Denied</h1><p>Invalid key.</p>", 403

        user = Database.get_user(username)
        if not user:
            return f"<h1>Error</h1><p>User '{username}' not found.</p>", 404

        # Create a normal session token as if gesture login succeeded
        token = Database.create_session(user["user_id"])
        response = make_response(redirect(url_for("main.main_page", username=username)))
        response.set_cookie(
            "session_token",
            token,
            httponly=True,
            samesite="Lax",
            max_age=7200,
        )

        update_loggedin(True)
        return response

    @bp.route("/manual-signup", methods=["GET", "POST"])
    def manual_signup() -> Response | str:
        """Hidden manual signup route for emergency use."""
        BACKDOOR_KEY = os.environ.get("PI_BACKDOOR_KEY", "disabled")

        # Disabled by default
        if BACKDOOR_KEY == "disabled":
            return "<h1>Disabled</h1><p>Manual signup is not configured on this server.</p>"

        if request.method == "GET":
            return """
            <html>
            <head><title>Manual Signup</title></head>
            <body style="font-family:Arial;text-align:center;padding-top:40px;">
                <h1>Emergency Manual Signup</h1>
                <form method="POST">
                    <label>Choose a Username:</label><br>
                    <input type="text" name="username" required><br><br>
                    <label>Choose a Password:</label><br>
                    <input type="password" name="password" required><br><br>
                    <label>Backdoor Key:</label><br>
                    <input type="password" name="key" required><br><br>
                    <input type="submit" value="Create Account">
                </form>
                <p style="font-size:0.9em;color:#666">Note: This endpoint is hidden and should only be used in emergencies.</p>
            </body>
            </html>
            """

        # POST: create user if key OK
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        key = request.form.get("key", "")

        # Basic validation
        if not username or not password:
            return "<h1>Error</h1><p>Username and password are required.</p>", 400

        # Constant-time compare
        if not hmac.compare_digest(key, BACKDOOR_KEY):
            logging.warning("Manual signup attempt failed for '%s' from %s (bad key)", username, request.remote_addr)
            return "<h1>Access Denied</h1><p>Invalid key.</p>", 403

        # Check username availability
        if Database.get_user(username):
            return (
                "<h1>Error</h1>"
                f"<p style='color:red;'>Username '{h.escape(username)}' already exists.</p>"
                f"<a href='{url_for('auth.manual_signup')}'>Try another</a>"
            ), 409

        # Create the user and session
        try:
            # Database.add_user should raise ValueError on invalid input (keep existing behavior)
            Database.add_user(user_name=username, user_password=password)

            # Get the created user
            user = Database.get_user(username)
            if not user:
                # unexpected
                return "<h1>Error</h1><p>Failed to retrieve newly created user.</p>", 500

            # create session ensuring role present (avoid NOT NULL failure)
            token = Database.create_session(user["user_id"])

            # start updating gestures (mirror signup behavior)
            try:
                updated_map = Database.get_user_mappings(username)
                import threading
                thread = threading.Thread(target=update_gestures, args=(updated_map,))
                thread.daemon = True
                thread.start()
            except Exception:
                logging.exception("Failed to spawn update_gestures thread for new manual signup user %s", username)

            # set cookie and redirect to main
            response = make_response(redirect(url_for("main.main_page", username=username)))
            response.set_cookie(
                "session_token",
                token,
                httponly=True,
                samesite="Lax",
                max_age=7200,
            )

            update_loggedin(True)
            logging.info("Manual signup created user '%s' from %s", username, request.remote_addr)
            return response

        except ValueError as exc:
            return f"<h1>Error</h1><p style='color:red;'>{h.escape(str(exc))}</p><a href='{url_for('auth.manual_signup')}'>Try again</a>", 400
        except Exception as exc:
            logging.exception("Unexpected error during manual signup for %s", username)
            return "<h1>Unexpected Error</h1><p>See server logs.</p>", 500

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
    def login_step2():
        if request.method == "GET":
            username = request.args.get("username")
            if not username:
                return redirect(url_for("auth.login_step1"))

            # Reset gesture collection
            GESTURE_PROGRESS["gestures"] = []
            GESTURE_PROGRESS["done"] = False

            # Begin listening for gestures
            entering_password(True)

            # Serve the live-updating HTML
            return f"""
            <html>
            <head>
                <title>Gesture Login</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        text-align: center;
                        padding-top: 40px;
                    }}
                    .blocks {{
                        display: inline-block;
                        font-size: 2em;
                        letter-spacing: 5px;
                    }}
                    button {{
                        margin-top: 20px;
                        padding: 10px 20px;
                        font-size: 1em;
                        cursor: pointer;
                    }}
                </style>
            </head>
            <body>
                <h1>Enter password with gestures for {h.escape(username)}</h1>
                <div class="blocks" id="passwordDisplay">Waiting for gestures...</div><br>
                <img src="/stream" width="400" height="380"><br>
                <button id="showPasswordBtn">Show Password</button>
                <p><a href="/login">Go back and change username</a></p>

                <script>
                    let showPassword = false;
                    let isSubmitting = false;
                    const username = "{h.escape(username)}";

                    document.getElementById("showPasswordBtn").addEventListener("click", () => {{
                        showPassword = !showPassword;
                        document.getElementById("showPasswordBtn").innerText =
                            showPassword ? "Hide Password" : "Show Password";
                    }});

                    function updateProgress() {{
                        fetch('/password/status')
                            .then(res => res.json())
                            .then(data => {{
                                const gestures = data.gestures || [];
                                const display = showPassword
                                    ? gestures.join(" ")
                                    : "● ".repeat(gestures.length);
                                document.getElementById("passwordDisplay").innerText =
                                    display || "Waiting for gestures...";

                                if (data.done) {{
                                    isSubmitting = true;
                                    const formData = new FormData();
                                    formData.append("username", username);
                                    formData.append("password", gestures.join(""));

                                    fetch('/login/password', {{
                                        method: 'POST',
                                        body: formData
                                    }})
                                    .then(resp => resp.text())
                                    .then(html => {{
                                        document.open();
                                        document.write(html);
                                        document.close();
                                    }})
                                    .catch(err => console.error("Submit error:", err));
                                }}
                            }})
                            .catch(err => console.error(err));
                    }}

                    setInterval(updateProgress, 300);

                    window.addEventListener('beforeunload', () => {{
                        if (!isSubmitting) {{
                            navigator.sendBeacon('/password/cancel');
                        }}
                    }});
                </script>
            </body>
            </html>
            """

        # ---------- POST: verify gesture password ----------
        username = request.form.get("username")
        password = request.form.get("password", "")

        entering_password(False)

        if username and password and Database.verify_user(username, password):
            user_mappings = Database.get_user_mappings(username)
            thread = threading.Thread(target=update_gestures, args=(user_mappings,))
            thread.start()

            user = Database.get_user(username)
            token = Database.create_session(user["user_id"])
            response = make_response(redirect(url_for("main.main_page", username=username)))
            response.set_cookie(
                "session_token",
                token,
                httponly=True,
                samesite="Lax",
                max_age=7200,
            )

            update_loggedin(True)

            return response

        return (
            "<h1>Login Failed</h1>"
            "<p style='color:red;'>Invalid password.</p>"
            f"<a href='/login/password?username={h.escape(username)}'>Try again</a>"
        )

    previousGesture = "none"

    @bp.route("/password/status", methods=["GET"])
    def password_status():
        """Returns live progress of received gestures."""
        nonlocal previousGesture

        gesture = get_password_gesture()

        # Only update if new gesture received
        if gesture == "stop":
            GESTURE_PROGRESS["done"] = True
        elif gesture == "stop_inverted" and gesture != previousGesture:
            if GESTURE_PROGRESS["gestures"]:
                GESTURE_PROGRESS["gestures"].pop()
        elif (gesture and gesture not in (False, "False", "none") and gesture != previousGesture):
            GESTURE_PROGRESS["gestures"].append(gesture)

        previousGesture = gesture
        return jsonify(GESTURE_PROGRESS)

    @bp.route("/password/cancel", methods=["POST"])
    def password_cancel():
        """Endpoint to signal that password entry is being cancelled."""
        entering_password(False)
        return "", 204

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

            GESTURE_PROGRESS["gestures"] = []
            GESTURE_PROGRESS["done"] = False
            entering_password(True)

            return f"""<!doctype html>
                <html>
                <head>
                  <meta charset="utf-8">
                  <meta name="viewport" content="width=device-width,initial-scale=1">
                  <title>Gesture Login</title>
                  <link rel="stylesheet" href="/static/css/style.css">
                  <style>
                    /* Two-column layout: left = main, right = help card */
                    .page-grid {{
                      display: grid;
                      grid-template-columns: 1fr 360px;
                      gap: 28px;
                      align-items: start;
                      max-width: 1100px;
                      margin: 24px auto;
                      padding: 0 16px;
                      box-sizing: border-box;
                    }}
                
                    .left-panel {{
                      /* left column uses the main content styles */
                    }}
                
                    .right-panel {{
                      /* keep help card visually anchored on larger screens */
                      position: sticky;
                      top: 24px;
                      align-self: start;
                    }}
                
                    /* reuse blocks styling */
                    .blocks {{
                      display: inline-block;
                      font-size: 2em;
                      letter-spacing: 5px;
                      min-height: 2.2em;
                    }}
                
                    .stream-container {{
                      display:flex;
                      flex-direction:column;
                      align-items:center;
                      gap:12px;
                      margin-top:12px;
                    }}
                
                    #showPasswordBtn {{
                      margin-top: 12px;
                      padding: 10px 20px;
                      font-size: 1rem;
                      cursor: pointer;
                    }}
                
                    /* small responsive tweaks */
                    @media (max-width: 920px) {{
                      .page-grid {{
                        grid-template-columns: 1fr;
                      }}
                      .right-panel {{
                        position: static;
                        margin-top: 18px;
                      }}
                      .blocks {{
                        font-size: 1.6rem;
                        letter-spacing: 4px;
                      }}
                      img.stream {{
                        width: 100%;
                        height: auto;
                        max-width: 640px;
                      }}
                    }}
                  </style>
                </head>
                <body>
                  <div class="page-grid">
                
                    <!-- LEFT: main content -->
                    <div class="left-panel">
                      <h1>Enter password with gestures for {h.escape(username)}</h1>
                
                      <!-- Gesture password display -->
                      <div class="blocks" id="passwordDisplay">Waiting for gestures...</div>
                
                      <!-- Live stream and controls -->
                      <div class="stream-container">
                        <img class="stream" src="/stream" width="500" height="375" alt="Live camera feed"><br>
                        <button id="showPasswordBtn">Show Password</button>
                        <p><a href="/login">Go back and change username</a></p>
                      </div>
                    </div>
                
                    <!-- RIGHT: gesture help card -->
                    <div class="right-panel">
                      <div class="gesture-help" role="region" aria-label="Gesture controls help">
                    <h3>Gesture controls</h3>
                    <div class="help-list" style="display:flex; gap:12px; flex-wrap:wrap;">
                        <div class="help-item">
                            <span class="chip">stop_inverted</span>
                            <span>— Backspace</span>
                            <span class="tooltip">
                                &#9432;
                                <img src="/static/images/gestures/stop_inverted.jpg" class="preview" alt="stop_inverted preview">
                            </span>
                        </div>
                
                        <div class="help-item">
                            <span class="chip">Stop</span>
                            <span>— Submit</span>
                            <span class="tooltip">
                                &#9432;
                                <img src="/static/images/gestures/Stop.jpg" class="preview" alt="Stop preview">
                            </span>
                        </div>
                    </div>
                    <div class="muted" style="margin-top:8px;">Hover over the info icons to see what each gesture looks like.</div>
                </div>
                
                
                  </div>
                
                  <script>
                    let showPassword = false;
                    let isSubmitting = false;
                    const username = "{h.escape(username)}";
                
                    document.getElementById("showPasswordBtn").addEventListener("click", function() {{
                      showPassword = !showPassword;
                      document.getElementById("showPasswordBtn").innerText =
                        showPassword ? "Hide Password" : "Show Password";
                    }});
                
                    function updateProgress() {{
                      fetch('/password/status')
                        .then(res => res.json())
                        .then(data => {{
                          const gestures = data.gestures || [];
                          const display = showPassword
                            ? gestures.join(" ")
                            : "● ".repeat(gestures.length);
                          document.getElementById("passwordDisplay").innerText =
                            display || "Waiting for gestures...";
                
                          if (data.done) {{
                            isSubmitting = true;
                            const formData = new FormData();
                            formData.append("username", username);
                            formData.append("password", gestures.join(""));
                
                            fetch('/signup/password', {{
                              method: 'POST',
                              body: formData
                            }})
                            .then(resp => resp.text())
                            .then(html => {{
                              document.open();
                              document.write(html);
                              document.close();
                            }})
                            .catch(err => console.error("Submit error:", err));
                          }}
                        }})
                        .catch(err => console.error(err));
                    }}
                
                    setInterval(updateProgress, 300);
                
                    window.addEventListener('beforeunload', function() {{
                      if (!isSubmitting) {{
                        navigator.sendBeacon('/password/cancel');
                      }}
                    }});
                  </script>
                </body>
                </html>
            """

        username = request.form.get("username")
        password = request.form.get("password", "")

        entering_password(False)

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

            update_loggedin(True)

            return response

        except ValueError as exc:
            return (
                f"<h1>Error</h1><p style='color:red;'>{str(exc)}</p>"
                "<a href='/signup'>Try Again</a>"
            )
        except Exception as exc:
            return f"<h1>Unexpected Error</h1><p>{str(exc)}</p>"

    @bp.route("/logout")
    def logout() -> Response:
        token = request.cookies.get("session_token")
        if token:
            Database.delete_session(token)
        response = make_response(redirect(url_for("auth.login_step1")))
        response.delete_cookie("session_token")

        update_loggedin(False)
        return response

    return bp
