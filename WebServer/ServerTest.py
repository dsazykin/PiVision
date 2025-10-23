from flask import Flask, request, redirect, url_for, Response, jsonify, render_template_string
import Database, json, os, cv2, time
from functools import wraps
from flask import make_response
import secrets

app = Flask(__name__)

def get_session_token():
    return request.cookies.get("session_token")

def require_login(f):
    """Decorator to require a valid session."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = get_session_token()
        session = Database.get_session(token) if token else None
        if not session:
            return redirect(url_for("login"))
        # Pass session info into the route
        request.session = session
        return f(*args, **kwargs)
    return wrapper

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

temp_dir = os.path.join(project_root, "WebServerStream")
os.makedirs(temp_dir, exist_ok=True)

FRAME_PATH = os.path.join(temp_dir, "latest.jpg")
JSON_PATH = os.path.join(temp_dir, "latest.json")

@app.after_request
def add_global_css(response: Response):
    # Only modify HTML responses
    if response.content_type == "text/html; charset=utf-8":
        css_link = f'<link rel="stylesheet" type="text/css" href="{url_for("static", filename="style.css")}">'
        # Inject right after <head> if present, else at start
        html = response.get_data(as_text=True)
        if "<head>" in html:
            html = html.replace("<head>", f"<head>{css_link}")
        else:
            html = css_link + html
        response.set_data(html)
    return response

@app.route("/")
def index():
    Database.initialize_database()
    return """
        <h1>Welcome to the Gesture Control Web App</h1>
        <p>This web interface lets you manage users and gesture recognition.</p>
        <a href="/login"><button>Login</button></a>
        <a href="/signup"><button>Sign Up</button></a>
    """


# --- LOGIN PAGE ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if Database.verify_user(username, password):
            user = Database.get_user(username)
            token = Database.create_session(user["user_id"], user["role"])

            resp = make_response(redirect(url_for("main_page", username=username)))
            resp.set_cookie("session_token", token, httponly=True, samesite="Lax", max_age=7200)
            return resp
        else:
            return """
                <h1>Login Failed</h1>
                <p style='color:red;'>Invalid username or password.</p>
                <a href='/login'>Try again</a>
            """
    return """
        <h1>Login</h1>
        <form method="POST">
            <label>Username:</label><br>
            <input type="text" name="username" required><br><br>
            <label>Password:</label><br>
            <input type="password" name="password" required><br><br>
            <input type="submit" value="Login">
        </form>
        <br><a href="/signup">Don't have an account? Sign up here</a>
    """

@app.route("/logout")
def logout():
    token = get_session_token()
    if token:
        Database.delete_session(token)
    resp = make_response(redirect(url_for("login")))
    resp.delete_cookie("session_token")
    return resp

@app.route("/testing")
def test():
    html = """
    <html><head><title>Pi Vision Gestures</title>
    <script>
    async function updateGesture(){
        const res = await fetch('/gesture');
        const data = await res.json();
        document.getElementById('g').innerText = data.gesture;
        document.getElementById('c').innerText = (data.confidence*100).toFixed(1)+'%';
    }
    setInterval(updateGesture,500); window.onload=updateGesture;
    </script></head>
    <body style="text-align:center;font-family:sans-serif;margin-top:40px">
      <h1 id="g">Loading...</h1>
      <p>Confidence: <span id="c">--%</span></p>
      <a href="/video">View Live Stream ▶</a>
    </body></html>"""
    return render_template_string(html)

# TODO: alter this an api Endpoint so it differentiates the json and frame api.
@app.route('/gesture')
def gesture():
    try:
        with open(JSON_PATH) as f:
            data = json.load(f)
    except Exception:
        data = {"gesture": "None", "confidence": 0.0}
    return jsonify(data)
def gen():
    while True:
        try:
            frame = cv2.imread(FRAME_PATH)
            if frame is None:
                time.sleep(0.05)
                continue
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')
        except Exception:
            time.sleep(0.05)

# we don't visit this page, it acts as an API andpoint
@app.route('/stream')
def stream():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# This is the page where users can see the camera feed on the webServer
@app.route('/video')
def video():
    return """<html><body style="text-align:center">
              <h2>Processed Gesture Feed</h2>
              <img src="/stream" width="640" height="480">
              <br><a href="/login">Back to login page</a>
              </body></html>"""

# This page is the "Homescreen", the page after login.
@app.route("/main/<username>")
@require_login
def main_page(username):
    html = """
    <html><head><title>Pi Vision Gestures</title>
    <script>
    async function updateGesture(){
        const res = await fetch('/gesture');
        const data = await res.json();
        document.getElementById('g').innerText = data.gesture;
        document.getElementById('c').innerText = (data.confidence*100).toFixed(1)+'%';
    }
    setInterval(updateGesture,500); window.onload=updateGesture;
    </script></head>
    """
    html += f"""
        <div class='homepage_container_div'>
            <div class='homepage_content_div'>
                <body style="text-align:center;font-family:sans-serif;margin-top:40px">
                <h1>Welcome, {username}</h1>
                <p>Choose an action:</p>
                <a href="/mappings/{username}"><button>Edit Gesture Mappings</button></a><br><br>
                <a href="/delete/{username}"><button style='color:red;'>Delete My Account</button></a><br><br>
                <a href="/logout"><button>Log Out</button></a>
            </div>
    """
    html += """
            <div class='homepage_content_div'>
                <h1 id="g">Loading...</h1>
                <p>Confidence: <span id="c">--%</span></p>
                <a href="/video">View Live Stream ▶</a>
            </div>
        </div>
    </body></html>"""
    return html 

# # Personal Mappings page, you can also change mappings
# @app.route("/mappings/<username>", methods=["GET", "POST"])
# def mappings(username):
#     if request.method == "POST":
#         gesture = request.form.get("gesture")
#         new_action = request.form.get("action")
#         Database.update_gesture_mapping(username, gesture, new_action)
#         return redirect(url_for("mappings", username=username))

#     mappings = Database.get_user_mappings(username)

#     html = f"<h1>Gesture Mappings for {username}</h1>"
#     for gesture, action in mappings.items():
#         html += f"""
#         <form method='POST'>
#             <label>{gesture}:</label>
#             <input type='text' name='action' value='{action}' required>
#             <input type='hidden' name='gesture' value='{gesture}'>
#             <input type='submit' value='Update'><br><br>
#         </form>
#         """
#     html += f"""
#     <br>
#     <form method="POST" action="/reset_mappings/{username}" onsubmit="return confirm('Are you sure you want to reset all mappings to default?');">
#         <input type="submit" value="Revert to Default Mappings" style="background-color:red; color:white; padding:8px; border:none; border-radius:4px; cursor:pointer;">
#     </form>
#     <br><a href='/start/{username}'>Start Recognition</a>
#     """

#     return html

@app.route("/mappings/<username>", methods=["GET", "POST"])
#@require_login
def mappings(username):
    token = Database.get_user_token()
    Database.verify_session(token, username)

    if request.method == "POST":
        gesture = request.form.get("gesture")
        new_action = request.form.get("action")
        new_duration = request.form.get("duration")
        Database.update_gesture_mapping(username, gesture, new_action, new_duration)
        return redirect(url_for("mappings", username=username))

    mappings = Database.get_user_mappings(username)

    html = f"""
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

            // Keyboard key pressed
            function onKey(e) {{
                e.preventDefault();
                let key = e.key.toLowerCase();
                if (key === ' ') key = 'space';
                if (key === 'meta') key = 'winleft';
                stopListening(key);
            }}

            // Mouse click (left/right/middle)
            function onClick(e) {{
                e.preventDefault();
                let name = '';
                if (e.button === 0) name = 'left_click';
                else if (e.button === 1) name = 'middle_click';
                else if (e.button === 2) name = 'right_click';
                stopListening(name);
            }}

            // Mouse movement direction (up, down, left, right)
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

            // Scroll wheel detection (scroll_up / scroll_down)
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
    """

    for gesture, (action, duration) in mappings.items():
        html += f"""
        <tr>
            <form method="POST">
                <td><strong>{gesture}</strong></td>
                <td>
                    <input type="text" id="input_{gesture}" name="action" value="{action}" required readonly>
                    <button type="button" onclick="startListening('{gesture}')">Edit</button>
                </td>
                <td>
                    <select name="duration">
                        <option value="press" {'selected' if duration == 'press' else ''}>press</option>
                        <option value="hold" {'selected' if duration == 'hold' else ''}>hold</option>
                    </select>
                </td>
                <td>
                    <input type="hidden" name="gesture" value="{gesture}">
                    <input type="submit" value="Update">
                </td>
            </form>
        </tr>
        """

    html += f"""
    </table>

    <br>
    <form method="POST" action="/reset_mappings/{username}"
          onsubmit="return confirm('Are you sure you want to reset all mappings to default?');">
        <input type="submit" value="Revert to Default Mappings"
               style="background-color:red; color:white; padding:8px; border:none; border-radius:4px; cursor:pointer;">
    </form>
    """

    return html


@app.route("/reset_mappings/<username>", methods=["POST"])
def reset_mappings(username):
    success = Database.reset_user_mappings(username)
    if success:
        return redirect(url_for("mappings", username=username))
    else:
        return f"<h1>Error</h1><p>Could not reset mappings for {username}.</p>"

@app.route("/database")
def showDatabase():
    databaseinfo = []

    for user_name, role in Database.get_all_users():
        mappings = Database.get_user_mappings(user_name)
        password = Database.get_user_password(user_name)
        if isinstance(password, bytes):
            password = password.decode('utf-8')
        print(f"Hashed password for {user_name}: {password}")
        databaseinfo.append({
            "user_name": user_name,
            "role": role,
            "mappings": mappings,
            "password": password
        })

    # Build HTML
    html = "<h1>Database Page</h1><div class='db_container_div'>"
    for user in databaseinfo:
        html += "<div class='db_entry_div'>"
        html += f"<h2>User: {user['user_name']} (Role: {user['role']})</h2><ul>"
        for gesture, action in user['mappings'].items():
            html += f"<li>{gesture}: {action}</li>"
        html += f"</ul><p><strong>Hashed Password:</strong> {user['password']}</p></div>"
    html += "</div>"
    return html


# Signup page
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        try:
            Database.add_user(user_name=username, user_password=password)
            return f"""
                <h1>Signup Successful</h1>
                <p>User '{username}' created successfully.</p>
                <a href="/login"><button>Go to Login</button></a>
            """
        except ValueError as e:
            return f"<h1>Error</h1><p>{str(e)}</p><a href='/signup'>Try Again</a>"
        except Exception as e:
            return f"<h1>Unexpected Error</h1><p>{str(e)}</p>"

    return """
        <h1>Sign Up</h1>
        <form method="POST">
            <label>Username:</label><br>
            <input type="text" name="username" required><br><br>
            <label>Password:</label><br>
            <input type="password" name="password" required><br><br>
            <input type="submit" value="Create Account">
        </form>
        <br><a href="/login">Already have an account? Login here</a>
    """

@app.route("/sessions", methods=["GET"])
def show_sessions():
    sessions = Database.get_all_sessions()

    if not sessions:
        return "<h2>No active sessions found.</h2>"

    html = """
    <h1>Active Sessions</h1>
    <style>
        table {
            border-collapse: collapse;
            margin-top: 20px;
            width: 90%;
            font-family: Arial, sans-serif;
        }
        th, td {
            border: 1px solid #ccc;
            padding: 10px;
            text-align: center;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .token-cell {
            max-width: 320px;
            overflow-wrap: anywhere;
            font-family: monospace;
            color: #333;
        }
        h1 {
            font-family: Arial, sans-serif;
            text-align: center;
        }
        body {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
    </style>

    <table>
        <tr>
            <th>ID</th>
            <th>User</th>
            <th>Role</th>
            <th>Session Token</th>
            <th>Created At</th>
            <th>Expires At</th>
        </tr>
    """

    for s in sessions:
        html += f"""
        <tr>
            <td>{s['session_id']}</td>
            <td>{s['user_name']}</td>
            <td>{s['role']}</td>
            <td class="token-cell">{s['session_token']}</td>
            <td>{s['created_at']}</td>
            <td>{s['expires_at']}</td>
        </tr>
        """

    html += "</table>"

    html += """
    <br><a href='/'><button style='padding:10px 20px; border:none; background-color:#4CAF50; color:white; border-radius:5px; cursor:pointer;'>Back Home</button></a>
    """

    return html


# Only used to delete Users. Currently you're only able to delete your own accout with a button
# Or you need to type in the url with someone else's name to delete that account.
@app.route("/delete/<username>")
def delete_user(username):
    deleted = Database.delete_user(username)
    if deleted == 0:
        return f"<h1>Deletion Failed</h1><p style='color:red;'>User '{username}' not found.</p>"
    return f"<h1>Account Deleted</h1><p>User '{username}' has been removed.</p><a href='/'>Return Home</a>"

app.run(host="0.0.0.0", port=5000, threaded = True, debug= True)