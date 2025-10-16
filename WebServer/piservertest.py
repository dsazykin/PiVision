from flask import Flask, request, redirect, url_for, Response, jsonify, render_template_string
import database, threading, json, os, cv2, time

app = Flask(__name__)

FRAME_PATH = r"/home/group31/Projects/project/temp/latest.jpg"
JSON_PATH = r"/home/group31/Projects/project/temp/latest.json"

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
    database.initialize_database()
    return """
        <h1>Welcome to the Gesture Control Web App</h1>
        <p>This web interface lets you manage users and start gesture recognition.</p>
        <a href="/login"><button>Login</button></a>
        <a href="/signup"><button>Sign Up</button></a>
    """


# --- LOGIN PAGE ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if database.verify_user(username, password):
            return redirect(url_for("main_page", username=username))
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
    return redirect(url_for("index"))

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


@app.route('/stream')
def stream():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return """<html><body style="text-align:center">
              <h2>Processed Gesture Feed</h2>
              <img src="/stream" width="640" height="480">
              <br><a href="/">Back</a>
              </body></html>"""

@app.route("/main/<username>")
def main_page(username):
    return f"""
        <h1>Welcome, {username}</h1>
        <p>Choose an action:</p>
        <a href="/start/{username}"><button>Start Recognition</button></a><br><br>
        <a href="/mappings/{username}"><button>Edit Gesture Mappings</button></a><br><br>
        <a href="/delete/{username}"><button style='color:red;'>Delete My Account</button></a><br><br>
        <a href="/logout"><button>Log Out</button></a>
    """

@app.route("/mappings/<username>", methods=["GET", "POST"])
def mappings(username):
    if request.method == "POST":
        gesture = request.form.get("gesture")
        new_action = request.form.get("action")
        database.update_gesture_mapping(username, gesture, new_action)
        return redirect(url_for("mappings", username=username))

    mappings = database.get_user_mappings(username)

    html = f"<h1>Gesture Mappings for {username}</h1>"
    html += "<form method='POST'>"
    for gesture, action in mappings.items():
        html += f"""
        <label>{gesture}:</label>
        <input type='text' name='action' value='{action}' required>
        <input type='hidden' name='gesture' value='{gesture}'>
        <input type='submit' value='Update'><br><br>
        """
    html += "</form>"
    html += f"<br><a href='/start/{username}'>Start Recognition</a>"
    return html

    
@app.route("/database")
def showDatabase():
    databaseinfo = []

    for user_name, role in database.get_all_users():
        mappings = database.get_user_mappings(user_name)
        databaseinfo.append({
            "user_name": user_name,
            "role": role,
            "mappings": mappings
        })

    # Build HTML
    html = "<h1>Database Page</h1><div class='db_container_div'>"
    for user in databaseinfo:
        html += "<div class='db_entry_div'>"
        html += f"<h2>User: {user['user_name']} (Role: {user['role']})</h2><ul>"
        for gesture, action in user['mappings'].items():
            html += f"<li>{gesture}: {action}</li>"
        html += "</ul></div>" # ul + entry divs
    html += "</div>" # container div
    return html

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        try:
            database.add_user(user_name=username, user_password=password)
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


@app.route("/delete/<username>")
def delete_user(username):
    deleted = database.delete_user(username)
    if deleted == 0:
        return f"<h1>Deletion Failed</h1><p style='color:red;'>User '{username}' not found.</p>"
    return f"<h1>Account Deleted</h1><p>User '{username}' has been removed.</p><a href='/'>Return Home</a>"

    

app.run(host="0.0.0.0", port=5000, threaded = True)