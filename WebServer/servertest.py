import database

from flask import Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Hello from Pi!</h1><p>This page is served by Flask on your Raspberry Pi 5.</p>"

@app.route("/main")
def init():
    database.initialize_database()
    return "<h1>title</h1>"
    
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
    html = "<h1>Database Page</h1>"
    for user in databaseinfo:
        html += f"<h2>User: {user['user_name']} (Role: {user['role']})</h2><ul>"
        for gesture, action in user['mappings'].items():
            html += f"<li>{gesture}: {action}</li>"
        html += "</ul>"

    return html

@app.route("/signup")
def addUser():
    database.add_user(user_name= "Paul", user_password= "Paul123")
    return "<h1>title</h1>"

app.run(host="0.0.0.0", port=5000)