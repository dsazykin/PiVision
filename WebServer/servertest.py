import database

from flask import Flask, request, redirect
app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Hello from Pi!</h1><p>This page is served by Flask on your Raspberry Pi 5.</p>"

@app.route("/main")
def init():
    database.initialize_database()
    html = "<h1>Database intitialization page</h1>"
    html += "<p style='color:green;'> Database correctly initialized.</p>"
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
    html = "<h1>Database Page</h1>"
    for user in databaseinfo:
        html += f"<h2>User: {user['user_name']} (Role: {user['role']})</h2><ul>"
        for gesture, action in user['mappings'].items():
            html += f"<li>{gesture}: {action}</li>"
        html += "</ul>"

    return html

@app.route("/signup", methods = ["GET", "POST"])
def signup():
    html = "<h1> Signup Page</h1>"
    if request.method == "POST":
        # Read data from the form
        username = request.form.get("username")
        password = request.form.get("password")

        try:
            database.add_user(user_name=username, user_password=password)
            return f"<h1>Signup Successful</h1><p>User '{username}' has been created.</p>"
        except ValueError as e:
            # Handle 'user already exists'
            return f"<h1>Error</h1><p>{str(e)}</p>"
        except Exception as e:
            return f"<h1>Unexpected Error</h1><p>{str(e)}</p>"
    
    html += """
        <form method="POST">
            <label>Username:</label><br>
            <input type="text" name="username" required><br><br>
            <label>Password:</label><br>
            <input type="password" name="password" required><br><br>
            <input type="submit" value="Sign Up">
        </form>
    """
    return html

@app.route("/delete", methods = ["GET", "POST"])
def delUser():
    html = "<h1> User Deletion page</h1>"
    if request.method == "POST":
        # Read data from the form
        username = request.form.get("username")
        try:
            deleted = database.delete_user(username)
            if deleted == 0:
                return f"<h1> Deletion Failed</h1><br/><p style='color:red;'> Deletion of user '{username}' was not succesfull</p>"
            else:
                return f"<h1> Deletion Successful</h1><p style='color:green;'>User '{username}' has been deleted.</p>"
        except ValueError as e:
            # Handle 'user already exists'
            return f"<h1>Error</h1><p>{str(e)}</p>"
        except Exception as e:
            return f"<h1>Unexpected Error</h1><p>{str(e)}</p>"
    
    html += """
        <form method="POST">
            <label>Username to be deleted:</label><br>
            <input type="text" name="username" required><br><br>
            <input type="submit" value="DELETE">
        </form>
    """
    return html
    

app.run(host="0.0.0.0", port=5000)