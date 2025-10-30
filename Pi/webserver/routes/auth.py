"""Authentication routes."""
from __future__ import annotations

from flask import Blueprint, Response, make_response, redirect, request, url_for

import Database
import json, os
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
                "Creator", "Nomad", "ZeroKool", "Seeker", "SyntaxError404"
                , "User"
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
            return "Please select a username."

        # GET request - display all usernames in a dropdown
        all_users = Database.get_all_users() 
        user_options = "".join(f'<option value="{u}">{u}</option>' for u in all_users)
        
        return f"""
            <h1>Select Username</h1>
            <form method="POST">
                <label>Username:</label><br>
                <select name="username_select" required>
                    {user_options}
                </select><br><br>
                <input type="submit" value="Next">
            </form>
            <br><a href="/signup">Don't have an account? Sign up here</a>
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
                return redirect(url_for("auth.login_step1")) #Go back if no username
                
            return f"""
                <h1>Login: Enter Password for {username}</h1>
                <form method="POST">
                    <input type="hidden" name="username" value="{username}">
                    <label>Password:</label><br>
                    <input type="password" name="password" required><br><br>
                    <input type="submit" value="Login">
                </form>
                <br><a href="/login">Go back and change username</a>
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
                    with open(BOOLEAN_PATH, 'w') as f:
                        json.dump({"loggedIn": True}, f)
                except Exception as e:
                    print("Error writing JSON: ", e)
                    
                return response
                
            return (
                "<h1>Login Failed</h1>"
                "<p style='color:red;'>Invalid password.</p>"
                f"<a href='/login/password?username={username}'>Try again</a>"
            )

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
        
        return f"""
            <h1>Sign Up (Step 1 of 2)</h1>
            <p>We've generated a unique name for you. Feel free to change it!</p>
            <form method="POST">
                <label>Choose a Username:</label><br>
                <input type="text" name="username" value="{default_username}" required><br><br>
                <input type="submit" value="Next: Set Password">
            </form>
            <br><a href="/login">Already have an account? Login here</a>
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

            return f"""
                <h1>Sign Up (Step 2 of 2)</h1>
                <p>Creating account for **{username}**</p>
                <form method="POST">
                    <input type="hidden" name="username" value="{username}">
                    <label>Choose a Password:</label><br>
                    <input type="password" name="password" required><br><br>
                    <input type="submit" value="Create Account">
                </form>
                <br><a href="/signup">Go back and change username</a>
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
                    with open(BOOLEAN_PATH, 'w') as f:
                        json.dump({"loggedIn": True}, f)
                except Exception as e:
                    print("Error writing JSON: ", e)
                    
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
        try:
            with open(BOOLEAN_PATH, 'w') as f:
                json.dump({"loggedIn": False}, f)
        except Exception as e:
            print("Error writing JSON: ", e)
        return response

    return bp
