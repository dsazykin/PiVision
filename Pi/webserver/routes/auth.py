"""Authentication routes."""
from __future__ import annotations

from flask import Blueprint, Response, make_response, redirect, request, url_for

import Database

from ..middleware import SessionManager


def create_blueprint(session_manager: SessionManager) -> Blueprint:
    bp = Blueprint("auth", __name__)

    @bp.route("/login", methods=["GET", "POST"])
    def login() -> Response | str:
        active_session = session_manager.get_active_session()
        if active_session:
            return redirect(
                url_for("main.main_page", username=active_session["user_name"])
            )

        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            if username and password and Database.verify_user(username, password):
                user = Database.get_user(username)
                token = Database.create_session(user["user_id"], user["role"])
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
                return response
            return (
                "<h1>Login Failed</h1>"
                "<p style='color:red;'>Invalid username or password.</p>"
                "<a href='/login'>Try again</a>"
            )

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

    @bp.route("/signup", methods=["GET", "POST"])
    def signup() -> Response | str:
        active_session = session_manager.get_active_session()
        if active_session:
            return redirect(
                url_for("main.main_page", username=active_session["user_name"])
            )

        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            if not username or not password:
                return "<h1>Error</h1><p>Username and password are required.</p>"
            try:
                Database.add_user(user_name=username, user_password=password)
                return f"""
                    <h1>Signup Successful</h1>
                    <p>User '{username}' created successfully.</p>
                    <a href="/login"><button>Go to Login</button></a>
                """
            except ValueError as exc:
                return (
                    f"<h1>Error</h1><p>{str(exc)}</p>"
                    "<a href='/signup'>Try Again</a>"
                )
            except Exception as exc:  # pragma: no cover - unexpected error path
                return f"<h1>Unexpected Error</h1><p>{str(exc)}</p>"

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

    @bp.route("/logout")
    def logout() -> Response:
        token = request.cookies.get("session_token")
        if token:
            Database.delete_session(token)
        response = make_response(redirect(url_for("auth.login")))
        response.delete_cookie("session_token")
        return response

    return bp
