"""Web server entry point for the Pi deployment."""
from __future__ import annotations

from Pi.webserver import create_app
from dotenv import load_dotenv
from waitress import serve

app = create_app()

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
    load_dotenv()
