"""Web server entry point for the Pi deployment."""
from __future__ import annotations

from Pi.webserver import create_app
from dotenv import load_dotenv

app = create_app()

if __name__ == "__main__":  
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=True)
    load_dotenv()
