"""Routes for laptop server downloads."""
from __future__ import annotations

import os

from flask import Blueprint, abort, request, send_file, url_for

from ..middleware import SessionManager
from Pi.webserver.config.paths import CONNECTION_SOFTWARE_PATH


def create_blueprint(session_manager: SessionManager) -> Blueprint:
    bp = Blueprint("downloads", __name__)

    @bp.route("/download-software")
    def download_page():
        exists = os.path.exists(CONNECTION_SOFTWARE_PATH)
        status_message = (
            "PiVision Connection Software is available for download."
            if exists
            else "PiVision Connection Software could not be found on the server."
        )

        download_button = (
            f"<a href='{url_for('downloads.download_file')}'><button class='btn-primary'>Download PiVision Connection Software</button></a>"
            if exists
            else "<button class='btn-disabled' disabled>Download Unavailable</button>"
        )

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Download | PiVision</title>
            <link rel="stylesheet" href="{{{{ url_for('static', filename='css/style.css') }}}}">
            <style>
                body {{
                    background: #f3f4f6;
                    font-family: system-ui, sans-serif;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 16px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    text-align: center;
                    width: 100%;
                    max-width: 480px;
                }}
                .brand {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 12px;
                    margin-bottom: 24px;
                }}
                .brand-mark {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: linear-gradient(135deg, #4f46e5, var(--primary, #6366f1));
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    padding: 10px 16px;
                    color: white;
                    font-weight: 700;
                    font-size: 1.1rem;
                    letter-spacing: 0.5px;
                    white-space: nowrap;
                }}
                h1 {{
                    margin-bottom: 16px;
                    font-size: 1.4rem;
                    color: #1f2937;
                }}
                p {{
                    color: #4b5563;
                    margin-bottom: 24px;
                }}
                .btn-primary {{
                    background: linear-gradient(135deg, #4f46e5, #6366f1);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    cursor: pointer;
                    font-weight: 600;
                    transition: background 0.2s ease-in-out;
                }}
                .btn-primary:hover {{
                    background: linear-gradient(135deg, #4338ca, #4f46e5);
                }}
                .btn-disabled {{
                    background: #9ca3af;
                    color: #f9fafb;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    cursor: not-allowed;
                    font-weight: 600;
                }}
                .back-link {{
                    margin-top: 24px;
                    display: inline-block;
                }}
                .back-link button {{
                    background: #e5e7eb;
                    color: #111827;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    cursor: pointer;
                    font-weight: 600;
                    transition: background 0.2s;
                }}
                .back-link button:hover {{
                    background: #d1d5db;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="brand">
                    <div class="brand-mark">PiVision</div>
                </div>
                <h1>Download PiVision Connection Software</h1>
                <p>{status_message}</p>
                {download_button}
                <div class="back-link">
                    <a href="{url_for('main.index')}">
                        <button>Back to Home</button>
                    </a>
                </div>
            </div>
        </body>
        </html>
        """

    @bp.route("/download-software/file")
    def download_file():
        if not os.path.exists(CONNECTION_SOFTWARE_PATH):
            abort(404)
        return send_file(
            CONNECTION_SOFTWARE_PATH,
            as_attachment=True,
            download_name="PiVision Connection Software.exe",
        )

    return bp
