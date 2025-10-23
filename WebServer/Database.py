import sqlite3, bcrypt

DB_PATH = "users.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():
    """Create all necessary tables if they don't exist."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Create User table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT UNIQUE NOT NULL,
            user_password TEXT NOT NULL,
            role TEXT CHECK(role IN ('user', 'admin')) NOT NULL DEFAULT 'user'
        )
        """)

        # Create Gesture mapping table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS gesture_mappings (
            gesture_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            gesture_name TEXT NOT NULL,
            mapped_action TEXT NOT NULL,
            duration TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, gesture_name)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            role TEXT CHECK(role IN ('user', 'admin')) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)

        # Default gestures: [mapped_action, duration]
        default_gestures = {
            "call": ["esc", "press"],
            "dislike": ["scroll_down", "hold"],
            "fist": ["delete", "press"],
            "four": ["tab", "press"],
            "like": ["scroll_up", "hold"],
            "mute": ["volume_toggle", "press"],
            "ok": ["enter", "press"],
            "one": ["left_click", "press"],
            "palm": ["space", "press"],
            "peace": ["winleft", "press"],
            "peace_inverted": ["alt", "hold"],
            "rock": ["w", "press"],
            "stop": ["mouse_up", "hold"],
            "stop_inverted": ["mouse_down", "hold"],
            "three": ["mouse_right", "hold"],
            "three2": ["mouse_left", "hold"],
            "two_up": ["right_click", "press"],
            "two_up_inverted": ["ctrl", "hold"]
        }

        # Insert defaults only if not already present
        for gesture, (action, duration) in default_gestures.items():
            cursor.execute("""
            INSERT OR IGNORE INTO gesture_mappings (user_id, gesture_name, mapped_action, duration)
            VALUES (NULL, ?, ?, ?)
            """, (gesture, action, duration))

        conn.commit()


def update_gesture_mapping(username, gesture_name, new_action, new_duration):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE gesture_mappings
            SET mapped_action = ?, duration = ?
            WHERE gesture_name = ?
              AND user_id = (SELECT user_id FROM users WHERE user_name = ?)
        """, (new_action, new_duration, gesture_name, username))
        conn.commit()


def reset_user_mappings(username):
    """Reset all a user's gesture mappings back to the system defaults."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            # Get user_id
            cursor.execute("SELECT user_id FROM users WHERE user_name = ?", (username,))
            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] reset_user_mappings: user {username} not found")
                return False
            user_id = result[0]

            # Delete all custom mappings
            cursor.execute("DELETE FROM gesture_mappings WHERE user_id = ?", (user_id,))
            conn.commit()

            # Copy defaults safely
            cursor.execute("""
                INSERT OR REPLACE INTO gesture_mappings (user_id, gesture_name, mapped_action, duration)
                SELECT ?, gesture_name, mapped_action, duration
                FROM gesture_mappings
                WHERE user_id IS NULL
            """, (user_id,))

            conn.commit()
            return True
    except Exception as e:
        print(f"[ERROR] reset_user_mappings failed: {e}")
        return False


def get_user_mappings(user_name):
    """Get all gesture mappings for a user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT g.gesture_name, g.mapped_action, g.duration
        FROM gesture_mappings g
        JOIN users u ON g.user_id = u.user_id
        WHERE u.user_name = ?
        """, (user_name,))
        rows = cursor.fetchall()
        return {gesture: (action, duration) for gesture, action, duration in rows}


def get_all_users():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_name, role FROM users")
        return cursor.fetchall()


def delete_user(user_name):
    with get_connection() as conn:
        cursor = conn.cursor()

        # Delete gesture mappings first
        cursor.execute("""
        DELETE FROM gesture_mappings
        WHERE user_id = (
            SELECT user_id FROM users WHERE user_name = ?
        )
        """, (user_name,))

        # Delete user
        cursor.execute("DELETE FROM users WHERE user_name = ?", (user_name,))
        deleted_rows = cursor.rowcount

        conn.commit()
        return deleted_rows


def get_user(username):
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_name=?", (username,))
        return cursor.fetchone()


def get_user_password(username):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_password FROM users WHERE user_name=?", (username,))
        row = cursor.fetchone()
        return row[0] if row else None


def add_user(user_name, user_password, role="user"):
    with get_connection() as conn:
        cursor = conn.cursor()

        # Prevent duplicates
        cursor.execute("SELECT user_id FROM users WHERE user_name = ?", (user_name,))
        if cursor.fetchone():
            raise ValueError(f"User '{user_name}' already exists.")

        # Hash password
        hashed_pw = bcrypt.hashpw(user_password.encode("utf-8"), bcrypt.gensalt())

        cursor.execute("""
        INSERT INTO users (user_name, user_password, role)
        VALUES (?, ?, ?)
        """, (user_name, hashed_pw, role))
        user_id = cursor.lastrowid

        # Copy default gestures with duration
        cursor.execute("""
        INSERT OR IGNORE INTO gesture_mappings (user_id, gesture_name, mapped_action, duration)
        SELECT ?, gesture_name, mapped_action, duration
        FROM gesture_mappings
        WHERE user_id IS NULL
        """, (user_id,))

        conn.commit()


def verify_user(username, password):
    """Check if username exists and password matches."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_password FROM users WHERE user_name=?", (username,))
        row = cursor.fetchone()
        if not row:
            return False
        stored_hash = row[0]
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash)
    
import secrets
from datetime import datetime, timedelta

# --- SESSION MANAGEMENT ---
def create_session(user_id, role):
    """Create a new session and return its token."""
    token = secrets.token_hex(32)
    expires_at = (datetime.utcnow() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO sessions (user_id, session_token, role, expires_at)
        VALUES (?, ?, ?, ?)
        """, (user_id, token, role, expires_at))
        conn.commit()
    return token

def verify_session(token, req_user_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        user_name = cursor.execute("SELECT user_name FROM sessions WHERE token=?", (token,))
        user_id = cursor.execute("SELECT user_id FROM users WHERE user_name=?", (user_name,))
    if (user_id == req_user_id):
        return True
    return False

def get_session(token):
    """Return session data if valid, else None."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
        SELECT s.*, u.user_name
        FROM sessions s
        JOIN users u ON s.user_id = u.user_id
        WHERE s.session_token = ? AND s.expires_at > CURRENT_TIMESTAMP
        """, (token,))
        return cursor.fetchone()
    
def get_all_sessions():
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
        SELECT s.session_id, s.session_token, s.role, s.created_at, s.expires_at,
               u.user_name
        FROM sessions s
        JOIN users u ON s.user_id = u.user_id
        ORDER BY s.created_at DESC
        """)
        return cursor.fetchall()



def delete_session(token):
    """Remove a specific session."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_token=?", (token,))
        conn.commit()


def cleanup_expired_sessions():
    """Remove all expired sessions."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP")
        conn.commit()
