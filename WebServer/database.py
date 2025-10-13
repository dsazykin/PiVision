import sqlite3

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
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, gesture_name)
        )
        """)

        # Insert default gestures (for user_id = NULL)
        default_gestures = {
            "call": "Esc",
            "dislike": "Scroll down",
            "fist": "Delete",
            "four": "Tab",
            "like": "Scroll up",
            "mute": "Toggle sound on/off",
            "ok": "Enter",
            "one": "Left click",
            "palm": "Space",
            "peace": "Windows key",
            "peace_inverted": "Alt",
            "rock": "w",
            "stop": "mouse_up",
            "stop_inverted": "mouse_down",
            "three": "mouse_right",
            "three2": "mouse_left",
            "two_up": "Right click",
            "two_up_inverted": "Ctrl"
        }

        # Insert only if not already present (default mappings have user_id IS NULL)
        for gesture, action in default_gestures.items():
            cursor.execute("""
            INSERT OR IGNORE INTO gesture_mappings (user_id, gesture_name, mapped_action)
            VALUES (NULL, ?, ?)
            """, (gesture, action))

        conn.commit()


def add_user(user_name, user_password, role="user"):
    with get_connection() as conn:
        cursor = conn.cursor()

        # Prevent duplicate usernames
        cursor.execute("SELECT user_id FROM users WHERE user_name = ?", (user_name,))
        if cursor.fetchone():
            raise ValueError(f"User '{user_name}' already exists.")

        # Add new user
        cursor.execute("""
        INSERT INTO users (user_name, user_password, role)
        VALUES (?, ?, ?)
        """, (user_name, user_password, role))
        user_id = cursor.lastrowid

        # Copy default gestures safely
        cursor.execute("""
        INSERT OR IGNORE INTO gesture_mappings (user_id, gesture_name, mapped_action)
        SELECT ?, gesture_name, mapped_action
        FROM gesture_mappings
        WHERE user_id IS NULL
        """, (user_id,))

        conn.commit()



def add_or_update_mapping(user_name, gesture_name, mapped_action):
    """Update a specific gesture mapping for a user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # Get user_id
        cursor.execute("SELECT user_id FROM users WHERE user_name = ?", (user_name,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"User '{user_name}' not found.")
        user_id = row[0]

        # Insert or update mapping
        cursor.execute("""
        INSERT INTO gesture_mappings (user_id, gesture_name, mapped_action)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, gesture_name)
        DO UPDATE SET mapped_action = excluded.mapped_action
        """, (user_id, gesture_name, mapped_action))
        conn.commit()


def get_user_mappings(user_name):
    """Get all gesture mappings for a user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT g.gesture_name, g.mapped_action
        FROM gesture_mappings g
        JOIN users u ON g.user_id = u.user_id
        WHERE u.user_name = ?
        """, (user_name,))
        rows = cursor.fetchall()
        return {gesture: action for gesture, action in rows}


def get_all_users():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_name, role FROM users")
        return cursor.fetchall()
