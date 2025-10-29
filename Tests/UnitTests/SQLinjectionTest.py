import unittest
import Pi.Database as db

username = "bob"
password = "bobisCool!"

username_injection = "' OR 1=1; SELECT user_password FROM users WHERE user_name= 'bob'"
password_injection = "' OR 1=1; SELECT user_password FROM users WHERE user_name= 'bob'"

class TestSignup(unittest.TestCase):

    def setUp(self):
        # full reset
        db.initialize_database()

        # remove "testusers" if in database.
        existing_user = db.get_user(username)
        if existing_user:
            db.delete_user(username)
        return super().setUp()

    def test_valid_signup(self):
        """Verify that signup creates a user and stores a hashed password."""
        db.add_user(username, password)

        # Assert that {username} now has an entry in the users table
        user = db.get_user(username)
        self.assertIsNotNone(user, "User should exist after signup")

        # Assert the password stored is a bcrypt hash (not the plaintext)
        stored_pw = db.get_user_password(username)
        self.assertNotEqual(stored_pw, password)
        self.assertTrue(stored_pw.startswith(b"$2b$") or stored_pw.startswith(b"$2a$"),
                        "Stored password should be a bcrypt hash")

    def test_has_session(self):
        """Verify that creating a session actually adds it to the sessions table."""
        # Add the user
        db.add_user(username, password)
        user = db.get_user(username)

        # Create a session for that user
        token = db.create_session(user["user_id"])
        self.assertIsNotNone(token, "create_session() should return a session token")

        # Retrieve all sessions
        sessions = db.get_all_sessions()
        self.assertGreater(len(sessions), 0, "There should be at least one session")

        # Check that one of the sessions belongs to our username
        user_has_session = any(row["user_name"] == username for row in sessions)
        self.assertTrue(user_has_session, f"User {username} should have a session")

        # verify that token exists in the table
        tokens = [row["session_token"] for row in sessions]
        self.assertIn(token, tokens, "Returned token should match one in database")
        

# We will now create another "account". This will be an SQL injection attempting to remove the user bob
    def test_injection_valid_signup(self):
        db.add_user(username_injection, password)

        injection = db.get_user(username_injection)
        self.assertIsNotNone(injection, "User should exist after signup")

        stored_pw = db.get_user_password(username_injection)
        self.assertNotEqual(stored_pw, password_injection)
        self.assertTrue(stored_pw.startswith(b"$2b$") or stored_pw.startswith(b"$2a$"),
                        "Stored password should be a bcrypt hash")

    def test_injection_has_session(self):

        db.add_user(username_injection, password_injection)
        user = db.get_user(username_injection)

        token = db.create_session(user["user_id"])
        self.assertIsNotNone(token, "create_session() should return a session token")

        sessions = db.get_all_sessions()
        self.assertGreater(len(sessions), 0, "There should be at least one session")

        injection_has_session = any(row["user_name"] == username_injection for row in sessions)
        self.assertTrue(injection_has_session, f"User {username_injection} should have a session")

        tokens = [row["session_token"] for row in sessions]
        self.assertIn(token, tokens, "Returned token should match one in database")

    def test_check_Bob(self):
        # We will now check if the SQL injection deleted our user Bob or if Bob is still part of the database. 
        self.assertTrue(db.get_user(username))

if __name__ == '__main__':
    unittest.main()
