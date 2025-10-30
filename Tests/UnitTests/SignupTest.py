import unittest
import Pi.Database as db

username = "TestUsername"
password = "TestPassword"
other_password = "OtherTestPassword"


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
        
    @unittest.expectedFailure    
    def test_double_user(self):
        db.add_user(username, password)
        usertable_size = db.get_all_usernames()

        #try to add a user with the same username:
        db.add_user(username, other_password)
        newsize = db.get_all_usernames()

        # Size of database should be the same because new user shouldn't be added.
        self.assertEqual(usertable_size, newsize)

        # Test if password of the user is still the same of the original user.
        db_pass = db.get_user_password(username)
        og_pass = db.hash_password(password)
        self.assertEqual(db_pass, og_pass)

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


if __name__ == '__main__':
    unittest.main()
