import unittest
import os

import bcrypt

import Pi.Database as db

# From SignupTest.py
username_signup = "TestUsername"
password_signup = "TestPassword"
other_password_signup = "OtherTestPassword"

# From SQLinjectionTest.py
username_injection = "bob"
password_injection = "bobisCool!"
injection = "' OR 1=1; DELETE FROM users WHERE user_name= 'bob'"


class TestSignup(unittest.TestCase):

    def setUp(self):
        # full reset
        db.initialize_database()

        # remove "testusers" if in database.
        existing_user = db.get_user(username_signup)
        if existing_user:
            db.delete_user(username_signup)
        return super().setUp()

    def test_valid_signup(self):
        """Verify that signup creates a user and stores a hashed password."""
        db.add_user(username_signup, password_signup)

        # Assert that {username} now has an entry in the users table
        user = db.get_user(username_signup)
        self.assertIsNotNone(user, "User should exist after signup")

        # Assert the password stored is a bcrypt hash (not the plaintext)
        stored_pw = db.get_user_password(username_signup)
        self.assertNotEqual(stored_pw, password_signup)
        self.assertTrue(stored_pw.startswith(b"$2b$") or stored_pw.startswith(b"$2a$"),
                        "Stored password should be a bcrypt hash")

    def test_double_user(self):
        db.add_user(username_signup, password_signup)
        usertable_size = db.get_all_users()

        #try to add a user with the same username:
        try:
            db.add_user(username_signup, other_password_signup)
            newsize = db.get_all_users()
        except Exception:
            newsize = db.get_all_users()


        # Size of database should be the same because new user shouldn't be added.
        self.assertEqual(usertable_size, newsize)

        # Test if password of the user is still the same of the original user.
        db_pass = db.get_user_password(username_signup)
        self.assertTrue(bcrypt.checkpw(password_signup.encode("utf-8"), db_pass))

    def test_has_session(self):
        """Verify that creating a session actually adds it to the sessions table."""
        # Add the user
        db.add_user(username_signup, password_signup)
        user = db.get_user(username_signup)

        # Create a session for that user
        token = db.create_session(user["user_id"])
        self.assertIsNotNone(token, "create_session() should return a session token")

        # Retrieve all sessions
        sessions = db.get_all_sessions()
        self.assertGreater(len(sessions), 0, "There should be at least one session")

        # Check that one of the sessions belongs to our username
        user_has_session = any(row["user_name"] == username_signup for row in sessions)
        self.assertTrue(user_has_session, f"User {username_signup} should have a session")

        # verify that token exists in the table
        tokens = [row["session_token"] for row in sessions]
        self.assertIn(token, tokens, "Returned token should match one in database")

    def tearDown(self):
        """Clean up database and test data after each test."""
        try:
            # Delete the test user (and related mappings/sessions)
            db.delete_user(username_signup)
        except Exception as e:
            print(f"[Warning] Cleanup failed: {e}")

        super().tearDown()


class TestSQLInjection(unittest.TestCase):

    def setUp(self):
        # full reset
        db.initialize_database()

        # remove users and injections if in database.
        existing_user = db.get_user(username_injection)
        if existing_user:
            db.delete_user(username_injection)
        existing_injection = db.get_user(injection)
        if existing_injection:
            db.delete_user(injection)
        return super().setUp()

    def test_uname_injection(self):
        # Add bob to the database
        db.add_user(username_injection, password_injection)

        # add a 'user' to the database, but code injected to remove bob
        db.add_user(injection, password_injection)

        # The injected code should be an entry in the database.
        injection_user = db.get_user(injection)
        self.assertIsNotNone(injection_user, "User should exist after signup")

        # Bob should still be in the database/ code injection should fail.
        bob = db.get_user(username_injection)
        self.assertIsNotNone(bob, "Bob should still be in the database")

    def tearDown(self):
        # Remove test users if they exist
        try:
            db.delete_user(username_injection)
            db.delete_user(injection)
        except Exception as e:
            # Non-fatal; print a warning so it shows up in test logs if cleanup fails
            print(f"[Warning] Cleanup failed: {e}")

        super().tearDown()


class TestGestureMapping(unittest.TestCase):

    def setUp(self):
        # Ensure a clean database for each test
        if os.path.exists(db.DB_PATH):
            os.remove(db.DB_PATH)
        db.initialize_database()

        # Create two test users
        db.add_user("UserA", "pwA")
        db.add_user("UserB", "pwB")

        self.userA = db.get_user("UserA")
        self.userB = db.get_user("UserB")

    def test_correct_change(self):
        """
        Verify that:
        The user initially has default gestures assigned.
        Updating a gesture mapping changes the DB entry for that user.
        Only that user's values are altered.
        Default (NULL user_id) gesture mappings remain unchanged.
        """

        # Verify default gestures exist for userA and userB
        gestures_A_before = db.get_user_mappings_by_user_id(self.userA["user_id"])
        gestures_B_before = db.get_user_mappings_by_user_id(self.userB["user_id"])
        default_count = len(gestures_A_before)
        self.assertGreater(default_count, 0, "UserA should have default gestures assigned")
        self.assertEqual(default_count, len(gestures_B_before), "Both users should start with same default mappings")

        # Pick one gesture to modify
        gesture_to_update = list(gestures_A_before.keys())[0]
        old_action, old_duration = gestures_A_before[gesture_to_update]
        new_action = old_action + "_modified"
        new_duration = "double_" + old_duration

        # Update gesture mapping for UserA only
        db.update_gesture_mapping_by_user_id(
            self.userA["user_id"], gesture_to_update, new_action, new_duration
        )

        # Fetch mappings again
        gestures_A_after = db.get_user_mappings_by_user_id(self.userA["user_id"])
        gestures_B_after = db.get_user_mappings_by_user_id(self.userB["user_id"])

        # Verify UserA’s mapping changed
        updated_action, updated_duration = gestures_A_after[gesture_to_update]
        self.assertEqual(updated_action, new_action, "UserA's mapped_action should be updated")
        self.assertEqual(updated_duration, new_duration, "UserA's duration should be updated")

        # Verify UserB’s mappings stayed identical
        self.assertEqual(gestures_B_before, gestures_B_after, "UserB's mappings should remain unchanged")

        # Verify default (NULL user_id) gestures still exist
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM gesture_mappings WHERE user_id IS NULL")
            remaining_defaults = cur.fetchone()[0]
        self.assertGreater(remaining_defaults, 0, "Default gesture mappings must remain in database")

    def tearDown(self):
        """Remove the test users created during setup."""
        try:
            # Delete users and their mappings if they exist
            db.delete_user("UserA")
            db.delete_user("UserB")
        except Exception as e:
            print(f"[Warning] Cleanup failed: {e}")


if __name__ == "__main__":
    unittest.main()
