import unittest
import Pi.Database as db

username = "bob"
password = "bobisCool!"

injection = "' OR 1=1; DELETE FROM users WHERE user_name= 'bob'"


class TestSignup(unittest.TestCase):

    def setUp(self):
        # full reset
        db.initialize_database()

        # remove users and injections if in database.
        existing_user = db.get_user(username)
        if existing_user:
            db.delete_user(username)
        existing_injection = db.get_user(injection)
        if existing_injection:
            db.delete_user(injection)
        return super().setUp()


# We will now create another "account". This will be an SQL injection attempting to remove the user bob
    def test_uname_injection(self):
        # Add bob to the database
        db.add_user(username, password)

        # add a 'user' to the database, but code injected to remove bob
        db.add_user(injection, password)

        # The injected code should be an entry in the database.
        injection_user = db.get_user(injection)
        self.assertIsNotNone(injection_user, "User should exist after signup")

        # Bob should still be in the database/ code injection should fail.
        bob = db.get_user(username)
        self.assertIsNotNone(bob, "Bob should still be in the database")

    def tearDown(self):
        # Remove test users if they exist
        try:
            db.delete_user(username)
            db.delete_user(injection)
        except Exception as e:
            # Non-fatal; print a warning so it shows up in test logs if cleanup fails
            print(f"[Warning] Cleanup failed: {e}")

        super().tearDown()


if __name__ == '__main__':
    unittest.main()
