import unittest
import Pi.Database as db

username = "TestAccount"
password = "TestAccount"

class TestSignup(unittest.TestCase):

    def test_valid_signup(self):
        # Signup the test account
        db.add_user(username, password)

        # assert that {username} now has an entry in the users table
        self.assertTrue(db.get_user(username))

        # assert the password in the database is the same as the provided password
        self.assertEqual(db.get_user_password(username), db.hash_password(password))

    def test_has_session(self, username):
        # Signup the test account
        db.add_user(username, password)

        # assert that {username} has an entry in the sessions table
        user = db.get_user(username)
        db.create_session(user["user_id"])

        sessions = db.get_all_sessions()
        user_session = sessions.__contains__(username)
        self.assertTrue(user_session)


if __name__ == '__main__':
    unittest.main()
