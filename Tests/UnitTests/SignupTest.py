import unittest
import Pi.Database as db

class TestSignup(unittest.TestCase):

    def test_valid_login(self, username="TestAccount", password="TestAccount"):
        # Signup the test account
        db.add_user(username, password, role)

        # assert that {username} now has an entry in the database
        self.assertTrue(db.get_user(username))

        # assert the password in the database is the same as the provided password
        self.assertEqual(db.get_user_password(username), db.hash_password(password))

    def test_login_no_password_provided(self):
        self.assertEqual(add(-2, -3), -5)

    def test_login_no_username_provided(self):
        self.assertEqual(add(-2, 3), 1)

if __name__ == '__main__':
    unittest.main()
