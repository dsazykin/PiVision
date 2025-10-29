import unittest
import Pi.Database as db

username = "bob"
password = "bobisCool!"

username_injection = "' OR 1=1; SELECT user_password FROM users WHERE user_name= 'bob'"
password_injection = "' OR 1=1; SELECT user_password FROM users WHERE user_name= 'bob'"

class TestInjection(unittest.TestCase):

# Inserting user "bob" into the database 
    def setUp(self):
        db.initialize_database()
        return super().setUp()

    def test_valid_signup(self):
        # Signup Bob's account
        db.add_user(username, password)

        # assert that {username} now has an entry in the users table
        self.assertTrue(db.get_user(username))

        # assert the password in the database is the same as the provided password
        self.assertEqual(db.get_user_password(username), db.hash_password(password))

    def test_has_session(self, username):
        # Signup Bob's account
        db.add_user(username, password)

        # assert that {username} has an entry in the sessions table
        user = db.get_user(username)
        db.create_session(user["user_id"])

        sessions = db.get_all_sessions()
        user_session = sessions.__contains__(username)
        self.assertTrue(user_session)
    
#  Inserting the injection code into the database 
    def test_valid_signup_injection(self):
        # Signup the injection account
        db.add_user(username_injection, password_injection)
        self.assertTrue(db.get_user(username_injection))
        self.assertEqual(db.get_user_password(username_injection), db.hash_password(password_injection))

    def test_injection_has_session(self, username_injection):
        # Signup the injection account
        db.add_user(username_injection, password_injection)
        user = db.get_user(username_injection)
        db.create_session(user["user_id"])

        sessions = db.get_all_sessions()
        user_session = sessions.__contains__(username_injection)
        self.assertTrue(user_session)
    
    def test_check_Bob(self):
        # We will now check if the SQL injection deleted our user Bob or if Bob is still part of the database. 
        self.assertTrue(db.get_user(username))

if __name__ == '__main__':
    unittest.main()
