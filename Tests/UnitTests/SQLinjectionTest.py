import unittest
import Pi.Database as db

class TestInjection(unittest.TestCase):

    def setUp(self):
        db.initialize_database()
        return super().setUp()


if __name__ == '__main__':
    unittest.main()
