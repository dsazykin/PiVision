import unittest
import os
import Pi.Database as db

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


if __name__ == "__main__":
    unittest.main()
