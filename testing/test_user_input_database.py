"""
Testing for user input database
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from user_input_database import UserInputDatabase

class TestUserInputDatabase(unittest.TestCase):
    """
    Unit testing for the user input database
    """

    def test_uuid_generator_format(self):
        """
        Test proper return and format of uuid
        """
        db = UserInputDatabase('')
        generated = db.uuid_generator()
        try:
            # Try parsing the UUID to ensure it's valid
            uuid_obj = uuid.UUID(str(generated))
            assert str(uuid_obj) == str(generated), "Invalid UUID format"
            print("Test passed: UUID format is valid")
        except ValueError:
            assert False, "Generated UUID is invalid"

    def test_uuid_generator_uniqueness(self):
        """
        Test proper uniqueness of uuid
        """
        db = UserInputDatabase('')
        uuids = {db.uuid_generator() for _ in range(1000)}  # Generate 1000 UUIDs
        assert len(uuids) == 1000, "UUIDs are not unique"
        print("Test passed: UUIDs are unique")

    @patch('sqlite3.connect')
    def test_add_image(self, mock_connect):
        """ Tests the image is successfully added to database with proper format """
        conn = MagicMock()
        cursor = MagicMock()
        mock_connect.return_value = conn
        conn.cursor.return_value = cursor

        tdc = UserInputDatabase('')

        # mock parameters
        data = ['TestID', 'TestGenus', 'TestSpecies', '12345', 'test_view', 'test_certainty']
        image_binary = b'test_binary_data'

        tdc.add_image(data, image_binary)

        # check that img is added to database
        cursor.execute.assert_called_with(
            '''
            INSERT INTO TrainingData (UserID, Genus, Species, UniqueID, View, Certainty, Image) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', data + [image_binary]
        )

        # check for committing and closing db
        conn.commit.assert_called()
        conn.close.assert_called()


    @patch('sqlite3.connect')
    def test_init(self, mock_connect):
        """ Test that database is properly built with respective columns """
        conn = MagicMock()
        cursor = MagicMock()
        mock_connect.return_value = conn
        conn.cursor.return_value = cursor
        UserInputDatabase('')

        mock_connect.assert_called_once_with('user_input_database.db')
        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()
        conn.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
