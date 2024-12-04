""" test_training_data_converter.py """
import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from training_data_converter import TrainingDataConverter

class TestTrainingDataConverter(unittest.TestCase):
    """
    Unit testing for training data converter
    """
    def test_img_to_binary(self):
        """ Test that img properly is stored as binary value """
        tdc = TrainingDataConverter("")
        # expected binary for mock file
        test_binary = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00'

        # call mock function with fake file path
        with patch('builtins.open', mock_open(read_data=test_binary)) as mock_file:
            res = tdc.img_to_binary('fake_image_path.png')
            self.assertEqual(res, test_binary)
            mock_file.assert_called_once_with('fake_image_path.png', 'rb')

    @patch('sqlite3.connect')
    def test_build_db(self, mock_connect):
        """ Test that database is properly built with respective columns """
        conn = MagicMock()
        cursor = MagicMock()
        mock_connect.return_value = conn
        conn.cursor.return_value = cursor
        tdc = TrainingDataConverter("")
        tdc.build_db('test_database.db')
        mock_connect.assert_called_once_with('test_database.db')
        cursor.execute.assert_called_once_with('''
            CREATE TABLE IF NOT EXISTS TrainingData (
                Genus TEXT,
                Species TEXT,
                UniqueID TEXT PRIMARY KEY,
                View TEXT,
                Image BLOB
            )
        ''')
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()
