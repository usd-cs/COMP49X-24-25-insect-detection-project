import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from training_data_converter import TrainingDataConverter

class TestTrainingDataConverter(unittest.TestCase):

    def testImgToBinary(self):
        tdc = TrainingDataConverter("")
        # expected binary for mock file
        testBinary = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00'

        # call mock function with fake file path
        with patch('builtins.open', mock_open(read_data=testBinary)) as mockFile:
            res = tdc.imgToBinary('fake_image_path.png')
            self.assertEqual(res, testBinary)
            mockFile.assert_called_once_with('fake_image_path.png', 'rb')
        
    @patch('sqlite3.connect')
    def testBuildDb(self, mock_connect):
        conn = MagicMock()
        cursor = MagicMock()
        mock_connect.return_value = conn
        conn.cursor.return_value = cursor
        tdc = TrainingDataConverter("")
        tdc.buildDb('test_database.db')
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