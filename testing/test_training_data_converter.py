""" test_training_data_converter.py """
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import sys
import os
import io
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
        tdc.db = 'test_database.db'
        tdc.build_db()
        mock_connect.assert_called_once_with('test_database.db')
        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    @patch('sqlite3.connect')
    def test_add_img_success(self, mock_connect):
        """ Tests the image is successfully added to database with proper format """
        conn = MagicMock()
        cursor = MagicMock()
        mock_connect.return_value = conn
        conn.cursor.return_value = cursor

        tdc = TrainingDataConverter("")
        tdc.db = 'test_database.db'

        # mock parameters
        data = ('TestGenus', 'TestSpecies', '12345', 'test_view')
        image_binary = b'test_binary_data'

        tdc.add_img(data, image_binary)

        # check that img is added to database
        cursor.execute.assert_called_once_with(
            '''
            INSERT INTO TrainingData (Genus, Species, UniqueID, View, Image) 
            VALUES (?, ?, ?, ?, ?)
            ''', data + (image_binary,)
        )

        # check for committing and closing db
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('training_data_converter.TrainingDataConverter.img_to_binary')
    @patch('training_data_converter.TrainingDataConverter.add_img')
    def test_valid_conversion(
        self, mock_add_img, mock_img_to_binary, mock_listdir, mock_path_exists
        ):
        """ Tests that valid conversion stores all images with labels """
        # mock directory path
        mock_path_exists.return_value = True

        #mock directory files
        mock_listdir.return_value = [
            'genus species 123 5XEXT view1.png', 
            'genus species 124 5XEXT view2.jpg'
        ]
        # mock binary data conversion
        mock_img_to_binary.return_value = b'binary_data'

        tdc = TrainingDataConverter('test_dir_path')
        tdc.conversion('test_database.db')

        # verify all images converted
        mock_img_to_binary.assert_any_call(
            os.path.join('test_dir_path', 'genus species 123 5XEXT view1.png')
            )
        mock_img_to_binary.assert_any_call(
            os.path.join('test_dir_path', 'genus species 124 5XEXT view2.jpg')
            )

        mock_add_img.assert_has_calls([
            call(('genus', 'species', '123', 'view1'), b'binary_data'),
            call(('genus', 'species', '124', 'view2'), b'binary_data')
        ])

    @patch('os.path.exists')
    def test_conversion_without_directory(self, mock_path_exists):
        """ Tests that function prints that directory doesn't exist if true """
        # mock that directory doesn't exist
        mock_path_exists.return_value = False

        tdc = TrainingDataConverter('None')

        # check output when running function
        captured_output = io.StringIO()
        sys.stdout = captured_output

        tdc.conversion('test_database.db')

        sys.stdout = sys.__stdout__

        self.assertIn('Directory, None, does not exist.', captured_output.getvalue())

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_conversion_with_invalid_file(self, mock_listdir, mock_path_exists):
        """ Tests that function prints that file is invalid if true """
        mock_path_exists.return_value = True
        mock_listdir.return_value = ['invalid.png']

        tdc = TrainingDataConverter('test_dir_path')

        # check output when running function
        captured_output = io.StringIO()
        sys.stdout = captured_output

        tdc.conversion('test_database.db')

        # Verify the invalid naming format message is printed
        self.assertIn('File, invalid.png, has invalid naming format.', captured_output.getvalue())

    def test_valid_name(self):
        """ Tests that parsing returns proper results with valid input """
        test_name = "data/Genus Species 12345 5XEXT side.jpg"
        expected = ("Genus", "Species", "12345", "side")
        tdc = TrainingDataConverter("")
        result = tdc.parse_name(test_name)
        self.assertEqual(result, expected)

    def test_invalid_name_too_short(self):
        """ Tests that parsing function returns None when given invalid input """
        test_name = "data/Genus Species"
        tdc = TrainingDataConverter("")
        result = tdc.parse_name(test_name)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
