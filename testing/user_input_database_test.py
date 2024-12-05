"""
Testing for user input database
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import sys
import os
import io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from user_input_database import user_input_database

class test_user_input_database(unittest.TestCase):

    def test_uuid_generator():
        """
        Test proper return of uuid
        """
        pass

    def test_add_image():
        pass

    def test_export_dbase():
        pass

    def test_init():
        pass