"""
Creates and manages the user-input database
"""

import sqlite3
import uuid


class user_input_database:

    def __init__(self, stored_dbase):
        """
        Sets up the database when the class is called. Creates an empty dbase 
        if no input, initializes based on the previous version if given an input
        """
        self.user_input_db = 'user_input_database.db'

    def build_user_db(self):
        """
        Initialize database for successful user inputs
        """
        table = sqlite3.connect(self.user_input_db)
        cursor = table.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS SuccessUserInput (
                UserID TEXT,
                Genus TEXT,
                Species TEXT,
                UniqueID TEXT PRIMARY KEY,
                View TEXT,
                Image BLOB,
                Certainty TEXT
            )
        ''')
        table.commit()
        table.close()
        
    def uuid_generator(self):
        """
        generates uuids for new images in the database
        Returns: new uuid
        """
        return uuid.uuid4()
    
    def add_image(self):
        """
        Adds a new input image to the database
        Returns: None
        """
        pass

    def export_dbase(self, filename):
        """
        Exports the dbase to be stored when called
        Returns: None
        """

