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
        pass
        
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

