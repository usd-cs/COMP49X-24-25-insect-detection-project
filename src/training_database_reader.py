""" training_database_reader.py """
import sqlite3
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class DatabaseReader:
    """
    Reads from SQLite database, and creates a pandas dataframe
    """
    def __init__(self, database, table="TrainingData", query=None):
        """
        Initialize DatabaseReader and loads data into a Pandas DataFrame. 
        
        Arguements:
            database (str): path to the SQLite database file.
            table (str): name of the table in the database to query.
            query (str): SQL query to execute(this is optional).
        """
        self.database = database
        self.table = table
        # if query is specified, set it to the specified, else set to default query if query=None
        self.query = query or f"SELECT Genus, Species, UniqueID, View FROM {self.table}"
        self.dataframe = self.load_data()

    def load_data(self):
        pass