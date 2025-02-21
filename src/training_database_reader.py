""" training_database_reader.py """
import os
import sys
import sqlite3
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class DatabaseReader:
    """
    Reads from SQLite database, and creates a pandas dataframe
    """
    def __init__(self, database, connection=None, table="TrainingData", query=None):
        """
        Initialize DatabaseReader and loads data into a Pandas DataFrame. 
        
        Arguements:
            database (str): path to the SQLite database file.
            connection (sqlite3.Connection): shared SQLite connection(this is optional).
            table (str): name of the table in the database to query.
            query (str): SQL query to execute(this is optional).
        """
        self.database = database
        self.connection = connection
        self.table = table
        # if query is specified, set it to the specified, else set to default query if query=None
        self.query = query or f"SELECT Genus, Species, UniqueID, View, Image FROM {self.table}"
        self.dataframe = self.load_data()

    def load_data(self):
        """
        Loads data from the database using the provided query. If database is invalid or empty,
        this method returns an empty DataFrame.
        
        Returns: 
            pd.DataFrame: the queried data from the SQLite database as a Pandas DataFrame.
        """
        try:
            if self.connection:
                # use the provided connection
                return pd.read_sql_query(self.query, self.connection)
            # create a new connection if none is provided
            with sqlite3.connect(self.database) as conn:
                return pd.read_sql_query(self.query, conn)
        except (sqlite3.DatabaseError, pd.io.sql.DatabaseError) as e:
            # if error is raised, then print error and return empty DataFrame
            print(f"Error reading database: {e}")
            return pd.DataFrame()

    def get_dataframe(self):
        """
        Simple getter method that returns the objects DataFrame
        """
        return self.dataframe

if __name__ == "__main__":
    db_path = input("Please input the file path of the SQLite database: ")
    reader = DatabaseReader(db_path)
    print("Process Completed")
