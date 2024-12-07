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
        """
        Loads data from the database using the provided query. If database is invalid or empty,
        this method returns an empty DataFrame.
        
        Returns: 
            pd.DataFrame: the queried data from the SQLite database as a Pandas DataFrame.
        """
        try:
            # connect to database specifiec by database instance
            with sqlite3.connect(self.database) as connection:
                # return DataFrame object
                return pd.read_sql_query(self.query, connection)
        except (sqlite3.DatabaseError, pd.io.sql.DatabaseError) as e:
            # if error is raised, then print error and return empty DataFrame
            print(f"Error reading database: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    database = input("Please input the file path of the SQLite database: ")
    reader = DatabaseReader(database)
    print("Process Completed")
