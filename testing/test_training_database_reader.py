import unittest
import sqlite3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from training_database_reader import DatabaseReader

class TestDatabaseReader(unittest.TestCase):
    """
    Unit testing for training database reader
    """
    @classmethod
    def setUpClass(cls):
        """
        Mock a temporary SQLite database for testing DatabaseReader class
        """
        cls.test_db = "test_database.db"

        # connect to the database
        connection = sqlite3.connect(cls.test_db)
        cursor = connection.cursor()

        # initialize the test table with the same expected format
        cursor.execute("""
            CREATE TABLE TrainingData (
                Genus TEXT,
                Species TEXT,
                UniqueID TEXT PRIMARY KEY,
                View TEXT
            )
        """)

        # insert sample data
        sample_data = [
            ("GenusA", "SpeciesA", "ID1", "View1"),
            ("GenusB", "SpeciesB", "ID2", "View2"),
            ("GenusC", "SpeciesC", "ID3", "View3"),
        ]
        cursor.executemany("""
            INSERT INTO TrainingData (Genus, Species, UniqueID, View)
            VALUES (?, ?, ?, ?)
        """, sample_data)

        connection.commit()
        connection.close()

    @classmethod
    def tearDownClass(cls):
        """
        Clean up by removing the temporary database.
        """
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_load_valid_data(self):
        """
        Test that DatabaseReader loads data from a valid database and table.
        Use correctly formatted mocked database.
        """
        # create DatabaseReader obejct with mocked database
        reader = DatabaseReader(self.test_db)
        df = reader.dataframe

        # assert the DataFrame contents(number of rows, columns exist, and data is expected)
        self.assertEqual(len(df), 3)
        self.assertIn("Genus", df.columns)
        self.assertEqual(df.iloc[0]["Genus"], "GenusA")

    def test_custom_query(self):
        """
        Test DatabaseReader using a custom SQL query.
        """
        # create DatabaseReader object with example query
        query = "SELECT Genus, Species FROM TrainingData WHERE Genus = 'GenusA'"
        reader = DatabaseReader(self.test_db, query=query)
        df = reader.dataframe

        # assert the DataFrame contents(number of rows and correct data)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Genus"], "GenusA")

    def test_invalid_table(self):
        """
        Test handling of reading from a non-existent table.
        """
        # create DatabaseReader object with an invalid table name 
        reader = DatabaseReader(self.test_db, table="InvalidTable")
        df = reader.dataframe

        # assert empty dataframe with returned(meaning error was raised while loading dataframe)
        self.assertTrue(df.empty)

    def test_invalid_database(self):
        """
        Test handling of an invalid database path.
        """

        reader = DatabaseReader("fake.db")
        df = reader.dataframe

        # verify an empty DataFrame is returned(meaning error was raised while loading dataframe)
        self.assertTrue(df.empty)

    def test_empty_table(self):
        """
        Test DatabaseReader correctly handles an empty table.
        """
        # create an empty table
        connection = sqlite3.connect(self.test_db)
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS EmptyTable (Column1 TEXT)")
        connection.commit()
        connection.close()

        # create reader object using the empty table
        reader = DatabaseReader(self.test_db, table="EmptyTable")
        df = reader.dataframe

        # assert an empty DataFrame is returned(meaning error was raised while loading dataframe)
        self.assertTrue(df.empty)

if __name__ == "__main__":
    unittest.main()