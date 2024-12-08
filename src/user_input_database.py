"""
Creates and manages the user-input database
"""

import sqlite3
import uuid


class UserInputDatabase:
    """
    Creates and manages the user input database. This database has two extra fields which are not
    needed in the training database. This database is also separate from the training database
    """

    def __init__(self, dataset_dir_path, backup_file=None):
        """
        Sets up the database when the class is called. Creates an empty dbase 
        if no input, initializes based on the previous version if given an input
        """
        self.dir_path = dataset_dir_path
        self.user_input_db = 'user_input_database.db'

        #if no data exists on startup as given by input, initialize empty db
        if backup_file is None:
            self.build_user_db()

        #initialize db via file contents if given a backup
        else:
            conn = sqlite3.connect(self.user_input_db)
            cursor = conn.cursor()

            with open(backup_file, "r", encoding="utf-8") as file:
                sql_contents = file.read()

            try:
                cursor.executescript(sql_contents)
            except sqlite3.Error as e:
                print(f"An error occured: {e}")

            conn.commit()
            conn.close()


    def img_to_binary(self, image_path):
        """
        **Taken from training data converter**

        Translate given file path of image to binary
        Return: binary file in bytes
        """
        with open(image_path, 'rb') as file:
            binary_file = file.read()
        return binary_file

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

    def add_image(self, image_data, image_binary):
        """
        Adds a new input image to the database
        Returns: None
        """
        table = sqlite3.connect(self.user_input_db)
        cursor = table.cursor()
        try:
            # ensure array contains correct amount of data
            if len(image_data) != 6:
                raise ValueError("image_data invalid, needs: [genus, species, unique_id, view]")

            #inserts data into the db
            cursor.execute('''
            INSERT INTO TrainingData (UserID, Genus, Species, UniqueID, View, Certainty, Image) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', image_data + [image_binary])

            table.commit()
            print(f"Inserted Image UniqueID: {image_data[3]}")

        #Check in case there is a duplicate uuid
        except sqlite3.IntegrityError:
            print(f"Image labeled, UniqueID {image_data[3]}, already exists.")
        finally:
            table.close()

    def export_dbase(self, filename = "user_input_db_backup.sql"):
        """
        Exports the dbase to be stored when called
        Returns: None
        """
        conn = sqlite3.connect(self.user_input_db)

        with open(filename, "w", encoding="utf-8") as file:
            for line in conn.iterdump():
                file.write(f"{line}\n")

        conn.close()
