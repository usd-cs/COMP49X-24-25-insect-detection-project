""" training_data_converter.py """
import sqlite3

class TrainingDataConverter:
    """
    Converter from file directory of images to sqlite database
    """
    def __init__(self, dataset_dir_path):
        """ Initialize converter with file path reference to directory"""
        self.dir_path = dataset_dir_path

    def img_to_binary(self, image_path):
        """ translate given file path of image to binary """
        with open(image_path, 'rb') as file:
            binary_file = file.read()
        return binary_file

    def build_db(self, db_name):
        table = sqlite3.connect(db_name)
        cursor = table.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS TrainingData (
                Genus TEXT,
                Species TEXT,
                UniqueID TEXT PRIMARY KEY,
                View TEXT,
                Image BLOB
            )
        ''')
        table.commit()
        table.close()
