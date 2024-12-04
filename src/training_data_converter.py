import sqlite3

class TrainingDataConverter:
    
    def __init__(self, dataSetDirPath):
        """ Initialize converter with file path reference to directory"""
        self.dirPath = dataSetDirPath

    def imgToBinary(self, imagePath):
        """ translate given file path of image to binary """
        with open(imagePath, 'rb') as file:
            binaryFile = file.read()
        return binaryFile
    
    def buildDb(self, dbName):
        table = sqlite3.connect(dbName)
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