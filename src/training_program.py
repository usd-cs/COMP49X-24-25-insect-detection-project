""" training_program.py """
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class TrainingProgram:
    """
    Reads 4 subsets of pandas database from DatabaseReader, and trains and saves 4 models
    according to their respective image angles.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # subsets to save database reading to
        self.caud_subset = self.get_subset("CAUD", self.dataframe)
        self.dors_subset = self.get_subset("DORS", self.dataframe)
        self.fron_subset = self.get_subset("FRON", self.dataframe)
        self.late_subset = self.get_subset("LATE", self.dataframe)

    def get_subset(self, view_type, dataframe):
        """
        Reads database and pulls subset where View column is equal to parameter, view_type
        
        Args: view_type (string): View type column value (e.g., 'CAUD', 'DORS', 'FRON', 'LATE')
       
        Return: pd.DataFrame: Subset of database if column value valid, otherwise empty dataframe
        """
        return dataframe[dataframe["View"] == view_type] if not dataframe.empty else pd.DataFrame()

    def get_caudal_view(self):
        """
        Getter method for caudal view
        Return: previously read caudal subset
        """
        return self.caud_subset

    def get_dorsal_view(self):
        """
        Getter method for dorsal view
        Return: previously read dorsal subset
        """
        return self.dors_subset

    def get_frontal_view(self):
        """
        Getter method for frontal view
        Return: previously read frontal subset
        """
        return self.fron_subset

    def get_lateral_view(self):
        """
        Getter method for lateral view
        Return: previously read lateral subset
        """
        return self.late_subset
