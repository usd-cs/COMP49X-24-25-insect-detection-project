"""post_eval_stack_training.py"""
import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split

class PostTrainingStacking:
    """
    Class that takes the trained models post training and uses a linear regression model
    to classify the outputs and predictions and combine them into a new output and prediction
    based on training created by evaluating test images
    """

    def __init__(self, dataframe):
        """
        Initializer that takes in the already trained models in evaluation mode
        in a list and sets up the new model and device
        """
        self.df = dataframe
        self.meta_model = None
        self.device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')

    def train_meta_model(self, label):
        """
        Trains meta model for the label ("Genus" or "Species") and saves
        new meta model to file Genus_meta.pth or Species_meta.pth

        Returns: None
        """
        #separate training data from classification (x,y)
        x = self.df.drop(columns=[label]).values
        y = self.df[label].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        #convert to tensors for torch
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)

        #determine shape based on input data
        input_dim = x_train.shape[1]
        output_dim = len(np.unique(y)) + 1
        self.meta_model = torch.nn.Linear(input_dim, output_dim)
        self.meta_model = self.meta_model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.01)

        num_epochs = 200
        for epoch in range(num_epochs):
            self.meta_model.train()
            outputs = self.meta_model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        self.meta_model.eval()
        with torch.no_grad():
            test_outputs = self.meta_model(x_test_tensor)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test_tensor).float().mean()
            print(f'PyTorch Logistic Regression Accuracy: {accuracy:.4f}')

        #save to src/models/label_meta.pth
        label_filename = label + "_meta.pth"
        meta_filename = os.path.join("src/models", label_filename)
        torch.save(self.meta_model.state_dict(), meta_filename)
