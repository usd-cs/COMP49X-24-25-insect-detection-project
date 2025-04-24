""" training_program.py """
import os
import sys
import json
from io import BytesIO
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformation_classes import HistogramEqualization

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, unspecified-encoding too-many-public-methods
class TrainingProgram:
    """
    Reads 4 subsets of pandas database from DatabaseReader, and trains and saves 4 models
    according to their respective image angles.
    """
    def __init__(self, dataframe, class_column , num_classes):
        """
        Initialize dataset, image height, and individual model training
        """
        self.dataframe = dataframe
        self.height = 300
        self.num_classes = num_classes
        # subsets to save database reading to
        self.caud_subset = self.get_subset("CAUD", self.dataframe)
        self.dors_subset = self.get_subset("DORS", self.dataframe)
        self.fron_subset = self.get_subset("FRON", self.dataframe)
        self.late_subset = self.get_subset("LATE", self.dataframe)
        # Set device to a CUDA-compatible gpu, else mps, else cpu
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_built()
            else 'cpu')
        self.caud_model = self.load_caud_model()
        self.dors_model = self.load_dors_model()
        self.fron_model = self.load_fron_model()
        self.late_model = self.load_late_model()
        # Dictionary variables
        self.class_column = class_column
        self.class_index_dictionary = {}
        self.class_string_dictionary = {}
        self.class_set = set()
        # model accuracy dictionary
        self.model_accuracies = {
            "caud" : 0,
            "dors" : 0,
            "fron" : 0,
            "late" : 0
        }

        classes = dataframe.iloc[:, self.class_column].values
        class_to_idx = {label: idx for idx, label in enumerate(sorted(set(classes)))}
        for class_values in classes:
            if class_to_idx[class_values] not in self.class_set:
                self.class_index_dictionary[class_to_idx[class_values]] = class_values
                self.class_string_dictionary[class_values] = class_to_idx[class_values]
                self.class_set.add(class_to_idx[class_values])

        # Create transformation method dictionary
        self.transformations = {
            "caud": transforms.Compose([
        transforms.Resize((self.height, self.height)),
        transforms.ToTensor(),
        HistogramEqualization(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            "dors": transforms.Compose([
        transforms.Resize((self.height, self.height)),
        transforms.ToTensor(),
        HistogramEqualization(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            "fron": transforms.Compose([
        transforms.Resize((self.height, self.height)),
        transforms.ToTensor(),
        HistogramEqualization(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            "late": transforms.Compose([
        transforms.Resize((self.height, self.height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        }

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

    def get_train_test_split(self, df):
        """
        Gets train and test split for given dataframe
        Returns: List of train and test data
        """
        image_binaries = df.iloc[:, -1].values
        classes = df.iloc[:, self.class_column].values
        labels = [self.class_string_dictionary[label] for label in classes]
        # Split subset into training and testing sets
        # x: images, y: species
        train_x, test_x, train_y, test_y = train_test_split(
        image_binaries, labels, test_size=0.2, random_state=42)
        return [train_x, test_x, train_y, test_y]

    def training_evaluation_caudal(self, num_epochs, train_loader, test_loader):
        """
        Code for training algorithm and evaluating model
        """
        # Model Training
        self.caud_model.train()
         # define loss function, optimization function, and image transformation
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.caud_model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Clear previous gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.caud_model(inputs)
                loss = criterion(outputs, labels)

                # Backpropagation pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        # evaluate testing machine
        self.caud_model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.caud_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        if total != 0:
            accuracy = correct / total
            print(f"Accuracy: {100 * accuracy:.2f}%")
            # Compute and print F1 score
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            self.model_accuracies["caud"] = f1
            print(f"Weighted F1 Score: {100 * f1:.2f}%")

    def training_evaluation_dorsal(self, num_epochs, train_loader, test_loader):
        """
        Code for training algorithm and evaluating model
        """
        # Model Training
        self.dors_model.train()
         # define loss function, optimization function, and image transformation
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.dors_model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Clear previous gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.dors_model(inputs)
                loss = criterion(outputs, labels)

                # Backpropagation pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        # evaluate testing machine
        self.dors_model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.dors_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if total != 0:
            accuracy = correct / total
            print(f"Accuracy: {100 * accuracy:.2f}%")
            # Compute and print F1 score
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            self.model_accuracies["dors"] = f1
            print(f"Weighted F1 Score: {100 * f1:.2f}%")

    def training_evaluation_frontal(self, num_epochs, train_loader, test_loader):
        """
        Code for training algorithm and evaluating model
        """
        # Model Training
        self.fron_model.train()
         # define loss function, optimization function, and image transformation
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.fron_model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Clear previous gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.fron_model(inputs)
                loss = criterion(outputs, labels)

                # Backpropagation pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        # evaluate testing machine
        self.fron_model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.fron_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if total != 0:
            accuracy = correct / total
            print(f"Accuracy: {100 * accuracy:.2f}%")
            # Compute and print F1 score
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            self.model_accuracies["fron"] = f1
            print(f"Weighted F1 Score: {100 * f1:.2f}%")

    def training_evaluation_lateral(self, num_epochs, train_loader, test_loader):
        """
        Code for training algorithm and evaluating model
        """
        # Model Training
        self.late_model.train()
         # define loss function, optimization function, and image transformation
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.late_model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Clear previous gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.late_model(inputs)
                loss = criterion(outputs, labels)

                # Backpropagation pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        # evaluate testing machine
        self.late_model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.late_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if total != 0:
            accuracy = correct / total
            print(f"Accuracy: {100 * accuracy:.2f}%")
            # Compute and print F1 score
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            self.model_accuracies["late"] = f1
            print(f"Weighted F1 Score: {100 * f1:.2f}%")

    def train_caudal(self, num_epochs):
        """
        Trains model with subset of caudal image views
        and save model to respective save file.
        Return: None
        """
        # Get training and testing data
        train_x, test_x, train_y, test_y = self.get_train_test_split(self.get_caudal_view())
        # Define image transformations, placeholder for preprocessing
        transformation = self.transformations["caud"]

        # Create DataLoaders
        train_dataset = ImageDataset(train_x, train_y, transform=transformation)
        test_dataset = ImageDataset(test_x, test_y, transform=transformation)
        training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        testing_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.training_evaluation_caudal(num_epochs, training_loader, testing_loader)

    def train_dorsal(self, num_epochs):
        """
        Trains model with subset of dorsal image views
        and save model to respective save file.
        Return: None
        """
        # Get training and testing data
        train_x, test_x, train_y, test_y = self.get_train_test_split(self.get_dorsal_view())
        # Define image transformations, placeholder for preprocessing
        transformation = self.transformations["dors"]

        # Create DataLoaders
        train_dataset = ImageDataset(train_x, train_y, transform=transformation)
        test_dataset = ImageDataset(test_x, test_y, transform=transformation)
        training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        testing_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.training_evaluation_dorsal(num_epochs, training_loader, testing_loader)

    def train_frontal(self, num_epochs):
        """
        Trains model with subset of frontal image views
        and save model to respective save file.
        Return: None
        """
        # Get training and testing data
        train_x, test_x, train_y, test_y = self.get_train_test_split(self.get_frontal_view())
        # Define image transformations, placeholder for preprocessing
        transformation = self.transformations["fron"]

        # Create DataLoaders
        train_dataset = ImageDataset(train_x, train_y, transform=transformation)
        test_dataset = ImageDataset(test_x, test_y, transform=transformation)
        training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        testing_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.training_evaluation_frontal(num_epochs, training_loader, testing_loader)

    def train_lateral(self, num_epochs):
        """
        Trains model with subset of lateral image views
        and save model to respective save file.
        Return: None
        """
        # Get training and testing data
        train_x, test_x, train_y, test_y = self.get_train_test_split(self.get_lateral_view())
        # Define image transformations, placeholder for preprocessing
        transformation = self.transformations["late"]

        # Create DataLoaders
        train_dataset = ImageDataset(train_x, train_y, transform=transformation)
        test_dataset = ImageDataset(test_x, test_y, transform=transformation)
        training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        testing_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.training_evaluation_lateral(num_epochs, training_loader, testing_loader)

    def load_caud_model(self):
        """
        Loads model to be trained and saved for caudal image views
        Return: ResNet model
        """
        model = models.resnet50()
        num_features = model.fc.in_features
        # number of classifications tentative
        model.fc = torch.nn.Linear(num_features, self.num_classes)
        model = model.to(self.device)

        return model

    def load_dors_model(self):
        """
        Loads model to be trained and saved for dorsal image views
        Return: ResNet model
        """
        model = models.resnet50()
        num_features = model.fc.in_features
        # number of classifications tentative
        model.fc = torch.nn.Linear(num_features, self.num_classes)
        model = model.to(self.device)

        return model

    def load_fron_model(self):
        """
        Loads model to be trained and saved for frontal image views
        Return: ResNet model
        """
        model = models.resnet50()
        num_features = model.fc.in_features
        # number of classifications tentative
        model.fc = torch.nn.Linear(num_features, self.num_classes)
        model = model.to(self.device)

        return model

    def load_late_model(self):
        """
        Loads model to be trained and saved for lateral image views
        Return: ResNet model
        """
        model = models.resnet50()
        num_features = model.fc.in_features
        # number of classifications tentative
        model.fc = torch.nn.Linear(num_features, self.num_classes)
        model = model.to(self.device)

        return model

    def save_models(self, model_filenames = None, height_filename = None,
                    class_dict_filename = None, accuracy_dict_filename = None):
        """
        Saves trained models to their respective files and image height file
        
        Returns: None
        """

        # Update/Initialize Model Accuracy Dictionary
        # update_flags indicates which models weights need to be updated and saved
        update_flags = self.update_accuracies(accuracy_dict_filename)

        if "caud" in model_filenames and model_filenames["caud"] and update_flags["caud"]:
            caud_file = model_filenames["caud"]
            caud_filename = os.path.join("src/models", caud_file)
            torch.save(self.caud_model.state_dict(), caud_filename)
            self.save_transformation(self.transformations["caud"], 0)
            print(f"Caudal Model weights saved to {caud_filename}")

        if "dors" in model_filenames and model_filenames["dors"] and update_flags["dors"]:
            dors_file = model_filenames["dors"]
            dors_filename = os.path.join("src/models", dors_file)
            torch.save(self.dors_model.state_dict(), dors_filename)
            self.save_transformation(self.transformations["dors"], 1)
            print(f"Dorsal Model weights saved to {dors_filename}")

        if "fron" in model_filenames and model_filenames["fron"] and update_flags["fron"]:
            fron_file = model_filenames["fron"]
            fron_filename = os.path.join("src/models", fron_file)
            torch.save(self.fron_model.state_dict(), fron_filename)
            self.save_transformation(self.transformations["fron"], 2)
            print(f"Frontal Model weights saved to {fron_filename}")

        if "late" in model_filenames and model_filenames["late"] and update_flags["late"]:
            late_file = model_filenames["late"]
            late_filename = os.path.join("src/models", late_file)
            torch.save(self.late_model.state_dict(), late_filename)
            self.save_transformation(self.transformations["late"], 3)
            print(f"Lateral Model weights saved to {late_filename}")

        # Handle dict_filename similarly if needed
        if class_dict_filename:
            class_dict_filename = os.path.join("src/models", class_dict_filename)
            with open(class_dict_filename, "w") as file:
                json.dump(self.class_index_dictionary, file, indent=4)
            print(f"Dictionary saved to {class_dict_filename}")

        if height_filename:
            height_filename = os.path.join("src/models", height_filename)
            with open(height_filename, "w") as file:
                file.write(str(self.height))
            print(f"Height saved to {height_filename}.")

    def update_accuracies(self, accuracy_dict_filename = None):
        """
        Reads in the previously saved model accuracies(if exists), and updates and saves 
        accuracy dictionary if accuracies increased during training. If model accuracies
        dictionary does not exist, then it initializes with training values or 0 if that
        model was not trained.
        
        Returns: update_flags - dictionary that tracks which models should update their weights
        """

        model_names = ["caud", "dors", "fron", "late"]
        update_flags = {}
        try:
            with open(accuracy_dict_filename, 'r') as f:
                accuracy_dict = json.load(f)

            for model in model_names:
                accuracy = accuracy_dict.get(model, 0)
                if accuracy < self.model_accuracies[model]:
                    # accuracy from most recent train is better than saved, so update
                    update_flags[model] = True
                    print(f"Updated Accuracy in Dictionary - Improved for {model} model.")
                elif accuracy >= self.model_accuracies[model]:
                    # accuracy did not improve from previously saved accuracy
                    self.model_accuracies[model] = accuracy
                    update_flags[model] = False
                    print(f"No Improvement to Accuracy for {model} model.")

        except FileNotFoundError:
            for model in model_names:
                update_flags[model] = True
            print(f"Accuracy File Not Found - Initializing at {accuracy_dict_filename}")

        with open(accuracy_dict_filename, "w") as file:
            json.dump(self.model_accuracies, file, indent=4)
        print(f"Model accuracies saved to {accuracy_dict_filename}.")

        return update_flags

    def save_transformation(self, transformation, angle):
        """
        Takes a transformation and angle input and saves the transformation
        to a file for use in the evaluation method program

        Returns: None
        """
        if angle == 0:
            with open("caud_transformation.pth", "wb") as f:
                dill.dump(transformation, f)

        elif angle == 1:
            with open("dors_transformation.pth", "wb") as f:
                dill.dump(transformation, f)

        elif angle == 2:
            with open("fron_transformation.pth", "wb") as f:
                dill.dump(transformation, f)

        elif angle == 3:
            with open("late_transformation.pth", "wb") as f:
                dill.dump(transformation, f)

# Custom Dataset class for loading images from binary data
class ImageDataset(Dataset):
    """
    Dataset class structure to hold image, transformation,
    and species label
    Arguments:
        image_binaries (0'b): image file in binary values
        label (str): species label of image
        transform (transforms.Compose): transform of image to be able 
        to input into model
    """
    def __init__(self, image_binaries, label, transform=None):
        """
        Initialize values
        """
        self.image_binaries = image_binaries
        self.label = torch.tensor(label, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        """
        Return: length of image binary data
        """
        return len(self.image_binaries)

    def __getitem__(self, idx):
        """
        Return: image and respective label
        """
        image_binary = self.image_binaries[idx]
        image = Image.open(BytesIO(image_binary))

        if self.transform:
            image = self.transform(image)

        return image, self.label[idx]
