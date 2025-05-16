# COMP49X-24-25-insect-detection-project

Convolutional Neural Network repository for training and testing seed beetle image classification models.

## Description

### Features 3 training programs

#### training_program.py: 
Our current implementation for the web app. Builds a seperate model for each 4 image angle. Has a parameter setting for training species or genus.

#### alt_training_program.py: 
Alternate models trained in the same way, but combining multiple image angles together (lateral and caudal, lateral and dorsal, all).

#### post_eval_stack_training.py: 
Stacking model training for the combined outputs of the current implementation models.

#### transformation_classes.py: 
Contains pre-processing transformation classes to be used in both training and evaluation.

### Data Conversion/Loading

#### training_data_converter.py: 
Data converter that filters and transfers image data provided by Dr. Morse into an sqlite3 database

#### training_database_reader.py: 
Data reader that reads sqlite3 database into a pandas dataframe.

#### stack_dataset_creator.py: 
Modifies default dataframe to be able to input into stacking model for training and evaluation.

#### model_loader.py: 
Loads currently saved models in the models repository.

#### user_input_database.py: 
Currently unused. Program to create database that pulls in user-submitted images in order to grow the main training database.

### Model Evaluation

#### evaluation_method.py: 
Uses necessary models based off input angles provided to evaluate and classify beetle species.

#### genus_evaluation_method.py: 
Uses necessary models based off input angles provided to evaluate and classify beetle genus.

### Simulators

#### simulator.py: 
End-to-end testing simulator that runs full database conversion and reading, training based on user-input choice, and evaluation.

#### eval_simulator.py: 
Testing evaluation of currently implemented models using user input of images in the local dataset directory.

#### alt_training_simulator: 
Runs alt_training_program.py and trains models to assess accuracies.

#### stack_simulator.py: 
Testing simulator for training and evaluation of stack model.

#### globals.py: 
Contains global variable names for simulator file name references.

## Getting Started

### Dependencies

* pandas
* torch
* torchvision
* scikit-learn
* dill
* pylint
* python 3.11

### Installing

Evaluation simulators will only run properly with image dataset downloaded.

### Executing program

#### simulator.py

* Run in VSCode
* Repeatedly enter number corresponding to which models you'd like to train
* Wait for training and automatic evaluation results

#### eval_simulator.py

* Run in VSCode
* Enter specimen id of specimen in local dataset that you'd like to evaluate e.g. GEM_187675032

#### stack_simulator.py and alt_training_simulator.py

* Run in VSCode
* Automatically runs training and testing without input
