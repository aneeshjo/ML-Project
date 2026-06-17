import os
import sys
from src.exception import CustomException
from src.logging_config import logger
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Saves a Python object to a file using dill.
    
    Parameters:
    - file_path: str, the path where the object should be saved
    - obj: the Python object to be saved
    """
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Save the object to the specified file path using dill
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        
        logger.info(f"Object saved successfully at {file_path}.")
    
    except Exception as e:
        logger.error("Error occurred while saving the object.")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluates multiple machine learning models and returns their performance scores.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target variable
    - X_test: Testing features
    - y_test: Testing target variable
    - models: A dictionary of model names and their corresponding model instances
    
    Returns:
    - A dictionary containing the R2 scores of each model
    """
    try:
        model_report = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            model_report[model_name] = score
            logger.info(f"{model_name} R2 Score: {score}")
        
        return model_report
    
    except Exception as e:
        logger.error("Error occurred during model evaluation.")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a Python object from a file using dill.
    
    Parameters:
    - file_path: str, the path to the file from which the object should be loaded
    
    Returns:
    - The Python object loaded from the specified file path
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        
        logger.info(f"Object loaded successfully from {file_path}.")
        return obj
    
    except Exception as e:
        logger.error("Error occurred while loading the object.")
        raise CustomException(e, sys)