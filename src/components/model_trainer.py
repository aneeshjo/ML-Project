import os
import sys
from src.exception import CustomException
import dill
from src.logging_config import logger
from src.utils import save_object, evaluate_models

import numpy as np
import pandas as pd 
from dataclasses import dataclass
from sklearn.pipeline import Pipeline

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for model training.
    It defines the path to save the trained model object.
    """
    
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')   

class ModelTrainer:
    """
    Class responsible for training the machine learning model.
    It defines a method to train the model and save the trained model object to a file.
    """
    
    def __init__(self):
        # Initialize the model trainer configuration
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Initiates the model training process.
        
        Parameters:
        - train_array: numpy array, the training data
        - test_array: numpy array, the testing data
        
        Returns:
        - Tuple containing the paths to the trained model file and the trained model object
        """
        
        try:
            logger.info("Starting model training process.")
            
            # Split the training and testing arrays into features and target variable
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Define a list of candidate models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "LightGBM Regressor": LGBMRegressor()
            }

            model_report = {}
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            
            # Train each model and evaluate its performance on the test set
            best_model_name = None
            best_model_score = float('-inf')
            best_model = None

            

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]    
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model found with R2 score greater than 0.6.", sys)

            logger.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}.")

            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )   

            logger.info(f"Model saved successfully at {self.model_trainer_config.trained_model_file_path}.")
            return (
                self.model_trainer_config.trained_model_file_path,
                best_model
            )

                     
        except Exception as e:
            logger.error("Error occurred during model training.")
            raise CustomException(e, sys)
    