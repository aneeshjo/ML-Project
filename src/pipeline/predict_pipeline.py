import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object
from src.logging_config import logger

class PredictPipeline:
    """
    Class responsible for making predictions using a trained machine learning model.
    It loads the preprocessor and model objects, applies the preprocessor to the input features,
    and returns the predictions.
    """
    
    def __init__(self):
        pass

    def predict(self, features):
        """
        Makes predictions based on the input features.
        
        Parameters:
        - features: The input features for which predictions are to be made
        
        Returns:
        - The predicted values
        """
        try:
            logger.info("Loading preprocessor and model objects for prediction.")
            
            # Load the preprocessor object
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            preprocessor = load_object(preprocessor_path)
            
            # Load the trained model object
            model_path = os.path.join('artifacts', 'model.pkl')
            model = load_object(model_path)
            
            # Apply the preprocessor to the input features
            data_scaled = preprocessor.transform(features)
            
            # Make predictions using the trained model
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            logger.error("Error occurred during prediction.")
            raise CustomException(e, sys)
        
class CustomData:
    """
    Class representing custom input data for making predictions.
    It defines the structure of the input data and provides a method to convert it into a DataFrame.
    """
    
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, 
                 lunch: str, test_preparation_course: str, reading_score: float, 
                 writing_score: float):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        """
        Converts the custom data into a pandas DataFrame.
        
        Returns:
        - A DataFrame containing the input data
        """
        try:
            logger.info("Converting custom data to DataFrame.")
            
            # Create a dictionary with the input data
            custom_data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            # Create a DataFrame from the dictionary
            df = pd.DataFrame(custom_data_dict)
            logger.info("Custom data converted to DataFrame.")
            return df

        except Exception as e:
            logger.error("Error occurred while converting custom data to DataFrame.")
            raise CustomException(e, sys)