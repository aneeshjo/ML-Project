import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logging_config import logger
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    It defines the path to save the preprocessor object.
    """
    
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Class responsible for data transformation.
    It creates a preprocessing pipeline for both numerical and categorical features,
    and saves the preprocessor object to a file.
    """
    
    def __init__(self):
        # Initialize the data transformation configuration
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a ColumnTransformer object that applies different transformations
        to numerical and categorical features.
        
        Returns:
        - preprocessor: ColumnTransformer object with the defined transformations
        """
        try:
            logger.info("Creating data transformer object.")
            
            # Define the numerical and categorical columns
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 
                                   'parental_level_of_education', 'lunch', 
                                   'test_preparation_course']
            # Define the transformation pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
                ('scaler', StandardScaler())  # Scale features to have mean=0 and variance=1
            ])
            # Define the transformation pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent
                ('one_hot_encoder', OneHotEncoder(drop='first'))  # Encode categorical features as one-hot vectors
            ])
            # Create the ColumnTransformer object
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            logger.error("Error occurred while creating data transformer object.")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process.
        
        Parameters:
        - train_path: str, path to the training data file
        - test_path: str, path to the testing data file
        
        Returns:
        - Tuple containing:
            - preprocessor object
            - transformed training features
            - transformed testing features
            - training target variable
            - testing target variable
        """
        try:
            logger.info("Starting data transformation process.")
            
            # Read the training and testing data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info("Training and testing data read successfully.")
            
            # Define the target variable and drop it from the feature set
            target_column_name = 'math_score'
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
            
            # Get the preprocessor object
            preprocessor_obj = self.get_data_transformer_object()
            
            # Fit and transform the training features, and transform the testing features
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            logger.info("Data transformation completed successfully.")

            transformed_train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            transformed_test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info("Transformed training and testing arrays created successfully.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                transformed_train_arr,
                transformed_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            logger.error("Error occurred during data transformation.")
            raise CustomException(e, sys)