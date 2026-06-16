import os
import sys
from src.exception import CustomException
from src.logging_config import logger
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_trasformation import DataTransformation, DataTransformationConfig


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    It defines the paths for train, test, and raw data files.
    """
    
    def __init__(self):
        # Define the directory to store the ingested data
        self.ingested_data_dir = os.path.join('artifacts', 'data_ingestion')
        
        # Define the path for the training data file
        self.train_data_path = os.path.join(self.ingested_data_dir, 'train.csv')
        
        # Define the path for the testing data file
        self.test_data_path = os.path.join(self.ingested_data_dir, 'test.csv')
        
        # Define the path for the raw data file
        self.raw_data_path = os.path.join(self.ingested_data_dir, 'data.csv')


class DataIngestion:
    """
    Class responsible for data ingestion.
    It reads the raw data, splits it into training and testing sets,
    and saves them to the specified paths.
    """
    
    def __init__(self):
        # Initialize the data ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, file_path):
        """
        Initiates the data ingestion process.
        
        Parameters:
        - file_path: str, path to the raw data file
        
        Returns:
        - Tuple of paths to the training and testing data files
        """
        
        logger.info("Starting data ingestion process.")
        
        try:
            # Read the raw data from the specified file path
            df = pd.read_csv(file_path)
            logger.info("Raw data read successfully.")
            
            # Create the directory for ingested data if it doesn't exist
            os.makedirs(self.ingestion_config.ingested_data_dir, exist_ok=True)
            
            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"Raw data saved at {self.ingestion_config.raw_data_path}.")
            
            # Split the data into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logger.info("Data split into training and testing sets successfully.")
            
            # Save the training and testing sets to CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logger.info(f"Training data saved at {self.ingestion_config.train_data_path}.")
            logger.info(f"Testing data saved at {self.ingestion_config.test_data_path}.")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            # Log the error and raise a custom exception with detailed information
            logger.error("Error occurred during data ingestion.")
            # The CustomException will capture the error message along with the file name and line number where the error occurred
            raise CustomException(e, sys)
if __name__ == "__main__":
    # Example usage of the DataIngestion class
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion(file_path='notebook/data/stud.csv')
    logger.info(f"Data ingestion completed. Train data path: {train_path}, Test data path: {test_path}.")

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path=train_path, test_path=test_path)