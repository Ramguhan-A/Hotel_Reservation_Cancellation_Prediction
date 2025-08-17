import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from config.paths_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_function import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]
        
        os.makedirs(RAW_DIR, exist_ok = True)
        
        logger.info(f"Data ingestion started with {self.bucket_name} and file {self.bucket_file_name}")
        
    def download_csv_from_gcp(self):
        
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)
            
            blob.download_to_filename(RAW_FILE_PATH)  # Dowload the file
            
            logger.info(f"CSV file successfully downloaded to {RAW_FILE_PATH}")
            
        except Exception as e:
            logger.error("Error while downlaoding the file")
            raise CustomException("Failed to download csv file", e)
        
    def split_data(self):
        
        try:
            logger.info("Spliting the file....")
            
            data = pd.read_csv(RAW_FILE_PATH)
            
            train_data, test_data = train_test_split(data, test_size= 1 - self.train_test_ratio, random_state=42) # combined split
        
            train_data.to_csv(TRAIN_FILE_PATH) # saving the file
            test_data.to_csv(TEST_FILE_PATH)
            
            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")
        
        except Exception as e:
            logger.error("Error occured while splitting teh data")
            
            raise CustomException("Failed to split data into training and test sets",e)
    
    def run(self):
        
        try:
            logger.info("Starting data ingestion process")
            
            self.download_csv_from_gcp()
            self.split_data()
            
            logger.info("Data ingestion completed successfully")
            
        except Exception as e:
            
            logger.error(f"CustomException : {str(e)}")
            
        finally:
            logger.info("Data ingestion completed")
            

# if __name__ == "__main__":
    
#     data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
#     data_ingestion.run()