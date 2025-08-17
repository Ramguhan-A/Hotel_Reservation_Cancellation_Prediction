import os
from src.logger import get_logger
from src.custom_exception import CustomException
import pandas
import yaml
import pandas as pd


logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"yaml File not found on the given path")
        
        with open(file_path,"r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("yaml File reading successfull")
            
            return config
    except Exception as e:
        logger.error("There is some error occured while reading the file")
        raise CustomException("Error occured while reading the yaml file", e)


#################################################################################################

def load_data(path):
    try:
        logger.info("Loading data")
        return pd.read_csv(path)
    
    except Exception as e:
        logger.error(f"Error while loading the data {e}")
        raise CustomException("Failed to laod the data",e)