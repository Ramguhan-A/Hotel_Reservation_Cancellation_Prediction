import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_function import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier # for feature selection
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:
    
    def __init__(self, train_path, test_path, processed_dir, config_path):
        
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
    def preprocess_data(self, df):
        try:
            logger.info("Starting our Data Processing step")
            
            # 1. Drop columns and duplicates
            logger.info("Dropping columns...")
            
            df.drop(columns=["Unnamed: 0","Booking_ID"], inplace=True)
            
            logger.info("Dropping duplicates...")
            
            df.drop_duplicates(inplace=True)
            
            # 2. Label encoding
            cat_columns = self.config["data_processing"]["categorical_columns"]
            num_columns = self.config["data_processing"]["numerical_columns"]
            
            logger.info("Applying Lable Encoding on Categorical columns...")
            
            label_encoder = LabelEncoder()

            label_mappings = {}

            for col in cat_columns:
                
                df[col] = label_encoder.fit_transform(df[col])
                label_mappings[col] = {label: code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
            
            logger.info("Label Mappings are: ")
            
            for col, mapping in label_mappings.items():
                logger.info(f"{col}: {mapping}")
            
            # 3. Skewness fix in numerical features (log-transform)
            logger.info("Skewness handling...")
            
            skew_threshold = self.config["data_processing"]["skewness_threshold"]  # threshold is 5
            
            skewness = df[num_columns].apply(lambda x: x.skew())
            
            for column in skewness[skewness>skew_threshold].index:
                if (df[column] <= -1).any():  # to prevent negative values on the feature (if any)
                    logger.warning(f"Skipping log transform for {column} due to negative values")
                    continue
                df[column] = np.log1p(df[column])
                
            return df
        
        except Exception as e:
            logger.error(f"Error occured in data preprocessing pipeline {e}")
            raise CustomException("Error while preprocess data", e)
    
    def balance_data(self, df):
        try:
            logger.info("Handling Imbalanced data...")
            X = df.drop(columns = 'booking_status')
            y = df["booking_status"]
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X,y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df["booking_status"] = y_resampled
            
            logger.info("Data balanced succesfully")
            
            return balanced_df
        
        except Exception as e:
            logger.error(f"Error during applying SMOTE {e}")
            raise CustomException("Error while balancing data", e)
        
    def feature_selection(self,df):
        
        try:
            logger.info("Selecting Features ...")
            
            X = df.drop(columns = "booking_status")
            y = df["booking_status"]
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X,y)
            
            # 4. Feature selection
            
            feature_importance = model.feature_importances_
            
            feature_importance_df = pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": feature_importance
                }
            )
            
            important_feature = feature_importance_df.sort_values(by="importance", ascending= False)
            
            no_of_features = self.config["data_processing"]["no_of_important_features"]  # get value from yaml file for reusability

            top_features = important_feature["feature"].head(no_of_features).values
            
            top_10_df = df[top_features.tolist() + ["booking_status"]]
            
            logger.info(f"Feature selection of top {no_of_features} are {top_features}... selected successfully")
            
            return top_10_df
        
        except Exception as e:
            logger.error(f"Error occured while selecting features {e}")
            
            raise CustomException("Error occured on feature selection", e)
        
    
    def save_data(self, df, file_path):
        try:
            logger.info("Saving the preprocessed dataframe as CSV")
            
            df.to_csv(file_path, index=False)
            
            logger.info(f"Data saved successfully on {file_path}")
            
        except Exception as e:
            logger.error("Error occured while saving the dataframe as CSV")
            raise CustomException("Error occured while saving the data", e)
        
    def pre_process(self):
        try:
            logger.info("loading the data from RAW directry...")
            
            # Loading the data
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
            
            # Preprocess the data
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
            
            # Balance the data (only on train data)
            train_df = self.balance_data(train_df)

            
            # Feature selection
            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]  # Ensure same columns from train_df
            
            # Save the data
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)
            
            logger.info("Data processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during preprocessing pipeline {e}")
            raise CustomException("Error occred on data preprocessing pipeline", e)
        

# if __name__ == "__main__":
    
#     processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
#     processor.pre_process()