import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_function import read_yaml, load_data
from scipy.stats import randint
import numpy as np

# Experiment tracking
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        
    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)
            
            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)
            
            # Spliting train data
            logger.info("Splitting train data...")
            
            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]
            
            # Splitting test data
            logger.info("Splitting test data...")
            
            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]
            
            logger.info("Data splitted successfully for Model Training")
            
            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            
            logger.error(f"Error While loading and splitting train and test data {e}")
            raise CustomException(f"Error while load and split data", e)
        
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing our model")
            
            lgbm_model = lgb.LGBMClassifier(random_state=RANDOM_SEARCH_PARAMS["random_state"])
            
            logger.info("Tuning the model using Hyper paramters...")
            
            # Hyper-parameter tuning
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions= self.params_dist,
                n_iter = self.random_search_params["n_iter"],
                cv= self.random_search_params["cv"],
                n_jobs= self.random_search_params["n_jobs"],
                verbose = self.random_search_params["verbose"],
                random_state= self.random_search_params["random_state"],
                scoring= self.random_search_params["scoring"]
            )
            
            logger.info("Fitting the model (hyper paramter tuning)...")
            
            # Fitting the model
            random_search.fit(X_train, y_train)
            
            logger.info("Hyperparameter tuning completed")
            
            # Best paramter and best estimator
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best paramters are : {best_params}")
            
            return best_lgbm_model
        
        except Exception as e:
            
            logger.error(f"Error occured while hyper parameter tuning {e}")
            raise CustomException("Failed to train model", e)
        
    def evaluate_model(self, model, X_test, y_test):
            
        try:
                
            # Predicting X_test
            y_pred = model.predict(X_test)

            # Evaluating the metrics
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
                
            logger.info(f"Accuracy score {accuracy}")
            logger.info(f"Precision score {precision}")
            logger.info(f"Recall score {recall}")
            logger.info(f"F1 score {f1}")
                
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
                }
        except Exception as e:
                
            logger.error(f"Error while evaluating the model {e}")
            raise CustomException("Error while evaluating the model", e)
        
    def save_model(self, model):
            try:
                
                os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
                
                logger.info("Saving the model")
                
                joblib.dump(model, self.model_output_path)
                
                logger.info(f"Model saved in {self.model_output_path}")
            
            except Exception as e:
                
                logger.error(f"Error while saving the model, {e}")
                raise CustomException("failed to save the model", e)
            
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting our Model trianing pipeline")
                
                logger.info("Starting MLFLOW experiment tracking")
                
                logger.info("Logging the training and testing dataset to MLFLOW")
                
                # ML flow
                mlflow.log_artifact(self.train_path, artifact_path = "datasets")
                mlflow.log_artifact(self.test_path, artifact_path = "datasets")
                
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)
                
                logger.info("Logging the model into ML flow")
                
                # ML flow
                mlflow.log_artifact(self.model_output_path)  # Model
                
                logger.info("Saving the model parameters and metrics to MLflow")
                model_params = best_lgbm_model.get_params()
                
                # Sanitize the metrics dictionary before logging
                loggable_params = {}
                for param_name, param_value in model_params.items():
                    if isinstance(param_value, (np.float64, np.float32, np.int64, np.int32)):  # Checking whether the value is in numpy
                        loggable_params[param_name] = param_value.item()
                        
                    else:
                        loggable_params[param_name] = param_value
                        
                mlflow.log_params(loggable_params) # Saving Params (Mlflow)
    
                        
                mlflow.log_metrics(metrics)    # saving metrics (Mlflow)
                
                logger.info("Model training completed...")
                
            
        except Exception as e:
            logger.error(f"Error while model training pipeline {e}")
            raise CustomException("Failed during model training ", e)
        

# if __name__ == "__main__":
#     trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
#     trainer.run()