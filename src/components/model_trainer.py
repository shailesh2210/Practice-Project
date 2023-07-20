import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluation_model, save_obj

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBoostRegressor

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self , train_arr , test_arr):
        try:
            logging.info("Model Training Started!")
            x_train , y_train  , x_test , y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            logging.info("Initiate models")

            models = {
                "linear_reg":LinearRegression(),
                "decision_tree":DecisionTreeRegressor(),
                "random_forest": RandomForestRegressor()
                # "xg_boost": XGBoostRegressor()
            }

            logging.info("Hyper-Para-Meter")

            params = {
                "linear_reg":{
                   'fit_intercept':[True],
                   'n_jobs':[2 , 3, 4, 5]
                },
                 "descion_tree":{
                    "class_weight":["balanced"],
                    "criterion":['gini',"entropy","log_loss"],
                    "splitter":['best','random'],
                    "max_depth":[3,4,5,6],
                    "min_samples_split":[2,3,4,5],
                    "min_samples_leaf":[1,2,3],
                    "max_features":["auto","sqrt","log2"]
                },
                "random_forest":{
                    "class_weight":["balanced"],
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                },
                # "xg_boost":{
                #     'n_estimators': [10 ,20, 50, 30],
                #     "learning_rate":[0.001 , 0.01 , 1 , 10],
                #     "max_depth":[9, 8, 5]
                # }
            }

            model_report:dict = evaluation_model(x_train=x_train ,y_train=y_train,
                            x_test=x_test,y_test=y_test,models=models, params=params)
            
            logging.info("Finding the best model score")
            # to get the best model score 
            best_model_score = max(sorted(model_report.values()))

            logging.info("finding the best model name")
            # to get the best model name 
            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info("Saving the model file ")
            # saving the model.pkl file 


            save_obj(
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )

            logging.info("Model Training Completed")

            
        except Exception as e:
            raise CustomException(e, sys)