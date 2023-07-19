import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self , train_arr , test_arr):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)