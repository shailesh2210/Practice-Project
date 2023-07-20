import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj

class PredictPipeline():
    def __init__(self):
        pass

    def predict(self , features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_obj(file_path=preprocessor_path)
            model = load_obj(file_path=model_path)

            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)

            return pred
        
        except Exception as e:
            raise C


class CustomData():
    def __init__(self ,
                 age:int,
                 sex:int,
                 bmi:int,
                 children:int,
                 smoker:int,
                 region:int):
        
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def get_data_as_df(self):
        try:
            output_as_df = {
                "age" :[self.age],
                "sex":[self.sex],
                "bmi":[self.bmi],
                "children":[self.children],
                "smoker":[self.smoker],
                "region":[self.region]
            }

            data = pd.DataFrame(output_as_df)

            return data
        
        except Exception as e:
            raise CustomException(e, sys)

        