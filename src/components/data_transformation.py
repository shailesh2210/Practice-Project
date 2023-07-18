import os
import sys
import pandas as pd
import numpy as np

sys.path.append(r"D:\Practice Machine Learning\Practice-Project")

from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_file_path_obj = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation_obj(self):
        try:
            logging.info("Data Transformation Started!...")
            numerical_features = ["age"	,"bmi","children","expenses"]
            categorical_features = ["sex","smoker","region"]

            # logging.info(f"Numerical Pipeline Completed" , {numerical_features})
            cat_pipeline = Pipeline(
                steps=[
                    ("encoding", OneHotEncoder()),
                ]
            )
            logging.info("Encoding Completed")
            # logging.info(f"Categorical Pipeline Completed",{categorical_features})

            preprocessor = ColumnTransformer([
                ("cat_pipeline", cat_pipeline , categorical_features)
            ]
            )
            logging.info("Column Transformation Completed")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initaite_data_transformation(self, train_path , test_path):
        try:
            logging.info("Reading the data set train ad test data")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            preprocesor_obj = self.get_transformation_obj()

            target_column = "expenses"
            numerical_columns = ["age"	,"bmi","children","expenses"]

            input_features_train_df = train_data.drop([target_column], axis=1)
            target_feature_train_df = train_data[target_column]

            input_features_test_df = test_data.drop([target_column], axis=1)
            target_feature_test_df = test_data[target_column]

            input_features_train_arr = preprocesor_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocesor_obj.transform(input_features_test_df)

            logging.info("Applying preprocessing Object")

            train_arr = np.c_[
                input_features_train_arr , np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_features_test_arr , np.array(target_feature_test_df)
            ]
        
            save_obj(
                file_path = self.data_transformation_config.preprocessor_file_path_obj,
                obj = preprocesor_obj
            )

            logging.info("Saving the file object")
            logging.info("Data Transformation Completed")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path_obj
            )
            
        except Exception as e:
            raise CustomException(e,sys)



