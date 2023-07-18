import os
import sys
sys.path.append(r"D:\Practice Machine Learning\Practice-Project")
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException 
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_file_path = os.path.join("artifacts" , "train.csv")
    test_file_path = os.path.join("artifacts" ,"test.csv")
    raw_file_path = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion Started....")
        try:
            # reading the dataset 
            df = pd.read_csv(os.path.join("notebook/", "insurance.csv"))
            logging.info("Reading the dataset")

            # making dirs 
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_file_path), exist_ok = True )
            logging.info("Making dir")

            df.to_csv(self.data_ingestion_config.raw_file_path,index=False, header=True)
            logging.info("saving the raw data")

            # splitting into train and test 
            train_set , test_set = train_test_split(df , test_size=0.2 , random_state=42)
            logging.info("splitting data into train and test")

            train_set.to_csv(self.data_ingestion_config.train_file_path , index = False, header = True)
            test_set.to_csv(self.data_ingestion_config.test_file_path,  index = False, header = True)
            logging.info("saving train and test data")

            logging.info("Data ingestion Completed Successfully")
            return(
                self.data_ingestion_config.train_file_path,
                self.data_ingestion_config.test_file_path
            )

        except Exception as e:
            logging.info("Error occured in file name")
            raise CustomException(e ,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data , test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initaite_data_transformation(train_data ,test_data)


