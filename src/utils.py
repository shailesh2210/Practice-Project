import os
import sys
from src.logger import logging
from src.exception import CustomException
import pickle
import dill

from sklearn.metrics import accuracy_score , r2_score

sys.path.append(r"D:\Practice Machine Learning\Practice-Project")

def save_obj(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path , exist_ok = True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj , file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluation_model(x_train , y_train , x_test , y_test , models):
    try:
        report = {}

        for i in range(len(models)):
            models = list(models.values())[i]

            models.fit(x_train, y_train)
            y_pred = models.predict(x_test)
            accuracy = r2_score(y_test , y_pred)

            print(accuracy)

            report[list(models.values())[i]] = accuracy

            return report
        
    except Exception as e:
        raise CustomException(e,sys)
        