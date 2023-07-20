import os
import sys
from src.logger import logging
from src.exception import CustomException
import pickle
import dill

from sklearn.model_selection import GridSearchCV
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
    
def evaluation_model(x_train , y_train , x_test , y_test , models, params):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]


            gs = GridSearchCV(model , param , cv=10 , n_jobs=5)
            gs.fit(x_train , y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            accuracy = r2_score(y_test , y_pred)
            print(accuracy)

            report[list(models.values())[i]] = accuracy

            return report
        
    except Exception as e:
        raise CustomException(e,sys)
        