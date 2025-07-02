import os  
import sys 
import pandas as pd 
import dill
import numpy as np 
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # create the directory if it doesn't exist

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train , Y_train , X_test , Y_test , models ):
  try :
    report  = {}

    for i in range(len(list(models))):
        model = list(models.values())[i]

        model.fit(X_train,Y_train)

        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_test)

        train_model_score = r2_score (Y_train , y_train_predict)
        test_model_score = r2_score (Y_test , y_test_predict)

        report[list(models.keys())[i]] = test_model_score

    return report

  except Exception as e : 
    raise CustomException(e,sys) 