import os  
import sys 
from dataclasses import dataclass 
from catboost import CatBoostRegressor 

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException 
from src.logger import logging 
from src.utils import save_object , evaluate_models

@dataclass 
class ModelTrainerConfig :
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer :
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
     try:
         logging.info("splitting test and train input data")

         x_train,y_train,x_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

         models = {
            "Random Forest " : RandomForestRegressor(),
            "Linear Regression" : LinearRegression(),
            "Gradient Boosting ": GradientBoostingRegressor(),
            "Decision Tree" : DecisionTreeRegressor(),
            "k-neighbors ": KNeighborsRegressor(),
            "XGBClassifier ": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Classifier": AdaBoostRegressor()
        }

         model_report :dict = evaluate_models(X_train=x_train , Y_train= y_train,X_test= x_test, Y_test = y_test, models=models)

         best_model_Score = max(sorted(model_report.values()))

         best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_Score)]
        
         best_model = models[best_model_name]

         if best_model_Score < 0.6 :
             raise CustomException("no best model found ")
         logging.info(f"best model found on both training and testing dataset")
        
         save_object (file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

         predicted = best_model.predict(x_test)
         return r2_score(y_test , predicted)
    

     except Exception as e  :
        raise CustomException(e,sys)
        