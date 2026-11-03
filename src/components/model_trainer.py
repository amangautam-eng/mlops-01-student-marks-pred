import os
import sys

from dataclasses import dataclass

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Model training Started')

            xtrain,ytrain,xtest,ytest=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "Linear Regression":LinearRegression(),
                "random forest":RandomForestRegressor(),

            }

            model_report:dict=evaluate_model(x_train=xtrain,
                                             y_train=ytrain,
                                             x_test=xtest,
                                             y_test=ytest,
                                             models=models)
            
            # To get best model score
            best_model_score=max(sorted(model_report.values()))

            #To get best model name
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            if best_model_score<0.6:
                raise CustomException("no best model")
            
            best_model=models[best_model_name]

            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(xtest)
            score=r2_score(ytest,predicted)

            return score
        except Exception as e:
            raise CustomException(e,sys)
            
