import os
import sys
import numpy as np
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.dsProject.logger import logging
from src.dsProject.exceptions import CustomException
from src.dsProject.utils import save_object, evaluate_models
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

@dataclass
class Model_Trainer_Config:
    trained_model_file_path = os.path.join("Artifacts","Model_tainer.pkl")

class Model_Trainer:
    def __init__(self):
        self.model_trainer_config = Model_Trainer_Config()

    def eval_metrics(self,true,predicted):
        rmse = np.sqrt(mean_squared_error(true,predicted))
        mae = mean_absolute_error(true,predicted)
        r2 = r2_score(true,predicted)

        return rmse,mae,r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split train & test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "CatBoosting Regressor" : CatBoostRegressor(verbose=False),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Liner Regression" : LinearRegression(),
                "Decision Tree" : DecisionTreeRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "K Neighbors Regressor" : KNeighborsRegressor()
            }

            params = {
                "Decision Tree" : {
                    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter' : ['best', 'random'],
                    'max_features' : ['sqrt', 'log2']                                  
                },
                "Random Forest" : {
                    'criterion' : ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    'max_features' : ['sqrt', 'log2'],
                    'n_estimators' : [8,16,32,64,128,256]
                },
                "Gradient Boosting" :{
                    'loss' : ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate' : [.1,.01,.05,.001],
                    'subsample' : [0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion' : ['friedman_mse', 'squared_error'],
                    'max_features' : ['auto','sqrt','log2'],
                    'n_estimators' : [8,16,32,64,128,256]
                },
                "Linear Regression" : {},
                "XGBRegressor" : {
                    'learning_rate' : [.1,.01,.05,.001],
                    'n_estimators' : [8,16,32,64,128,256]
                },
                "CatBoosting Regressor" : {
                    'learning_rate' : [0.01,0.05,0.1],
                    'depth' : [6,8,10],
                    'iterations' : [30,50,100]
                },
                "AdaBoost Regressor" : {
                    'learning_rate' : [.1,.01,.05,.001],
                    'loss' : ['linear','square','exponential'],
                    'n_estimators' : [8,16,32,64,128,256]
                },
                "K Neighbors Regressor" : {
                    'n_neighbors' : [5,7,9,11,13,15],
                    'weights' : ['uniform','distance'],
                    'metric' : ['minkowski','euclidean','manhattan']
                }
            }

            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models,params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print("The best model is: " + best_model_name)
            model_names = list(params.keys())
            actual_model = ""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]

            mlflow.set_registry_uri("https://dagshub.com/prathameshj27/DataScience_Project.mlflow")
            tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme


            ##MLflow

            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)
                (rmse,mae,r2) = self.eval_metrics(y_test,predicted_qualities)

                mlflow.log_params(best_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2",r2)

            # Model registry does not work with file store
            if tracking_uri_type_store != "file":
                mlflow.sklearn.log_model(best_model,"sklearn-model", registered_model_name=actual_model)
            else:
                mlflow.sklearn.log_model(best_model, "sklearn-model")

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both train & test datasets")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r_square = r2_score(y_test, predicted)
            return r_square


        except Exception as e:
            raise CustomException(e, sys)

