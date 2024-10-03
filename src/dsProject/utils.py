import os
import sys
from src.dsProject.exceptions import CustomException
from src.dsProject.logger import logging
import pandas as pd
import pymysql
from dotenv import load_dotenv
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    logging.info("Reading data from MySql initiated")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
            )

        logging.info("Connection Established")
        df=pd.read_sql_query("select * from student", mydb)
        print(df.head())
        return df

    except Exception as ex:
        raise CustomException(ex,sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model,param,cv=5)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)    ##Train Model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
            return report

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        print("inside load_object function of utils and file path is ", file_path)
        with open(file_path, "rb") as file_obj:
            print("Inside open file path")
            return pickle.load(file_obj)
            
    except Exception as e:
        print("error inside load object",e)
        raise CustomException(e,sys)