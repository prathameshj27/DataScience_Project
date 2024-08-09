import os
import sys
from src.dsProject.exceptions import CustomException
from src.dsProject.logger import logging
import pandas as pd
import pymysql
from dotenv import load_dotenv

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