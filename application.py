from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.dsProject.pipelines.prediction_pipeline import CustomData, Predict_Pipeline

application = Flask(__name__)

#app=application

@application.route("/")
def index():
    print("Index Page")
    return render_template("index.html")

@application.route("/predictdata", methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        print("inside get request of predict data")
        return render_template("home.html")
    else:
        print("Inside post request of predictdata")
        data = CustomData(
            gender = request.form.get("gender"),
            race_ethnicity = request.form.get("ethnicity"),
            parental_level_of_education = request.form.get("parental_level_of_education"),
            lunch = request.form.get("lunch"),
            test_preparation_course = request.form.get("test_preparation_course"),
            writing_score = float(request.form.get("writing_score")),
            reading_score = float(request.form.get("reading_score"))            
        )
        
        print(data)
        pred_df = data.get_data_as_dataframe()
        print("Form input data:\n",pred_df)
        predict_pipeline = Predict_Pipeline()
        result = predict_pipeline.predict(pred_df)
        print(f"Predicted Maths Score is: ", result[0])
        return render_template("home.html", result = result[0])

    
if __name__ == "__main__":
    application.run(host='0.0.0.0')