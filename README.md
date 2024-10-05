# End-to end Data Science Project - Student Math Score Prediction

## Project Overview

This project aims to predict the math scores of students based on various input features such as gender, race/ethnicity, parental level of education, and more. Using a student performance dataset sourced from Kaggle, we create an end-to-end machine learning pipeline that includes data ingestion, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, model evaluation and deployment of web application on AWS cloud.


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Xgboost
- Catboost
- MySQL
- DVC (Data Version Control)
- MLflow
- Flask
- AWS (EC2, Elastic Beanstalk, CodePipeline)
- GitHub

## Data Source

The dataset used for this project can be found on https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data

## Phases

1. **Data Ingestion**: Runs the data ingestion script to read data from MySQL DB and perform a train-test split.

2. **Exploratory Data Analysis (EDA)**: Uses Jupyter notebooks to explore the data.

3. **Feature Engineering**: Process and engineer features.

4. **Model Training**: Trains the model using various algorithms.

5. **Hyperparameter Tuning**: Optimizes hyperparameters.

6. **Model Evaluation**: Evaluates the best model among several other machine learning models.

7. **Run the Flask App**: Run the Flask web application to get the predicted math score based on input features and the selected model.


## Results

The best model achieved a satisfactory score on the test dataset, with detailed results and performance metrics documented in the Jupyter notebooks.

## Deployment

The Flask web application is deployed on AWS Elastic Beanstalk, utilizing CodePipeline for continuous delivery from the GitHub repository.

## Future Work

- Explore additional machine learning algorithms.
- Implement advanced feature engineering techniques.
- Expand the web application with more input features and analytics.
