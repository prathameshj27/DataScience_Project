# End-to-end Data Science Project - Student Math Score Prediction

## Project Overview

This project aims to predict the math scores of students based on various input features such as gender, race/ethnicity, parental level of education, and more. Using a student performance dataset sourced from Kaggle, I was able to develop an end-to-end machine learning pipeline that includes data ingestion, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, model evaluation and deployment of web application on AWS cloud.

## Table of Contents
Project Structure
Technologies Used
Dataset
Installation
Usage
Results
Deployment
Tracking and Version Control
Future work
Contributing

## Project Structure
```
Data Science Project/
│
├── Artifacts/
│   ├── raw.csv            # Raw dataset
│   ├── test.csv           # Test split
│   ├── train.csv          # Train split
│   ├── preprocessor.pkl   # Data Preprocessing object 
│   ├── modeltainer.pkl    # ML model object
│
├── notebooks/          
│   ├── EDA STUDENT PERFORMANCE.ipynb  # Jupyter notebooks for EDA
│   ├── MODEL TRAINING.ipynb           # Jupyter notebooks for Model Training
│
├── src/               # Application source code
│    ├── dsProject/
│    │    ├── components/
│    │    │    ├── data_ingestion.py        # Data ingestion script        
│    │    │    ├── data_transformation.py   # Data transformation script
│    │    │    ├── model_trainer.py         # Model training & model evaluation script
│    │           
│    │    ├── pipelines/
│    │    │    ├── prediction_pipeline.py   # Model prediction script 
│    │
│    │    ├── exceptions.py    # Exception handling code
│    │    ├── logger.py        # Logging code
│    │    ├── utils.py         # General utility code
│
├── templates/      # HTML template files for web application
│    ├── home.html
│    ├── index.html
│
├── application.py      # Flask web application for maths score prediction
├── main.py             # Main application source code file    
├── requirements.txt    # Python package dependencies
├── README.md           # Project documentation
├── setup.py            # Setup file for packages creation
├── template.py         # template file for project folder structure creation
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib, Seaborn (For EDA)
- Xgboost
- Catboost
- MySQL
- DVC (Data Version Control)
- MLflow (For experiment tracking)
- Flask
- AWS (EC2, Elastic Beanstalk, CodePipeline)
- GitHub

## Dataset

The dataset used for this project can be found on https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
    git clone https://github.com/prathameshj27/DataScience_Project.git
    cd Data Science Project

2. Create a virtual environment and activate it:
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:
    pip install -r requirements.txt

## Usage

1. **Run the main.py file**: Executes the entire workflow of the Data Science project - Data Ingestion, Data Transformation, Model Training, Hyperparameter Tuning and Model Evaluation sequentially to get the best ML model object & its R2 score as an output

2. **Run the Flask App**: The Flask app allows users to input various features and receive a predicted math score based on the trained model.


## Results

The best model achieved a satisfactory score on the test dataset, with detailed results and performance metrics documented in the Jupyter notebooks.

## Deployment

The Flask application is deployed to AWS using Elastic Beanstalk. CodePipeline is configured for continuous delivery, automatically deploying new updates from the GitHub repository.

## Tracking and Version Control
DVC: Used to track changes in the dataset.
MLflow: Used for tracking experiments and model performance metrics.
Git: Version control

## Future Work

- Explore additional machine learning algorithms.
- Implement advanced feature engineering techniques.
- Expand the web application with more input features and analytics.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements.
