House Pricing Prediction with MLflow

This project implements a machine learning pipeline for predicting house prices while leveraging MLflow for experiment tracking and offline evaluation. Below is an overview of the process used in training, logging, and evaluation:

1. Training and Logging

Models Used:

Linear Regression

Random Forest Regressor

Steps:

Train the models using the house pricing dataset.

Log key components using MLflow:

Model parameters (e.g., features, algorithm type).

Metrics (e.g., Mean Squared Error, R² Score).

Artifacts, such as prediction results and serialized models.

Save and register models for further analysis or deployment.

2. Offline Evaluation

Purpose:

Validate the performance of saved models on test datasets.

Steps:

Load saved models from MLflow by referencing their unique run_id.

Evaluate models using metrics such as:

Mean Squared Error (MSE)

R² Score

Compare results to identify the best-performing model.
