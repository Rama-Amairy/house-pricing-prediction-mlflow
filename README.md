# House Pricing Prediction with MLflow

This project implements a machine learning pipeline for predicting house prices while leveraging **MLflow** for experiment tracking and **offline evaluation**. Below is an overview of the process used in training, logging, and evaluation:

---

## **1. Training and Logging**
- **Models Used**:
  - Linear Regression
  - Random Forest Regressor
- **Steps**:
  1. Train the models using the house pricing dataset.
  2. Log key components using MLflow:
     - Model parameters (e.g., features, algorithm type).
     - Metrics (e.g., Mean Squared Error, R² Score).
     - Artifacts, such as prediction results and serialized models.
  3. Save and register models for further analysis or deployment.

---

## **2. Offline Evaluation**
- **Purpose**:
  - Validate the performance of saved models on test datasets.
- **Steps**:
  1. Load saved models from MLflow by referencing their unique `run_id`.
  2. Evaluate models using metrics such as:
     - Mean Squared Error (MSE)
     - R² Score
  3. Compare results to identify the best-performing model.

---

