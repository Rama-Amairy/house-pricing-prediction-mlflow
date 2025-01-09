import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import time
from mlflow_setup import setup_mlflow
from offline_evaluation import offline_evaluate

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, features):
    start_time = time.time()
    with mlflow.start_run():
        # Log Parameters
        mlflow.log_param("features", features)
        mlflow.log_param("model", model_name)
        mlflow.log_param("datasets_used", "housing_price_dataset.csv")
        mlflow.set_tag("source", "train.py")

        # Train Model
        model.fit(X_train, y_train)

        # Make Predictions
        y_pred = model.predict(X_test)

        # Evaluate Model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log Metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Prepare Input Example for Logging
        input_example = pd.DataFrame([{
            "SquareFeet": X_test.iloc[0]["SquareFeet"],
            "Bedrooms": X_test.iloc[0]["Bedrooms"],
            "Bathrooms": X_test.iloc[0]["Bathrooms"],
            "YearBuilt": X_test.iloc[0]["YearBuilt"]
        }])

        # Save and Log Model with Input Example
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        # Record duration
        duration = time.time() - start_time
        mlflow.log_metric("duration", duration)

        # Print Results
        print(f"Model: {model_name}")
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")
        print(f"Duration: {duration} seconds")

def main():
    # Load Dataset from Kaggle path
    data_path = "app/scr/data/housing_price_dataset.csv"
    data = pd.read_csv(data_path)

    for col in ["Bedrooms", "Bathrooms", "YearBuilt"]:
        data[col] = data[col].astype(float)

    features = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt"]
    target = "Price"
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    setup_mlflow("House Price Prediction")

    linear_model = LinearRegression()
    train_and_log_model(linear_model, "LinearRegression", X_train, X_test, y_train, y_test, features)

    rf_model = RandomForestRegressor(random_state=42)
    train_and_log_model(rf_model, "RandomForestRegressor", X_train, X_test, y_train, y_test, features)

    print("\nOffline Evaluation:")
    offline_evaluate("runs:/414a308e06a94fd8b6ed2b172ee4cfc6/model", X_test, y_test)
    offline_evaluate("runs:/bbb8440df5244104bf1739e79b089323/model", X_test, y_test)

if __name__ == "__main__":
    main()
