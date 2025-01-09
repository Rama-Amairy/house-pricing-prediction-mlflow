from sklearn.metrics import mean_squared_error, r2_score
import mlflow

def offline_evaluate(model_uri, X_test, y_test):
    model = mlflow.sklearn.load_model(model_uri)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Offline Evaluation - Model: {model_uri}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
