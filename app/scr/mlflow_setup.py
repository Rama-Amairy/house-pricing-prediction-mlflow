import mlflow

def setup_mlflow(experiment_name: str, tracking_uri: str = "http://localhost:5000"):
    """
    Sets up MLflow with the given experiment name and tracking URI.

    :param experiment_name: Name of the experiment in MLflow.
    :param tracking_uri: URI of the MLflow tracking server (default: localhost).
    """
    # Set the MLflow tracking server
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set the experiment name
    mlflow.set_experiment(experiment_name)

    print(f"MLflow is set up for experiment: {experiment_name}, URI: {tracking_uri}")
