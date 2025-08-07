import mlflow
from project.config import Config
from contextlib import contextmanager

def setup_mlflow():
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

@contextmanager
def start_mlflow_run(run_name):
    setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        try:
            yield run
        finally:
            mlflow.end_run()

def log_tuning_results(trial, metrics):
    mlflow.log_params(trial.params)
    mlflow.log_metrics({
        "val_loss": metrics[0],
        "sharpe_ratio": metrics[1],
        "profit_factor": metrics[2]
    })

def log_training_results(model_type, hparams, metrics):
    mlflow.log_params(hparams)
    mlflow.log_metrics(metrics)