import mlflow
from mlflow import MlflowClient
import os
EXPERIMENT_NAME = "fraud_detection_dag"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
from mlflow.tracking import MlflowClient
def load_latest_model() -> mlflow.pyfunc.PyFuncModel:
    client = MlflowClient(tracking_uri=None)  # lit MLFLOW_TRACKING_URI env
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment {EXPERIMENT_NAME} not found")

    # récupère la dernière run au statut FINISHED triée sur `start_time`
    runs = client.search_runs(
        exp.experiment_id,
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No successful run found")
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    return mlflow.pyfunc.load_model(model_uri), run_id
