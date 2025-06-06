from airflow.models import DagBag

def test_dag_can_be_loaded():
    dagbag = DagBag(dag_folder="/opt/airflow/dags", include_examples=False)
    dag = dagbag.get_dag("train_and_log_fraud_model")
    assert dag is not None
    assert not dagbag.import_errors
