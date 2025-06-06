# File: dags/example_dag.py

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

with DAG(
    dag_id="hello_world",
    start_date=datetime(2025, 6, 3),
    schedule_interval="@daily",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:

    hello = BashOperator(
        task_id="say_hello",
        bash_command="echo 'Hello, Airflow!'"
    )
