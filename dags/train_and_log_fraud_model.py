import os
from datetime import timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ------------------------------------------------------------------------
# Variables d’environnement (NE PAS écrire dans os.environ ici)
# ------------------------------------------------------------------------
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:///opt/airflow/mlruns")      # ex. https://…hf.space
DATA_CSV = os.getenv("DATA_CSV_PATH", "/opt/airflow/data/transactions.csv")         # ex. /opt/airflow/…

def _require(value, name: str) -> str:
    """Lève une erreur claire si la variable n’est pas définie."""
    if not value:
        raise RuntimeError(
            f"{name} manquant : définissez-le via variable d'environnement "
            f"ou dans le workflow CI."
        )
    return value
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
# ------------------------------------------------------------------------
# DAG
# ------------------------------------------------------------------------
default_args = {"owner": "toi", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="train_and_log_fraud_model",
    description="Daily training + MLflow logging for fraud model",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval="0 2 * * *",   # chaque jour à 02 h 00
    catchup=False,
    tags=["fraud"],
) as dag:

    # -------------------- 1. VALIDATION DES DONNÉES --------------------
    EXPECTED_COLS = [
        "amount",
        "transaction_type",
        "country",
        "is_fraud",
    ]

    def validate_df(df: pd.DataFrame) -> None:
        """Vérifie (1) schéma, (2) valeurs manquantes, (3) cible binaire."""
        # 1️⃣ colonnes présentes
        missing = set(EXPECTED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        # 2️⃣ pas de valeurs manquantes
        if df[EXPECTED_COLS].isna().any().any():
            raise ValueError("Valeurs manquantes détectées")

        # 3️⃣ cible binaire
        labels = set(df["is_fraud"].unique())
        if labels != {0, 1}:
            raise ValueError(f"'is_fraud' doit contenir 0/1, j’ai trouvé {labels}")

    def validate_data_callable(**_):
        import pandas as pd
        df = pd.read_csv(DATA_CSV)
        """Lit le CSV (chemin dans DATA_CSV_PATH) et applique validate_df."""
        csv_path = _require(DATA_CSV, "DATA_CSV_PATH")
        df = pd.read_csv(csv_path)
        validate_df(df)

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data_callable,
    )

    # ---------------- 2. ENTRAÎNEMENT & LOG DANS MLFLOW ----------------
    def train_model_callable(**_):
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(_require(MLFLOW_URI, "MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("fraud_detection_dag")

        # --- jeu de données simulé ---
        df = pd.DataFrame({
            "transaction_id": np.arange(1, 5001),
            "amount": np.round(np.random.exponential(scale=50, size=5000), 2),
            "transaction_type": np.random.choice(
                ["purchase", "withdrawal", "transfer"], size=5000, p=[0.7, 0.2, 0.1]
            ),
            "country": np.random.choice(["FR", "US", "DE", "ES", "IT"], size=5000),
            "is_fraud": np.random.choice([0, 1], size=5000, p=[0.98, 0.02]),
        })

        X = pd.get_dummies(df[["amount", "transaction_type", "country"]], drop_first=True)
        y = df["is_fraud"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        with mlflow.start_run(run_name="rf_from_airflow"):
            params = {"n_estimators": 100, "max_depth": 10}
            mlflow.log_params(params)

            rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)

            y_pred   = rf.predict(X_test)
            y_proba  = rf.predict_proba(X_test)[:, 1]
            mlflow.log_metrics({
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc":  roc_auc_score(y_test, y_proba),
            })

            mlflow.sklearn.log_model(rf, artifact_path="model")

    train_task = PythonOperator(
        task_id="train_model_task",
        python_callable=train_model_callable,
    )

    # ------------------------- ORCHESTRATION --------------------------
    validate_task >> train_task
