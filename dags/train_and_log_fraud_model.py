import os
from datetime import timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import pandas as pd
# ------------------------------------------------------------------
# Config générale / variables d’environnement
# ------------------------------------------------------------------
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
# → ajoute, si nécessaire, tes variables USERNAME / PASSWORD dans ton docker-compose

default_args = {
    "owner": "toi",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ------------------------------------------------------------------
# DAG
# ------------------------------------------------------------------
with DAG(
    dag_id="train_and_log_fraud_model",
    description="Daily training + MLflow logging for fraud model",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval="0 2 * * *",   # chaque jour à 02h00
    catchup=False,
    tags=["fraud"],
) as dag:

    # ------------------- 1. VALIDATION DES DONNÉES -------------------
    EXPECTED_COLS = [
        "amount",
        "transaction_type",
        "country",
        "is_fraud",
    ]

    def validate_data_callable():
        """
        Charge le jeu de données local (ou depuis S3/DB) et valide :
        1. colonnes attendues présentes
        2. pas de valeurs manquantes
        3. cible binaire
        """
        import pandas as pd

        df = pd.read_csv("/opt/airflow/data/transactions.csv")  # adapte la source

        # 1️⃣ colonnes
        missing = set(EXPECTED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        # 2️⃣ valeurs manquantes
        if df[EXPECTED_COLS].isna().any().any():
            raise ValueError("Valeurs manquantes détectées")

        # 3️⃣ cible binaire
        VALID_LABELS = {0, 1}
        labels = set(df["is_fraud"].unique())
        if labels != VALID_LABELS:
            raise ValueError(f"'is_fraud' doit contenir exactement {VALID_LABELS}, "
                     f"or j’ai trouvé {labels}")
       

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data_callable,
    )

    # ------------------- 2. ENTRAÎNEMENT & LOG MLflow -------------------
    def train_model_callable():
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        import mlflow
        import mlflow.sklearn

        # --- Charger ou simuler les données ---
        df = pd.DataFrame(
            {
                "transaction_id": np.arange(1, 5001),
                "amount": np.round(np.random.exponential(scale=50, size=5000), 2),
                "transaction_type": np.random.choice(
                    ["purchase", "withdrawal", "transfer"], size=5000, p=[0.7, 0.2, 0.1]
                ),
                "country": np.random.choice(["FR", "US", "DE", "ES", "IT"], size=5000),
                "is_fraud": np.random.choice([0, 1], size=5000, p=[0.98, 0.02]),
            }
        )

        df_encoded = pd.get_dummies(
            df[["amount", "transaction_type", "country"]], drop_first=True
        )
        X = df_encoded
        y = df["is_fraud"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- MLflow ---
        mlflow.set_experiment("fraud_detection_dag")
        with mlflow.start_run(run_name="rf_from_airflow"):
            mlflow.log_params({"n_estimators": 100, "max_depth": 10})
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)[:, 1]
            mlflow.log_metrics(
                {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_proba),
                }
            )
            mlflow.sklearn.log_model(rf, "model")  # log artefact

    train_task = PythonOperator(
        task_id="train_model_task",
        python_callable=train_model_callable,
    )

    # ------------------- 3. ORDRE DES TÂCHES -------------------
    validate_task >> train_task
